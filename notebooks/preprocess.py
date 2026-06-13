"""
Módulo de Preprocesamiento de Datos (ETL)
-----------------------------------------
Este script contiene las funciones core para la limpieza, armonización y 
Feature Engineering de los datos crudos.

Su propósito es ser importado en pipelines de producción o en otros scripts
(como API o entrenamiento continuo), abstrayendo la lógica del Jupyter Notebook.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple

def armonizar_cabeceras(df: pd.DataFrame, tipo_origen: str) -> pd.DataFrame:
    """
    Renombra las columnas de los datasets nacionales para estandarizarlas
    con la estructura base del MINSA-SIEN.
    
    Args:
        df (pd.DataFrame): Dataset crudo.
        tipo_origen (str): 'ANEMIA_DA' o 'TAMIZAJE'.
        
    Returns:
        pd.DataFrame: Dataset con cabeceras armonizadas.
    """
    if tipo_origen == 'ANEMIA_DA':
        map_cols = {
            'EDAD_REGISTRO': 'EdadMeses',
            'GENERO': 'Sexo',
            'DIAGNOSTICO': 'Dx_anemia',
            'LATITUD': 'Latitud_Origen',
            'LONGITUD': 'Longitud_Origen'
        }
    elif tipo_origen == 'TAMIZAJE':
        map_cols = {
            'Edad': 'EdadMeses',
            'Diagnostico': 'Dx_anemia'
        }
    else:
        return df # SIEN no requiere mapeo
        
    return df.rename(columns=map_cols)


def estandarizar_target(val: str) -> float:
    """
    Convierte textos diagnósticos dispares (SIEN, CIE-10) en un target binario.
    
    Args:
        val (str): Diagnóstico original.
        
    Returns:
        float: 0.0 (Sano) o 1.0 (Anemia). Retorna NaN si es irreconocible.
    """
    val_str = str(val).strip().upper()
    if pd.isna(val) or val_str == 'NAN':
        return np.nan
        
    if 'NORMAL' in val_str:
        return 0.0
    if 'ANEMIA' in val_str or '85018' in val_str:
        return 1.0
        
    return np.nan


def aplicar_filtros_clinicos(df: pd.DataFrame, es_tamizaje: bool = False) -> pd.DataFrame:
    """
    Aplica filtros de alcance del proyecto (Edad <= 59.99 meses) y 
    limpieza de outliers biológicos (Hemoglobina entre 4 y 20).
    
    Args:
        df (pd.DataFrame): Dataset a limpiar.
        es_tamizaje (bool): Flag para aplicar corrección de 'Tipo_edad' en años.
        
    Returns:
        pd.DataFrame: Dataset filtrado.
    """
    df = df.copy()
    
    # 1. Corrección especial para archivo de Tamizaje (Edades en Años a Meses)
    if es_tamizaje and 'Tipo_edad' in df.columns:
        filtro_anos = df['Tipo_edad'] == 'A'
        df.loc[filtro_anos, 'EdadMeses'] = pd.to_numeric(df.loc[filtro_anos, 'EdadMeses'], errors='coerce') * 12
        
    # 2. Asegurar tipo numérico y filtrar edad (Menores de 5 años)
    df['EdadMeses'] = pd.to_numeric(df['EdadMeses'], errors='coerce')
    df = df[df['EdadMeses'] <= 59.99]
    
    # 3. Limpieza de Outliers de Hemoglobina (si existe la columna)
    if 'Hemoglobina' in df.columns:
        df['Hemoglobina'] = pd.to_numeric(df['Hemoglobina'], errors='coerce')
        df = df[(df['Hemoglobina'] >= 4) & (df['Hemoglobina'] <= 20)]
        
    # 4. Aplicar Target Binario
    if 'Dx_anemia' in df.columns:
        df['Target'] = df['Dx_anemia'].apply(estandarizar_target)
        
    return df


def enriquecer_geografia(df: pd.DataFrame, df_ubigeo: pd.DataFrame, tipo_origen: str) -> pd.DataFrame:
    """
    Cruza el dataset clínico con el padrón de INEI para inyectar altitud, 
    índices de pobreza y desarrollo humano.
    
    Args:
        df (pd.DataFrame): Dataset clínico.
        df_ubigeo (pd.DataFrame): Padrón estandarizado de distritos.
        tipo_origen (str): 'SIEN', 'TAMIZAJE' o 'ANEMIA_DA'.
        
    Returns:
        pd.DataFrame: Dataset enriquecido (Merge Left).
    """
    columnas_a_inyectar = ['altitude', 'latitude', 'longitude', 'idh_2019', 'pct_pobreza_total']
    
    if tipo_origen == 'SIEN':
        # SIEN almacena el código INEI en la columna mal llamada 'UbigeoREN'
        df['UbigeoREN'] = pd.to_numeric(df['UbigeoREN'], errors='coerce')
        df_merged = df.merge(
            df_ubigeo[['inei'] + columnas_a_inyectar],
            left_on='UbigeoREN', right_on='inei', how='left'
        )
    elif tipo_origen == 'TAMIZAJE':
        # TAMIZAJE utiliza 'id_ubigeo' numérico compatible con INEI
        df['id_ubigeo'] = pd.to_numeric(df['id_ubigeo'], errors='coerce')
        df_merged = df.merge(
            df_ubigeo[['inei'] + columnas_a_inyectar],
            left_on='id_ubigeo', right_on='inei', how='left'
        )
    elif tipo_origen == 'ANEMIA_DA':
        # ANEMIA_DA carece de IDs; se cruza por coincidencia de texto
        df['DEPARTAMENTO'] = df['DEPARTAMENTO'].astype(str).str.upper().str.strip()
        df['PROVINCIA'] = df['PROVINCIA'].astype(str).str.upper().str.strip()
        df['DISTRITO'] = df['DISTRITO'].astype(str).str.upper().str.strip()
        
        df_merged = df.merge(
            df_ubigeo[['departamento', 'provincia', 'distrito'] + columnas_a_inyectar],
            left_on=['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO'],
            right_on=['departamento', 'provincia', 'distrito'],
            how='left'
        )
    else:
        return df

    return df_merged


def ejecutar_pipeline_etl(rutas_sien: List[Path], ruta_da: Path, ruta_tam: Path, ruta_ubigeo: Path, dir_salida: Path) -> None:
    """
    Función orquestadora (Factory) que ejecuta todo el flujo ETL de principio a fin,
    replicando el proceso del notebook exploratorio.
    """
    print("Iniciando Pipeline de Preprocesamiento (Producción)...")
    
    # 1. Preparar Diccionario Geográfico Base
    df_ubigeo = pd.read_csv(ruta_ubigeo, low_memory=False)
    df_ubigeo['inei'] = pd.to_numeric(df_ubigeo['inei'], errors='coerce')
    for col in ['departamento', 'provincia', 'distrito']:
        df_ubigeo[col] = df_ubigeo[col].astype(str).str.upper().str.strip()

    dfs_procesados_sien = []
    
    # 2. Procesar Regiones SIEN
    print("Procesando archivos regionales SIEN...")
    for ruta in rutas_sien:
        df_sien = pd.read_csv(ruta, low_memory=False).drop_duplicates()
        df_sien = aplicar_filtros_clinicos(df_sien, es_tamizaje=False)
        df_sien = enriquecer_geografia(df_sien, df_ubigeo, 'SIEN')
        dfs_procesados_sien.append(df_sien)
        
    df_regiones_enriquecido = pd.concat(dfs_procesados_sien, ignore_index=True)
    df_regiones_enriquecido = df_regiones_enriquecido.dropna(subset=['Target'])

    # 3. Procesar Archivos Nacionales
    print("Procesando archivos nacionales...")
    df_da = pd.read_csv(ruta_da, sep=';', low_memory=False).drop_duplicates()
    df_da = armonizar_cabeceras(df_da, 'ANEMIA_DA')
    df_da = aplicar_filtros_clinicos(df_da, es_tamizaje=False)
    df_da = enriquecer_geografia(df_da, df_ubigeo, 'ANEMIA_DA')
    
    df_tam = pd.read_csv(ruta_tam, low_memory=False).drop_duplicates()
    df_tam = armonizar_cabeceras(df_tam, 'TAMIZAJE')
    df_tam = aplicar_filtros_clinicos(df_tam, es_tamizaje=True)
    df_tam = enriquecer_geografia(df_tam, df_ubigeo, 'TAMIZAJE')

    # 4. Consolidar Dataset Global Limpio
    columnas_comunes = ['EdadMeses', 'Sexo', 'Target', 'altitude', 'latitude', 'longitude', 'idh_2019', 'pct_pobreza_total']
    
    df_global_limpio = pd.concat([
        df_regiones_enriquecido[[c for c in columnas_comunes if c in df_regiones_enriquecido.columns]],
        df_da[[c for c in columnas_comunes if c in df_da.columns]],
        df_tam[[c for c in columnas_comunes if c in df_tam.columns]]
    ], ignore_index=True)
    
    df_global_limpio = df_global_limpio.dropna(subset=columnas_comunes)

    # --- NUEVO: FEATURE SELECTION (PODA DE COLUMNAS) ---
    # Eliminamos columnas crudas (IDs, nombres, fechas) que ya fueron explotadas 
    # en el preprocesamiento y que causarían problemas de memoria en el Machine Learning.
    columnas_basura = [
        'Microred', 'EESSS', 'Dist_EESS', 'UbigeoPN', 'ProvinciaPN', 'DistritoPN', 
        'CentroPobladoPN', 'FechaHemoglobina', 'FechaAtencion', 'FechaNacimiento', 
        'DistritoREN', 'UbigeoREN', 'DISTRITO', 'PROVINCIA', 'DEPARTAMENTO', 'id_ubigeo'
    ]
    df_regiones_enriquecido = df_regiones_enriquecido.drop(columns=columnas_basura, errors='ignore')

    # 5. Exportar
    dir_salida.mkdir(parents=True, exist_ok=True)
    df_regiones_enriquecido.to_csv(dir_salida / 'df_regiones_enriquecido.csv', index=False)
    df_global_limpio.to_csv(dir_salida / 'df_global_limpio.csv', index=False)
    print("Pipeline ejecutado con éxito. Archivos exportados a data/processed/.")

if __name__ == '__main__':
    from config import (
        DIR_SIEN_REGIONAL, 
        FILE_ANEMIA_DA, 
        FILE_TAMIZAJE, 
        REGIONES_SELECCIONADAS,
        PROCESSED_DIR,
        FILE_UBIGEO
    )
    
    rutas_regionales = [DIR_SIEN_REGIONAL / r for r in REGIONES_SELECCIONADAS]
    
    ejecutar_pipeline_etl(
        rutas_sien=rutas_regionales,
        ruta_da=FILE_ANEMIA_DA,
        ruta_tam=FILE_TAMIZAJE,
        ruta_ubigeo=FILE_UBIGEO,
        dir_salida=PROCESSED_DIR
    )
