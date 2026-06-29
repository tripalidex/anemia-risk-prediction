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
from typing import List


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
        return df  # SIEN no requiere mapeo

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
    Aplica filtros de alcance del proyecto:
    - Edad <= 59.99 meses.
    - Hemoglobina entre 4 y 20, solo como filtro de calidad.
    - Construcción del Target binario a partir de Dx_anemia.

    Importante:
    Hemoglobina y Dx_anemia pueden usarse en esta fase de limpieza,
    pero no deben quedar en el dataset final de entrenamiento.

    Args:
        df (pd.DataFrame): Dataset a limpiar.
        es_tamizaje (bool): Flag para aplicar corrección de 'Tipo_edad' en años.

    Returns:
        pd.DataFrame: Dataset filtrado.
    """
    df = df.copy()

    # 1. Corrección especial para archivo de Tamizaje.
    # Si Tipo_edad = 'A', la edad viene en años y se convierte a meses.
    if es_tamizaje and 'Tipo_edad' in df.columns:
        filtro_anos = df['Tipo_edad'] == 'A'
        df.loc[filtro_anos, 'EdadMeses'] = (
            pd.to_numeric(df.loc[filtro_anos, 'EdadMeses'], errors='coerce') * 12
        )

    # 2. Asegurar tipo numérico y filtrar menores de 5 años.
    if 'EdadMeses' not in df.columns:
        raise ValueError("No se encontró la columna obligatoria 'EdadMeses'.")

    df['EdadMeses'] = pd.to_numeric(df['EdadMeses'], errors='coerce')
    df = df[df['EdadMeses'] <= 59.99]

    # 3. Limpieza de outliers biológicos de Hemoglobina.
    # Se usa solo como filtro clínico, no como variable predictora final.
    if 'Hemoglobina' in df.columns:
        df['Hemoglobina'] = pd.to_numeric(df['Hemoglobina'], errors='coerce')
        df = df[(df['Hemoglobina'] >= 4) & (df['Hemoglobina'] <= 20)]

    # 4. Aplicar Target binario.
    # Dx_anemia se usa para crear Target, pero luego se elimina del dataset final.
    if 'Dx_anemia' in df.columns:
        df['Target'] = df['Dx_anemia'].apply(estandarizar_target)

    return df


def enriquecer_geografia(
    df: pd.DataFrame,
    df_ubigeo: pd.DataFrame,
    tipo_origen: str
) -> pd.DataFrame:
    """
    Cruza el dataset clínico con el padrón de INEI para inyectar altitud,
    índices de pobreza y desarrollo humano.

    Args:
        df (pd.DataFrame): Dataset clínico.
        df_ubigeo (pd.DataFrame): Padrón estandarizado de distritos.
        tipo_origen (str): 'SIEN', 'TAMIZAJE' o 'ANEMIA_DA'.

    Returns:
        pd.DataFrame: Dataset enriquecido.
    """
    columnas_a_inyectar = [
        'altitude',
        'latitude',
        'longitude',
        'idh_2019',
        'pct_pobreza_total'
    ]

    if tipo_origen == 'SIEN':
        # SIEN almacena el código INEI en la columna mal llamada 'UbigeoREN'.
        if 'UbigeoREN' not in df.columns:
            raise ValueError("No se encontró la columna 'UbigeoREN' para el origen SIEN.")

        df['UbigeoREN'] = pd.to_numeric(df['UbigeoREN'], errors='coerce')

        df_merged = df.merge(
            df_ubigeo[['inei'] + columnas_a_inyectar],
            left_on='UbigeoREN',
            right_on='inei',
            how='left'
        )

    elif tipo_origen == 'TAMIZAJE':
        # TAMIZAJE utiliza 'id_ubigeo' numérico compatible con INEI.
        if 'id_ubigeo' not in df.columns:
            raise ValueError("No se encontró la columna 'id_ubigeo' para el origen TAMIZAJE.")

        df['id_ubigeo'] = pd.to_numeric(df['id_ubigeo'], errors='coerce')

        df_merged = df.merge(
            df_ubigeo[['inei'] + columnas_a_inyectar],
            left_on='id_ubigeo',
            right_on='inei',
            how='left'
        )

    elif tipo_origen == 'ANEMIA_DA':
        # ANEMIA_DA carece de IDs; se cruza por coincidencia de texto.
        columnas_texto = ['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO']

        for col in columnas_texto:
            if col not in df.columns:
                raise ValueError(f"No se encontró la columna '{col}' para el origen ANEMIA_DA.")

            df[col] = df[col].astype(str).str.upper().str.strip()

        df_merged = df.merge(
            df_ubigeo[['departamento', 'provincia', 'distrito'] + columnas_a_inyectar],
            left_on=['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO'],
            right_on=['departamento', 'provincia', 'distrito'],
            how='left'
        )

    else:
        return df

    return df_merged


def eliminar_columnas_con_fuga(df: pd.DataFrame, nombre_df: str) -> pd.DataFrame:
    """
    Elimina columnas que no deben llegar al entrenamiento del modelo.

    Hemoglobina y Dx_anemia se consideran fuga de datos:
    - Hemoglobina está directamente ligada al diagnóstico clínico de anemia.
    - Dx_anemia es la columna diagnóstica usada para construir Target.

    Args:
        df (pd.DataFrame): Dataset procesado.
        nombre_df (str): Nombre del dataset para mensajes de validación.

    Returns:
        pd.DataFrame: Dataset sin columnas con fuga de datos.
    """
    columnas_basura = [
        # Identificadores, textos administrativos y columnas crudas.
        'Microred',
        'EESSS',
        'Dist_EESS',
        'UbigeoPN',
        'ProvinciaPN',
        'DistritoPN',
        'CentroPobladoPN',
        'FechaHemoglobina',
        'FechaAtencion',
        'FechaNacimiento',
        'DistritoREN',
        'UbigeoREN',
        'DISTRITO',
        'PROVINCIA',
        'DEPARTAMENTO',
        'id_ubigeo',
        'inei',
        'departamento',
        'provincia',
        'distrito',
        'Latitud_Origen',
        'Longitud_Origen',
        'Tipo_edad',

        # Columnas eliminadas específicamente por Data Leakage.
        'Hemoglobina',
        'Hbc',
        'Dx_anemia'
    ]

    df_limpio = df.drop(columns=columnas_basura, errors='ignore')

    columnas_prohibidas = [
        'Hemoglobina',
        'Hbc',
        'Dx_anemia'
    ]

    for col in columnas_prohibidas:
        if col in df_limpio.columns:
            raise ValueError(
                f"Data Leakage detectado: la columna '{col}' sigue en {nombre_df}."
            )

    columnas_sospechosas = [
        col for col in df_limpio.columns
        if any(
            patron in col.lower()
            for patron in ['hemo','hb','hbc','anemia', 'diagnostico', 'dx']
        )
        and col != 'Target'
    ]

    if columnas_sospechosas:
        print(f"Advertencia: revisar posibles columnas sospechosas en {nombre_df}:")
        print(columnas_sospechosas)

    print(f"Validación Data Leakage OK en {nombre_df}.")
    return df_limpio


def ejecutar_pipeline_etl(
    rutas_sien: List[Path],
    ruta_da: Path,
    ruta_tam: Path,
    ruta_ubigeo: Path,
    dir_salida: Path
) -> None:
    """
    Función orquestadora que ejecuta todo el flujo ETL de principio a fin,
    replicando el proceso del notebook exploratorio.
    """
    print("Iniciando Pipeline de Preprocesamiento...")

    # 1. Preparar diccionario geográfico base.
    df_ubigeo = pd.read_csv(ruta_ubigeo, low_memory=False)

    columnas_ubigeo_requeridas = [
        'inei',
        'departamento',
        'provincia',
        'distrito',
        'altitude',
        'latitude',
        'longitude',
        'idh_2019',
        'pct_pobreza_total'
    ]

    faltantes_ubigeo = [
        col for col in columnas_ubigeo_requeridas
        if col not in df_ubigeo.columns
    ]

    if faltantes_ubigeo:
        raise ValueError(
            f"Faltan columnas obligatorias en el archivo de ubigeo: {faltantes_ubigeo}"
        )

    df_ubigeo['inei'] = pd.to_numeric(df_ubigeo['inei'], errors='coerce')

    for col in ['departamento', 'provincia', 'distrito']:
        df_ubigeo[col] = df_ubigeo[col].astype(str).str.upper().str.strip()

    dfs_procesados_sien = []

    # 2. Procesar regiones SIEN.
    print("Procesando archivos regionales SIEN...")

    for ruta in rutas_sien:
        print(f"Procesando: {ruta.name}")

        df_sien = pd.read_csv(ruta, low_memory=False).drop_duplicates()
        df_sien = aplicar_filtros_clinicos(df_sien, es_tamizaje=False)
        df_sien = enriquecer_geografia(df_sien, df_ubigeo, 'SIEN')

        dfs_procesados_sien.append(df_sien)

    if not dfs_procesados_sien:
        raise ValueError("No se encontraron archivos SIEN para procesar.")

    df_regiones_enriquecido = pd.concat(dfs_procesados_sien, ignore_index=True)

    if 'Target' not in df_regiones_enriquecido.columns:
        raise ValueError("No se generó la columna 'Target' en los datos SIEN.")

    df_regiones_enriquecido = df_regiones_enriquecido.dropna(subset=['Target'])

    # 3. Procesar archivos nacionales.
    print("Procesando archivo nacional ANEMIA_DA...")

    df_da = pd.read_csv(ruta_da, sep=';', low_memory=False).drop_duplicates()
    df_da = armonizar_cabeceras(df_da, 'ANEMIA_DA')
    df_da = aplicar_filtros_clinicos(df_da, es_tamizaje=False)
    df_da = enriquecer_geografia(df_da, df_ubigeo, 'ANEMIA_DA')

    print("Procesando archivo nacional TAMIZAJE...")

    df_tam = pd.read_csv(ruta_tam, low_memory=False).drop_duplicates()
    df_tam = armonizar_cabeceras(df_tam, 'TAMIZAJE')
    df_tam = aplicar_filtros_clinicos(df_tam, es_tamizaje=True)
    df_tam = enriquecer_geografia(df_tam, df_ubigeo, 'TAMIZAJE')

    # 4. Consolidar dataset global limpio.
    columnas_comunes = [
        'EdadMeses',
        'Sexo',
        'Target',
        'altitude',
        'latitude',
        'longitude',
        'idh_2019',
        'pct_pobreza_total'
    ]

    df_global_limpio = pd.concat(
        [
            df_regiones_enriquecido[
                [c for c in columnas_comunes if c in df_regiones_enriquecido.columns]
            ],
            df_da[
                [c for c in columnas_comunes if c in df_da.columns]
            ],
            df_tam[
                [c for c in columnas_comunes if c in df_tam.columns]
            ]
        ],
        ignore_index=True
    )

    # Se eliminan filas incompletas solo en las columnas finales del modelo.
    columnas_presentes_global = [
        c for c in columnas_comunes
        if c in df_global_limpio.columns
    ]

    df_global_limpio = df_global_limpio.dropna(subset=columnas_presentes_global)

    # 5. Feature Selection y control de Data Leakage.
    print("Aplicando control de Data Leakage...")

    df_regiones_enriquecido = eliminar_columnas_con_fuga(
        df_regiones_enriquecido,
        nombre_df='df_regiones_enriquecido'
    )

    df_global_limpio = eliminar_columnas_con_fuga(
        df_global_limpio,
        nombre_df='df_global_limpio'
    )

    # 6. Validaciones finales.
    if 'Target' not in df_regiones_enriquecido.columns:
        raise ValueError("El dataset regional quedó sin columna Target.")

    if 'Target' not in df_global_limpio.columns:
        raise ValueError("El dataset global quedó sin columna Target.")

    print("Distribución del Target en df_regiones_enriquecido:")
    print(df_regiones_enriquecido['Target'].value_counts(dropna=False))
    print(df_regiones_enriquecido['Target'].value_counts(normalize=True, dropna=False))

    print("Distribución del Target en df_global_limpio:")
    print(df_global_limpio['Target'].value_counts(dropna=False))
    print(df_global_limpio['Target'].value_counts(normalize=True, dropna=False))

    print("Columnas finales df_regiones_enriquecido:")
    print(df_regiones_enriquecido.columns.tolist())

    print("Columnas finales df_global_limpio:")
    print(df_global_limpio.columns.tolist())

    # 7. Exportar.
    dir_salida.mkdir(parents=True, exist_ok=True)

    df_regiones_enriquecido.to_csv(
        dir_salida / 'df_regiones_enriquecido.csv',
        index=False
    )

    df_global_limpio.to_csv(
        dir_salida / 'df_global_limpio.csv',
        index=False
    )

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

    rutas_regionales = [
        DIR_SIEN_REGIONAL / r
        for r in REGIONES_SELECCIONADAS
    ]

    ejecutar_pipeline_etl(
        rutas_sien=rutas_regionales,
        ruta_da=FILE_ANEMIA_DA,
        ruta_tam=FILE_TAMIZAJE,
        ruta_ubigeo=FILE_UBIGEO,
        dir_salida=PROCESSED_DIR
    )