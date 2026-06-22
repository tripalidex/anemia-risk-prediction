import numpy as np
import pandas as pd

from config import (
    DIR_SIEN_REGIONAL,
    FILE_ANEMIA_DA,
    FILE_TAMIZAJE,
    FILE_UBIGEO,
    MAPEO_ANEMIA_DA,
    MAPEO_TAMIZAJE,
    PROCESSED_DIR,
    REGIONES_SELECCIONADAS,
)

COLUMNAS_BASURA = ["Mircroed", "EESSS", "Dist_EESSS", "UbigeoPN", "ProvinciaPN", "DistritoPN",
                   "CentroPobladoPN", "FechaHemoglobina", "FechaAtencion", "FechaNacimiento",
                   "DistritoREN", "UbigeoREN", "DISTRITO", "PROVINCIA", "DEPARTAMENTO", "id_ubigeo",]

COLUMNAS_FUGA_DATOS = ["Hemoglobina", "Dx_anemia"]

COLUMNAS_GEO = ["altitude", "latitude", "longitude", "idh_2019", "pct_pobreza_total"]

COLUMNAS_COMUNES = ["EdadMeses", "Sexo", "Target"] + COLUMNAS_GEO

def cargar_datos() -> tuple[list[pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Iniciando carga y limpieza de duplicados...")

    dfs_sien = []
    for region in REGIONES_SELECCIONADAS:
        df_temp = pd.read_csv(DIR_SIEN_REGIONAL / region, low_memory=False)
        antes = len(df_temp)
        df_temp = df_temp.drop_duplicates()
        print(f"{region}: {antes - len(df_temp)} duplicados eliminados")
        dfs_sien.append(df_temp)

    df_da = pd.read_csv(FILE_ANEMIA_DA, sep=";", low_memory=False).drop_duplicates

    df_tamizaje = pd.read_csv(FILE_TAMIZAJE, low_memory=False).drop_duplicates
    print(f"ANMEIA_DA cargado. Shape: {df_da.shape}")
    print(f"TAMIZAJE cargado. Shape: {df_tamizaje.shape}")

    df_ubigeo = pd.read_csv(FILE_UBIGEO, low_memory=False)

    return dfs_sien, df_da, df_tamizaje, df_ubigeo

def armonizar_cabeceras(df_da: pd.DataFrame, df_tamizaje: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_da = df_da.rename(columns=MAPEO_ANEMIA_DA)
    df_tamizaje = df_tamizaje.rename(columns=MAPEO_TAMIZAJE)
    print("Cabeceras de ANEMIA_DA y TAMIZAJE armonizadas al estándar SIEN.")
    return df_da, df_tamizaje

def _aplicar_filtros(df: pd.DataFrame, nombre: str) -> pd.DataFrame: 
    inicial = len(df)

    df("EdadMeses") = pd.to_numeric(df["EdadMeses"], errors="coerce")
    df = df[df["EdadMeses"] <= 59.99]

    if "Hemoglobina" in df.columns:
        df["Hemoglobina"] = pd.to_numeric(df["Hemoglobina"], errors="coerce")
        df = df[(df["Hemoglobina"] >= 4) & (df["Hemoglobina"] <= 20)]

    print(f"Filtros en {nombre}: retenidos {len(df)} de {inicial} registros.")
    return df

def aplicar_filtros_alcance(
        dfs_sien: list[pd.DataFrame], df_da: pd.DataFrame, df_tamizaje: pd.DataFrame
) -> tuple[list[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    
    if "Tipo_edad" in df_tamizaje.columns:
        filtro_anos = df_tamizaje["Tipo_edad"] == "A"
        df_tamizaje.loc[filtro_anos, "EdadMeses"] = df_tamizaje.loc[filtro_anos, "EdadMeses"] * 12

        dfs_sien = [_aplicar_filtros(df, f"SIEN REGIÓN {1 + 1}") for i, df in enumerate(dfs_sien)]
        df_da = _aplicar_filtros(df_da, "ANEMIA_DA")
        df_tamizaje = _aplicar_filtros(df_tamizaje, "TAMIZAJE")
        return dfs_sien, df_da, df_tamizaje
    
def _estandarizar_target(val) -> float:
    val = str(val).strip().upper()
    if pd.isna(val) or val == "NAN":
        return np.nan
    if "NORMAL" in val:
        return 0
    if "ANEMIA" in val or "85018" in val:
        return 1
    return np.nan

def construir_target(
        dfs_sien: list[pd.DataFrame], df_da: pd.DataFrame, df_tamizaje: pd.DataFrame
) -> tuple[list[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    for df in dfs_sien:
        df["Target"] = df["Dx_anemia"].apply(_estandarizar_target)
    df_da["Target"] = df_da["Dx_anemia"].apply(_estandarizar_target)
    df_tamizaje["Target"] = df_tamizaje["Dx_anemia"].apply(_estandarizar_target)
    print("Target binario estandarizado en todos los datasets.")
    return dfs_sien, df_da, df_tamizaje

def enriquecer_geografia(
    dfs_sien: list[pd.DataFrame],
    df_da: pd.DataFrame,
    df_tamizaje: pd.DataFrame,
    df_ubigeo: pd.DataFrame,
) -> tuple[list[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    df_ubigeo["inei"] = pd.to_numeric(df_ubigeo["inei"], errors="coerce")
    df_ubigeo["renlec"] = pd.to_numeric(df_ubigeo["renlec"], errors="coerce")
    for col in ("departamento", "provincia", "distrito"):
        df_ubigeo[col] = df_ubigeo[col].str.upper().str.strip()

    for i, df in enumerate(dfs_sien):
        df["UbigeoREN"] = pd.to_numeric(df["UbigeoREN"], errors="coerce")
        dfs_sien[i] = df.merge(
            df_ubigeo[["inei"] + COLUMNAS_GEO],
            left_on="UbigeoREN", right_on="inei", how="left"
        )

    df_tamizaje["id_ubigeo"] = pd.to_numeric(df_tamizaje["id_ubigeo"], errors="coerce")
    df_tamizaje = df_tamizaje.merge(
        df_ubigeo[["inei"] + COLUMNAS_GEO],
        left_on="id_ubigeo", right_on="inei", how="left"
    )

    for col in ("DEPARTAMENTO", "PROVINCIA", "DISTRITO"):
        df_da[col] = df_da[col].str.upper().str.strip()
    df_da = df_da.merge(
        df_ubigeo[["departamento", "provincia", "distrito"] + COLUMNAS_GEO],
        left_on=["DEPARTAMENTO", "PROVINCIA", "DISTRITO"],
        right_on=["departamento", "provincia", "distrito"],
        how="left"
    )

    print("Bases de datos enriquecidas con variable soscioeconómicas y especiales.")
    return dfs_sien, df_da, df_tamizaje
        
def construir_dataset_regiones(
    dfs_sien: list[pd.DataFrame], incluir_columnas_fuga: bool = False
) -> pd.DataFrame:
    df = pd.concat(dfs_sien, ignore_index=True)
    df = df.dropna(subset=["Target"])
    df = df.drop(columns=COLUMNAS_BASURA, errors="ignore")

    if not incluir_columnas_fuga:
        eliminadas = [c for c in COLUMNAS_FUGA_DATOS if c in df.columns]
        df = df.drop(columns=eliminadas, errors="ignore")
        if eliminadas:
            print(f"[FIX FUGA] Columnas excluidas del dataset de modelado : {eliminadas}")

    print(f"Dataset Regional Enriquecido creado. Shape: {df.shape}")
    return df

def construir_dataset_global(
        df_regiones_sin_podar: pd.DataFrame, df_da: pd.DataFrame, df_tamizaje:pd.DataFrame
) -> pd.DataFrame:
    
    df_sien_common = df_regiones_sin_podar[[c for c in COLUMNAS_COMUNES if c in df_regiones_sin_podar.columns]]
    df_da_common = df_da[[c for c in COLUMNAS_COMUNES if c in df_da.columns]]
    df_tam_common = df_tamizaje[[c for c in COLUMNAS_COMUNES if c in df_tamizaje.columns]]

    df_global = pd.concat([df_sien_common, df_da_common, df_tam_common], ignore_index=True)
    df_global = df_global.dropna(subset=COLUMNAS_COMUNES)

    print(f"Dataset Global Limpio creado. Shape: {df_global.shape}")
    return df_global

def main(incluir_columns_fuga: bool = False) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dfs_sien, df_da, df_tamizaje, df_ubigeo = cargar_datos()
    df_da, df_tamizaje = armonizar_cabeceras(df_da, df_tamizaje)
    dfs_sien, df_da, df_tamizaje = aplicar_filtros_alcance(dfs_sien, df_da, df_tamizaje)
    dfs_sien, df_da, df_tamizaje = enriquecer_geografia(dfs_sien, df_da, df_tamizaje, df_ubigeo)

    df_regiones_sin_podar = pd.concat(dfs_sien, ignore_index=True).dropna(subset=["Target"])
    df_global = construir_dataset_global(df_regiones_sin_podar, df_da, df_tamizaje)

    df_regiones = construir_dataset_regiones(dfs_sien, incluir_columnas_fuga=incluir_columns_fuga)

    print("Iniciando Exportación de datasets listos para Machine Learning...")
    sufijo = "_CONFUGA" if incluir_columns_fuga else ""
    ruta_regiones = PROCESSED_DIR / f"df_regiones_enriquecido{sufijo}.csv"
    ruta_global = PROCESSED_DIR / f"df_global_limpo.csv"

    df_regiones.to_csv(ruta_regiones, index=False)
    df_global.to_csv(ruta_global, index=False)

    print(f"[OK] {ruta_regiones.name} exportado correctamente.")
    print(f"[OK] {ruta_global.name} exportado correctamente")

if __name__ == "__main__":
    main()
