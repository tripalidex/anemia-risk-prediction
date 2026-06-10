from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Rutas a los diferentes orígenes de datos
DIR_SIEN_REGIONAL = RAW_DIR
FILE_ANEMIA_DA = RAW_DIR / "ANEMIA_DA.csv"
FILE_TAMIZAJE = RAW_DIR / "TB_DIGTEL_ANEMIA_TAMIZAJE.csv"
FILE_UBIGEO = RAW_DIR / "ubigeo" / "ubigeo_distrito.csv"

# Regiones seleccionadas para el SIEN (representatividad geográfica)
REGIONES_SELECCIONADAS = [
    "Niños SAN MARTIN.csv",
    "Niños TACNA.csv",
    "Niños PUNO.csv",
    "Niños CUSCO.csv",
    "Niños LORETO.csv",
    "Niños PIURA.csv"
]

# Diccionarios de mapeo para resolver la incompatibilidad de columnas
# Objetivo: Llevar todos a [Sexo, EdadMeses, Dx_anemia, Departamento]
MAPEO_ANEMIA_DA = {
    "GENERO": "Sexo",
    "EDAD_REGISTRO": "EdadMeses",
    "DIAGNOSTICO": "Dx_anemia",
    "DEPARTAMENTO": "Departamento",
    "LONGITUD": "Longitud",
    "LATITUD": "Latitud",
    "GRADO_SEVERIDAD": "Severidad"
}

MAPEO_TAMIZAJE = {
    "Sexo": "Sexo",
    "Edad": "EdadMeses",
    "Diagnostico": "Dx_anemia",
    "id_ubigeo": "Ubigeo"
}

# Columnas finales esperadas tras la unificación
COLUMNAS_FINALES = [
    "Fuente", # Para saber de qué CSV vino
    "Sexo", 
    "EdadMeses", 
    "Departamento",
    "Dx_anemia",
    "Hemoglobina", # Solo en SIEN, el resto se imputará
    "AlturaREN",   # Solo en SIEN, el resto se imputará geográficamente
    "Juntos",      # Solo en SIEN
    "Qaliwarma",   # Solo en SIEN
    "SIS"          # Solo en SIEN
]
