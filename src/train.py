"""
Script de Entrenamiento MLOps (train.py)
----------------------------------------
Entrena modelos de clasificación (Regresión Logística y Árboles de Decisión)
utilizando dos datasets paralelos: Masivo (Global) y Socio-Médico (Regional).
Aplica preprocesamiento y balanceo de clases con SMOTE.
"""

import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# ¡CRUCIAL! Usamos el Pipeline de imblearn, no el de sklearn, 
# para garantizar que SMOTE se aplique SOLO en el entrenamiento y no en Test/Producción.
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'models'

def build_pipeline(classifier, numeric_features, categorical_features) -> ImbPipeline:
    """Construye un pipeline estricto de MLOps: Limpieza -> Codificación -> SMOTE -> Modelo"""
    
    # 1. Transformaciones Numéricas: Imputar posibles nulos residuales y Escalar
    num_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Transformaciones Categóricas: Imputar nulos y One-Hot Encoding
    cat_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Ensamblaje del Preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numeric_features),
            ('cat', cat_transformer, categorical_features)
        ])

    # 4. Pipeline Final
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', classifier)
    ])
    
    return pipeline

def train_and_save(df_path: Path, dataset_name: str):
    """Función modular para cargar, dividir, entrenar y exportar modelos de un dataset."""
    print(f"\n=== Iniciando Entrenamiento Pipeline: {dataset_name} ===")
    print(f"Cargando dataset...")
    df = pd.read_csv(df_path, low_memory=False)
    
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    # Detección automática de tipos de columnas (con parche para Pandas 3.0/4.0)
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    print("Dividiendo datos 80/20...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    modelos = {
        'RegresionLogistica': LogisticRegression(max_iter=1000, random_state=42),
        'ArbolDecision': DecisionTreeClassifier(max_depth=10, random_state=42)
    }
    
    for nombre, clf in modelos.items():
        print(f"Entrenando {nombre} con SMOTE...")
        pipeline = build_pipeline(clf, numeric_features, categorical_features)
        pipeline.fit(X_train, y_train)
        
        # Exportar el modelo entrenado
        ruta_modelo = MODEL_DIR / f"{dataset_name}_{nombre}.joblib"
        joblib.dump(pipeline, ruta_modelo)
        print(f"[OK] Modelo guardado en: {ruta_modelo.name}")

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Entrenar Pipeline A (Masivo)
    ruta_masivo = DATA_DIR / 'df_global_limpio.csv'
    train_and_save(ruta_masivo, "Masivo")
    
    # 2. Entrenar Pipeline B (Socio-Médico Profundo)
    ruta_regional = DATA_DIR / 'df_regiones_enriquecido.csv'
    train_and_save(ruta_regional, "SocioMedico")
    
    print("\n[OK] Fase de Entrenamiento finalizada con exito! Modelos listos en carpeta /models.")

if __name__ == '__main__':
    main()
