"""
Script de Entrenamiento MLOps Optimizado (train.py)
---------------------------------------------------
Entrena modelos de clasificación (Regresión Logística y Árboles de Decisión)
utilizando dos datasets paralelos: Masivo (Global) y Socio-Médico (Regional).
Aplica preprocesamiento, balanceo de clases con SMOTE y Optimización de 
Hiperparámetros mediante GridSearchCV enfocándose en maximizar el RECALL.
"""

import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Usamos el Pipeline de imblearn para que SMOTE se aplique SOLO en el entrenamiento
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'models'

def build_pipeline(classifier, numeric_features, categorical_features) -> ImbPipeline:
    """Construye un pipeline estricto de MLOps: Limpieza -> Codificación -> SMOTE -> Modelo"""
    
    # 1. Transformaciones Numéricas: Imputar posibles nulos residuales y Escalar
    numeric_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 2. Transformaciones Categorizadas: Imputar con constante y aplicar One-Hot Encoding
    categorical_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='DESCONOCIDO')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Preprocesador global
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Ensamblar el Pipeline Completo incluyendo SMOTE
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', classifier)
    ])
    
    return pipeline

def ejecutar_entrenamiento_optimo(df_path: Path, dataset_name: str):
    """Carga los datos, define el espacio de búsqueda e inicia GridSearch optimizando para Recall."""
    print(f"\n=== Iniciando Optimización para Dataset: {dataset_name} ===")
    
    if not df_path.exists():
        print(f"[ERROR] No se encontró el archivo: {df_path}")
        return

    df = pd.read_csv(df_path, low_memory=False)
    
    if 'Target' not in df.columns:
        print("[ERROR] La columna 'Target' no existe en el DataFrame.")
        return

    X = df.drop(columns=['Target'])
    y = df['Target']
    
    # Identificar variables de manera dinámica
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    print("Dividiendo datos en entrenamiento y prueba (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Definición de los clasificadores base
    modelos_base = {
        'RegresionLogistica': LogisticRegression(max_iter=1000, random_state=42),
        'ArbolDecision': DecisionTreeClassifier(random_state=42)
    }
    
    # Definición de las mallas de hiperparámetros para la optimización
    # Nota: Como usamos un Pipeline, agregamos el prefijo 'classifier__' para mapear las variables
    param_grids = {
        'RegresionLogistica': {
            'classifier__C': [0.01, 0.1, 1, 10]
        },
        'ArbolDecision': {
            'classifier__max_depth': [5, 10, 15, None],
            'classifier__criterion': ['gini', 'entropy']
        }
    }
    
    for nombre, clf in modelos_base.items():
        print(f"\n--- Optimizando {nombre} con GridSearchCV ---")
        pipeline_base = build_pipeline(clf, numeric_features, categorical_features)
        
        # Configuramos la búsqueda en cuadrícula priorizando RECALL
        grid_search = GridSearchCV(
            estimator=pipeline_base,
            param_grid=param_grids[nombre],
            cv=5,
            scoring='recall',
            n_jobs=-1,
            verbose=1
        )
        
        print(f"Ejecutando GridSearch (cv=5) enfocado en Recall...")
        grid_search.fit(X_train, y_train)
        
        print(f"[MEJOR RECALL EN CV]: {grid_search.best_score_:.4f}")
        print(f"[MEJORES PARÁMETROS]: {grid_search.best_params_}")
        
        # El mejor modelo entrenado se extrae automáticamente
        mejor_modelo = grid_search.best_estimator_
        
        # Exportar el pipeline completo y optimizado
        ruta_modelo = MODEL_DIR / f"{dataset_name}_{nombre}.joblib"
        joblib.dump(mejor_modelo, ruta_modelo)
        print(f"[OK] Modelo optimizado guardado en: {ruta_modelo.name}")

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Optimizar Pipeline A (Masivo)
    ruta_masivo = DATA_DIR / 'df_global_limpio.csv'
    ejecutar_entrenamiento_optimo(ruta_masivo, "Masivo")
    
    # 2. Optimizar Pipeline B (Socio-Médico / Regional)
    ruta_regional = DATA_DIR / 'df_regiones_enriquecido.csv'
    ejecutar_entrenamiento_optimo(ruta_regional, "Regional")

if __name__ == '__main__':
    main()
