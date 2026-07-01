import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

def get_test_data(df_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Reproduce el split exacto del train.py para obtener el 20% de datos de prueba."""
    df = pd.read_csv(df_path, low_memory=False)
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    # El random_state=42 asegura que el corte sea idéntico al del entrenamiento
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test

def evaluar_modelo(dataset_name: str, modelo_nombre: str, X_test: pd.DataFrame, y_test: pd.Series):
    """Carga un modelo binario, realiza inferencia y exporta sus métricas."""
    ruta_modelo = MODEL_DIR / f"{dataset_name}_{modelo_nombre}.joblib"
    
    if not ruta_modelo.exists():
        print(f"[ERROR] No se encontro el modelo: {ruta_modelo.name}. Ejecutaste train.py?")
        return
        
    print(f"\nEvaluando: [{dataset_name}] {modelo_nombre}")
    pipeline = joblib.load(ruta_modelo)
    
    # Realizar predicciones
    y_pred = pipeline.predict(X_test)

    # Probabilidades para la curva ROC
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Generar reportes
    reporte = classification_report(y_test, y_pred)
    matriz = confusion_matrix(y_test, y_pred)

    # Heatmap de la Matriz de Confusión

    plt.figure(figsize=(6, 5))
    
    sns.heatmap(
    matriz,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Anemia", "Anemia"],
    yticklabels=["No Anemia", "Anemia"]
)
    
    plt.title(f"Matriz de Confusión - {dataset_name} ({modelo_nombre})")
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")

    plt.tight_layout()

    plt.savefig(
         FIGURES_DIR / f"confusion_{dataset_name}_{modelo_nombre}.png"
    )

    plt.close()

    # CURVA ROC-AUC

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="darkorange", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="navy")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Curva ROC - {dataset_name} ({modelo_nombre})")
    plt.legend(loc="lower right")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / f"roc_{dataset_name}_{modelo_nombre}.png"
)
    
    plt.close()

    # Gráfico de Importancia de Variables

    if modelo_nombre == "ArbolDecision":
        
        # Obtener el clasificador del Pipeline
        modelo = pipeline.named_steps["classifier"]

        # Obtener el preprocesador para recuperar los nombres de las variables
        preprocessor = pipeline.named_steps["preprocessor"]

         # Variables transformadas (después del OneHotEncoder)
        feature_names = preprocessor.get_feature_names_out()

        # Importancia de cada variable
        importancia = modelo.feature_importances_

        importancia_df = (
            pd.DataFrame({
                "Variable": feature_names,
                "Importancia": importancia
        })
        .sort_values(by="Importancia", ascending=False)
        .head(15)
    )
        
        plt.figure(figsize=(10, 6))

        sns.barplot(
            data=importancia_df,
            x="Importancia",
            y="Variable"
    )
        
        plt.title(f"Top 15 Variables Más Importantes\n{dataset_name}")
        plt.xlabel("Importancia")
        plt.ylabel("Variable")

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / f"variables_{dataset_name}_{modelo_nombre}.png"
    )
        
        plt.close()

    
    resultado_texto = f"=== EVALUACIÓN ACADÉMICA ===\n"
    resultado_texto += f"Dataset: {dataset_name}\n"
    resultado_texto += f"Algoritmo: {modelo_nombre}\n"
    resultado_texto += "="*28 + "\n"
    resultado_texto += f"\n1. MATRIZ DE CONFUSIÓN (TN, FP | FN, TP):\n{matriz}\n"
    resultado_texto += f"\n2. REPORTE DE CLASIFICACIÓN (Enfocado en Recall Clase 1):\n{reporte}\n"
    
    print(resultado_texto)
    
    # Exportar el reporte a un archivo plano
    ruta_reporte = REPORTS_DIR / f"reporte_{dataset_name}_{modelo_nombre}.txt"
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write(resultado_texto)
    print(f"[OK] Reporte guardado en: {ruta_reporte.relative_to(BASE_DIR)}")

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Iniciando Fase de Evaluación MLOps (Inferencia sobre conjunto Test)...")
    
    # 1. Cargar datos de prueba
    ruta_masivo = DATA_DIR / 'df_global_limpio.csv'
    X_test_mas, y_test_mas = get_test_data(ruta_masivo)
    
    ruta_regional = DATA_DIR / 'df_regiones_enriquecido.csv'
    X_test_reg, y_test_reg = get_test_data(ruta_regional)

    # 2. Evaluar Dataset Masivo (Nacional)
    evaluar_modelo("Masivo", "RegresionLogistica", X_test_mas, y_test_mas)
    evaluar_modelo("Masivo", "ArbolDecision", X_test_mas, y_test_mas)
    
    # 3. Evaluar Dataset Socio-Médico (SIEN Regional)
    evaluar_modelo("SocioMedico", "RegresionLogistica", X_test_reg, y_test_reg)
    evaluar_modelo("SocioMedico", "ArbolDecision", X_test_reg, y_test_reg)
    
    print("\n[INFO] Todas las evaluaciones finalizadas!")

if __name__ == '__main__':
    main()