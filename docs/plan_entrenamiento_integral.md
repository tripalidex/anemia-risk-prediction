# Plan de Entrenamiento Integral (Uso Simultáneo de Datasets)

Alineado con las directrices del "Avance 01" y la justificación de alta interpretabilidad médica, el presente plan detalla la arquitectura para la Fase 3, donde entrenaremos algoritmos clásicos utilizando **ambos datasets preprocesados de manera paralela** para obtener conclusiones profundas desde dos frentes distintos.

---

## 1. Estrategia de Modelado Dual

Para maximizar el impacto de la investigación, el entrenamiento se dividirá en dos pipelines paralelos, cada uno alimentado por un dataset distinto, pero sujetos a las mismas reglas algorítmicas:

### Pipeline A: "El Modelo Masivo"
*   **Dataset:** `df_global_limpio.csv` (~400,000 registros).
*   **Predictores:** Variables base (Edad, Sexo, Altitud, Pobreza, IDH).
*   **Objetivo Académico:** Evaluar la capacidad de predecir la anemia a escala macro (nacional), usando únicamente determinantes estructurales, para observar cómo la pobreza geográfica y la altitud interactúan con el diagnóstico.

### Pipeline B: "El Modelo Socio-Médico Profundo"
*   **Dataset:** `df_regiones_enriquecido.csv` (~338,000 registros).
*   **Predictores:** 44 variables clínicas y sociales (Programas Juntos, Qaliwarma, tipo de seguro SIS, afiliaciones de la madre, etc.).
*   **Objetivo Académico:** Extraer reglas médicas y sociales complejas. El Árbol de Decisión nos revelará si pertenecer a ciertos programas sociales de alimentación reduce drásticamente el riesgo de anemia infantil.

---

## 2. Decisiones Técnicas y Algorítmicas

En estricto apego al marco teórico del documento de investigación, se descartan los modelos de "Caja Negra" y se fija la siguiente configuración para ambos Pipelines (A y B):

1.  **Algoritmos de Clasificación:**
    *   **Regresión Logística:** Actuará como "Modelo Base" (Baseline) para entender los pesos de las variables.
    *   **Árboles de Decisión:** Actuará como "Modelo Principal", generando las reglas transparentes que exigen los especialistas en salud.
2.  **Preprocesamiento Interno (Evitando Fuga de Datos):**
    *   *StandardScaler* para normalizar variables numéricas (como Altitud).
    *   *One-Hot Encoding* para variables categóricas.
3.  **Balanceo de Clases:**
    *   Se aplicará la técnica **SMOTE (Sobremuestreo Sintético)** para balancear la gran cantidad de pacientes sanos frente a los enfermos, garantizando que el modelo preste atención al riesgo de anemia.
    *   *Regla de Oro:* SMOTE se aplicará **únicamente** sobre la muestra de entrenamiento (80%), preservando el 20% de prueba intacto para la validación final.

---

## 3. Flujo de Ejecución (Código en `src/`)

Para implementar esto en Python bajo estándares de MLOps, modificaremos dos archivos clave:

### Archivo 1: `src/train.py` (El Constructor)
Este script realizará las siguientes operaciones de forma automática:
1. Cargará ambos datasets.
2. Dividirá cada dataset en 80% (Estudio) y 20% (Test).
3. Aplicará las transformaciones (Escalado/One-Hot) y el SMOTE al 80% de estudio.
4. Entrenará la Regresión Logística y el Árbol de Decisión para ambos datasets.
5. Exportará los dos campeones resultantes (ej. `modelo_masivo.joblib` y `modelo_social.joblib`) a la carpeta `models/`.

### Archivo 2: `src/predict.py` (El Evaluador)
Su rol será "evaluar" a los modelos campeones usando el 20% de datos que se escondió (Test).
1. Construirá las Matrices de Confusión.
2. Imprimirá los reportes de clasificación.
3. **Prioridad Absoluta:** Analizará exhaustivamente el **Recall (Sensibilidad)** de la Clase 1 (Anemia), asegurando que los falsos negativos se mantengan en el mínimo posible, según exige la metodología de salud pública.
