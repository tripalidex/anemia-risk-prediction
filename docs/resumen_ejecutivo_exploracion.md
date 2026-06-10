# Resumen Ejecutivo: Análisis Exploratorio de Datos (SIEN Regiones)

**Proyecto:** Predicción de Riesgo de Anemia en Menores de 5 Años  
**Fase:** 01 - Exploración y Descubrimiento de Datos (EDA)  
**Alcance Analizado:** 6 Regiones SIEN (San Martín, Tacna, Puno, Cusco, Loreto, Piura) y Datasets Nacionales (`ANEMIA_DA`, `TAMIZAJE`).

Tras la ejecución exhaustiva del proceso de validación y carga total (`01_exploracion.ipynb`), se presentan los hallazgos críticos que determinarán la estrategia de preprocesamiento y modelado de Machine Learning:

---

## 1. Hallazgo Principal: Calidad Excepcional de Completitud (Nulos)
Contrario a la expectativa habitual en datasets gubernamentales, el análisis global demostró una **calidad de completitud excepcional**.
*   **Archivos Regionales (SIEN):** Arrojaron un **0.0% de valores nulos** en las columnas vitales evaluadas (incluyendo `EdadMeses`, `Hemoglobina` y `AlturaREN`).
*   **Archivos Nacionales:** `ANEMIA_DA` posee un 0.0% de nulos y `TAMIZAJE` presentó un máximo de ~0.57% de valores faltantes.
*   **Impacto Estratégico:** Esto es una excelente noticia arquitectónica. Significa que la fase de imputación no requerirá técnicas destructivas (no tendremos que descartar miles de filas). Se empleará imputación espacial (mediana departamental) únicamente para cruzar datos faltantes de variables externas como la altitud en los archivos nacionales.

## 2. Alerta Operativa: Presencia de Registros Duplicados
Se ha confirmado empíricamente la existencia de múltiples registros 100% idénticos en las bases de datos regionales, lo cual evidencia errores sistémicos o humanos al momento de la captura en los centros de salud.
*   **Métricas Regionales:**
    *   Loreto: 100 registros duplicados
    *   Piura: 94 registros duplicados
    *   San Martín: 84 registros duplicados
    *   Cusco: 39 registros duplicados
    *   Puno: 23 registros duplicados
    *   Tacna: 9 registros duplicados
*   **Impacto Estratégico:** Entrenar un modelo con datos clonados genera sesgos artificiales y *overfitting* (sobreajuste). Por ello, el primer filtro obligatorio de la fase ETL será aplicar un barrido de deduplicación global (`drop_duplicates()`).

## 3. Reto de Modelado (Alerta Roja): Desbalance de Clases
El hallazgo técnico más desafiante es la distribución poblacional de la variable objetivo (el diagnóstico de anemia).
*   **Distribución (Muestra San Martín):**
    *   Sano (Normal): ~46,000 niños.
    *   Anemia Leve/Moderada: ~8,000 niños.
*   **Impacto Estratégico:** Existe una proporción aproximada de **6 niños sanos por cada 1 niño con anemia**. Ante este fuerte desbalance de clases, un algoritmo de Machine Learning tradicional tenderá a predecir "Sano" el 100% de las veces para maximizar su precisión artificialmente. 
*   **Solución:** Se hace mandatorio implementar técnicas avanzadas de balanceo en la fase de entrenamiento, tales como la asignación de **Pesos de Clase (Class Weights)** en algoritmos como XGBoost/Random Forest, o el uso de **Sobremuestreo (SMOTE)**.

## 4. Estructura y Taxonomía de Variables
Se verificó que los orígenes de datos no poseen compatibilidad nativa en sus esquemas de base de datos.
*   **Incompatibilidad de Nomenclatura:** `ANEMIA_DA` registra la edad como `EDAD_REGISTRO` y el sexo como `GENERO`, mientras que `SIEN` utiliza `EdadMeses` y `Sexo`. 
*   **Formato de Diagnósticos:** El target no está unificado. `TAMIZAJE` utiliza códigos médicos CIE-10 (ej. `85018.00`), `ANEMIA_DA` usa cadenas de texto extensas ('ANEMIA POR DEFICIENCIA DE HIERRO...'), y `SIEN` usa formatos legibles cortos ('Anemia Leve').
*   **Impacto Estratégico:** La unificación exigirá un diccionario maestro de mapeo. Se concluye que **la mejor estrategia es un Mapeo Inclusivo y una estandarización a un Target Binario** (`0 = Sano`, `1 = Riesgo de Anemia`).

## 5. Ampliación del Alcance del Proyecto
La estadística descriptiva confirmó la viabilidad y riqueza de los datos hasta los **59.99 meses** de edad. 
*   **Impacto Estratégico:** Se modifica oficialmente el alcance del proyecto, expandiéndolo de "menores de 3 años" a **"niños menores de 5 años"** para aprovechar al máximo el volumen del dataset, maximizando la capacidad de generalización del modelo.

---

**Conclusión Final:**
Los datos base (`data/raw`) poseen la integridad y calidad necesarias para sostener un modelo predictivo robusto. Los desafíos no radican en la "limpieza de basura", sino en la **arquitectura de unificación** y en el **tratamiento matemático del desbalance de clases**. El proyecto cuenta con total luz verde para iniciar la Fase de Preprocesamiento (`02_preprocesamiento.ipynb`).
