# Resumen Ejecutivo: Fase 2 - Preprocesamiento y ETL

**Proyecto:** Predicción de Riesgo de Anemia en Menores de 5 Años  
**Fase:** 02 - Preprocesamiento, Limpieza y Feature Engineering (Data Preparation)  
**Entregables:** `notebooks/02_preprocesamiento.ipynb` y `src/preprocess.py`  

Tras finalizar la fase de limpieza y transformación de los 8 archivos crudos (MINSA y SIEN Regional), se consolidan las siguientes justificaciones arquitectónicas y resultados para sustentar el avance a la fase de Machine Learning:

---

## 1. Arquitectura MLOps: La Dualidad de Archivos (.ipynb vs .py)
Se tomó la decisión estratégica de aplicar el estándar *Cookiecutter Data Science*, separando la lógica de preprocesamiento en dos entornos distintos:
*   **El Laboratorio (`02_preprocesamiento.ipynb`):** Es un entorno narrativo paso a paso. Se utiliza para experimentar visualmente, trazar el flujo de datos según la metodología de Aurélien Géron y permitir una fácil auditoría y presentación de la tesis/proyecto.
*   **La Fábrica de Producción (`src/preprocess.py`):** Es la refactorización modular del código usando las mejores prácticas (PEP-8, funciones puras, Type Hints). Su propósito es que la lógica de limpieza pueda ser consumida instantáneamente por un servidor web u otro script en el futuro (despliegue en producción) de forma automatizada, sin depender de un entorno interactivo como Jupyter.

## 2. Justificación de las Decisiones de Preprocesamiento (ETL)
Para convertir los datos crudos dispares en un modelo de Machine Learning consumible, se tomaron tres decisiones críticas:

1.  **Filtros de Alcance Estrictos:** Se aplicó un recorte poblacional estricto descartando todos los registros donde la edad supere los 59.99 meses. Además, se limpió el "ruido biológico", purgando lecturas de hemoglobina irreales (menores a 4 o mayores a 20 g/dL).
2.  **Mapeo de Target Binario:** Puesto que las fuentes mezclaban códigos CIE-10 (`85018.00`) con textos narrativos ("ANEMIA LEVE", "NORMAL"), se construyó un diccionario unificador colapsando el diagnóstico en un sistema Binario: **`0 = Sano`** y **`1 = Riesgo de Anemia`**. Esto optimiza el aprendizaje del algoritmo.
3.  **Feature Engineering (Enriquecimiento Socioespacial):** *Decisión estrella del proyecto.* En lugar de imputar o ignorar la variable de altitud (crucial para la anemia), se cruzaron todos los expedientes clínicos con el **Padrón de Ubigeos (INEI/RENIEC)**. Esto permitió inyectar exitosamente la `altitud`, la `latitud` y fuertes predictores como el `Índice de Pobreza` (pct_pobreza_total) y el `Índice de Desarrollo Humano` (idh_2019) directo al expediente de cada niño.

## 3. Resultados Obtenidos (Carpeta `data/processed/`)
En lugar de forzar un solo dataset asumiendo pérdida de variables, la arquitectura generó **DOS** sets de experimentación altamente pulidos:

### A. Dataset "Regiones Enriquecido" (`df_regiones_enriquecido.csv`)
*   **Características:** Alto volumen de variables sociodemográficas (afiliación al SIS, Qaliwarma, Juntos). Solo incluye la data del SIEN.
*   **Volumen:** 338,421 observaciones y 44 predictores.
*   **Tasa de Éxito de Enriquecimiento:** Se logró un cruce perfecto del **99.95%** de los datos geográficos (solo 163 nulos residuales en altitud tras cruzar por código INEI).

### B. Dataset "Global Limpio" (`df_global_limpio.csv`)
*   **Características:** Alto volumen poblacional. Fusiona los 8 archivos (SIEN + ANEMIA_DA + TAMIZAJE) pero de manera restrictiva, quedándose únicamente con ~8 columnas maestras comunes y eliminando el 100% de los nulos.
*   **Volumen:** **397,042 observaciones** inmaculadas (0 nulos).
*   **Balance del Target:** Registra ~294k niños sanos frente a **~102k casos de anemia**. La inyección de la data nacional duplicó la captura de la clase minoritaria, lo cual será vital para entrenar los árboles de decisión en la siguiente fase.

---
**Conclusión de Fase:**  
Los datos procesados son ahora matemáticamente puros, estandarizados y están altamente enriquecidos con factores socioeconómicos. Se autoriza formalmente el inicio de la **Fase 3: Entrenamiento de Modelos Base y Ensamblados (Logistic Regression / Decision Trees / XGBoost).**
