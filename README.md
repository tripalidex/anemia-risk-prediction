# Predicción del Riesgo de Anemia en Menores de 5 Años 🩸

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-orange?style=flat-square&logo=scikit-learn)
![MLOps](https://img.shields.io/badge/Architecture-MLOps-success?style=flat-square)

Este repositorio contiene el código fuente, la investigación y los modelos predictivos para evaluar el riesgo de anemia en infantes peruanos menores de 5 años. El proyecto utiliza datos provenientes del SIEN (Sistema de Información del Estado Nutricional) del MINSA e INEI, cruzando datos clínicos con características sociodemográficas, altitud e índices de desarrollo humano.

El objetivo principal es proveer a la salud pública de una herramienta algorítmica capaz de identificar el riesgo de anemia de forma proactiva, basándose en determinantes sociales de la salud (SDOH).

---

## 🏗️ Arquitectura del Proyecto (MLOps)

El proyecto sigue una arquitectura orientada a producción inspirada en *Cookiecutter Data Science*. Separa estrictamente la exploración teórica (Notebooks) de la ejecución productiva (Scripts modulares).

```text
anemia-risk-prediction/
├── data/                  # (Ignorado en GitHub por confidencialidad)
│   ├── raw/               # Archivos CSV crudos (MINSA, SIEN, INEI)
│   └── processed/         # Datasets limpios generados por preprocess.py
├── docs/                  # Documentación académica y Resúmenes Ejecutivos
├── models/                # Cerebros exportados (.joblib) generados por train.py
├── notebooks/             # Entorno de experimentación interactiva (EDA y Prototipos)
├── reports/               # Métricas de evaluación exportadas en formato .txt
├── src/                   # Código fuente de producción
│   ├── config.py          # Centralización de rutas absolutas
│   ├── preprocess.py      # Pipeline de extracción, limpieza y enriquecimiento (ETL)
│   ├── train.py           # Motor de modelado, balanceo SMOTE y exportación
│   └── predict.py         # Motor de inferencia y validación contra datos ocultos
├── pyproject.toml         # Gestor de dependencias de Python (compatible con uv)
└── README.md              # Este documento
```

---

## ⚙️ Guía de Instalación (Usando `uv`)

Para garantizar la reproducibilidad y evitar los clásicos problemas de dependencias, este proyecto utiliza [**`uv`**](https://github.com/astral-sh/uv), el gestor de paquetes de Python de ultra-alta velocidad escrito en Rust.

### Paso 1: Instalar `uv`
Si aún no tienes `uv` instalado en tu sistema, instálalo vía PowerShell (Windows):
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```
*(Para Mac/Linux, consulta la [documentación oficial](https://github.com/astral-sh/uv)).*

### Paso 2: Clonar y Sincronizar el Proyecto
Abre tu terminal, posicionate en la carpeta raíz del proyecto y sincroniza las dependencias. `uv` creará automáticamente el entorno virtual (`.venv`) y descargará `scikit-learn`, `pandas`, `imbalanced-learn`, etc., exactamente en las versiones estipuladas.

```bash
git clone <URL_DEL_REPOSITORIO>
cd anemia-risk-prediction
uv sync
```

---

## 🚀 Guía de Ejecución Rápida

El ciclo de vida del proyecto está dividido en tres fases orquestadas secuencialmente. Debes ejecutarlas estrictamente en este orden. 

*(Asegúrate de que tus archivos de datos crudos se encuentren ubicados en `data/raw/` antes de iniciar).*

### 1. Preprocesamiento (Extracción y Limpieza)
Cruza los datos del MINSA con los padrones del INEI, purga las variables de alta cardinalidad (nombres, hospitales) para evitar fugas de memoria, y exporta los datos listos.
```bash
uv run python src/preprocess.py
```
*(Deberás ver un mensaje de éxito indicando que los archivos CSV han sido creados en `data/processed/`).*

### 2. Entrenamiento de Modelos
Carga los datasets preprocesados, aplica **Sobremuestreo Sintético (SMOTE)** para balancear la cantidad de niños sanos vs enfermos, y entrena los algoritmos base (Regresión Logística y Árboles de Decisión).
```bash
uv run python src/train.py
```
*(Los modelos resultantes se exportarán automáticamente a la carpeta `models/`).*

### 3. Evaluación y Predicción
Toma los "cerebros" recién entrenados (el modelo oculto del 20%) y los evalúa contra datos nunca antes vistos, priorizando la métrica del **Recall** para detectar la mayor cantidad de riesgos.
```bash
uv run python src/predict.py
```
*(Los resultados y las Matrices de Confusión se guardarán como archivos de texto en la carpeta `reports/`).*
