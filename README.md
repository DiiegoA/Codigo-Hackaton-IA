# 🚀 Core MLOps Pipeline - Hackathon IA 2026

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-Optimized-orange?style=for-the-badge)
![Optuna](https://img.shields.io/badge/Optuna-Bayesian_Search-blueviolet?style=for-the-badge)
![Polars](https://img.shields.io/badge/Polars-Ultra_Fast-yellow?style=for-the-badge)
![MLOps](https://img.shields.io/badge/MLOps-Production_Ready-success?style=for-the-badge)

Arquitectura de Machine Learning de grado de producción diseñada para procesar, limpiar, enriquecer y modelar datos tabulares complejos (ej. telemetría de Microsoft Clarity). Este pipeline está construido bajo la filosofía de **"Cero Fugas de Datos" (Zero Data Leakage)**, optimización extrema de memoria (RAM Shields) y resiliencia ante desbalances de clases severos.

---

## 🧠 Filosofía Arquitectónica

El código no es un simple script de ciencia de datos; es un **PipelineManager** centralizado (Cerebro MLOps) que transporta el estado, los artefactos y las reglas de ruteo dinámico desde la ingesta hasta la inferencia.

- **Regla del Muro de Hierro:** El pipeline aprende los parámetros exclusivamente en la matriz de `Train` (`.fit()`) y los aplica a ciegas sobre `Test` (`.transform()`).
- **Inmunidad al Pánico (Blind Thresholding):** Integración de optimización por PR-AUC (Average Precision) y F-Beta (F0.5) para evitar modelos perezosos en datasets con desbalance extremo (ej. 1:30).
- **Escudos Anti-Crasheos:** Downcasting matemático dinámico, sanitización léxica (Regex) y manejo universal de NaNs/NaTs.

---

## ⚙️ Fases del Pipeline

### 📥 FASE 1: Ingesta y El Muro de Hierro

- **Ingesta Multi-Core:** Uso de `Polars` (Rust backend) con fallback a `Pandas` (C-Engine) para leer millones de filas en segundos.
- **Guillotina Temprana:** Eliminación de clones absolutos, variables con >90% de nulos, constantes (Varianza Cero) y _Target Leakage_ automático.
- **Autopsia de Datos:** Escáneres forenses visuales para identificar nulos camuflados (`"?"`, `"N/A"`) e IDs disfrazados.
- **Muro de Hierro:** Separación estricta de variables (StratifiedKFold/GroupKFold) para garantizar validaciones puras.

### 🧹 FASE 2: Pre-procesamiento y Sanitización

- **Downcasting Matemático:** Compresión de Float64 a Float32/Int8 para un ahorro de RAM superior al 40%.
- **Normalización Textual:** Fusión tipográfica por mayoría (NLP heurístico) para variables categóricas.
- **Rare Labeling:** Agrupación de colas largas (alta cardinalidad) protegiendo Nulos algorítmicos.
- **Sonda Visual Automática:** Dashboards automatizados (KDE, Boxplots, Pairplots con _RAM Shield_) para análisis exploratorio profundo.

### 🧬 FASE 3: Ingeniería de Características (Feature Engineering)

- **Auto-Discovery de Ratios:** Análisis del léxico de las columnas para realizar divisiones matemáticas con sentido de negocio.
- **Missingness Flags:** Creación de rastreadores booleanos que le indican al modelo dónde faltaba información.
- **Ingeniería Temporal:** Descomposición de Datetimes y trigonometría (Seno/Coseno) para variables cíclicas.
- **Embeddings GGPL (Opcional):** Proyección no lineal usando árboles de decisión para enriquecer el espacio vectorial.

### ⚖️ FASE 4: Justicia Algorítmica (Fairness)

- **Auditoría DIR (Disparate Impact Ratio):** Detección de sesgos en variables protegidas (edad, género, ubicación).
- **Reweighing (IPF):** Corrección iterativa de pesos para garantizar predicciones éticas sin alterar los datos originales.

### 🪓 FASE 5: El Tribunal Supremo (Selección de Variables)

- **Guillotina Colineal:** Extracción de características redundantes (Correlación de Spearman > 98%).
- **Boruta-SHAP y Diamond FDR:** Batallas entre variables reales y "Clones de Sombra" para eliminar el ruido predictivo garantizando un _False Discovery Rate_ bajo.

### 🔮 FASE 6: Evolución Bayesiana (Forja del Oráculo)

- **Optuna Framework:** Búsqueda en el hiperespacio de 40 a 100 mutaciones con Validación Cruzada Estricta.
- **Cost-Sensitive Learning:** Ajuste dinámico de `scale_pos_weight` inyectado en LightGBM.
- **Threshold Moving:** Búsqueda del corte matemático perfecto utilizando la curva Precisión-Recall en lugar del estático 0.50.

---

## 📦 Instalación y Requisitos

Asegúrate de contar con un entorno virtual (recomendado Python 3.9+) y ejecuta:

```bash
pip install pandas numpy scikit-learn lightgbm optuna shap matplotlib seaborn polars pydantic
```
