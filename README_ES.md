# 🔍 Análisis de Sentimiento de Clientes — Reviews de Amazon

> **Proyecto de NLP end-to-end que analiza 568.000+ reviews reales de Amazon — combinando baseline de TextBlob, vectorización TF-IDF y Regresión Logística para clasificar el sentimiento del cliente y generar insights de negocio accionables.**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)](https://scikit-learn.org)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-green)](https://www.nltk.org)
[![TextBlob](https://img.shields.io/badge/TextBlob-Sentiment-yellow)](https://textblob.readthedocs.io)
[![Estado](https://img.shields.io/badge/Estado-Activo-brightgreen)]()

---

## 🧠 El Problema de Negocio

Toda empresa de e-commerce recibe miles de reviews de clientes cada mes. Leerlas manualmente es imposible a escala. Sin un sistema automatizado que las clasifique, los equipos de producto y operaciones no pueden distinguir quejas aisladas de problemas sistémicos que requieren acción inmediata.

La pregunta crítica:

> *¿Qué están diciendo realmente los clientes — y qué problemas se repiten lo suficiente como para justificar una acción correctiva?*

Este proyecto construye un pipeline automatizado de clasificación de sentimiento que procesa reviews a escala, identifica los principales drivers de insatisfacción y traduce el output del modelo en recomendaciones concretas de negocio.

---

## ✅ La Solución

Un pipeline de NLP end-to-end que limpia y preprocesa texto crudo de reviews, evalúa un baseline basado en reglas (TextBlob), entrena un modelo supervisado de clasificación (TF-IDF + Regresión Logística) y presenta las palabras más predictivas de cada categoría de sentimiento.

> *El modelo supervisado detecta el 73% de las reviews negativas — más de 3 veces la tasa de detección del 22% del baseline de TextBlob — dándole a los equipos operativos una señal confiable para actuar sobre la insatisfacción del cliente antes de que escale.*

---

## 📐 Arquitectura del Sistema

```
┌─────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│  568K Reviews       │───▶│  Preprocesamiento    │───▶│  Baseline TextBlob   │
│  Amazon Food        │    │  NLTK · Stopwords    │    │  Análisis Polaridad  │
└─────────────────────┘    └──────────────────────┘    └──────────┬───────────┘
                                                                    │
                                              ┌─────────────────────▼──────────┐
                                              │  TF-IDF + Regresión Logística  │
                                              │  Clasificación Supervisada      │
                                              │  Insights de Negocio · WordCloud│
                                              └────────────────────────────────┘
```

---

## 🔄 Metodología — Framework STAR

### Situación
Una plataforma de e-commerce con 568.000+ reviews de clientes no tenía ningún sistema automatizado para clasificar el sentimiento a escala. La revisión manual era imposible, y las señales críticas de insatisfacción — ocultas en reviews negativas — pasaban desapercibidas, impactando en las decisiones de calidad de producto y en la reputación de la marca.

### Tarea
Construir un pipeline de NLP que:
- Preprocese texto crudo de reviews a escala
- Evalúe un baseline basado en reglas e identifique sus limitaciones
- Entrene un modelo supervisado que supere al baseline en detección de reviews negativas
- Identifique las palabras y temas clave que impulsan cada categoría de sentimiento
- Entregue insights de negocio accionables a partir del output del modelo

### Acción

**1 — Análisis Exploratorio**
- Analizó 568.454 reviews — score promedio 4.18, fuerte sesgo positivo (78% positivas)
- Identificó desbalance crítico: reviews de 1 estrella (52K) superan a las de 2 estrellas (29K) — los clientes muy insatisfechos son más vocales que los moderadamente insatisfechos
- Creó etiquetas de sentimiento: Positivo (4-5★), Neutro (3★), Negativo (1-2★)

**2 — Preprocesamiento de Texto (NLTK)**
- Eliminó etiquetas HTML, caracteres especiales y puntuación
- Convirtió a minúsculas y eliminó stopwords en inglés
- Procesó 100.000 reviews para el modelado — suficiente para entrenamiento robusto

**3 — Análisis WordCloud**
- Visualizó las palabras más frecuentes por categoría de sentimiento
- Hallazgo clave: las reviews negativas se agrupan alrededor de "horrible", "worst", "disappointed", "waste money", "return" — indicando desajuste entre expectativas y experiencia post-compra
- Las reviews neutras muestran lenguaje ambivalente: "okay", "decent", "however", "nothing special"

**4 — Baseline TextBlob**
- Aplicó scoring de sentimiento basado en reglas (polaridad -1 a +1)
- Resultado: 71% de accuracy pero solo 22% de recall en reviews negativas — el 78% de las quejas pasan desapercibidas

**5 — Modelo Supervisado: TF-IDF + Regresión Logística**

| Métrica | Baseline TextBlob | Regresión Logística |
|---------|------------------|---------------------|
| Accuracy | 71% | **79%** |
| F1 Negativo | 0,32 | **0,66** |
| F1 Neutro | 0,17 | **0,35** |
| F1 Positivo | 0,85 | **0,89** |

TF-IDF con 10.000 features y bigramas (ngram_range 1-2), combinado con balanceo de clases, más que duplicó la detección de reviews negativas del 22% al 73%.

### Resultados

**Resultado Clave #1:** El modelo supervisado alcanza 79% de accuracy y detecta el 73% de las reviews negativas — comparado con el 22% del baseline. Por cada 1.000 quejas, el modelo identifica correctamente 730 vs 220 con el baseline.

**Resultado Clave #2:** Los principales predictores negativos — "horrible", "worst", "disappointed", "never buy", "waste money", "return" — revelan que los clientes insatisfechos no solo están enojados, sino que están activamente advirtiendo a otros que no compren.

**Resultado Clave #3:** La aparición de "waste money" y "return" entre las palabras más negativas indica que el desajuste precio-calidad y la fricción en el proceso de devolución son problemas sistémicos, no incidentes aislados.

---

## 📊 Análisis y Visualizaciones

**Distribución de scores y sentimientos:**

![Score Distribution](img/score_distribution.png)

![Sentiment Distribution](img/sentiment_distribution.png)

**Palabras más frecuentes por sentimiento — Análisis WordCloud:**

![WordClouds](img/wordclouds.png)

**Distribución de polaridad TextBlob por sentimiento:**

![Polarity Distribution](img/polarity_distribution.png)

**Evaluación del modelo — Matriz de Confusión y comparación F1:**

![Model Evaluation](img/model_evaluation.png)

**Palabras más predictivas por categoría de sentimiento:**

![Top Words](img/top_words_by_sentiment.png)

**Resumen ejecutivo — rendimiento del modelo e impacto de negocio:**

![Executive Summary](img/executive_summary.png)

---

## 🔍 Insights Clave de Negocio

| Insight | Acción Recomendada |
|---------|-------------------|
| 73% de reviews negativas detectadas correctamente (vs 22% baseline) | Implementar el modelo para alertar al equipo de producto sobre reviews negativas en tiempo real |
| "Waste money" y "return" son los principales predictores negativos | Investigar la percepción precio-calidad y la fricción en el proceso de devolución |
| "Never buy" indica clientes que desalientan activamente a otros | Monitorear y responder públicamente estas reviews para proteger la reputación de marca |
| Reviews de 1 estrella superan a las de 2 — la insatisfacción es polarizada | Priorizar la resolución de reviews de 1 estrella — tienen mayor riesgo de churn y referencia negativa |
| Reviews neutras usan "however" y "unfortunately" — casi conversiones | Pequeñas mejoras de calidad podrían convertir clientes neutros en promotores |

---

## 🛠️ Stack Tecnológico

| Capa | Tecnología | Propósito |
|------|------------|-----------|
| Fuente de Datos | Kaggle — Amazon Fine Food Reviews | 568.454 reviews reales de clientes |
| Preprocesamiento | Python · NLTK · Regex | Limpieza de texto, eliminación de stopwords |
| Baseline | TextBlob | Scoring de sentimiento basado en reglas |
| Visualización | WordCloud · Matplotlib · Seaborn | Patrones de sentimiento y frecuencia de palabras |
| Modelado | scikit-learn · TfidfVectorizer · LogisticRegression | Clasificación supervisada de sentimiento |
| Evaluación | Accuracy · Precision · Recall · F1 · Matriz de Confusión | Evaluación orientada al negocio |
| Entorno | Jupyter Notebook | Pipeline completo y reproducible |

---

## 📁 Estructura del Repositorio

```
sentiment-analysis/
│
├── notebooks/
│   └── 01_sentiment_analysis.ipynb   # Pipeline completo: EDA, preprocesamiento, modelado, evaluación
├── data/
│   └── Reviews.csv                   # No incluido — ver sección Dataset
├── img/
│   ├── score_distribution.png        # Distribución de ratings 1-5 estrellas
│   ├── sentiment_distribution.png    # División Positivo · Neutro · Negativo
│   ├── wordclouds.png                # Palabras más frecuentes por sentimiento
│   ├── polarity_distribution.png     # Análisis baseline TextBlob
│   ├── model_evaluation.png          # Matriz de confusión + comparación F1
│   ├── top_words_by_sentiment.png    # Palabras más predictivas por categoría
│   └── executive_summary.png        # Resumen ejecutivo de impacto de negocio
├── requirements.txt                  # Dependencias Python
└── LICENSE                           # Licencia MIT
```

---

## 📊 Dataset

Este proyecto usa el dataset **Amazon Fine Food Reviews**, disponible en [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).

Para ejecutar el notebook localmente:
1. Descargá `Reviews.csv` desde el link de arriba
2. Colocalo en la carpeta `data/`
3. Ejecutá el notebook

El dataset contiene 568.454 reviews reales de productos alimenticios de Amazon, incluyendo score (1-5) y texto completo de la review.

---

## 👤 Autor

**Andrés Navarro**
Analista de Datos · NLP · Machine Learning · Python

[![GitHub](https://img.shields.io/badge/GitHub-AndyNavarro77-black?logo=github)](https://github.com/AndyNavarro77)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Conectar-blue?logo=linkedin)](https://www.linkedin.com/in/andr%C3%A9s-navarro77/)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visitar-orange?logo=netlify)](https://andres-navarro-portfolio.netlify.app/)

---

*Construido para demostrar pensamiento de NLP end-to-end — desde el preprocesamiento de texto crudo hasta la clasificación supervisada — traduciendo el output del modelo en inteligencia de negocio accionable sobre 568.000+ reviews reales de clientes.*