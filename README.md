# Daily Sports Activities - Clasificacion con Machine Learning

Proyecto de Machine Learning para la clasificacion de actividades deportivas diarias utilizando tecnicas de aprendizaje supervisado y metodos de ensemble.

## Descripcion

Este proyecto implementa un pipeline completo de ML que incluye:
- Analisis exploratorio de datos (EDA)
- Preprocesamiento y limpieza
- Extraccion de caracteristicas
- Reduccion de dimensionalidad (PCA, KPCA)
- Modelado con multiples algoritmos
- Metodos de ensemble (Voting)

## Estructura del Proyecto

```
Daily_Sports_Activities/
├── Data/                           # Datos y modelos entrenados
│   ├── dataset.csv                 # Dataset original
│   ├── dataset_preprocesado.csv    # Datos limpios
│   ├── dataset_final_features.csv  # Features finales
│   ├── modelo_pca.pkl              # Transformador PCA
│   ├── modelo_kpca.pkl             # Transformador Kernel PCA
│   ├── modelo_scaler.pkl           # Escalador
│   └── modelo_rf_select.pkl        # Random Forest selector
│
├── Notebooks del Pipeline
│   ├── 01_Carga_y_Analisis_Descriptivo.ipynb
│   ├── 02_Preprocesamiento_y_Limpieza.ipynb
│   ├── 03_Extraccion_de_Caracteristicas.ipynb
│   ├── 04_Param_LR_MLP.ipynb       # Parametrizacion LR/MLP
│   ├── 04.Param_KPCA.ipynb         # Parametrizacion KPCA
│   ├── 05_LogisticRgression.ipynb  # Logistic Regression
│   ├── 05_MLPerceptron.ipynb       # Multi-Layer Perceptron
│   ├── 05_RandomForest.ipynb       # Random Forest
│   ├── 06_AdaBoost-*.ipynb         # AdaBoost (4 variantes)
│   └── 06_Ensemble_*               # Metodos de Ensemble
│
├── requirements.txt
├── LICENSE
└── README.md
```

## Pipeline de Machine Learning

### 1. Analisis Exploratorio (01_Carga_y_Analisis_Descriptivo.ipynb)
- Carga del dataset
- Estadisticas descriptivas
- Visualizacion de distribuciones
- Analisis de correlaciones

### 2. Preprocesamiento (02_Preprocesamiento_y_Limpieza.ipynb)
- Manejo de valores faltantes
- Deteccion y tratamiento de outliers
- Normalizacion y estandarizacion
- Balanceo de clases

### 3. Feature Engineering (03_Extraccion_de_Caracteristicas.ipynb)
- Extraccion de caracteristicas temporales
- Caracteristicas estadisticas
- Seleccion de variables

### 4. Reduccion de Dimensionalidad
- PCA (04_Param_LR_MLP.ipynb)
- Kernel PCA (04.Param_KPCA.ipynb)

### 5. Modelos Individuales
- Logistic Regression (05_LogisticRgression.ipynb)
- Multi-Layer Perceptron (05_MLPerceptron.ipynb)
- Random Forest (05_RandomForest.ipynb)

### 6. AdaBoost (4 variantes)
- Sin reduccion dimensional (06_AdaBoost-sin.ipynb)
- Con PCA (06_AdaBoost-PCA.ipynb)
- Con KPCA (06_AdaBoost-KPCA.ipynb)
- Con Random Forest (06_AdaBoost-Bosque.ipynb)

### 7. Metodos de Ensemble
- Hard Voting (3 variantes: sin/PCA/KPCA)
- Soft Voting (2 variantes: sin/KPCA)

## Requisitos

```bash
pip install -r requirements.txt
```

Dependencias principales:
- Python 3.8+
- Jupyter Notebook
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn

## Datos

El dataset contiene actividades deportivas diarias con las siguientes caracteristicas:
- dataset.csv: Datos originales (413MB)
- dataset_preprocesado.csv: Datos limpios y preprocesados
- dataset_final_features.csv: Features finales para modelado

## Modelos Entrenados

Los modelos entrenados estan guardados en formato .pkl en la carpeta Data/:
- modelo_pca.pkl: Transformador PCA entrenado
- modelo_kpca.pkl: Transformador Kernel PCA entrenado
- modelo_scaler.pkl: Escalador (StandardScaler)
- modelo_rf_select.pkl: Random Forest para seleccion de features

## Resultados

Los notebooks contienen:
- Metricas de evaluacion (accuracy, precision, recall, F1)
- Matrices de confusion
- Curvas ROC
- Comparacion de modelos
- Analisis de importancia de features

## Uso

1. Clonar el repositorio:
```bash
git clone https://github.com/IsaiasRdzc/Daily_Sports_Activities.git
cd Daily_Sports_Activities
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecutar notebooks en orden:
```bash
jupyter notebook
```

4. Seguir el pipeline desde 01_Carga_y_Analisis_Descriptivo.ipynb

## Licencia

Este proyecto esta bajo la Licencia MIT - ver el archivo LICENSE para mas detalles.

## Autor

IsaiasRdzc
