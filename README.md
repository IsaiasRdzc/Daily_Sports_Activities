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
├── notebooks/
│   ├── 01_analisis_exploratorio/
│   │   └── 01_Carga_y_Analisis_Descriptivo.ipynb
│   ├── 02_preprocesamiento/
│   │   └── 02_Preprocesamiento_y_Limpieza.ipynb
│   ├── 03_feature_engineering/
│   │   ├── 03_Extraccion_de_Caracteristicas.ipynb
│   │   ├── 04_Param_LR_MLP.ipynb
│   │   └── 04.Param_KPCA.ipynb
│   ├── 04_modelos_individuales/
│   │   ├── 05_LogisticRgression.ipynb
│   │   ├── 05_MLPerceptron.ipynb
│   │   └── 05_RandomForest.ipynb
│   └── 05_ensembles/
│       ├── 06_AdaBoost-Bosque.ipynb
│       ├── 06_AdaBoost-KPCA.ipynb
│       ├── 06_AdaBoost-PCA.ipynb
│       ├── 06_AdaBoost-sin.ipynb
│       ├── 06_Ensemble_HardW_Voting/
│       ├── 06_Ensemble_HardW_Voting_KPCA/
│       ├── 06_Ensemble_HardW_Voting_Sin/
│       ├── 06_Ensemble_Soft_Voting/
│       └── 06_Ensemble_Soft_Voting_KPCA/
│
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   └── processed/
│       ├── dataset_preprocesado.csv
│       └── dataset_final_features.csv
│
├── models/
│   ├── modelo_pca.pkl
│   ├── modelo_kpca.pkl
│   ├── modelo_scaler.pkl
│   └── modelo_rf_select.pkl
│
├── requirements.txt
├── LICENSE
└── README.md
```

## Pipeline de Machine Learning

### 1. Analisis Exploratorio (notebooks/01_analisis_exploratorio/)
- Carga del dataset
- Estadisticas descriptivas
- Visualizacion de distribuciones
- Analisis de correlaciones

### 2. Preprocesamiento (notebooks/02_preprocesamiento/)
- Manejo de valores faltantes
- Deteccion y tratamiento de outliers
- Normalizacion y estandarizacion
- Balanceo de clases

### 3. Feature Engineering (notebooks/03_feature_engineering/)
- Extraccion de caracteristicas temporales
- Caracteristicas estadisticas
- Seleccion de variables
- Parametrizacion de modelos (LR, MLP, KPCA)

### 4. Modelos Individuales (notebooks/04_modelos_individuales/)
- Logistic Regression
- Multi-Layer Perceptron
- Random Forest

### 5. Metodos de Ensemble (notebooks/05_ensembles/)
- AdaBoost (4 variantes: sin/PCA/KPCA/Bosque)
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

El dataset contiene actividades deportivas diarias:
- data/raw/dataset.csv: Datos originales (413MB)
- data/processed/dataset_preprocesado.csv: Datos limpios y preprocesados
- data/processed/dataset_final_features.csv: Features finales para modelado

## Modelos Entrenados

Los modelos entrenados estan guardados en formato .pkl en la carpeta models/:
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

4. Seguir el pipeline desde notebooks/01_analisis_exploratorio/01_Carga_y_Analisis_Descriptivo.ipynb

## Licencia

Este proyecto esta bajo la Licencia MIT - ver el archivo LICENSE para mas detalles.

## Autor

IsaiasRdzc
