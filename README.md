## Audio Engagement Prediction (Kaggle Competition)

Machine learning project focused on predicting how long users listen to audio content using metadata and textual information.

### 1. Fina model notebook - model.ipynb
### 2. EDA notebooks - EDA/autoviz_analysis.ipynb; EDA/tsne_umap.ipynb

### Overview

The project builds a regression pipeline to estimate **audio listening time** based on audio metadata, textual descriptions, and user interaction signals. The workflow includes exploratory data analysis, feature engineering, text processing, and gradient boosting models optimized for performance.

### Key Features

- Developed a **regression model** to predict user audio listening duration
- Performed **Exploratory Data Analysis (EDA)** to identify patterns in listening behavior
- Implemented **feature engineering** to transform raw metadata into predictive features
- Applied **TF-IDF vectorization** for text-based features such as titles and descriptions
- Used **dimensionality reduction techniques** (t-SNE, UMAP) to visualize high-dimensional feature spaces
- Trained models using **LightGBM** and **CatBoost** with **cross-validation**
- Improved model performance using **hyperparameter optimization with Optuna (Bayesian optimization)**
- Accelerated data processing and training using **GPU-based workflows with RAPIDS and CUDA**
