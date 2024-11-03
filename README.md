# Cancer_diagnosis
A machine learning project that predicts patient's cancer diagnosis with given data

Introduction
This project leverages a cancer dataset to develop and evaluate predictive models. The goal is to identify important features associated with cancer diagnoses and to build a model capable of distinguishing between cancer and healthy samples. The notebook explores the data, performs preprocessing, and applies machine and deep learning models to achieve high AUC in cancer prediction.


Requirements
The following libraries are required to run the notebook, and they are typically available in Google Colab:
- **Python** 3.10
- **Libraries**: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tensorflow`
Additional libraries can be installed within the notebook using `!pip install`.
Usage


Notebook Overview
This notebook is organized into the following main sections:

Data Loading:
- Loads the cancer dataset, typically from a CSV file or a similar format.
- Provides an initial preview of the data, examining its shape, feature names, and the first few rows to understand its structure.
  
Data Preprocessing:
- Standardizes or normalizes features as needed for optimal model performance.
  
Class Imbalance Handling
Exploratory Data Analysis (EDA):
- Visualizes the distribution of key features, using histograms, scatter plots, and box plots to highlight potential patterns.
- Examine correlations between features and the target variable to identify important predictors.
  
Feature Extraction
- Feature Selection using GBM
- Performs dimensionality reduction or feature selection methods to retain only the most relevant features using PCA and LDA
  
Model Training:
- Trains multiple machine learning and deep learning models (e.g., logistic regression, support vector machine, neural networks) on the preprocessed data.
- Configures model parameters and applies validation to assess stability and accuracy.
  
Model Evaluation:
- Evaluates models on the test data using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
- Visualizes model performance with confusion matrices and ROC curves to help assess classification effectiveness.

Predictions and Interpretation:
- Make predictions on test data.
- Hyperparameter Tuning
- Optimizes model parameters (manual hyperparameter tuning for SVM and MLP and RandomizedSearchCV for Logistic Regression and Voting Classifier) to achieve the best possible performance.
