# FY_Project
Using Deep Learning continuous prediction of Mortality rate in Intensive Care Unit.

# Heart Disease Prediction using Machine Learning and Deep Learning

## Project Overview

This project aims to predict heart disease using a comprehensive dataset and various machine learning and deep learning models. The goal is to develop a robust prediction system that can assist in early diagnosis and intervention, ultimately improving patient outcomes. The project involves data preprocessing, feature selection, training multiple classification models, and evaluating their performance.

## Dataset

The dataset `data01.csv` contains various clinical and demographic features related to heart disease. Key steps performed on the data include:

*   **Loading:** Data is loaded from `/content/drive/MyDrive/123/data01.csv`.
*   **Handling Missing Values:** Missing values in several columns (e.g., `BMI`, `heart rate`, `Systolic blood pressure`, `Diastolic blood pressure`, `Respiratory rate`, `temperature`, `SP O2`, `Neutrophils`, `Basophils`, `Lymphocyte`, `PT`, `Creatine kinase`, `glucose`, `PH`, `Lactic acid`, `PCO2`) were imputed using either the mean or mode.
*   **Dropping Rows:** Any remaining rows with missing values were dropped.
*   **Feature Engineering/Removal:** Irrelevant columns like `group` and `ID` were removed.
*   **Class Distribution:** An analysis of the `outcome` variable showed an imbalanced dataset (approximately 86.3% for class 0 and 13.7% for class 1).

## Methodology

### 1. Feature Selection

To reduce dimensionality and improve model performance, `SelectPercentile` with `mutual_info_classif` was used to select the top 35% of features. The selected features include:
`['gendera', 'deficiencyanemias', 'depression', 'COPD', 'heart rate', 'temperature', 'Urine output', 'Leucocyte', 'Lymphocyte', 'PT', 'Urea nitrogen', 'Blood sodium', 'Anion gap', 'PH', 'Bicarbonate', 'Lactic acid', 'PCO2']`

### 2. SMOTE Sampling

Due to the class imbalance, Synthetic Minority Over-sampling Technique (SMOTE) was applied to the reduced feature set (`X_reduced`) and target variable (`y`) to balance the classes.

### 3. Data Splitting

The balanced dataset was split into training and testing sets with an 80/20 ratio for traditional ML models and a 70/30 ratio for deep learning models, ensuring consistency with previous practices.

### 4. Model Training and Evaluation

Several machine learning and deep learning models were trained and evaluated. A `storeResults` function was implemented to keep track of Accuracy, Precision, Recall, and F1-Score for each model.

#### Traditional Machine Learning Models:

*   **Logistic Regression:** A baseline linear model.
*   **Random Forest:** An ensemble tree-based model.
*   **XGBoost:** A gradient boosting framework.
*   **Stacking Classifier:** Combines predictions from multiple base estimators (Random Forest, MLPClassifier) with a final estimator (LGBMClassifier).
*   **Voting Classifier:** Combines predictions from AdaBoostClassifier and RandomForestClassifier using a soft voting strategy.

#### Deep Learning Models:

*   **Recurrent Neural Network (RNN) with Attention:** A custom RNN model incorporating an attention mechanism for sequence processing. This model was carefully implemented and debugged to resolve `K.squeeze` errors by migrating to `tf.squeeze` and `tf.matmul` within a custom Keras `Layer`.
*   **LSTM Autoencoder (LSTM-AE) for Anomaly Detection:** An LSTM-based autoencoder was trained for unsupervised anomaly detection. It calculates reconstruction errors, establishes a threshold based on training data, and classifies test samples as anomalous if their error exceeds the threshold. This approach was specifically developed to correctly handle the output of the autoencoder for classification metrics.

## Results

The performance of all models is summarized in the following table:

| ML Model            | Accuracy | Precision | Recall | F1-Score |
| :------------------ | :------- | :-------- | :----- | :------- |
| Logistic Regression | 0.729    | 0.729     | 0.729  | 0.729    |
| Random Forest       | 0.892    | 0.893     | 0.892  | 0.892    |
| XGBoost             | 0.915    | 0.916     | 0.915  | 0.915    |
| Stacking Classifier | 0.892    | 0.892     | 0.892  | 0.892    |
| Voting Classifier   | 0.902    | 0.904     | 0.902  | 0.902    |
| RNN                 | 0.524    | 1.000     | 0.524  | 0.688    |
| LSTM-AE             | 0.546    | 0.299     | 0.546  | 0.386    |

### Visualizations

Bar charts are generated to visually compare the Accuracy, Precision, Recall, and F1-Score across all trained models.

## Conclusion

Based on the evaluation metrics, **XGBoost** emerged as the best-performing model for heart disease prediction among the tested classifiers, showing the highest Accuracy, Precision, Recall, and F1-Score.

The deep learning models (RNN and LSTM-AE), while providing insights into sequence data and anomaly detection, require further optimization or a different approach to match the performance of the best traditional machine learning models for this specific classification task.

## How to Reproduce

To reproduce this project:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2.  **Ensure you have the dataset:** Place `data01.csv` in the specified Google Drive path or update the path in the notebook.
3.  **Install dependencies:** All required libraries are imported at the beginning of the notebook cells. You might need to install them if not already present in your environment (e.g., `pip install pandas numpy scikit-learn tensorflow keras imbalanced-learn xgboost lightgbm`)
4.  **Run the Jupyter Notebook/Colab:** Execute the cells sequentially to perform data preprocessing, model training, and evaluation.

## Future Work

*   Hyperparameter tuning for all models, especially the deep learning architectures.
*   Exploration of more advanced deep learning models (e.g., more complex LSTM/GRU architectures, Transformers).
*   Investigation of different anomaly detection techniques with the LSTM-AE or alternative approaches.
*   Feature importance analysis for the best-performing models.
*   Deployment of the best model for real-time predictions.
