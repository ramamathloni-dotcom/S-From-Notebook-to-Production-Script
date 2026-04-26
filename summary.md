# Project Summary & Observations

## 1. What We Did
- Developed a modular machine learning pipeline using Python, `pandas`, and `scikit-learn`.
- Implemented a clean command-line interface (CLI) to allow flexible model evaluation.
- Utilized `StratifiedKFold` to ensure consistency across folds, preserving the class distribution of the churn target.
- Automated the logging process to track execution steps.

## 2. Results
After running the pipeline on the telecom dataset (1,500 rows):
- **DecisionTree F1-Mean:** ~0.2236
- **RandomForest F1-Mean:** ~0.0379

## 3. Key Observations
- **Performance Gap:** The Decision Tree classifier significantly outperformed the RandomForest model in terms of the F1-score.
- **Class Imbalance:** Our data validation showed a significant class imbalance (Churned: ~16%, Not Churned: ~84%). This explains the relatively low F1-scores, as the models are struggling to identify the minority class (churners).
- **Next Steps:**
  - Future iterations should focus on hyperparameter tuning (e.g., using GridSearchCV).
  - Techniques to handle imbalanced data, such as SMOTE or adjusting `class_weight`, are recommended to improve predictive performance on the churned class.