import argparse
import logging
import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def load_data(data_path):
    if not os.path.exists(data_path):
        logging.error(f"File not found at: {data_path}")
        sys.exit(1)
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Loaded data from {data_path} ({df.shape[0]} rows, {df.shape[1]} columns)")
        return df
    except Exception as e:
        logging.error(f"Failed to read CSV: {e}")
        sys.exit(1)

def validate_data(df):
    required = ['churned', 'customer_id']
    for col in required:
        if col not in df.columns:
            logging.error(f"Validation Error: Missing required column '{col}'")
            sys.exit(1)
    
    dist = df['churned'].value_counts(normalize=True)
    logging.info(f"Data validation successful. Class distribution:\n{dist}")
    return True

def train_and_evaluate(df, n_folds, random_seed):
    logging.info(f"Starting training with {n_folds} folds...")
    
    X = df.drop(columns=['churned', 'customer_id'])
    y = df['churned']
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    models = {
        "RandomForest": RandomForestClassifier(class_weight='balanced', random_state=random_seed),
        "DecisionTree": DecisionTreeClassifier(random_state=random_seed)
    }
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    results = []
    
    for name, model in models.items():
        logging.info(f"Evaluating {name}...")
        scores = cross_val_score(model, X_encoded, y, cv=cv, scoring='f1')
        results.append({"Model": name, "F1_Mean": scores.mean(), "F1_Std": scores.std()})
        logging.info(f"{name} F1 Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
    return pd.DataFrame(results)

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Telecom Churn Model Comparison Pipeline")
    parser.add_argument("--data-path", default="data/telecom_churn.csv", help="Path to input dataset (CSV)")
    parser.add_argument("--output-dir", default="./output", help="Directory for saving results")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true", help="Validate data without training")
    
    args = parser.parse_args()
    
    df = load_data(args.data_path)
    validate_data(df)
    
    if args.dry_run:
        logging.info("--- DRY RUN: Validation successful ---")
        sys.exit(0)
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = train_and_evaluate(df, args.n_folds, args.random_seed)
    
    save_path = os.path.join(args.output_dir, "comparison_results.csv")
    results.to_csv(save_path, index=False)
    logging.info(f"All done! Results saved to {save_path}")

if __name__ == "__main__":
    main()

