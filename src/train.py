import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
import xgboost as xgb
import shap

def train_pipeline(data_path, models_dir):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # 1. Clean Data
    df['TotalCharges'] = df['TotalCharges'].replace(' ', '0.0')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    
    # Map Churn to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Separate Features and Target
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']
    
    # Define column types
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = [col for col in X.columns if col not in num_cols]
    
    # 2. Train-Test Split (Stratified to maintain churn ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Churn distribution in Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
    
    # 3. Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), cat_cols)
        ]
    )
    
    # Fit and transform
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Get preprocessed feature names
    feature_names = preprocessor.get_feature_names_out()
    # Clean up names for cleaner display
    feature_names_clean = [
        name.replace('num__', '').replace('cat__', '') 
        for name in feature_names
    ]
    
    # Save clean feature names inside preprocessor metadata or a helper
    preprocessor.feature_names_ = feature_names_clean
    
    # 4. Modeling
    # A. Logistic Regression (Baseline) with class weighting
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_model.fit(X_train_proc, y_train)
    
    # B. XGBoost with scale_pos_weight to handle imbalance
    print("Training XGBoost...")
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8
    )
    xgb_model.fit(X_train_proc, y_train)
    
    # 5. Evaluate Models
    print("Evaluating models...")
    models = {
        'Logistic Regression': lr_model,
        'XGBoost': xgb_model
    }
    
    metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_proc)
        y_prob = model.predict_proba(X_test_proc)[:, 1]
        
        # Calculate standard metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Calculate PR-AUC
        p_precision, p_recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(p_recall, p_precision)
        
        print(f"\n--- {name} Results ---")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
        
        metrics[name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc)
        }
        
    # 6. Fit SHAP Explainer (on XGBoost)
    print("Fitting SHAP TreeExplainer on XGBoost...")
    # Wrap model and data for SHAP
    explainer = shap.TreeExplainer(xgb_model, X_train_proc)
    
    # Pre-calculate SHAP values for test set for global statistics
    print("Calculating SHAP values for test set...")
    shap_values = explainer(X_test_proc)
    
    # Save everything
    os.makedirs(models_dir, exist_ok=True)
    
    with open(os.path.join(models_dir, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)
        
    with open(os.path.join(models_dir, 'lr_model.pkl'), 'wb') as f:
        pickle.dump(lr_model, f)
        
    with open(os.path.join(models_dir, 'xgboost_model.pkl'), 'wb') as f:
        pickle.dump(xgb_model, f)
        
    with open(os.path.join(models_dir, 'shap_explainer.pkl'), 'wb') as f:
        pickle.dump(explainer, f)
        
    # Save precomputed test set features and shap values for visualization speed
    with open(os.path.join(models_dir, 'shap_test_data.pkl'), 'wb') as f:
        pickle.dump({
            'X_test_proc': X_test_proc,
            'y_test': y_test.values,
            'shap_values': shap_values,
            'feature_names': feature_names_clean
        }, f)
        
    with open(os.path.join(models_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Pipeline components and models successfully saved to {models_dir}")

if __name__ == "__main__":
    train_pipeline("WA_Fn-UseC_-Telco-Customer-Churn.csv", "models")
