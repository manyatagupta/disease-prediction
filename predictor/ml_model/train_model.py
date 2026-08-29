import os
import sys
import django
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
# Set up Django environment
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'disease_prediction_project.settings')
django.setup()

from predictor.models import Disease, Symptom

DATASET_PATH = r"C:\Users\manya\Downloads\Final_Augmented_dataset_Diseases_and_Symptoms.csv"

def train_and_save_model():
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    
    # The first column is 'diseases', the rest are symptoms
    symptoms_cols = df.columns[1:].tolist()
    unique_diseases = df['diseases'].unique().tolist()
    
    print(f"Found {len(unique_diseases)} diseases and {len(symptoms_cols)} symptoms.")
    
    print("Populating database with symptoms and diseases...")
    # Add symptoms to DB
    for s_name in symptoms_cols:
        Symptom.objects.get_or_create(name=s_name.replace("_", " ").title())
        
    # Add diseases to DB
    for d_name in unique_diseases:
        Disease.objects.get_or_create(
            name=d_name,
            defaults={
                "description": f"AI predicted disease: {d_name}",
                "precautions": "Please consult a healthcare professional for accurate advice.",
                "recommended_doctor": "General Physician"
            }
        )
    print("Database populated successfully.")
    
    print("Preparing data for training...")
    print("Optimizing memory usage by downcasting integers...")
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
        
    print("Sampling 100,000 rows to prevent MemoryError but maintain high accuracy...")
    if len(df) > 100000:
        df = df.sample(n=100000, random_state=42)

    X = df.drop(columns=['diseases'])
    y = df['diseases']
    
    import time
    start_time = time.time()
    
    print("Hyperparameter tuning Random Forest with RandomizedSearchCV...")
    # Base classifier with balanced class weights to handle any imbalances
    # Using n_jobs=1 to avoid joblib memory duplication during multiprocessing
    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=1)
    
    # Define a smaller parameter grid for RandomizedSearch to save time
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [None, 30, 50],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # We use cv=3 to speed up the tuning process
    search = RandomizedSearchCV(
        estimator=rf, 
        param_distributions=param_grid, 
        n_iter=5, # Try 5 random combinations
        cv=3, 
        scoring='accuracy', 
        random_state=42, 
        n_jobs=1
    )
    
    search.fit(X, y)
    best_rf = search.best_estimator_
    print(f"Best hyperparameters found: {search.best_params_}")
    
    print("Extracting feature importances...")
    # Extract feature importances from the best uncalibrated model
    feature_importances = best_rf.feature_importances_
    
    print("Training final Calibrated model using 'isotonic' method (best for large datasets)...")
    # 'isotonic' calibration often works better than 'sigmoid' for large datasets
    # We use cv='prefit' because some rare diseases might have fewer than 3 samples in our 100k subset,
    # which would cause StratifiedKFold to fail during cv=3.
    clf = CalibratedClassifierCV(estimator=best_rf, method='isotonic', cv='prefit', n_jobs=1)
    clf.fit(X, y)
    
    end_time = time.time()
    print(f"Model tuned, trained and calibrated in {end_time - start_time:.2f} seconds.")
    
    print(f"Final Calibrated Model accuracy on training sample: {clf.score(X, y):.2f}")
    
    model_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save the calibrated model
    joblib.dump(clf, os.path.join(model_dir, "model.pkl"), compress=3)
    
    # Save the original columns as the expected features
    joblib.dump(symptoms_cols, os.path.join(model_dir, "symptoms_list.pkl"))
    
    # Save feature importances mapping
    importance_dict = dict(zip(symptoms_cols, feature_importances))
    joblib.dump(importance_dict, os.path.join(model_dir, "feature_importances.pkl"), compress=3)
    
    print("Model, symptoms list, and feature importances saved successfully.")

if __name__ == "__main__":
    train_and_save_model()