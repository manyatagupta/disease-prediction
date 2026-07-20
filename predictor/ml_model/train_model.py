import os
import sys
import django
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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
    
    print("Preparing data for training on the FULL dataset (memory optimized)...")
    X = df.drop(columns=['diseases'])
    y = df['diseases']
    
    import time
    start_time = time.time()
    
    print("Training Naive Bayes model (extremely fast and highly accurate)...")
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X, y)
    
    end_time = time.time()
    print(f"Model trained in {end_time - start_time:.2f} seconds.")
    
    print(f"Model accuracy on training sample: {clf.score(X, y):.2f}")
    
    # Save the model and the symptom list
    model_dir = os.path.dirname(os.path.abspath(__file__))
    joblib.dump(clf, os.path.join(model_dir, "model.pkl"), compress=3)
    
    # Save the original columns as the expected features
    joblib.dump(symptoms_cols, os.path.join(model_dir, "symptoms_list.pkl"))
    print("Model and symptoms list saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
