import os
import joblib
from django.apps import AppConfig
from django.conf import settings

class PredictorConfig(AppConfig):
    name = 'predictor'
    
    ml_model = None
    all_symptoms = None
    feature_importances = None

    def ready(self):
        # Prevent running this twice or running when not fully ready if not needed
        # but django handles ready() running once
        model_dir = os.path.join(settings.BASE_DIR, 'predictor', 'ml_model')
        model_path = os.path.join(model_dir, 'model.pkl')
        symptoms_list_path = os.path.join(model_dir, 'symptoms_list.pkl')
        importances_path = os.path.join(model_dir, 'feature_importances.pkl')
        
        try:
            if os.path.exists(model_path) and os.path.exists(symptoms_list_path):
                PredictorConfig.ml_model = joblib.load(model_path)
                PredictorConfig.all_symptoms = joblib.load(symptoms_list_path)
                
                if os.path.exists(importances_path):
                    PredictorConfig.feature_importances = joblib.load(importances_path)
                else:
                    print("Warning: Feature importances not found.")
                    
                print("ML Model loaded successfully.")
            else:
                print("Warning: ML model files not found. Predictor app started without loaded models.")
        except Exception as e:
            print(f"Error loading ML model during app initialization: {e}")
