import json
import numpy as np
import pandas as pd
from .apps import PredictorConfig
from .models import Disease

class PredictionError(Exception):
    pass

def make_prediction(selected_symptoms):
    """
    Takes a list of selected symptom names and returns prediction results.
    Returns a dictionary containing:
      - predicted_disease: str
      - confidence_score: float
      - disease_obj: Disease model instance (or None)
      - top_3: list of dicts with disease info
      - chart_labels: JSON string
      - chart_data: JSON string
      - low_confidence_warning: bool
      - key_symptoms: list of dicts representing the top contributing symptoms
    """
    clf = PredictorConfig.ml_model
    all_symptoms = PredictorConfig.all_symptoms
    feature_importances = PredictorConfig.feature_importances
    
    if clf is None or all_symptoms is None:
        raise PredictionError('Error: ML prediction model is not loaded. Please contact support.')
        
    # Prepare input vector
    formatted_selected = [s.replace("_", " ").title() for s in selected_symptoms]
    input_data = [1 if s in formatted_selected else 0 for s in all_symptoms]
    input_df = pd.DataFrame([input_data], columns=all_symptoms)
    
    # Predict
    predicted_disease = clf.predict(input_df)[0]
    probabilities = clf.predict_proba(input_df)[0]
    
    class_probs = list(zip(clf.classes_, probabilities))
    class_probs.sort(key=lambda x: x[1], reverse=True)
    top_5 = class_probs[:5]
    
    # We no longer normalize top 5 probabilities as we are using CalibratedClassifierCV
    # which gives us true confidence scores directly.
    confidence_score = round(top_5[0][1] * 100, 2)
    low_confidence_warning = confidence_score < 30.0
    
    # Identify key contributing symptoms
    key_symptoms = []
    if feature_importances:
        selected_importances = []
        for s in formatted_selected:
            if s in feature_importances:
                selected_importances.append({'name': s, 'importance': feature_importances[s]})
        
        # Sort by importance descending
        selected_importances.sort(key=lambda x: x['importance'], reverse=True)
        # Take top 3
        key_symptoms = selected_importances[:3]
    
    # Fetch top 3 disease info
    top_3 = []
    for i in range(min(3, len(top_5))):
        d_name = top_5[i][0]
        d_conf = round(top_5[i][1] * 100, 2)
        d_obj = Disease.objects.filter(name=d_name).first()
        if d_obj:
            top_3.append({
                'name': d_name,
                'confidence': d_conf,
                'severity': getattr(d_obj, 'severity', 'Moderate'),
                'description': d_obj.description,
                'precautions': d_obj.precautions,
                'specialist': getattr(d_obj, 'recommended_doctor', 'General Physician')
            })
            
    chart_labels = json.dumps([x[0] for x in top_5])
    chart_data = json.dumps([round(x[1] * 100, 2) for x in top_5])
    
    disease_obj = Disease.objects.filter(name=predicted_disease).first()
    
    return {
        'predicted_disease': predicted_disease,
        'confidence_score': confidence_score,
        'disease_obj': disease_obj,
        'top_3': top_3,
        'chart_labels': chart_labels,
        'chart_data': chart_data,
        'low_confidence_warning': low_confidence_warning,
        'key_symptoms': key_symptoms
    }
