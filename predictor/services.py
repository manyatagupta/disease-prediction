import json
import re
import difflib
import numpy as np
import pandas as pd
from .apps import PredictorConfig
from .models import Disease


class PredictionError(Exception):
    pass


# Clinical Emergency Red-Flag Catalog
EMERGENCY_RED_FLAGS = [
    {
        'category': 'Cardiac / Vascular Emergency',
        'keywords': ['chest pain', 'sharp chest pain', 'chest tightness', 'radiating pain', 'angina', 'crushing chest pain'],
        'flag': 'Acute chest discomfort or pressure (Potential acute cardiac syndrome / ischemia)',
        'urgency': 'Critical'
    },
    {
        'category': 'Acute Respiratory Distress',
        'keywords': ['shortness of breath', 'difficulty breathing', 'stridor', 'gasping', 'choking', 'cyanosis'],
        'flag': 'Severe acute respiratory compromise or severe oxygenation deficit',
        'urgency': 'Critical'
    },
    {
        'category': 'Neurological Deficit (Stroke / CNS)',
        'keywords': ['fainting', 'loss of consciousness', 'syncope', 'seizure', 'paralysis', 'slurred speech', 'facial droop', 'sudden numbness', 'coma'],
        'flag': 'Acute neurological deficit or altered consciousness (Potential stroke / TIA / intracranial event)',
        'urgency': 'Critical'
    },
    {
        'category': 'Severe Active Hemorrhage',
        'keywords': ['coughing up blood', 'hemoptysis', 'vomiting blood', 'hematemesis', 'rectal bleeding', 'black stool', 'melena'],
        'flag': 'Active internal hemorrhage or upper/lower gastrointestinal bleeding',
        'urgency': 'Urgent'
    },
    {
        'category': 'Systemic Airway / Anaphylaxis',
        'keywords': ['throat swelling', 'swelling of lips', 'difficulty swallowing', 'anaphylaxis'],
        'flag': 'Signs of acute upper airway edema or severe systemic anaphylaxis',
        'urgency': 'Critical'
    },
    {
        'category': 'Acute CNS / Meningeal Signs',
        'keywords': ['stiff neck', 'neck rigidity', 'photophobia'],
        'flag': 'Meningeal irritation markers with systemic presentation',
        'urgency': 'Urgent'
    }
]

def evaluate_emergency_triage(selected_symptoms):
    """
    Evaluates patient symptoms against clinical emergency red flags.
    Returns:
      - is_emergency: bool
      - emergency_level: 'CRITICAL', 'URGENT', or 'STANDARD'
      - emergency_flags: list of triggered warning strings
      - instructions: recommended immediate patient action
    """
    normalized_inputs = [normalize_symptom_name(s) for s in selected_symptoms]
    triggered_flags = []
    has_critical = False

    for rule in EMERGENCY_RED_FLAGS:
        for kw in rule['keywords']:
            kw_norm = normalize_symptom_name(kw)
            for user_sym in normalized_inputs:
                if kw_norm in user_sym or user_sym in kw_norm:
                    triggered_flags.append({
                        'category': rule['category'],
                        'flag': rule['flag'],
                        'urgency': rule['urgency'],
                        'trigger_symptom': user_sym.title()
                    })
                    if rule['urgency'] == 'Critical':
                        has_critical = True
                    break

    # De-duplicate flags by flag text
    seen_flags = set()
    unique_flags = []
    for f in triggered_flags:
        if f['flag'] not in seen_flags:
            seen_flags.add(f['flag'])
            unique_flags.append(f)

    if unique_flags:
        emergency_level = 'CRITICAL' if has_critical else 'URGENT'
        instructions = (
            "CRITICAL ALERT: Emergency clinical red flags have been identified. "
            "Please call local emergency medical services (911 / 112) or go to the nearest emergency room immediately. "
            "Do not attempt to drive yourself. Rest and remain in an upright, comfortable position while awaiting medical personnel."
            if has_critical else
            "URGENT NOTICE: High-priority symptoms detected. Please seek prompt medical evaluation at an urgent care center or clinic within 12 to 24 hours."
        )
        return True, emergency_level, unique_flags, instructions

    return False, 'STANDARD', [], "Standard outpatient evaluation recommended if symptoms persist."

def make_prediction(selected_symptoms):
    clf = PredictorConfig.ml_model
    all_symptoms = PredictorConfig.all_symptoms
    feature_importances = PredictorConfig.feature_importances
    if clf is None or all_symptoms is None:
        raise PredictionError('Error: ML prediction model is not loaded. Please contact support.')
    normalized_selected = set(s.replace('_', ' ').strip().lower() for s in selected_symptoms)
    input_data = [1 if s.replace('_', ' ').strip().lower() in normalized_selected else 0 for s in all_symptoms]
    input_df = pd.DataFrame([input_data], columns=all_symptoms)
    predicted_disease = clf.predict(input_df)[0]
    probabilities = clf.predict_proba(input_df)[0]
    class_probs = list(zip(clf.classes_, probabilities))
    class_probs.sort(key=lambda x: x[1], reverse=True)
    top_5 = class_probs[:5]
    confidence_score = round(top_5[0][1] * 100, 2)
    low_confidence_warning = confidence_score < 30.0
    key_symptoms = []
    if feature_importances:
        selected_importances = [{'name': s.title(), 'importance': feature_importances[s]} for s in all_symptoms if s.replace('_', ' ').strip().lower() in normalized_selected and s in feature_importances]
        selected_importances.sort(key=lambda x: x['importance'], reverse=True)
        key_symptoms = selected_importances[:3]
    top_3 = []
    for i in range(min(3, len(top_5))):
        d_name = top_5[i][0]
        d_obj = Disease.objects.filter(name=d_name).first()
        if d_obj:
            top_3.append({'name': d_name, 'confidence': round(top_5[i][1] * 100, 2), 'severity': getattr(d_obj, 'severity', 'Moderate'), 'description': d_obj.description, 'precautions': d_obj.precautions, 'specialist': getattr(d_obj, 'recommended_doctor', 'General Physician')})
    chart_labels = json.dumps([x[0] for x in top_5])
    chart_data = json.dumps([round(x[1] * 100, 2) for x in top_5])
    disease_obj = Disease.objects.filter(name=predicted_disease).first()
    return {'predicted_disease': predicted_disease, 'confidence_score': confidence_score, 'disease_obj': disease_obj, 'top_3': top_3, 'chart_labels': chart_labels, 'chart_data': chart_data, 'low_confidence_warning': low_confidence_warning, 'key_symptoms': key_symptoms}
