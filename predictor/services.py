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

# Common Clinical Synonyms & Colloquial Symptom Terms
COMMON_SYNONYMS = {
    'tummy ache': 'lower abdominal pain',
    'stomach ache': 'lower abdominal pain',
    'belly pain': 'lower abdominal pain',
    'stomach pain': 'lower abdominal pain',
    'abdominal ache': 'sharp abdominal pain',
    'abdominal pain': 'sharp abdominal pain',
    'throwing up': 'vomiting',
    'threw up': 'vomiting',
    'puking': 'vomiting',
    'nauseated': 'nausea',
    'heart palpitation': 'palpitations',
    'racing heart': 'palpitations',
    'short breath': 'shortness of breath',
    'breathless': 'shortness of breath',
    'breathlessness': 'shortness of breath',
    'hard to breathe': 'shortness of breath',
    'difficulty breathing': 'shortness of breath',
    'loose motions': 'diarrhea',
    'watery stools': 'diarrhea',
    'tiredness': 'fatigue',
    'exhaustion': 'fatigue',
    'tired': 'fatigue',
    'body ache': 'muscle pain',
    'body aches': 'muscle pain',
    'muscle aches': 'muscle pain',
    'joint aches': 'joint pain',
    'head ache': 'headache',
    'severe headache': 'headache',
    'high temperature': 'fever',
    'feverish': 'fever',
    'chills': 'chills',
    'coughing': 'cough',
    'phlegm': 'sputum',
    'dizzy': 'dizziness',
    'lightheaded': 'dizziness',
    'lightheadedness': 'dizziness',
    'runny nose': 'nasal congestion',
    'stuffy nose': 'nasal congestion',
    'throat ache': 'sore throat',
    'itchy skin': 'itching',
    'itchiness': 'itching',
    'rashes': 'skin rash',
    'rash': 'skin rash',
    'chest pressure': 'chest tightness',
    'burning chest': 'sharp chest pain',
}


def normalize_symptom_name(raw_name):
    """
    Cleans, strips, and removes punctuation from a symptom string.
    """
    s = re.sub(r'[^\w\s]', ' ', raw_name or '')
    return re.sub(r'\s+', ' ', s).strip().lower()


def resolve_symptoms(selected_symptoms, all_symptoms):
    """
    Resolves input symptoms against the model's feature set using:
    1. Direct normalized match
    2. Common synonym mapping
    3. Fuzzy string matching (difflib)
    Returns:
      - matched_features: set of matched feature names in all_symptoms
      - resolutions: list of tuples (original, resolved_feature, match_type)
    """
    all_symptoms_norm = {normalize_symptom_name(s): s for s in all_symptoms}
    matched_features = set()
    resolutions = []

    for sym in selected_symptoms:
        norm = normalize_symptom_name(sym)
        if not norm:
            continue

        # 1. Direct match
        if norm in all_symptoms_norm:
            matched_feature = all_symptoms_norm[norm]
            matched_features.add(matched_feature)
            resolutions.append((sym, matched_feature, 'exact'))
            continue

        # 2. Synonym match
        synonym_target = COMMON_SYNONYMS.get(norm)
        if synonym_target and normalize_symptom_name(synonym_target) in all_symptoms_norm:
            matched_feature = all_symptoms_norm[normalize_symptom_name(synonym_target)]
            matched_features.add(matched_feature)
            resolutions.append((sym, matched_feature, 'synonym'))
            continue

        # 3. Fuzzy match
        close_matches = difflib.get_close_matches(norm, list(all_symptoms_norm.keys()), n=1, cutoff=0.72)
        if close_matches:
            matched_feature = all_symptoms_norm[close_matches[0]]
            matched_features.add(matched_feature)
            resolutions.append((sym, matched_feature, 'fuzzy'))
        else:
            # Substring fallback
            found_sub = None
            for cand_norm, orig_feat in all_symptoms_norm.items():
                if norm in cand_norm or cand_norm in norm:
                    found_sub = orig_feat
                    break
            if found_sub:
                matched_features.add(found_sub)
                resolutions.append((sym, found_sub, 'substring'))
            else:
                # Token overlap fallback
                norm_words = set(norm.split())
                best_cand = None
                best_overlap = 0
                for cand_norm, orig_feat in all_symptoms_norm.items():
                    cand_words = set(cand_norm.split())
                    overlap = len(norm_words & cand_words)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_cand = orig_feat
                if best_overlap >= 1 and (best_overlap >= len(norm_words) * 0.5 or best_overlap >= 2):
                    matched_features.add(best_cand)
                    resolutions.append((sym, best_cand, 'token_overlap'))
                else:
                    resolutions.append((sym, None, 'unmatched'))

    return matched_features, resolutions


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


def calculate_clinical_risk(disease_name, confidence, is_emergency, symptoms_count, disease_severity):
    """
    Stratifies overall patient clinical risk based on triage flags, disease severity, and confidence.
    Returns:
      - risk_tier: 'CRITICAL', 'HIGH', 'MODERATE', or 'LOW'
      - urgency_timeframe: str
      - risk_summary: str
    """
    if is_emergency:
        return 'CRITICAL', 'Immediate Emergency Care (0-2 Hours)', 'Emergency red flags detected. Immediate clinical intervention required.'

    high_disease_keywords = [
        'heart', 'cardiac', 'infarction', 'stroke', 'failure', 'acute',
        'pulmonary', 'sepsis', 'shock', 'hepatitis', 'meningitis',
        'cancer', 'tumor', 'coma', 'aneurysm', 'hemorrhage'
    ]
    name_lower = (disease_name or '').lower()
    is_severe_condition = any(kw in name_lower for kw in high_disease_keywords) or (disease_severity or '').lower() in ['high', 'severe', 'critical']

    if is_severe_condition and confidence >= 50.0:
        return 'HIGH', 'Within 24 Hours', 'High-priority clinical condition identified. Prompt physician evaluation strongly advised.'
    elif is_severe_condition or confidence >= 80.0 or symptoms_count >= 6:
        return 'MODERATE', 'Within 2 to 3 Days', 'Moderate risk profile. Schedule an outpatient consultation with a healthcare provider.'
    else:
        return 'LOW', 'Within 5 to 7 Days if Persistent', 'Mild or non-acute presentation. Self-monitor and consult a physician if symptoms do not improve.'


def make_prediction(selected_symptoms):
    """
    Takes a list of selected symptom names and returns comprehensive diagnostic results.
    Includes:
      - predicted_disease: str
      - confidence_score: float
      - disease_obj: Disease model instance (or None)
      - top_3: list of dicts with disease info
      - chart_labels: JSON string
      - chart_data: JSON string
      - low_confidence_warning: bool
      - key_symptoms: list of dicts representing the top contributing symptoms
      - is_emergency: bool
      - emergency_level: str ('CRITICAL', 'URGENT', 'STANDARD')
      - emergency_flags: list of dicts
      - emergency_instructions: str
      - risk_tier: str ('CRITICAL', 'HIGH', 'MODERATE', 'LOW')
      - urgency_timeframe: str
      - risk_summary: str
      - resolved_symptoms_count: int
    """
    clf = PredictorConfig.ml_model
    all_symptoms = PredictorConfig.all_symptoms
    feature_importances = PredictorConfig.feature_importances

    if clf is None or all_symptoms is None:
        raise PredictionError('Error: ML prediction model is not loaded. Please contact support.')

    if not selected_symptoms:
        raise PredictionError('Please provide at least one symptom for evaluation.')

    # 1. Emergency Red-Flag Triage
    is_emergency, emergency_level, emergency_flags, emergency_instructions = evaluate_emergency_triage(selected_symptoms)

    # 2. Intelligent Symptom Resolution (Exact, Synonym, Fuzzy)
    matched_features, resolutions = resolve_symptoms(selected_symptoms, all_symptoms)

    # Prepare model input vector (1 if symptom feature matched, 0 otherwise)
    input_data = [1 if s in matched_features else 0 for s in all_symptoms]
    input_df = pd.DataFrame([input_data], columns=all_symptoms)

    # 3. Model Inference
    predicted_disease = clf.predict(input_df)[0]
    probabilities = clf.predict_proba(input_df)[0]

    class_probs = list(zip(clf.classes_, probabilities))
    class_probs.sort(key=lambda x: x[1], reverse=True)
    top_5 = class_probs[:5]

    confidence_score = round(top_5[0][1] * 100, 2)
    low_confidence_warning = confidence_score < 30.0

    # 4. Identify Key Contributing Symptoms
    key_symptoms = []
    if feature_importances:
        selected_importances = []
        for s in all_symptoms:
            if s in matched_features and s in feature_importances:
                selected_importances.append({'name': s.title(), 'importance': feature_importances[s]})

        selected_importances.sort(key=lambda x: x['importance'], reverse=True)
        key_symptoms = selected_importances[:4]

    # 5. Fetch Differential Diagnosis Info
    top_3 = []
    for i in range(min(4, len(top_5))):
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
    disease_severity = getattr(disease_obj, 'severity', 'Moderate') if disease_obj else 'Moderate'

    # 6. Patient Risk Stratification
    risk_tier, urgency_timeframe, risk_summary = calculate_clinical_risk(
        disease_name=predicted_disease,
        confidence=confidence_score,
        is_emergency=is_emergency,
        symptoms_count=len(selected_symptoms),
        disease_severity=disease_severity
    )

    return {
        'predicted_disease': predicted_disease,
        'confidence_score': confidence_score,
        'disease_obj': disease_obj,
        'top_3': top_3,
        'chart_labels': chart_labels,
        'chart_data': chart_data,
        'low_confidence_warning': low_confidence_warning,
        'key_symptoms': key_symptoms,
        'is_emergency': is_emergency,
        'emergency_level': emergency_level,
        'emergency_flags': emergency_flags,
        'emergency_instructions': emergency_instructions,
        'risk_tier': risk_tier,
        'urgency_timeframe': urgency_timeframe,
        'risk_summary': risk_summary,
        'resolved_symptoms_count': len(matched_features)
    }
