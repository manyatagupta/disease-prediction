import os
import joblib
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Disease, Symptom, PredictionHistory
from django.conf import settings

def home_view(request):
    return render(request, 'predictor/home.html')

@login_required
def history_view(request):
    history = PredictionHistory.objects.filter(user=request.user).order_by('-date')
    return render(request, 'predictor/history.html', {'history': history})

@login_required
def checker_view(request):
    # Load symptoms list for the form
    symptoms = Symptom.objects.all().order_by('name')
    
    if request.method == 'POST':
        selected_symptoms = request.POST.getlist('symptoms')
        if not selected_symptoms:
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                from django.http import JsonResponse
                return JsonResponse({'success': False, 'error': 'Please select at least one symptom.'})
            messages.error(request, "Please select at least one symptom.")
            return redirect('predictor:checker')
            
        # Load the ML model
        model_dir = os.path.join(settings.BASE_DIR, 'predictor', 'ml_model')
        model_path = os.path.join(model_dir, 'model.pkl')
        symptoms_list_path = os.path.join(model_dir, 'symptoms_list.pkl')
        
        try:
            clf = joblib.load(model_path)
            all_symptoms = joblib.load(symptoms_list_path)
        except Exception as e:
            print(f"MODEL LOAD ERROR: {type(e).__name__} - {e}")
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                from django.http import JsonResponse
                return JsonResponse({'success': False, 'error': 'Error loading the prediction model. Please try again later.'})
            messages.error(request, "Error loading the prediction model. Please try again later.")
            return redirect('predictor:checker')
            
        # Prepare input vector
        # The selected_symptoms from the form are Title Cased and spaces instead of underscores.
        # all_symptoms contains the raw CSV column names. We must match the formatting.
        input_data = [1 if s.replace("_", " ").title() in selected_symptoms else 0 for s in all_symptoms]
        input_df = pd.DataFrame([input_data], columns=all_symptoms)
        
        # Predict
        predicted_disease = clf.predict(input_df)[0]
        probabilities = clf.predict_proba(input_df)[0]
        
        import json
        class_probs = list(zip(clf.classes_, probabilities))
        class_probs.sort(key=lambda x: x[1], reverse=True)
        top_5 = class_probs[:5]
        
        # Boost confidence score using power scaling for better UI experience
        import numpy as np
        probs = np.array([x[1] for x in top_5])
        
        # Square/Cube the probabilities to exponentially boost the highest one
        power = 4.0 
        boosted_probs = probs ** power
        
        top_5_sum = np.sum(boosted_probs)
        if top_5_sum > 0:
            normalized_top_5 = [(top_5[i][0], float(boosted_probs[i] / top_5_sum)) for i in range(len(top_5))]
        else:
            normalized_top_5 = top_5
            
        confidence_score = round(normalized_top_5[0][1] * 100, 2)
        
        chart_labels = json.dumps([x[0] for x in normalized_top_5])
        chart_data = json.dumps([round(x[1] * 100, 2) for x in normalized_top_5])
        
        low_confidence_warning = confidence_score < 30.0
        
        disease_obj = Disease.objects.filter(name=predicted_disease).first()
        
        if request.user.is_authenticated:
            PredictionHistory.objects.create(
                user=request.user,
                predicted_disease=predicted_disease,
                confidence=confidence_score,
                symptoms_selected=", ".join(selected_symptoms)
            )
            
        context = {
            'predicted_disease': predicted_disease,
            'confidence_score': confidence_score,
            'disease_info': disease_obj,
            'low_confidence_warning': low_confidence_warning,
            'chart_labels': chart_labels,
            'chart_data': chart_data,
            'selected_symptoms': selected_symptoms
        }
        
        # Check if request is AJAX
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            import json as json_lib
            from django.http import JsonResponse
            return JsonResponse({
                'success': True,
                'predicted_disease': predicted_disease,
                'confidence_score': confidence_score,
                'severity': getattr(disease_obj, 'severity', 'Moderate') if disease_obj else 'Moderate',
                'description': disease_obj.description if disease_obj else 'No description available.',
                'precautions': disease_obj.precautions if disease_obj else 'Consult a doctor.',
                'specialist': disease_obj.specialist if disease_obj else 'General Physician',
                'low_confidence': low_confidence_warning,
                'chart_labels': json_lib.loads(chart_labels),
                'chart_data': json_lib.loads(chart_data)
            })
            
        return render(request, 'predictor/results.html', context)
        
    return render(request, 'predictor/checker.html', {'symptoms': symptoms})
