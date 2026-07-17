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
        confidence_score = round(max(probabilities) * 100, 2)
        
        low_confidence_warning = confidence_score < 30.0
        
        disease_obj = Disease.objects.filter(name=predicted_disease).first()
        
        if request.user.is_authenticated:
            PredictionHistory.objects.create(
                user=request.user,
                predicted_disease=predicted_disease,
                confidence_score=confidence_score,
                symptoms_submitted=", ".join(selected_symptoms)
            )
            
        context = {
            'predicted_disease': predicted_disease,
            'confidence_score': confidence_score,
            'disease_info': disease_obj,
            'low_confidence_warning': low_confidence_warning
        }
        return render(request, 'predictor/results.html', context)
        
    return render(request, 'predictor/checker.html', {'symptoms': symptoms})
