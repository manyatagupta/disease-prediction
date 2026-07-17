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
        prediction = clf.predict(input_df)[0]
        probabilities = clf.predict_proba(input_df)[0]
        confidence = max(probabilities) * 100
        
        # Fetch disease info
        disease = Disease.objects.filter(name=prediction).first()
        
        # Save to history
        PredictionHistory.objects.create(
            user=request.user,
            symptoms_selected=", ".join(selected_symptoms),
            predicted_disease=prediction,
            confidence=confidence
        )
        
        context = {
            'predicted_disease': prediction,
            'confidence': round(confidence, 2),
            'disease': disease,
            'selected_symptoms': selected_symptoms
        }
        return render(request, 'predictor/results.html', context)
        
    return render(request, 'predictor/checker.html', {'symptoms': symptoms})
