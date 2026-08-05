import os
import joblib
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Disease, Symptom, PredictionHistory
from django.conf import settings
from .apps import PredictorConfig
from django.core.paginator import Paginator
from django.db.models import Count, Avg

def home_view(request):
    return render(request, 'predictor/home.html')

@login_required
def history_view(request):
    all_history = PredictionHistory.objects.filter(user=request.user).order_by('-date')
    
    # Summary stats
    total_predictions = all_history.count()
    
    most_frequent = None
    if total_predictions > 0:
        most_frequent_data = all_history.values('predicted_disease').annotate(count=Count('predicted_disease')).order_by('-count').first()
        most_frequent = most_frequent_data['predicted_disease'] if most_frequent_data else None
        
    avg_conf = all_history.aggregate(Avg('confidence'))['confidence__avg']
    avg_confidence = round(avg_conf, 1) if avg_conf else 0.0

    # Pagination
    paginator = Paginator(all_history, 10) # 10 items per page
    page_number = request.GET.get('page')
    history_page = paginator.get_page(page_number)

    context = {
        'history': history_page,
        'total_predictions': total_predictions,
        'most_frequent_disease': most_frequent,
        'avg_confidence': avg_confidence
    }
    return render(request, 'predictor/history.html', context)

@login_required
def delete_history_item(request, item_id):
    if request.method == 'POST':
        item = PredictionHistory.objects.filter(id=item_id, user=request.user).first()
        if item:
            item.delete()
            messages.success(request, "Prediction record deleted successfully.")
        else:
            messages.error(request, "Record not found.")
    return redirect('predictor:history')

@login_required
def clear_history(request):
    if request.method == 'POST':
        PredictionHistory.objects.filter(user=request.user).delete()
        messages.success(request, "All prediction history has been cleared.")
    return redirect('predictor:history')

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
            
        # Use the pre-loaded ML model
        clf = PredictorConfig.ml_model
        all_symptoms = PredictorConfig.all_symptoms
        
        if clf is None or all_symptoms is None:
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                from django.http import JsonResponse
                return JsonResponse({'success': False, 'error': 'Error: ML prediction model is not loaded. Please contact support.'})
            messages.error(request, "Error: ML prediction model is not loaded. Please contact support.")
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
        
        # Using raw probabilities and normalizing the top 5 to sum to 100% for the chart
        import numpy as np
        probs = np.array([x[1] for x in top_5])
        
        top_5_sum = np.sum(probs)
        if top_5_sum > 0:
            normalized_top_5 = [(top_5[i][0], float(probs[i] / top_5_sum)) for i in range(len(top_5))]
        else:
            normalized_top_5 = top_5
            
        confidence_score = round(normalized_top_5[0][1] * 100, 2)
        
        # Extract Top 3 for UI
        top_3 = []
        for i in range(min(3, len(normalized_top_5))):
            d_name = normalized_top_5[i][0]
            d_conf = round(normalized_top_5[i][1] * 100, 2)
            d_obj = Disease.objects.filter(name=d_name).first()
            if d_obj:
                top_3.append({
                    'name': d_name,
                    'confidence': d_conf,
                    'severity': getattr(d_obj, 'severity', 'Moderate'),
                    'description': d_obj.description,
                    'precautions': d_obj.precautions,
                    'specialist': d_obj.specialist
                })
        
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
            'selected_symptoms': selected_symptoms,
            'top_3': top_3
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
                'chart_data': json_lib.loads(chart_data),
                'top_3': top_3
            })
            
        return render(request, 'predictor/results.html', context)
        
    return render(request, 'predictor/checker.html', {'symptoms': symptoms})
