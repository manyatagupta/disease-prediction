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
from django.http import JsonResponse

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
            
        from .services import make_prediction, PredictionError
        try:
            results = make_prediction(selected_symptoms)
        except PredictionError as e:
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                from django.http import JsonResponse
                return JsonResponse({'success': False, 'error': str(e)})
            messages.error(request, str(e))
            return redirect('predictor:checker')
        
        disease_obj = results['disease_obj']
        
        prediction_id = None
        if request.user.is_authenticated:
            history_obj = PredictionHistory.objects.create(
                user=request.user,
                predicted_disease=results['predicted_disease'],
                confidence=results['confidence_score'],
                symptoms_selected=", ".join(selected_symptoms)
            )
            prediction_id = history_obj.id
            
        context = {
            'predicted_disease': results['predicted_disease'],
            'confidence_score': results['confidence_score'],
            'disease_info': disease_obj,
            'low_confidence_warning': results['low_confidence_warning'],
            'chart_labels': results['chart_labels'],
            'chart_data': results['chart_data'],
            'selected_symptoms': selected_symptoms,
            'top_3': results['top_3'],
            'key_symptoms': results['key_symptoms'],
            'prediction_id': prediction_id
        }
        
        # Check if request is AJAX
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            import json as json_lib
            from django.http import JsonResponse
            return JsonResponse({
                'success': True,
                'predicted_disease': results['predicted_disease'],
                'confidence_score': results['confidence_score'],
                'severity': getattr(disease_obj, 'severity', 'Moderate') if disease_obj else 'Moderate',
                'description': disease_obj.description if disease_obj else 'No description available.',
                'precautions': disease_obj.precautions if disease_obj else 'Consult a doctor.',
                'specialist': getattr(disease_obj, 'recommended_doctor', 'General Physician') if disease_obj else 'General Physician',
                'low_confidence': results['low_confidence_warning'],
                'chart_labels': json_lib.loads(results['chart_labels']),
                'chart_data': json_lib.loads(results['chart_data']),
                'top_3': results['top_3'],
                'key_symptoms': results['key_symptoms']
            })
            
        return render(request, 'predictor/results.html', context)
        
    return render(request, 'predictor/checker.html', {'symptoms': symptoms})

@login_required
def submit_feedback(request):
    if request.method == 'POST':
        import json as json_lib
        try:
            data = json_lib.loads(request.body)
            prediction_id = data.get('prediction_id')
            is_helpful = data.get('is_helpful')
            
            if prediction_id and is_helpful is not None:
                history = PredictionHistory.objects.filter(id=prediction_id, user=request.user).first()
                if history:
                    history.is_helpful = is_helpful
                    history.save()
                    return JsonResponse({'success': True})
        except Exception:
            pass
    return JsonResponse({'success': False}, status=400)
