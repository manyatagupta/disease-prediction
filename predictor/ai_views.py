import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from groq import Groq

# Initialize Groq client
client = Groq(api_key=settings.GROQ_API_KEY)

@csrf_exempt
def chat_with_grok(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')
            
            if not user_message:
                return JsonResponse({'success': False, 'error': 'Message cannot be empty.'})
            
            system_prompt = """
            You are Grok, an advanced AI Health Assistant for MediPredict.
            You help users understand their symptoms, provide general precautions, diet plans, and medical advice.
            Important Rules:
            1. Always include a disclaimer that you are an AI and they should consult a real doctor for serious conditions.
            2. Be concise, empathetic, and professional.
            3. Use markdown formatting for readability (e.g., bullet points for precautions).
            """
            
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            
            ai_response = completion.choices[0].message.content
            
            return JsonResponse({'success': True, 'response': ai_response})
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
            
    return JsonResponse({'success': False, 'error': 'Invalid request method.'})

@csrf_exempt
def analyze_report(request):
    if request.method == 'POST':
        try:
            if 'report_file' not in request.FILES:
                return JsonResponse({'success': False, 'error': 'No file uploaded.'})
                
            report_file = request.FILES['report_file']
            
            # Simple text extraction for PDF
            text_content = ""
            if report_file.name.lower().endswith('.pdf'):
                import PyPDF2
                reader = PyPDF2.PdfReader(report_file)
                for page in reader.pages:
                    text_content += page.extract_text() + "\n"
            else:
                return JsonResponse({'success': False, 'error': 'Currently only PDF files are supported.'})
                
            if not text_content.strip():
                return JsonResponse({'success': False, 'error': 'Could not extract text from the PDF.'})
                
            system_prompt = """
            You are Grok, an advanced AI Health Assistant for MediPredict.
            Your task is to analyze the following text extracted from a medical report (blood test, lab results, etc.).
            1. Summarize the overall findings in plain English.
            2. Highlight any abnormal values or areas of concern.
            3. Provide general recommendations.
            4. Include a disclaimer to consult a doctor.
            Use Markdown for formatting.
            """
            
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Here is the report text:\n\n{text_content[:4000]}"}
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            
            ai_response = completion.choices[0].message.content
            return JsonResponse({'success': True, 'response': ai_response})
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
            
    # For GET requests, render the template
    from django.shortcuts import render
    return render(request, 'predictor/report_analyzer.html')
