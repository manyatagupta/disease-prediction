import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from groq import Groq

# Initialize Groq client
client = Groq(api_key=settings.GROQ_API_KEY)

from django.http import JsonResponse, StreamingHttpResponse

@csrf_exempt
def chat_with_grok(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            # Expecting a list of messages from the client
            messages = data.get('messages', [])
            
            if not messages:
                return JsonResponse({'success': False, 'error': 'Messages cannot be empty.'})
            
            system_prompt = """
            You are Grok, an advanced AI Health Assistant for MediPredict.
            You help users understand their symptoms, provide general precautions, diet plans, and medical advice.
            Important Rules:
            1. Always include a disclaimer that you are an AI and they should consult a real doctor for serious conditions.
            2. Be concise, empathetic, and professional.
            3. Use markdown formatting for readability (e.g., bullet points for precautions).
            """
            
            # Format messages for Groq API
            api_messages = [{"role": "system", "content": system_prompt}]
            for msg in messages:
                api_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
            
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=api_messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stream=True,
            )
            
            def event_stream():
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        yield f"data: {json.dumps({'content': content})}\n\n"
                yield "data: [DONE]\n\n"
                
            return StreamingHttpResponse(event_stream(), content_type='text/event-stream')
            
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
            file_name = report_file.name.lower()
            
            system_prompt = """
            You are Grok, an advanced AI Health Assistant for MediPredict.
            Your task is to analyze the medical report (blood test, lab results, etc.).
            1. Summarize the overall findings in plain English.
            2. Highlight any abnormal values or areas of concern.
            3. Provide general recommendations.
            4. Include a disclaimer to consult a doctor.
            Use Markdown for formatting.
            """
            
            if file_name.endswith('.pdf'):
                import PyPDF2
                text_content = ""
                reader = PyPDF2.PdfReader(report_file)
                for page in reader.pages:
                    text_content += page.extract_text() + "\n"
                    
                if not text_content.strip():
                    return JsonResponse({'success': False, 'error': 'Could not extract text from the PDF.'})
                    
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Here is the report text:\n\n{text_content[:4000]}"}
                    ],
                    temperature=0.3,
                    max_tokens=1500,
                )
                
            elif file_name.endswith(('.jpg', '.jpeg', '.png')):
                import base64
                encoded_string = base64.b64encode(report_file.read()).decode('utf-8')
                
                # Determine mime type
                mime_type = "image/png" if file_name.endswith('.png') else "image/jpeg"
                image_url = f"data:{mime_type};base64,{encoded_string}"
                
                completion = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": "Please analyze this medical report image and extract the key information as requested."},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]}
                    ],
                    temperature=0.3,
                    max_tokens=1500,
                )
            else:
                return JsonResponse({'success': False, 'error': 'Currently only PDF, JPG, and PNG files are supported.'})
            
            ai_response = completion.choices[0].message.content
            return JsonResponse({'success': True, 'response': ai_response})
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
            
    # For GET requests, render the template
    from django.shortcuts import render
    return render(request, 'predictor/report_analyzer.html')
