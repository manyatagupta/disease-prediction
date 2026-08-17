from django.shortcuts import redirect
from django.contrib.auth import login, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.contrib.auth.views import LoginView
from django.urls import reverse_lazy
from django.views.generic import CreateView
from django.views import View

class SignUpView(CreateView):
    form_class = UserCreationForm
    template_name = 'accounts/signup.html'
    success_url = reverse_lazy('predictor:home')

    def form_valid(self, form):
        response = super().form_valid(form)
        login(self.request, self.object)
        messages.success(self.request, 'Registration successful. Welcome!')
        return response

class CustomLoginView(LoginView):
    template_name = 'accounts/login.html'
    
    def form_valid(self, form):
        messages.success(self.request, f"You are now logged in as {form.get_user().username}.")
        return super().form_valid(form)
        
    def get_success_url(self):
        return reverse_lazy('predictor:home')

class CustomLogoutView(View):
    def post(self, request, *args, **kwargs):
        logout(request)
        messages.info(request, 'You have successfully logged out.')
        return redirect('predictor:home')
        
    def get(self, request, *args, **kwargs):
        logout(request)
        messages.info(request, 'You have successfully logged out.')
        return redirect('predictor:home')
