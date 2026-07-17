# AI Disease Prediction Web Application

This is a modern, responsive web application built with Python (Django) that leverages a Machine Learning model (Random Forest Classifier) to predict potential diseases based on a user's selected symptoms. The application uses a massive dataset containing 773 unique diseases and 377 symptoms.

## Features

* **Premium UI**: Built with custom Vanilla CSS featuring glassmorphism, dynamic gradients, and smooth micro-animations.
* **AI Symptom Checker**: Select from 377 different symptoms. The Random Forest model instantly predicts the most likely disease and provides a confidence score.
* **Actionable Advice**: Displays descriptions, suggested precautions, and the recommended specialist (e.g., General Physician, Neurologist) for the predicted disease.
* **User Authentication**: Secure sign up and login system (powered by Django Auth).
* **Prediction History**: Logged-in users have a personalized dashboard to track their past predictions and selected symptoms.

## Tech Stack

* **Backend**: Python, Django
* **Machine Learning**: `scikit-learn`, `pandas`, `numpy`, `joblib`
* **Frontend**: HTML5, Vanilla CSS, JavaScript
* **Database**: SQLite (Development)

## Dataset

The ML model is trained on the Kaggle *Disease Symptom Prediction* augmented dataset (`Final_Augmented_dataset_Diseases_and_Symptoms.csv`), ensuring a highly robust and accurate prediction engine (near 100% training accuracy).

## Local Setup & Installation

Follow these steps to run the project locally on your machine.

### Prerequisites
* Python 3.9+
* pip (Python Package Installer)

### 1. Clone the repository
```bash
git clone https://github.com/manyatagupta/disease-prediction.git
cd disease-prediction
```

### 2. Set up a virtual environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install django scikit-learn pandas numpy joblib
```

### 4. Run Migrations & Setup Database
```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. Train the Model & Populate Database
Run the provided script to parse the Kaggle dataset, populate the SQLite database with the symptoms and diseases, and serialize the trained Random Forest model.
*(Note: You will need the dataset CSV placed at the correct path in the script, or you can use the pre-trained `model.pkl` if included).*
```bash
python predictor/ml_model/train_model.py
```

### 6. Run the Development Server
```bash
python manage.py runserver
```
Visit `http://127.0.0.1:8000` in your browser to access the application.

## Disclaimer
*This tool is for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a physician or other qualified health provider with any questions regarding a medical condition.*
