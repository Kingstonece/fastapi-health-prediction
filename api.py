from fastapi import FastAPI, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import uvicorn

# Load trained model, scaler, and encoders
rf_model = joblib.load("health_risk_classifier.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Serve static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Function to predict risk
def predict_risk(heart_rate, respiratory_rate, body_temp, spo2, systolic_bp, diastolic_bp, age, gender, weight, height, hrv, pulse_pressure, bmi, map_value):
    # Encode gender
    gender_encoded = label_encoders['Gender'].transform([gender])[0]
    
    # Prepare input data
    input_data = np.array([[heart_rate, respiratory_rate, body_temp, spo2, systolic_bp, diastolic_bp, age, gender_encoded,
                            weight, height, hrv, pulse_pressure, bmi, map_value]])

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Predict risk category
    prediction = rf_model.predict(input_scaled)[0]

    # Decode the risk category
    risk_category = label_encoders['Risk Category'].inverse_transform([prediction])[0]

    return risk_category

# Serve the main webpage
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Handle form submission
@app.post("/predict/")
def get_risk_prediction(
    request: Request,
    heart_rate: int = Form(...),
    respiratory_rate: int = Form(...),
    body_temp: float = Form(...),
    spo2: float = Form(...),
    systolic_bp: int = Form(...),
    diastolic_bp: int = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    weight: float = Form(...),
    height: float = Form(...),
    hrv: float = Form(...),
    pulse_pressure: int = Form(...),
    bmi: float = Form(...),
    map_value: float = Form(...)
):
    # Get prediction
    risk_category = predict_risk(heart_rate, respiratory_rate, body_temp, spo2, systolic_bp, diastolic_bp, age, gender, weight, height, hrv, pulse_pressure, bmi, map_value)
    
    return templates.TemplateResponse("index.html", {"request": request, "result": risk_category})
