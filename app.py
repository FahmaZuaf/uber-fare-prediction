from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('uber-fare-predictor/uber_fare_model.pkl')
scaler = joblib.load('uber-fare-predictor/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        distance = float(request.form['distance_km'])
        passenger = int(request.form['passenger_count'])
        hour = int(request.form['hour'])
        day = int(request.form['day_of_week'])
        month = int(request.form['month'])

        features = np.array([[distance, passenger, hour, day, month]])
        scaled = scaler.transform(features)
        fare = model.predict(scaled)[0]
        return render_template('index.html', prediction_text=f"Estimated Fare: ${fare:.2f}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")
