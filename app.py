# app.py
from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Load the best model
best_model = joblib.load('model.pkl')

# Load the label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Flask App
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_date = request.form.get('date')
    try:
        input_date = pd.to_datetime(input_date, format='%d-%m-%Y')
    except ValueError:
        return render_template('index.html', error='Invalid date format. Use DD-MM-YYYY.')

    # Create features from the input date
    month = input_date.month
    day = input_date.day
    dayofyear = input_date.dayofyear

    # Create a DataFrame with the date features
    input_df = pd.DataFrame([[month, day, dayofyear]], columns=['month', 'day', 'dayofyear'])
    input_scaled = scaler.transform(input_df)  # Use the same scaler as before
    prediction_encoded = best_model.predict(input_scaled)[0]

    # Convert the numerical prediction back to the original string label
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]

    # Pass prediction to result.html
    return render_template('result.html', prediction=prediction)

@app.route('/result', methods=['GET'])
def result():
    # For simplicity, you can hardcode a prediction value or retrieve it from your model logic
    # For example, uncomment the line below to hardcode a prediction
    # prediction = "Sunny"
    return render_template('result.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
