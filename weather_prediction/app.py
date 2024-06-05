from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    precip_type = int(request.form['precip_type'])
    apparent_temperature = float(request.form['apparent_temperature'])
    humidity = float(request.form['humidity'])
    wind_speed = float(request.form['wind_speed'])
    wind_bearing = float(request.form['wind_bearing'])
    visibility = float(request.form['visibility'])
    loud_cover = float(request.form['loud_cover'])
    pressure = float(request.form['pressure'])

    # Prepare the feature array for prediction
    features = np.array([[precip_type, apparent_temperature, humidity, wind_speed, wind_bearing, visibility, loud_cover, pressure]])

    # Scale the features
    features_scaled = scaler.transform(features)

    # Make the prediction
    prediction = model.predict(features_scaled)

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
