from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# Assuming your features are in columns 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
# Assuming your target variable is in a column named 'label'
Y = df['label']

# Initialize the model
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(X, Y)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    N = int(request.form['N'])
    P = int(request.form['P'])
    K = int(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Scale the input data
    data = [[N, P, K, temperature, humidity, ph, rainfall]]
    # scaled_data = scaler.fit_transform(data)

    # Make prediction
    prediction = RF.predict(data)

    # Return the prediction result
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
