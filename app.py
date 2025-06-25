import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Load data, model, and label encoder
data = pd.read_csv('train.csv')
model = pickle.load(open('model.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

@app.route('/')
def index():
    locations = sorted(data['Location'].unique())
    return render_template('home.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    location = request.form.get('location')
    area = float(request.form.get('area'))
    bhk = int(request.form.get('bhk'))
    ne = request.form.get('toggle')
    gy = request.form.get('gym')
    ind = request.form.get('ind')
    ca = request.form.get('car')
    jog = request.form.get('jog')

    # Convert categorical inputs to numerical
    location_encoded = le.transform([location])[0]
    new = 1 if ne == 'on' else 0
    gym = 1 if gy == 'on' else 0
    jogg = 1 if jog == 'on' else 0
    car = 1 if ca == 'on' else 0
    indd = 1 if ind == 'on' else 0

    # Prepare input DataFrame
    input_data = pd.DataFrame([[area, location_encoded, bhk, new, gym, car, indd, jogg]],
                              columns=['Area', 'Location_Encoded', 'No. of Bedrooms', 'New/Resale',
                                       'Gymnasium', 'Car Parking', 'Indoor Games', 'Jogging Track'])

    # Make prediction and convert back to normal price
    predicted_price = model.predict(input_data)[0] * 1e6

    return str(np.round(predicted_price, 2))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
