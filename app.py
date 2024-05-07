from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            area = float(request.form['area'])  # Convert area to float
            new_resale = int(request.form['new_resale'])  # Get new/resale
            furnished = int(request.form['furnished'])  # Get furnished
        except ValueError:
            return "Invalid input. Please enter numeric value for area."

        # Prepare input data for prediction
        input_data = np.array([[area, new_resale, furnished]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
