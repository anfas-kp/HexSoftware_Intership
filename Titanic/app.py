from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model and preprocessed data
with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    age = float(request.form['age'])
    sibsp = float(request.form['sibsp'])
    parch = float(request.form['parch'])
    fare = float(request.form['fare'])

    # Create a DataFrame for the input
    input_data = pd.DataFrame([[age, sibsp, parch, fare]], columns=['age', 'sibsp', 'parch', 'fare'])

    # Make a prediction
    prediction = model.predict(input_data)[0]
    result = "Survived" if prediction == 1 else "Not Survived"

    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == '__main__':
    app.run(debug=True)