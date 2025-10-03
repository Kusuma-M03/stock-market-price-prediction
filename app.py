from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
with open(r"stockprediction_model1.pkl", "rb") as f:
    model, scaler = pickle.load(f)

# Home page
@app.route('/')
def home():
    return render_template("index.html", prediction=None)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from form
        features = [
            float(request.form['open']),
            float(request.form['high']),
            float(request.form['low']),
            float(request.form['volume']),
            float(request.form['hl_pct']),
            float(request.form['pct_change']),
            float(request.form['ma10']),
            float(request.form['ma50'])
        ]

        # Preprocess and predict
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]

        return render_template("index.html", prediction=round(prediction, 2))

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
