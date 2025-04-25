from flask import Flask, request, render_template
import numpy as np
import joblib

# Load saved models
scaler = joblib.load("scaler.joblib")
pca = joblib.load("pca.joblib")
gmm_model = joblib.load("gmm_model.joblib")
label_map = joblib.load("gmm_label_map.joblib")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            int(request.form['Unnamed: 0']),
            int(request.form['Age']),
            int(request.form['Job']),
            int(request.form['Sex']),
            int(request.form['Housing']),
            int(request.form['Saving_accounts']),
            int(request.form['Checking_account']),
            int(request.form['Duration']),
            int(request.form['Purpose']),
            int(request.form['Credit_amount'])
        ]

        credit_dur_ratio = features[8] / (features[6] + 1)
        features.append(credit_dur_ratio)

        input_scaled = scaler.transform([features])
        input_pca = pca.transform(input_scaled)
        pred_cluster = gmm_model.predict(input_pca)[0]

        mapped_label = label_map[pred_cluster]
        result = "Bad Credit" if mapped_label == 1 else "Good Credit"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return f"Prediction Error: {str(e)}"

if __name__ == '_main_':
    app.run(debug=True)
