import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load AdaBoost Model
MODEL_PATH = "AdaBoost.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# Embedded HTML Template with embedded CSS animations and responsive UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AdaBoost AI Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-border: rgba(255, 255, 255, 0.12);
            --primary-accent: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            --input-border: rgba(255, 255, 255, 0.15);
            --focus-ring: #818cf8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: var(--text-main);
            padding: 2rem 1rem;
            overflow-x: hidden;
            position: relative;
        }

        /* Ambient Glow Background Animation */
        body::before, body::after {
            content: '';
            position: absolute;
            width: 320px;
            height: 320px;
            border-radius: 50%;
            filter: blur(100px);
            z-index: 0;
            animation: float 12s infinite alternate ease-in-out;
        }

        body::before {
            background: rgba(99, 102, 241, 0.35);
            top: 10%;
            left: 15%;
        }

        body::after {
            background: rgba(168, 85, 247, 0.3);
            bottom: 10%;
            right: 15%;
            animation-delay: -6s;
        }

        @keyframes float {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(40px, 50px) scale(1.15); }
        }

        .container {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: fadeInScale 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes fadeInScale {
            0% { opacity: 0; transform: scale(0.95) translateY(20px); }
            100% { opacity: 1; transform: scale(1) translateY(0); }
        }

        .header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .header h1 {
            font-size: 2.25rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #cbd5e1;
            letter-spacing: 0.01em;
        }

        .input-group input, .input-group select {
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            color: #ffffff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.25s ease;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--focus-ring);
            box-shadow: 0 0 0 4px rgba(129, 140, 248, 0.2);
            background: rgba(15, 23, 42, 0.8);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background: var(--primary-accent);
            color: #ffffff;
            border: none;
            border-radius: 12px;
            padding: 0.9rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4);
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px -5px rgba(168, 85, 247, 0.5);
            opacity: 0.95;
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        /* Result Section */
        .result-card {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--card-border);
            text-align: center;
            display: none;
            animation: slideUp 0.5s ease-out forwards;
        }

        @keyframes slideUp {
            0% { opacity: 0; transform: translateY(15px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        .result-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .result-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #38bdf8;
            text-shadow: 0 0 12px rgba(56, 189, 248, 0.3);
        }

        .spinner {
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: #ffffff;
            animation: spin 0.8s linear infinite;
            display: none;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>AdaBoost Classification</h1>
        <p>Interactive Machine Learning Prediction Engine</p>
    </div>

    <form id="predictionForm">
        <div class="form-grid">
            <div class="input-group">
                <label for="Age">Age</label>
                <input type="number" id="Age" name="Age" placeholder="e.g. 35" required>
            </div>
            <div class="input-group">
                <label for="Gender">Gender (0 = Male, 1 = Female)</label>
                <input type="number" id="Gender" name="Gender" min="0" max="1" placeholder="0 or 1" required>
            </div>
            <div class="input-group">
                <label for="Tenure">Tenure (Months)</label>
                <input type="number" id="Tenure" name="Tenure" placeholder="e.g. 12" required>
            </div>
            <div class="input-group">
                <label for="Usage Frequency">Usage Frequency</label>
                <input type="number" id="Usage Frequency" name="Usage Frequency" placeholder="e.g. 18" required>
            </div>
            <div class="input-group">
                <label for="Support Calls">Support Calls</label>
                <input type="number" id="Support Calls" name="Support Calls" placeholder="e.g. 2" required>
            </div>
            <div class="input-group">
                <label for="Payment Delay">Payment Delay (Days)</label>
                <input type="number" id="Payment Delay" name="Payment Delay" placeholder="e.g. 5" required>
            </div>
            <div class="input-group">
                <label for="Subscription Type">Subscription Type (Encoded)</label>
                <input type="number" id="Subscription Type" name="Subscription Type" placeholder="e.g. 1" required>
            </div>
            <div class="input-group">
                <label for="Contract Length">Contract Length (Months)</label>
                <input type="number" id="Contract Length" name="Contract Length" placeholder="e.g. 12" required>
            </div>
            <div class="input-group">
                <label for="Total Spend">Total Spend ($)</label>
                <input type="number" step="0.01" id="Total Spend" name="Total Spend" placeholder="e.g. 450.50" required>
            </div>
            <div class="input-group">
                <label for="Last Interaction">Last Interaction (Days ago)</label>
                <input type="number" id="Last Interaction" name="Last Interaction" placeholder="e.g. 14" required>
            </div>

            <button type="submit" class="btn-submit" id="submitBtn">
                <span id="btnText">Predict Output</span>
                <div class="spinner" id="btnSpinner"></div>
            </button>
        </div>
    </form>

    <div class="result-card" id="resultCard">
        <div class="result-title">Model Output</div>
        <div class="result-value" id="resultValue">-</div>
    </div>
</div>

<script>
    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();

        const btnText = document.getElementById('btnText');
        const btnSpinner = document.getElementById('btnSpinner');
        const resultCard = document.getElementById('resultCard');
        const resultValue = document.getElementById('resultValue');

        // Animation UI Feedback
        btnText.innerText = "Processing...";
        btnSpinner.style.display = "block";
        resultCard.style.display = "none";

        const formData = new FormData(this);
        const data = {};
        formData.forEach((value, key) => { data[key] = parseFloat(value); });

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok) {
                resultValue.innerText = `Class: ${result.prediction}`;
            } else {
                resultValue.innerText = `Error: ${result.error}`;
            }
        } catch (err) {
            resultValue.innerText = "Error submitting values";
        } finally {
            btnText.innerText = "Predict Output";
            btnSpinner.style.display = "none";
            resultCard.style.display = "block";
        }
    });
</script>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model file AdaBoost.pkl not found on server'}), 500

    try:
        data = request.get_json()
        
        # Expected feature order matching model
        features = [
            'Age', 'Gender', 'Tenure', 'Usage Frequency', 
            'Support Calls', 'Payment Delay', 'Subscription Type', 
            'Contract Length', 'Total Spend', 'Last Interaction'
        ]
        
        input_data = [data.get(feat, 0) for feat in features]
        prediction = model.predict([input_data])[0]

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Default port for AWS App Runner or local testing
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
