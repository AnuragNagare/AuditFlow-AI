# app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# === LOAD TRAINED MODELS ===
BASE_DIR = r"D:\Git hub project\AuditFlow AI"
MODEL_DIR = os.path.join(BASE_DIR, "models")

anomaly_model = joblib.load(os.path.join(MODEL_DIR, "anomaly_model.pkl"))
fraud_model = joblib.load(os.path.join(MODEL_DIR, "fraud_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

print("AuditFlow AI models loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    df = pd.read_csv(file)

    # === FEATURE ENGINEERING (Same as training) ===
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['amount_log'] = np.log1p(df['amount'])
    df['vendor_freq'] = df.groupby('vendor')['txn_id'].transform('count')

    # Duplicate detection
    df = df.sort_values(['vendor', 'amount', 'date']).reset_index(drop=True)
    df['prev_date'] = df.groupby(['vendor', 'amount'])['date'].shift(1)
    df['days_diff'] = (df['date'] - df['prev_date']).dt.days
    df['is_duplicate'] = (df['days_diff'] <= 7).astype(int).fillna(0)

    # Features
    features = ['amount', 'amount_log', 'is_weekend', 'vendor_freq', 'is_duplicate', 'day_of_week']
    X = df[features]
    X_scaled = scaler.transform(X)

    # === AI PREDICTIONS ===
    fraud_probs = fraud_model.predict_proba(X_scaled)[:, 1] * 100
    anomaly_flags = anomaly_model.predict(X) == -1

    df['risk_score'] = fraud_probs.round(2)
    df['is_anomaly'] = anomaly_flags
    df['flag'] = np.where((df['risk_score'] > 70) | df['is_anomaly'], 'High Risk', 'Low Risk')

    # === AI REASONING ENGINE ===
    reasons = []
    for idx, row in df.iterrows():
        reason_parts = []
        
        if row['is_duplicate']:
            reason_parts.append("Duplicate transaction detected with same vendor & amount within 7 days")
        if row['amount'] > 100000:
            reason_parts.append("Very high value transaction above ₹1,00,000")
        if row['amount'] % 10000 == 0 and row['amount'] > 50000:
            reason_parts.append("Round figure amount (common in fraudulent entries)")
        if row['is_weekend']:
            reason_parts.append("Transaction on weekend — unusual for business expenses")
        if row['vendor_freq'] > df['vendor_freq'].quantile(0.95):
            reason_parts.append("Vendor appears unusually frequently")
        if row['risk_score'] > 80:
            reason_parts.append("XGBoost fraud model scored extremely high risk")
        if row['is_anomaly']:
            reason_parts.append("Isolation Forest flagged as statistical anomaly")

        if reason_parts:
            reason = " • ".join(reason_parts[:3])  # Top 3 reasons
        else:
            reason = "Low risk — normal business transaction pattern"
            
        reasons.append(reason)

    df['ai_reason'] = reasons

    # === FINAL RESULTS ===
    total = len(df)
    flagged = len(df[df['flag'] == 'High Risk'])

    high_risk_df = df[df['flag'] == 'High Risk'].copy()
    high_risk_df['date'] = high_risk_df['date'].dt.strftime('%Y-%m-%d')
    high_risk = high_risk_df.head(20).to_dict('records')

    return render_template('dashboard.html',
                         total=total,
                         flagged=flagged,
                         high_risk=high_risk,
                         percentage=f"{flagged/total*100:.1f}" if total > 0 else "0")

if __name__ == '__main__':
    app.run(debug=True)