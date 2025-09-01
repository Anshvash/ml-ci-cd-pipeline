# app.py
import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify

# --- Config ---
MODEL_PATH = os.getenv("MODEL_PATH", "model/iris_model.pkl")  # adjust filename if needed

# --- App ---
app = Flask(__name__)

# Load once at startup
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    # Fail fast with a helpful message
    raise RuntimeError(f"Could not load model from {MODEL_PATH}: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}, 200

@app.post("/predict")
def predict():
    """
    Accepts either:
    {"input": [[...feature vector...], [...]]}  # 2D list
    or
    {"input": [...feature vector...]}           # 1D list
    """
    try:
        payload = request.get_json(force=True)
        x = payload.get("input")
        if x is None:
            return jsonify(error="Missing 'input'"), 400

        # Normalize to 2D array
        if isinstance(x, list) and (len(x) > 0) and not isinstance(x[0], list):
            x = [x]

        X = np.array(x, dtype=float)
        preds = model.predict(X)
        # If your model returns numpy types, convert to Python
        preds = preds.tolist()
        return jsonify(prediction=preds), 200

    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    # Local dev only; Render will run with Gunicorn (see startCommand below)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
