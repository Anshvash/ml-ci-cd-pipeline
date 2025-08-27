import os
import json
from flask import Flask, request, jsonify
import joblib
import numpy as np

MODEL_PATH  =  os.get_env("MODEL_PATH", "model/iris_model.pkl")

app = Flask(__name__)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

@app.get("/health")
def health_check():
    return {"status":"ok"}, 200


@app.post("/predict")
def predict():
    """
    Accepts either:
    {"input":[[...feature vector...] , [...]]} #2d list
    or
    {"input": [...feature vector...]}    #1d list

    """
    
    try:
        payload = request.get_json(force=True)
        x= payload.get("input")
        if x is None:
            return {"error":"Missing 'input' in request"}, 400
        
        # normalize input to 2d array
        if isinstance(x, list) and (len(x)>0) and not isinstance(x[0], list):
            x = [x]

        X = np.array(x , dtype= float)
        preds = model.predict(X)
        preds = preds.tolist()
        return jsonify({"predictions": preds}) , 200
    except Exception as e:
        return jsonify(error= str(e)), 500
    
    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=int(os.environ.get("Port", 8000)))
        