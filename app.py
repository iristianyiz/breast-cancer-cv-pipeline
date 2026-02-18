# app.py — Flask inference API
import joblib
import numpy as np
import cv2
from flask import Flask, request, jsonify

from features import preprocess, extract_features

app = Flask(__name__)

MODEL_PATH = "model.pkl"
_model = None


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    try:
        load_model()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "missing file field"}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        return jsonify({"error": "unsupported file type"}), 400

    data = f.read()
    img_arr = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"error": "invalid image"}), 400

    gray = preprocess(img_bgr)
    feats = extract_features(gray).reshape(1, -1)

    model = load_model()

    # prediction
    pred = int(model.predict(feats)[0])

    # probability if supported
    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(feats)[0][1])  # P(malignant)

    return jsonify({
        "label": "malignant" if pred == 1 else "benign",
        "prob_malignant": prob,
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
