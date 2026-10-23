
import time
import mlflow
import mlflow.sklearn
from flask import Flask, jsonify
from prometheus_client import Counter, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from sklearn.datasets import load_iris

app = Flask(__name__)

# Prometheus monitoring metrics
PREDICTION_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

# Load the registered model from MLflow
MODEL_NAME = "IrisClassifier"

try:
    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}/1"
    )
except Exception:
    model = None

iris = load_iris()

@app.route("/")
def home():
    return jsonify({
        "application": "ML Model Deployment",
        "status": "running",
        "model_loaded": model is not None
    })

@app.route("/predict/<int:sample_id>")
def predict(sample_id):
    if model is None:
        return jsonify({
            "error": "Model not registered or unavailable. Train the model first."
        }), 503

    if sample_id < 0 or sample_id >= len(iris.data):
        return jsonify({"error": "Invalid sample ID"}), 400

    PREDICTION_COUNT.inc()
    prediction = model.predict([iris.data[sample_id]])[0]

    return jsonify({
        "sample_id": sample_id,
        "prediction": iris.target_names[int(prediction)].item()
    })

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {
        "Content-Type": CONTENT_TYPE_LATEST
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
