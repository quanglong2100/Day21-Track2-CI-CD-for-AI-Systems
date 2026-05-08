from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

# Config
GCS_BUCKET = os.environ.get("GCS_BUCKET")
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")

def download_model():
    """Tải file model.pkl từ GCS về máy khi server khởi động."""
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_MODEL_KEY)
        
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        blob.download_to_filename(MODEL_PATH)
        print(f"Model downloaded to {MODEL_PATH}")
    except Exception as e:
        print(f"Error downloading model: {e}")

# Try to download and load model on startup
if os.environ.get("GCS_BUCKET"):
    download_model()
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")
    
    # Get prediction from model
    prediction = int(model.predict([req.features])[0])
    
    # Map class to labels
    labels = {0: "thấp", 1: "trung bình", 2: "cao"}
    return {
        "prediction": prediction, 
        "label": labels.get(prediction, "unknown")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)