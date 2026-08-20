import sys
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ai.model_loader import ModelLoader
from ai.face_detector import FaceDetector
from ai.detector import process_frame_bytes

app = FastAPI(title="Face Mask Detection AI Service", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    """Pre-load TensorFlow Model and OpenCV Face Detector at startup."""
    print("[INFO] Starting FastAPI AI Service... Pre-loading models.")
    ModelLoader()
    FaceDetector()
    print("[SUCCESS] FastAPI AI Service startup complete!")


@app.get("/health")
def health_check():
    """Health check endpoint to verify AI service and model status."""
    loader = ModelLoader()
    return {
        "status": "ok",
        "model_loaded": loader.is_loaded
    }


@app.post("/predict")
async def predict_frame(file: UploadFile = File(...)):
    """
    Real-time frame prediction endpoint.
    Receives frame image file, runs multi-face detection & Keras CNN inference.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Image file required.")

    try:
        contents = await file.read()
        results = process_frame_bytes(contents)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    from ai.config import HOST, PORT
    uvicorn.run("ai.main:app", host=HOST, port=PORT, reload=True)
