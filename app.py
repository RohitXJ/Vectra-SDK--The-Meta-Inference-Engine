from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List
import os
import uuid
import threading
import time
import io
import torch
import traceback
from PIL import Image

from utils import storage
from main_service import run_fewshot_pipeline
from fastapi.staticfiles import StaticFiles
from data import IO
from core import backbone, embedding

app = FastAPI(title="Vectra-SDK--The Meta-Inference-Engine API")

# --- CORS CONFIG ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- BACKGROUND CLEANUP TASK ---
SESSION_RETRAIN_COUNTS = {}

def background_cleanup():
    """Background loop to clean up old sessions every 10 minutes."""
    while True:
        try:
            # Clean folders older than 1 hour (3600 seconds)
            storage.cleanup_old_sessions(max_age_seconds=3600)
        except Exception as e:
            print(f"Background Cleanup Error: {e}")
        # Wait 10 minutes before next run
        time.sleep(600)

# Start cleanup thread
cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
cleanup_thread.start()

# --- ENDPOINTS ---

@app.post("/session/new")
def create_session():
    """Generates a new session token for a user."""
    token = str(uuid.uuid4())
    storage.get_session_paths(token)
    SESSION_RETRAIN_COUNTS[token] = 0
    return {"token": token}

@app.post("/upload")
async def upload_images(
    token: str = Form(...),
    category: str = Form(...),  # "support" or "query"
    class_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Uploads multiple images for a specific class and category."""
    if category not in ["support", "query"]:
        raise HTTPException(status_code=400, detail="Category must be 'support' or 'query'")
    
    saved_paths = []
    for file in files:
        content = await file.read()
        path = storage.save_upload_image(token, category, class_name, file.filename, content)
        saved_paths.append(path)
    
    return {
        "message": f"Successfully uploaded {len(files)} images to {category}/{class_name}",
        "token": token
    }

@app.post("/train")
def train_model(
    token: str = Form(...),
    backbone_name: str = Form("resnet18"),
    use_unknown: str = Form("false")
):
    """Triggers the few-shot pipeline for the given session token."""
    try:
        count = SESSION_RETRAIN_COUNTS.get(token, 0)
        if count >= 4:
            raise HTTPException(status_code=403, detail="Maximum retries exceeded (3 allowed). Please start a new session.")
        SESSION_RETRAIN_COUNTS[token] = count + 1

        # Convert string form to bool
        use_unknown_bool = use_unknown.lower() == "true"

        results = run_fewshot_pipeline(token, backbone_name, use_unknown=use_unknown_bool)
        export_filename = os.path.basename(results["export_path"])
        
        return {
            "status": "success",
            "accuracy": f"{results['accuracy']:.2f}%",
            "labels": results["labels"],
            "model_filename": export_filename,
            "predicted_labels": results["predicted_labels"],
            "true_labels": results["true_labels"]
        }
    except Exception as e:
        print(f"ERROR in /train: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/eval")
async def evaluate_image(
    token: str = Form(...),
    file: UploadFile = File(...)
):
    """Evaluates a single image using the trained session model."""
    try:
        paths = storage.get_session_paths(token)
        export_filename = f"fewshot_model_{token}.pt"
        model_path = os.path.join(paths["export"], export_filename)

        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found. Please train first.")

        # 1. Load exported session config
        config = torch.load(model_path, map_location="cpu")
        prototypes = config["prototypes"]
        labels = config["labels"]
        backbone_name = config["backbone"]
        image_format = config["image_format"]
        use_unknown = config.get("use_unknown", False)
        unknown_threshold = config.get("unknown_threshold", None)

        # 2. Initialize encoder
        encoder = backbone.get_encoder(backbone_name, image_format)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = encoder.to(device)
        encoder.eval()

        # 3. Preprocess image
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        if image.mode != image_format:
            if image_format == 'L':
                image = image.convert('L')
            else:
                image = image.convert('RGB')
        
        transform = IO.get_transform(image_format)
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            query_embedding = encoder(input_tensor)
            if len(query_embedding.shape) > 2:
                query_embedding = query_embedding.view(query_embedding.size(0), -1)
            
            # Normalize embedding
            query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
            
            dummy_label = torch.zeros(1)
            preds, _ = embedding.compute_distances_and_predict(
                query_embedding.cpu(), dummy_label, prototypes,
                use_unknown=use_unknown, unknown_threshold=unknown_threshold
            )

        pred_idx = preds[0].item()
        result_label = labels[pred_idx] if pred_idx < len(labels) else "Unknown"

        return {"prediction": result_label}

    except Exception as e:
        print(f"ERROR in /eval: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/download/{token}")
def download_model(token: str):
    """Downloads the trained model for the given session."""
    paths = storage.get_session_paths(token)
    export_filename = f"fewshot_model_{token}.pt"
    file_path = os.path.join(paths["export"], export_filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model not found. Please train first.")
    
    return FileResponse(
        path=file_path,
        filename=export_filename,
        media_type='application/octet-stream'
    )

@app.delete("/session/{token}")
def delete_session(token: str):
    """Manually clears session data."""
    SESSION_RETRAIN_COUNTS.pop(token, None)
    success = storage.clear_session_data(token)
    if success:
        return {"message": f"Session {token} data cleared."}
    else:
        return {"message": f"Session {token} not found or already cleared."}

# Serve Frontend - Must be last
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
