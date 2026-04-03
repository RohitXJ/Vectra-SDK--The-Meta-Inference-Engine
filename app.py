from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List
import os
import uuid
import threading
import time

from utils import storage
from main_service import run_fewshot_pipeline

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

@app.get("/")
def read_root():
    return {"message": "Welcome to Vectra-SDK--The Meta-Inference-Engine API"}

@app.post("/session/new")
def create_session():
    """Generates a new session token for a user."""
    token = str(uuid.uuid4())
    storage.get_session_paths(token)
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
    backbone_name: str = Form("resnet18")
):
    """Triggers the few-shot pipeline for the given session token."""
    try:
        results = run_fewshot_pipeline(token, backbone_name)
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
        raise HTTPException(status_code=500, detail=str(e))

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
    success = storage.clear_session_data(token)
    if success:
        return {"message": f"Session {token} data cleared."}
    else:
        return {"message": f"Session {token} not found or already cleared."}
