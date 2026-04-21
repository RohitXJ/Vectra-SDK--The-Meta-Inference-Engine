# Vectra: The Meta-Inference Engine & SDK

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![AI Library](https://img.shields.io/badge/AI-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Custom-green.svg)](LICENSE)

**Vectra** is a full-lifecycle Few-Shot Learning platform designed to bridge the gap between complex meta-learning research and real-world application. It allows users to train, evaluate, and deploy high-performance image classifiers using only a handful of examples (5-10 images per class) instead of thousands.

---

## 🚀 Key Features

- **Few-Shot Learning Engine:** Utilizes Prototypical Networks to learn new categories with minimal data.
- **Out-of-Distribution (OOD) Detection:** Native "Unknown" category rejection using Euclidean distance thresholds.
- **Standalone SDK:** A `pip`-installable Python library for integrating Vectra models into any application.
- **Interactive Web Suite:** A modern FastAPI + Vanilla JS dashboard for data upload, training visualization, and live testing.
- **Live Stream Support:** Real-time camera inference with visual overlays for rapid prototyping.
- **Optimized for Deployment:** Dockerized environment with CPU-optimized inference for cloud hosting.

---

## 🛠 Project Structure

The project is divided into two primary components:

### 1. The Vectra Engine (Core)
The main application suite located in the root directory.
- `app.py`: FastAPI server handling the web interface and API endpoints.
- `main_service.py`: Orchestrates the training and export pipeline.
- `core/`: Contains the backbone architectures (ResNet, MobileNet, EfficientNet) and embedding logic.
- `frontend/`: The interactive UI for managing training sessions.

### 2. The Vectra SDK (Deployment)
Located in `vectra_sdk/`, this is a standalone library for developers.
- `vectra.inference`: The high-level class for loading `.pt` models and running predictions.
- `vectra.utils.vision`: Utilities for real-time video stream processing.

---

## 📦 Installation & Setup

### For Development (Local)
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Vectra-SDK.git
   cd Vectra-SDK
   ```
2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application:**
   ```bash
   uvicorn app:app --reload
   ```
   Access the UI at `http://127.0.0.1:8000`.

### For Production (Docker)
We use a multi-stage, CPU-optimized build to keep the image size small (~800MB).
```bash
docker build -t vectra-engine .
docker run -p 8000:8000 vectra-engine
```

---

## 📖 How it Works: The "Unknown" Category

Unlike standard classifiers that force every image into a known label, Vectra uses **Distance-Based Rejection**. 

1. **Prototypes:** For each class, the engine calculates a "Prototype" (mean vector) in a high-dimensional feature space.
2. **Euclidean Distance:** During inference, the engine calculates the distance between the query image and the nearest prototype.
3. **Thresholding:** If the `min_distance > threshold`, the image is classified as **"Unknown"**. This prevents false positives when the model sees objects it was never trained on.

---

## 💻 Using the SDK

Once you have exported a model (`.pt`) from the engine, you can use it in any Python project:

```python
from vectra.inference import VectraInference

# Initialize the SDK
sdk = VectraInference("my_model.pt")

# Predict from a path, PIL image, or OpenCV frame
result = sdk.predict("test_image.jpg")

print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.2f}")
```

---

## 🎓 Academic Context

This project was developed as a **Final Year Academic Project**. It demonstrates expertise in:
- **Artificial Intelligence:** Few-shot learning, computer vision, and meta-learning.
- **System Architecture:** Decoupling inference engines from client SDKs.
- **DevOps:** Containerization, resource optimization, and API design.

---

## 📜 License
This project is licensed under a Custom Research and Educational Use License. Commercial use is prohibited without prior permission.

**Author:** Rohit Gomes, Ankur Halder, Riyanka Bag, Bidisha Pal
