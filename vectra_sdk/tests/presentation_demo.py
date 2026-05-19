import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vectra.inference import VectraInference

# 1. Initialize Vectra Engine
sdk = VectraInference("test_model.pt")

# 2. Single Image Inference
res = sdk.predict("test_data/cat_1.jpg")
print(f"Prediction: {res['label']} ({res['confidence']:.2%})")

# 3. Batch Processing
images = ["test_data/cat_2.jpg", "test_data/dog_1.jpg", "test_data/unknown_1.jpg"]
results = sdk.predict_batch(images)

for img, res in zip(images, results):
    print(f"{img:20} -> {res['label']}")
