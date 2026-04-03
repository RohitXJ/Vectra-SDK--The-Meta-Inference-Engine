import os
import sys
import shutil

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main_service import run_fewshot_pipeline
from utils import storage

def simulate_backend_flow():
    # 1. Create a mock token
    token = "test_token_123"
    print(f"--- Simulating Flow for Token: {token} ---")
    
    # 2. Setup paths
    paths = storage.get_session_paths(token)
    print(f"Temp Root: {paths['root']}")
    
    # 3. Simulate image uploads from local temp_data to tokenized temp storage
    # We'll use the images already in temp_data/ for this simulation
    local_data_root = "temp_data"
    for category in ["support", "query"]:
        cat_src = os.path.join(local_data_root, category)
        if not os.path.exists(cat_src):
            print(f"Error: {cat_src} not found. Please ensure temp_data exists.")
            return
            
        for class_name in os.listdir(cat_src):
            class_path = os.path.join(cat_src, class_name)
            if not os.path.isdir(class_path): continue
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                with open(img_path, "rb") as f:
                    content = f.read()
                    storage.save_upload_image(token, category, class_name, img_name, content)
    
    print("✅ Simulated image uploads to tokenized temp storage.")
    
    # 4. Run the pipeline
    print("🚀 Running Few-Shot Pipeline...")
    try:
        results = run_fewshot_pipeline(token, backbone_name="resnet18")
        print("\n--- Pipeline Results ---")
        print(f"Accuracy: {results['accuracy']:.2f}%")
        print(f"Labels: {results['labels']}")
        print(f"Model Exported to: {results['export_path']}")
        print("✅ Pipeline executed successfully.")
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")

    # 5. Cleanup simulation
    # storage.clear_session_data(token)
    # print(f"Cleaned up {token} temp data.")

if __name__ == "__main__":
    simulate_backend_flow()
