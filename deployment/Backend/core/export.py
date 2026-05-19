import torch
import os
import zipfile

def export_model(run_config, export_dir="export/", filename="fewshot_model.pt", zip_output=False):
    os.makedirs(export_dir, exist_ok=True)
    model_path = os.path.join(export_dir, filename)
    
    # Ensure prototypes are on CPU for export
    if "prototypes" in run_config:
        run_config["prototypes"] = run_config["prototypes"].cpu()
        
    if not zip_output: 
        torch.save(run_config, model_path)
        print(f"Model Exported to {model_path}")
        return model_path
    else:
        # Save temp .pt then zip it
        torch.save(run_config, model_path)
        zip_path = model_path.replace(".pt", ".zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(model_path, arcname=filename)
        os.remove(model_path)
        print(f"Model Exported to {zip_path}")
        return zip_path
