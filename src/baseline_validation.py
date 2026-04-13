import os
import yaml
from ultralytics import YOLO

# --- Push EVERYTHING inside the shield! ---
if __name__ == '__main__':
    
    # 1. Use the combined dataset
    dataset_dir = "combined_dataset"
    print(f"Dataset active directory: {dataset_dir}")

    # 2. Create the YAML configuration using the combined dataset path
    yaml_content = {
        'path': dataset_dir,
        'train': os.path.join("images", "train"),
        'val': os.path.join("images", "val"),
        'test': os.path.join("images", "val"),  # Using val as test for baseline
        'names': {0: 'person'},
        'nc': 1
    }

    yaml_path = "dataset_local.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)

    print(f"Created configuration file: {yaml_path}")

    # 4. Load the trained model and evaluate
    model = YOLO('yolov8n.pt')

    print("\nEvaluating baseline...")
    metrics = model.val(data=yaml_path, split='test', classes=[0])

    print("\n--- Baseline Results ---")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")