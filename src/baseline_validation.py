import os
import yaml
import kagglehub
from ultralytics import YOLO

# 1. Dynamically locate the dataset
# If it's already downloaded, this just instantly returns the cached path
print("Locating dataset...")
base_path = kagglehub.dataset_download("rgbnihal/c2a-dataset")

# 2. Build the nested path dynamically (OS-independent)
dataset_dir = os.path.join(base_path, "C2A_Dataset", "new_dataset3")
print(f"Dataset active directory: {dataset_dir}")

# 3. Create the YAML configuration using the dynamic path
yaml_content = {
    'path': dataset_dir,
    'train': os.path.join("train", "images"),
    'val': os.path.join("test", "images"),    # Mapping 'val' to 'test' for baseline
    'test': os.path.join("test", "images"),
    'names': {0: 'person'},
    'nc': 1
}

yaml_path = "c2a_local.yaml"
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_content, f)

print(f"Created configuration file: {yaml_path}")

# 4. Load the generic pre-trained model and evaluate
model = YOLO('yolov8n.pt')

print("\nEvaluating baseline...")
metrics = model.val(data=yaml_path, split='test', classes=[0])

print("\n--- Baseline Results ---")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"mAP50:    {metrics.box.map50:.4f}")