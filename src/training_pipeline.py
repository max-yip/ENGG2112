import os
import cv2
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO
import kagglehub


# 1. Start by using a random subset for validation & experimentation for faster iterations and tuning, before using 100% dataset in the final training before deployment
# 2. Data profiling: Normalise object area distribution, tuning anchor boxes and ensuring the model doesn't miss distant or small humans.
# 3. Model Configuration: Target yaml configurations to humans detection
# 4. Fine-Tuning: Download pre-trained weights and fine-tunes them on the custom dataset.


# 1. Dynamically locate the dataset
print("Locating dataset...")
base_path = kagglehub.dataset_download("rgbnihal/c2a-dataset")

# 2. Build the nested path dynamically (OS-independent)
dataset_path = os.path.join(base_path, "C2A_Dataset", "new_dataset3")
print(f"Dataset active directory: {dataset_path}")

# Verify dataset structure
assert os.path.exists(os.path.join(dataset_path, "train", "images")), "Train images missing!"
assert os.path.exists(os.path.join(dataset_path, "All labels with Pose information", "labels")), "Pose labels missing!"
print("✅ Dataset verified!")

#Object Size Distribution
def get_object_sizes(label_dir):
    sizes = []
    for label_file in tqdm(os.listdir(label_dir)[:6129]):  # Sample for speed
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                _, _, _, w, h = map(float, line.strip().split())
                sizes.append(w * h)
    return sizes

sizes = get_object_sizes(f"{dataset_path}/train/labels")
small_threshold = np.percentile(sizes, 15)  # Dynamic threshold

plt.figure(figsize=(10, 5))
plt.hist(sizes, bins=50, alpha=0.7, label='All Objects')
plt.axvline(small_threshold, color='r', linestyle='--', label=f'Small Threshold ({small_threshold:.4f})')
plt.title("Normalized Object Area Distribution")
plt.xlabel("Width × Height (normalized)")
plt.ylabel("Count")
plt.legend()
plt.show()

#Base Config (All Data)
base_yaml = {
    "path": dataset_path,
    "train": os.path.join("train", "images"),
    "val": os.path.join("test", "images"),  # Using test as val for now
    "names": {0: "human"},
    "nc": 1,
    "augment": True,
    "small_objects": True
}
yaml_path = "c2a_local.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(base_yaml, f)

with open(yaml_path, "r") as f:
    print(f.read())

# Load model
model = YOLO("yolov8n.pt")

# Then train with optimized anchors
results = model.train(
    data=yaml_path,
    epochs=40,
    imgsz=640,
    batch=8,
    device='auto',  # Use auto for local
    project=".",
    name="stage1_all_data",
    exist_ok=True
)