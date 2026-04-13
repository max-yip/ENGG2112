import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from ultralytics import YOLO

def verify_dataset(dataset_path):
    """Checks if the required YOLO directory structure exists."""
    print(f"Dataset active directory: {dataset_path}")
    assert os.path.exists(os.path.join(dataset_path, "images", "train")), "Train images missing!"
    assert os.path.exists(os.path.join(dataset_path, "labels", "train")), "Train labels missing!"
    assert os.path.exists(os.path.join(dataset_path, "images", "val")), "Validation images missing!"
    assert os.path.exists(os.path.join(dataset_path, "labels", "val")), "Validation labels missing!"
    print("✅ Dataset verified!")

def profile_object_sizes(dataset_path, sample_limit=6129, show_plot=True):
    """Calculates object areas and plots their distribution to find small object thresholds."""
    label_dir = os.path.join(dataset_path, "labels", "train")
    sizes = []
    
    # Safely slice the directory in case there are fewer files than the limit
    files = os.listdir(label_dir)[:sample_limit]
    
    for label_file in tqdm(files, desc="Profiling Object Sizes"):
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                _, _, _, w, h = map(float, line.strip().split())
                sizes.append(w * h)
                
    small_threshold = np.percentile(sizes, 15)  # Dynamic threshold
    print(f"Calculated 15th percentile small threshold: {small_threshold:.4f}")

    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.hist(sizes, bins=50, alpha=0.7, label='All Objects')
        plt.axvline(small_threshold, color='r', linestyle='--', label=f'Small Threshold ({small_threshold:.4f})')
        plt.title("Normalized Object Area Distribution")
        plt.xlabel("Width × Height (normalized)")
        plt.ylabel("Count")
        plt.legend()
        plt.show()

    return sizes, small_threshold

def generate_yaml_config(dataset_path, yaml_path="combined_local.yaml"):
    """Generates the data.yaml file required by YOLO."""
    base_yaml = {
        "path": dataset_path,
        "train": os.path.join("images", "train"),
        "val": os.path.join("images", "val"),
        "names": {0: "human"},
        "nc": 1,
        "augment": True,
        "small_objects": True
    }
    
    with open(yaml_path, "w") as f:
        yaml.dump(base_yaml, f)

    print(f"✅ Generated YAML config at {yaml_path}:")
    with open(yaml_path, "r") as f:
        print(f.read())
        
    return yaml_path

def train_yolo(yaml_path, epochs=40, batch_size=8, fraction=0.1, project_dir=".", run_name="stage1_all_data"):
    """Loads the YOLO model and begins the training loop."""
    model = YOLO("yolov8n.pt")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("About to start training...")
    
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        fraction=fraction,
        device=device,
        project=project_dir,
        name=run_name,
        exist_ok=True
    )
    return results