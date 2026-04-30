import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from ultralytics import YOLO
import json
from pathlib import Path
from typing import List


class Experiment:
    """Stores the results of a single training run."""

    def __init__(self, 
                 name: str, 
                 model: str, 
                 map50: float, 
                 map50_95: float, 
                 epochs: int, 
                 fraction: float, 
                 img_size: int, 
                 P: float = None, 
                 R: float = None):
        self.name = name
        self.model = model          # e.g., 'yolov8n.pt', 'yolov8s.pt'
        self.map50 = map50          # Mean Average Precision @ IoU 0.50
        self.map50_95 = map50_95    # Mean Average Precision @ IoU 0.50:0.95
        self.epochs = epochs
        self.fraction = fraction
        self.img_size = img_size
        self.P = P  # Precision
        self.R = R  # Recall
        ### plus other metrics

    def __repr__(self):
        return (f"Experiment(name={self.name}, model={self.model}, "
                f"mAP50={self.map50:.3f}, mAP50-95={self.map50_95:.3f}, epochs={self.epochs})")

class ExperimentTracker:
    """Logs and queries a collection of experiments."""

    def __init__(self, filepath: str = "experiments.json"):
        self.runs: List[Experiment] = []
        self.filepath = filepath
        self.load()  # Load existing experiments on startup

    def save(self):
        """Save experiments to JSON file."""
        data = [
            {
                "name": exp.name,
                "model": exp.model,
                "map50": exp.map50,
                "map50_95": exp.map50_95,
                "epochs": exp.epochs,
                "fraction": exp.fraction,
                "img_size": exp.img_size,
                "P": exp.P,
                "R": exp.R
            }
            for exp in self.runs
        ]
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Experiments saved to {self.filepath}")

    def load(self):
        """Load experiments from JSON file."""
        if Path(self.filepath).exists():
            text = Path(self.filepath).read_text()
            if not text.strip():
                print(f"Empty experiment file: {self.filepath} — starting with no experiments.")
                self.runs = []
                return
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                print(f"Warning: invalid JSON in {self.filepath}. Starting with no experiments.")
                self.runs = []
                return
            self.runs = [Experiment(**exp_dict) for exp_dict in data]
            print(f"Loaded {len(self.runs)} experiments from {self.filepath}")
        else:
            self.runs = []

    def log(self, experiment: Experiment):
        """Add an Experiment to the tracker and save."""
        self.runs.append(experiment)
        self.save()  # Auto-save after each experiment

    def best_by(self, metric: str = "map50_95") -> Experiment:
        """Return the Experiment with the highest value for 'metric'."""
        if not self.runs:
            raise ValueError("No experiments logged yet.")
        # Uses python's built-in max function with a lambda to fetch the attribute
        return max(self.runs, key=lambda exp: getattr(exp, metric))
  
    def filter_by_model(self, model_name: str) -> List[Experiment]:
        """Return all experiments that used the given model architecture."""
        return [exp for exp in self.runs if exp.model == model_name]
 
    def __len__(self):
        return len(self.runs)


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
        "test": os.path.join("images", "test"),
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

def train_yolo(yaml_path, yolo_model="yolov26n.pt", img_size = 640, epochs=40, batch_size=8, fraction=0.1, project_dir=".", run_name="stage1_all_data"):
    """Loads the YOLO model and begins the training loop."""
    
    model = YOLO(yolo_model)
    model.load("yolo26n.pt")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("About to start training...")
    
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        fraction=fraction,
        device=device,
        project=project_dir,
        name=run_name,
        exist_ok=True
    )
    return results

def plot_results(tracker: ExperimentTracker):
    """Visualizes the YOLO tracking results using matplotlib."""
    if not tracker.runs:
        print("No data to plot.")
        return

    names = [exp.name for exp in tracker.runs]
    models = [exp.model for exp in tracker.runs]
    map50 = [exp.map50 for exp in tracker.runs]
    map50_95 = [exp.map50_95 for exp in tracker.runs]

    x = range(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 4.5))

    bars_map50 = ax.bar([i - width/2 for i in x], map50, width, label='mAP@50', color='#0D9488', alpha=0.88)
    bars_map50_95 = ax.bar([i + width/2 for i in x], map50_95, width, label='mAP@50-95', color='#F59E0B', alpha=0.88)

    # Annotate bars with values
    for bar in bars_map50:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=7.5)
    for bar in bars_map50_95:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{n}\n({m})" for n, m in zip(names, models)], fontsize=8)
    
    # Dynamically scale Y-axis based on data
    ax.set_ylim(min(map50_95) - 0.1, 1.0)
    ax.set_ylabel('mAP Score')
    ax.set_title('YOLO Experiment Tracker — mAP@50 vs mAP@50-95', fontsize=12, pad=10)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()