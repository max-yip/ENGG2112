import matplotlib.pyplot as plt
import matplotlib
from typing import List

from models.training_pipeline import generate_yaml_config, train_yolo

# Set plotting DPI for better resolution
matplotlib.rcParams['figure.dpi'] = 110

class YoloExperiment:
    """Stores the results of a single YOLO training run."""

    def __init__(self, name: str, model: str, map50: float, map50_95: float, epochs: int):
        self.name = name
        self.model = model          # e.g., 'yolov8n.pt', 'yolov8s.pt'
        self.map50 = map50          # Mean Average Precision @ IoU 0.50
        self.map50_95 = map50_95    # Mean Average Precision @ IoU 0.50:0.95
        self.epochs = epochs

    def __repr__(self):
        return (f"YoloExperiment(name={self.name}, model={self.model}, "
                f"mAP50={self.map50:.3f}, mAP50-95={self.map50_95:.3f}, epochs={self.epochs})")


class ExperimentTracker:
    """Logs and queries a collection of YOLO experiments."""

    def __init__(self):
        self.runs: List[YoloExperiment] = []

    def log(self, experiment: YoloExperiment):
        """Add an Experiment to the tracker."""
        self.runs.append(experiment)

    def best_by(self, metric: str = "map50_95") -> YoloExperiment:
        """Return the Experiment with the highest value for 'metric'."""
        if not self.runs:
            raise ValueError("No experiments logged yet.")
        # Uses python's built-in max function with a lambda to fetch the attribute
        return max(self.runs, key=lambda exp: getattr(exp, metric))
  
    def filter_by_model(self, model_name: str) -> List[YoloExperiment]:
        """Return all experiments that used the given model architecture."""
        return [exp for exp in self.runs if exp.model == model_name]
 
    def __len__(self):
        return len(self.runs)


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


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Run a real YOLO training pipeline and log the results
    # ---------------------------------------------------------
    tracker = ExperimentTracker()

    dataset_path = "combined_dataset"
    yaml_config_path = "combined_local.yaml"
    run_name = "tracked_real_training"
    epochs_to_run = 10

    # Generate or refresh the YAML config from the current dataset
    generate_yaml_config(dataset_path, yaml_path=yaml_config_path)

    # Train using the real pipeline and capture metrics
    results = train_yolo(
        yaml_path=yaml_config_path,
        epochs=epochs_to_run,
        batch_size=8,
        fraction=0.1,
        project_dir=".",
        run_name=run_name
    )

    final_map50 = getattr(results.box, "map50", None)
    final_map50_95 = getattr(results.box, "map", None)

    if final_map50 is None or final_map50_95 is None:
        raise RuntimeError("Unable to extract YOLO metrics from training results.")

    tracker.log(YoloExperiment(
        name=run_name,
        model="yolov8n",
        map50=final_map50,
        map50_95=final_map50_95,
        epochs=epochs_to_run
    ))

    print(f"Logged {len(tracker)} real YOLO experiments.")
    print(f"Best by mAP@50    : {tracker.best_by('map50')}")
    print(f"Best by mAP@50-95 : {tracker.best_by('map50_95')}")

    plot_results(tracker)
