from ultralytics import YOLO
import json
from pathlib import Path
import numpy as np

def test_model(
    model_path: str, 
    yaml_path: str, 
    img_size: int = 800, 
    batch_size: int = 8,
    output_name: str = "test_results",
    output_dir: str = "results"
):
    """
    Evaluates a trained YOLO model on the test dataset and extracts key metrics.
    
    Args:
        model_path: Path to the trained model weights
        yaml_path: Path to the dataset YAML config
        img_size: Image size for evaluation
        batch_size: Batch size for evaluation
        output_name: Name of the output subdirectory
        output_dir: Base directory to save results (default: "results")
    """
    print(f"Loading trained model from: {model_path}")
    
    # 1. Load the trained model
    try:
        model = YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model at {model_path}. Error: {e}")

    print(f"\nRunning evaluation on the TEST split using {yaml_path}...")
    
    # 2. Run validation specifically on the 'test' split
    # Note: 'split="test"' tells Ultralytics to look for the 'test:' key in your YAML
    metrics = model.val(
        data=yaml_path,
        split='test',
        imgsz=img_size,
        batch=batch_size,
        save_json=False, # Don't save the raw predictions JSON (too large)
        project=output_dir,
        name=output_name
    )

    # 3. Extract core metrics
    p = metrics.box.mp    # Mean Precision across all classes
    r = metrics.box.mr    # Mean Recall across all classes
    map50 = metrics.box.map50  # mAP at IoU=0.50
    map50_95 = metrics.box.map # mAP at IoU=0.50:0.95

    # 4. Calculate F1 Score
    # The F1 score is the harmonic mean of Precision and Recall.
    # We add a tiny epsilon (1e-6) to the denominator to prevent division by zero.
    f1_score = 2 * (p * r) / (p + r + 1e-6)

    # 5. Extract average IoU (if available) or approximate
    # Ultralytics doesn't output a single "average IoU of all bounding boxes" by default,
    # as mAP50-95 already aggregates performance across IoU thresholds. 
    # However, you can access the fitness score (which heavily weights map50-95).
    fitness = metrics.box.fitness()

    # 6. Compile results
    results_dict = {
        "model": model_path,
        "dataset_config": yaml_path,
        "image_size": img_size,
        "metrics": {
            "Precision": float(p),
            "Recall": float(r),
            "mAP@50": float(map50),
            "mAP@50-95": float(map50_95),
            "F1_Score": float(f1_score),
            "Fitness": float(fitness)
        }
    }

    # 7. Print and save results
    print("\n" + "="*40)
    print("🎯 TEST DATASET METRICS")
    print("="*40)
    print(f"Precision (P)  : {p:.4f}")
    print(f"Recall (R)     : {r:.4f}")
    print(f"F1 Score       : {f1_score:.4f}")
    print(f"mAP@50         : {map50:.4f}")
    print(f"mAP@50-95      : {map50_95:.4f}")
    print("="*40)

    # Save to JSON for easy parsing later
    output_file = Path(output_dir) / output_name / "test_metrics_summary.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=4)
        
    print(f"\nDetailed metrics saved to: {output_file}")
    print(f"Visual curves (PR curve, F1 curve) saved in: {output_dir}/{output_name}/")

    return results_dict

if __name__ == '__main__':
    # Define paths based on your previous training script
    # Update 'model_path' to point to wherever your best weights were saved.
    # By default, Ultralytics saves them in runs/detect/<run_name>/weights/best.pt
    
    TARGET_MODEL_PATH = "runs/detect/yolo26-1pass-fixed-dataset/weights/best.pt" 
    YAML_CONFIG = "combined_local.yaml"
    IMAGE_SIZE = 800
    OUTPUT_DIR = "model_results"  # Change this to save to a different directory
    
    test_model(
        model_path=TARGET_MODEL_PATH,
        yaml_path=YAML_CONFIG,
        img_size=IMAGE_SIZE,
        output_dir=OUTPUT_DIR
    )