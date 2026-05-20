from ultralytics import YOLO
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

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

    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = []

        if isinstance(existing_data, list):
            existing_data.append(results_dict)
        elif isinstance(existing_data, dict):
            existing_data = [existing_data, results_dict]
        else:
            existing_data = [results_dict]
    else:
        existing_data = [results_dict]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4)
        
    print(f"\nDetailed metrics saved to: {output_file}")
    print(f"Visual curves (PR curve, F1 curve) saved in: {output_dir}/{output_name}/")

    return results_dict


class PyTorchDetectionDataset(torch.utils.data.Dataset):
    """Custom PyTorch dataset for YOLO-format annotations for detection models."""
    
    def __init__(self, image_dir, label_dir, img_size=640):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.image_files = sorted(self.image_dir.glob("*.jpg")) + sorted(self.image_dir.glob("*.png"))
        self.transform = torch.nn.Identity()  # Will normalize after loading
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        import torchvision.io as tv_io
        import torchvision.transforms.functional as F
        
        img_path = self.image_files[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")
        
        # Read image
        image = tv_io.read_image(str(img_path))
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3, :, :]
        elif image.shape[0] != 3:
            raise RuntimeError(f"Unexpected channel count {image.shape[0]} in {img_path}")
        
        # Resize
        image = F.resize(image, [self.img_size, self.img_size], antialias=True)
        
        # Convert to float and normalize
        image = image.float() / 255.0
        # image = torch.nn.functional.normalize(image, dim=0)
        
        # Load YOLO annotations and convert to absolute coords
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        
                        # Convert from YOLO format (center, normalized) to absolute coords
                        x1 = ((cx - bw/2) * self.img_size)
                        y1 = ((cy - bh/2) * self.img_size)
                        x2 = ((cx + bw/2) * self.img_size)
                        y2 = ((cy + bh/2) * self.img_size)
                        
                        # Clamp to image bounds
                        x1 = max(0, min(x1, self.img_size - 1))
                        y1 = max(0, min(y1, self.img_size - 1))
                        x2 = max(x1 + 1, min(x2, self.img_size))
                        y2 = max(y1 + 1, min(y2, self.img_size))
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id + 1)  # 1-indexed labels
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels
        }
        
        return image, target


def collate_fn(batch):
    """Collate function for DataLoader."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def test_pytorch_model(
    model_path: str,
    model_type: str,
    dataset_path: str,
    img_size: int = 640,
    batch_size: int = 8,
    output_name: str = "test_results",
    output_dir: str = "model_results"
):
    """
    Evaluates a trained PyTorch detection model (Faster RCNN or RetinaNet) on the test dataset.
    
    Args:
        model_path: Path to the saved model weights (.pt file)
        model_type: "faster_rcnn" or "retinanet"
        dataset_path: Path to the dataset directory (with images/labels subdirs)
        img_size: Image size for evaluation
        batch_size: Batch size for evaluation
        output_name: Name of the output subdirectory
        output_dir: Base directory to save results
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading {model_type} model from: {model_path}")
    
    # 1. Load the model
    if model_type.lower() == "faster_rcnn":
        model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
    elif model_type.lower() == "retinanet":
        model = retinanet_resnet50_fpn(pretrained=False, num_classes=2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}. Error: {e}")
    
    model.to(device)
    model.eval()
    
    # 2. Create test dataset
    test_image_dir = Path(dataset_path) / "images" / "test"
    test_label_dir = Path(dataset_path) / "labels" / "test"
    
    test_dataset = PyTorchDetectionDataset(test_image_dir, test_label_dir, img_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"\nRunning evaluation on {len(test_dataset)} test images...")
    
    # 3. Evaluate model
    all_predictions = []
    all_targets = []
    metric = MeanAveragePrecision(iou_type="bbox")
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            predictions = model(images)
            
            # Format predictions for mAP calculation
            formatted_predictions = []
            formatted_targets = []
            
            for pred in predictions:
                formatted_predictions.append({
                    "boxes": pred["boxes"].cpu(),
                    "scores": pred["scores"].cpu(),
                    "labels": pred["labels"].cpu()
                })
            
            for target in targets:
                formatted_targets.append({
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu()
                })
            
            all_predictions.extend(formatted_predictions)
            all_targets.extend(formatted_targets)
    
    # 4. Compute metrics
    metric.update(all_predictions, all_targets)
    result = metric.compute()
    
    # Extract metrics
    map50 = result['map_50'].item() if result['map_50'] is not None else 0.0
    map50_95 = result['map'].item() if result['map'] is not None else 0.0
    
    # Calculate precision and recall from predictions
    total_tp = total_fp = total_fn = 0
    score_thresh = 0.5
    iou_thresh = 0.5
    
    from torchvision.ops import box_iou
    
    for pred, target in zip(all_predictions, all_targets):
        gt_boxes = target["boxes"]
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        
        keep = pred_scores >= score_thresh
        pred_boxes = pred_boxes[keep]
        
        if len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
            continue
        
        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue
        
        ious = box_iou(pred_boxes, gt_boxes)
        matched_gt = set()
        tp = fp = 0
        
        for i in range(len(pred_boxes)):
            max_iou, gt_idx = torch.max(ious[i], dim=0)
            if max_iou >= iou_thresh and gt_idx.item() not in matched_gt:
                tp += 1
                matched_gt.add(gt_idx.item())
            else:
                fp += 1
        
        fn = len(gt_boxes) - len(matched_gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # 5. Compile results
    results_dict = {
        "model": model_path,
        "model_type": model_type,
        "dataset_path": dataset_path,
        "image_size": img_size,
        "metrics": {
            "Precision": float(precision),
            "Recall": float(recall),
            "mAP@50": float(map50),
            "mAP@50-95": float(map50_95),
            "F1_Score": float(f1_score)
        }
    }
    
    # 6. Print and save results
    print("\n" + "="*40)
    print(f"🎯 TEST DATASET METRICS ({model_type.upper()})")
    print("="*40)
    print(f"Precision (P)  : {precision:.4f}")
    print(f"Recall (R)     : {recall:.4f}")
    print(f"F1 Score       : {f1_score:.4f}")
    print(f"mAP@50         : {map50:.4f}")
    print(f"mAP@50-95      : {map50_95:.4f}")
    print("="*40)
    
    # Save to JSON
    output_file = Path(output_dir) / output_name / "test_metrics_summary.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = []
        
        if isinstance(existing_data, list):
            existing_data.append(results_dict)
        elif isinstance(existing_data, dict):
            existing_data = [existing_data, results_dict]
        else:
            existing_data = [results_dict]
    else:
        existing_data = [results_dict]
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4)
    
    print(f"\nDetailed metrics saved to: {output_file}")
    
    return results_dict

if __name__ == '__main__':
    # Define paths based on your previous training script
    # Update 'model_path' to point to wherever your best weights were saved.
    # By default, Ultralytics saves them in runs/detect/<run_name>/weights/best.pt
    
    # # --- Test YOLO Model ---
    # print("\n" + "="*60)
    # print("TESTING YOLO MODEL")
    # print("="*60)
    # YOLO_MODEL_PATH = "runs/detect/yolo26s-640px/weights/best.pt" 
    YAML_CONFIG = "combined_local.yaml"
    IMAGE_SIZE = 640
    OUTPUT_DIR = "model_results"
    
    # test_model(
    #     model_path=YOLO_MODEL_PATH,
    #     yaml_path=YAML_CONFIG,
    #     img_size=IMAGE_SIZE,
    #     output_dir=OUTPUT_DIR,
    #     output_name="yolo_test_results"
    # )
    
    # --- Test Faster RCNN Model ---
    print("\n" + "="*60)
    print("TESTING FASTER RCNN MODEL")
    print("="*60)
    FASTER_RCNN_PATH = "runs/detect/faster_rcnn_10epochs/model.pt"
    DATASET_PATH = "combined_dataset"
    
    test_pytorch_model(
        model_path=FASTER_RCNN_PATH,
        model_type="faster_rcnn",
        dataset_path=DATASET_PATH,
        img_size=IMAGE_SIZE,
        output_name="faster_rcnn_test_results",
        output_dir=OUTPUT_DIR
    )
    
    # --- Test RetinaNet Model ---
    print("\n" + "="*60)
    print("TESTING RETINANET MODEL")
    print("="*60)
    RETINANET_PATH = "runs/detect/retinanet-stage1-10epoch/model.pt"
    
    test_pytorch_model(
        model_path=RETINANET_PATH,
        model_type="retinanet",
        dataset_path=DATASET_PATH,
        img_size=IMAGE_SIZE,
        output_name="retinanet_test_results",
        output_dir=OUTPUT_DIR
    )