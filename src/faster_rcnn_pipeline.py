import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

from lib.dataset import merge_datasets, verify_dataset
from lib.baseline_validation import run_baseline_validation
from lib.training_tools import ExperimentTracker, Experiment, plot_results

import torchvision.io as tv_io
import torchvision.transforms.functional as F


class YOLODataset(Dataset):
    """Custom PyTorch dataset for YOLO-format annotations."""
    
    def __init__(self, image_dir, label_dir, img_size=640):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.image_files = sorted(self.image_dir.glob("*.jpg")) + sorted(self.image_dir.glob("*.png"))
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")
        
        # 1. Native PyTorch Read
        import torchvision.io as tv_io
        import torchvision.transforms.functional as F
        
        # read_image returns a uint8 tensor (C, H, W)
        image = tv_io.read_image(str(img_path))
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3, :, :]
        elif image.shape[0] != 3:
            raise RuntimeError(f"Unexpected channel count {image.shape[0]} in {img_path}")
        
        # 2. Resize directly on the GPU/Tensor
        image = F.resize(image, [self.img_size, self.img_size], antialias=True)
        
        # 3. Convert to float and scale to [0, 1] BEFORE normalizing
        image = image.float() / 255.0
        
        # 4. Apply ImageNet normalization
        image = self.transform(image)
        
        # Load YOLO annotations (normalized coords) and convert to absolute
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
                        labels.append(class_id + 1)  # Faster R-CNN uses 1-indexed labels
        
        if len(boxes) == 0:
            # Add dummy box if no objects
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

def train_faster_rcnn(
    dataset_path: str,
    epochs: int = 10,
    batch_size: int = 8, # Try bumping this to 8 or 16 if your GPU has enough VRAM!
    img_size: int = 640,
    lr: float = 0.005,
    weight_decay: float = 0.0005,
    project_dir: str = ".",
    run_name: str = "faster_rcnn_stage1"
):
    """Train Faster R-CNN model on custom dataset with AMP and multiprocessing."""
    
    # 1. PREVENT OPENCV DEADLOCKS WITH MULTIPROCESSING
    cv2.setNumThreads(0) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    run_dir = Path(project_dir) / "runs" / "detect" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create train/val datasets
    train_image_dir = Path(dataset_path) / "images" / "train"
    train_label_dir = Path(dataset_path) / "labels" / "train"
    val_image_dir = Path(dataset_path) / "images" / "val"
    val_label_dir = Path(dataset_path) / "labels" / "val"
    
    train_dataset = YOLODataset(train_image_dir, train_label_dir, img_size)
    val_dataset = YOLODataset(val_image_dir, val_label_dir, img_size)
    
    # 2. OPTIMIZE DATALOADERS
    # Use multiple CPU cores and pin memory for faster CPU-to-GPU transfer
    num_workers = min(os.cpu_count(), 8) if os.name != 'nt' else 4 # Safe default for Windows
    print(f"Using {num_workers} CPU workers for data loading.")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Load pre-trained Faster R-CNN (Freezing first layer for slightly faster iteration)
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT", trainable_backbone_layers=3)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)  
    
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # 3. INITIALIZE AMP SCALER
    scaler = torch.amp.GradScaler('cuda')
    
    # Initialize MeanAveragePrecision metric
    metric = MeanAveragePrecision(iou_type="bbox")
    
    metrics_history = {
        "train_loss": [],
        "val_map50": [],
        "val_map": [],
        "val_precision": [],
        "val_recall": []
    }
    
    print("Starting training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss_sum = 0.0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            # Move data to GPU non-blocking
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad(set_to_none=True) # Slightly faster than zero_grad()
            
            # 4. USE AUTOMATIC MIXED PRECISION (AMP)
            with torch.amp.autocast('cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_sum += losses.item()
        
        avg_train_loss = train_loss_sum / len(train_loader)
        metrics_history["train_loss"].append(avg_train_loss)
        
        model.eval()
        all_predictions = []
        all_targets = []
        
        # Reset metric for this epoch
        metric.reset()
        
        # Initialize precision/recall calculation variables
        total_tp = total_fp = total_fn = 0
        score_thresh = 0.5  # confidence threshold for predictions
        pr_iou_thresh = 0.5  # IoU threshold for precision/recall
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                images = [img.to(device, non_blocking=True) for img in images]
                targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
                
                # AMP for validation forward pass too
                with torch.amp.autocast('cuda'):
                    predictions = model(images)
                
                # Format predictions and targets for MeanAveragePrecision
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
                
                # Calculate precision/recall for this batch
                for output, target in zip(predictions, targets):
                    gt_boxes = target["boxes"].to(device)
                    
                    # Predictions
                    boxes = output["boxes"].to(device)
                    scores = output["scores"].to(device)
                    
                    keep = scores >= score_thresh
                    boxes = boxes[keep]
                    scores = scores[keep]
                    
                    # ----- Precision / Recall calculation -----
                    if len(boxes) == 0:
                        total_fn += len(gt_boxes)
                        continue
                    
                    if len(gt_boxes) == 0:
                        # All predictions are false positives when no ground truths
                        total_fp += len(boxes)
                        continue
                    
                    ious = box_iou(boxes, gt_boxes)
                    matched_gt = set()
                    tp = fp = 0
                    
                    for i in range(len(boxes)):
                        max_iou, gt_idx = torch.max(ious[i], dim=0)
                        if max_iou >= pr_iou_thresh and gt_idx.item() not in matched_gt:
                            tp += 1
                            matched_gt.add(gt_idx.item())
                        else:
                            fp += 1
                    
                    fn = len(gt_boxes) - len(matched_gt)
                    
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
        
        # Update metric with all predictions and targets
        metric.update(all_predictions, all_targets)
        
        # Compute final metrics
        result = metric.compute()
        
        map50 = result['map_50'].item() if result['map_50'] is not None else 0.0
        map_value = result['map'].item() if result['map'] is not None else 0.0
        
        # Calculate precision and recall using IoU 0.5
        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        
        metrics_history["train_loss"].append(avg_train_loss)
        metrics_history["val_map50"].append(map50)
        metrics_history["val_map"].append(map_value)
        metrics_history["val_precision"].append(precision)
        metrics_history["val_recall"].append(recall)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} | mAP@50: {map50:.4f} | mAP: {map_value:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        lr_scheduler.step()
    
    # Save model
    model_path = run_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    class Results:
        def __init__(self, metrics):
            self.box = type('obj', (object,), {
                'map50': metrics['val_map50'][-1] if metrics['val_map50'] else 0,
                'map': metrics['val_map'][-1] if metrics['val_map'] else 0,
                'mp': np.mean([m for m in metrics['val_precision'][-5:]]) if metrics['val_precision'] else 0,
                'mr': np.mean([m for m in metrics['val_recall'][-5:]]) if metrics['val_recall'] else 0
            })()
    
    results = Results(metrics_history)
    return results, model, (train_loader, val_loader)

def collate_fn(batch):
    """Collate function for DataLoader."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def main():
    # --- 1. Download and merge datasets ---
    print("Step 1: Downloading and merging datasets...")
    merge_datasets()
    
    # --- 2. Run baseline validation ---
    # print("\nStep 2: Running baseline validation...")
    # run_baseline_validation()
    
    # ---------------------------------------------------------
    # Run a Faster R-CNN training pipeline and log the results
    # ---------------------------------------------------------
    tracker = ExperimentTracker(filepath="experiments.json")

    dataset_path = "combined_dataset"
    run_name = "faster_rcnn_20epochs"
    epochs_to_run = 20
    batch_size = 8
    img_size = 640
    learning_rate = 0.005

    # Train using Faster R-CNN
    results, model, dataloaders = train_faster_rcnn(
        dataset_path=dataset_path,
        epochs=epochs_to_run,
        batch_size=batch_size,
        img_size=img_size,
        lr=learning_rate,
        project_dir=".",
        run_name=run_name
    )

    final_map50 = getattr(results.box, "map50", None)
    final_map = getattr(results.box, "map", None)
    final_P = getattr(results.box, "mp", None)
    final_R = getattr(results.box, "mr", None)

    if final_map50 is None or final_map is None:
        raise RuntimeError("Unable to extract Faster R-CNN metrics from training results.")

    tracker.log(Experiment(
        name=run_name,
        model="faster_rcnn_resnet50",
        map50=final_map50,
        map50_95=final_map,
        epochs=epochs_to_run,
        fraction=1.0,
        img_size=img_size,
        P=final_P,
        R=final_R
    ))

    print(f"\nLogged {len(tracker)} Faster R-CNN experiments.")
    print(f"Best by mAP@50    : {tracker.best_by('map50')}")
    print(f"Best by mAP@50-95 : {tracker.best_by('map50_95')}")

    plot_results(tracker)


if __name__ == '__main__':
    main()
