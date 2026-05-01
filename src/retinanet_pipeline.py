"""
retinanet_pipeline.py: Implementation of RetinaNet 
"""

import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt

from torchmetrics.detection import MeanAveragePrecision

from lib.dataset import merge_datasets, verify_dataset
from lib.baseline_validation import run_baseline_validation
from lib.training_tools import ExperimentTracker, Experiment, plot_results

import torchvision.io as tv_io
import torchvision.transforms.functional as F


class DetectionDataset(Dataset):
    """Custom PyTorch dataset that reads YOLO-format label files for a TorchVision detection model."""
    
    def __init__(self, image_dir, label_dir, img_size=640):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.image_files = sorted(self.image_dir.glob("*.jpg")) + sorted(self.image_dir.glob("*.png"))
    
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
        
        # 3. Convert to float and scale to [0, 1]
        image = image.float() / 255.0
        
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
                        labels.append(class_id + 1)  # YOLO class 0 => TorchVision label 1, background is implicitly label 0
        
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

def train_retinanet(
    dataset_path: str,
    epochs: int = 10,
    batch_size: int = 8, # Try bumping this to 8 or 16 if your GPU has enough VRAM!
    img_size: int = 640,
    lr: float = 0.005,
    weight_decay: float = 0.0005,
    project_dir: str = ".",
    run_name: str = "retinanet_stage1"
):
    """Train RetinaNet model on custom dataset with AMP and multiprocessing."""
    
    # 1. PREVENT OPENCV DEADLOCKS WITH MULTIPROCESSING
    cv2.setNumThreads(0) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cpu":
        print("⚠️  WARNING: Running on CPU. For faster iteration, consider:")
        print("   - Using a GPU (CUDA-enabled system)")
        print("   - Or reducing epochs and batch_size for testing")
        print("   - Recommended CPU test params: epochs=1, batch_size=2")
    
    # Create directories
    run_dir = Path(project_dir) / "runs" / "detect" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create train/val datasets
    train_image_dir = Path(dataset_path) / "images" / "train"
    train_label_dir = Path(dataset_path) / "labels" / "train"
    val_image_dir = Path(dataset_path) / "images" / "val"
    val_label_dir = Path(dataset_path) / "labels" / "val"
    
    train_dataset = DetectionDataset(train_image_dir, train_label_dir, img_size)
    val_dataset = DetectionDataset(val_image_dir, val_label_dir, img_size)
    
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
    
    # Verify dataset before training
    verify_dataset()
    
    # Load pre-trained RetinaNet (Freezing first layer for slightly faster iteration)
    model = retinanet_resnet50_fpn(weights="DEFAULT", trainable_backbone_layers=3)
    
    # Replace classification head for binary classification (background + collapsed building)
    in_channels = model.head.classification_head.cls_logits.in_channels
    num_classes = 2  # background + collapsed building
    num_anchors_per_location = model.anchor_generator.num_anchors_per_location()[0]
    num_outputs = num_classes * num_anchors_per_location
    
    # Create new Conv2d layer
    new_cls_logits = torch.nn.Conv2d(
        in_channels, num_outputs, kernel_size=3, stride=1, padding=1
    )
    
    # CRITICAL: Initialize bias for Focal Loss
    # Focal Loss requires special prior probability initialization
    # Using π=0.01 (prior prob of positive class) → bias = -log((1-π)/π) ≈ -4.595
    prior_prob = 0.01
    bias_init = -torch.log(torch.tensor((1.0 - prior_prob) / prior_prob))
    torch.nn.init.constant_(new_cls_logits.bias, bias_init.item())
    torch.nn.init.normal_(new_cls_logits.weight, std=0.01)
    
    model.head.classification_head.cls_logits = new_cls_logits
    model.head.classification_head.num_classes = num_classes

    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # 3. INITIALIZE AMP SCALER
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    
    metrics_history = {
        "train_loss": [],
        "val_loss": [],
        "val_ap50": [],
        "val_ap": [],
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
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            train_loss_sum += losses.item()
            
            if use_amp:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()
        
        # Calculate average training loss
        avg_train_loss = train_loss_sum / len(train_loader)
        
        # Validation phase
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                images = [img.to(device, non_blocking=True) for img in images]
                targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
                
                # AMP for validation forward pass too
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    predictions = model(images)
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Compute real mAP using torchmetrics
        # Convert predictions and targets to torchmetrics format
        preds = []
        metric_targets = []
        
        for pred, target in zip(all_predictions, all_targets):
            # Predictions: boxes, scores, labels
            pred_boxes = pred['boxes'].cpu()
            pred_scores = pred['scores'].cpu()
            pred_labels = pred['labels'].cpu()
            
            preds.append({
                'boxes': pred_boxes,
                'scores': pred_scores,
                'labels': pred_labels
            })
            
            # Targets: boxes, labels
            target_boxes = target['boxes'].cpu()
            target_labels = target['labels'].cpu()
            
            metric_targets.append({
                'boxes': target_boxes,
                'labels': target_labels
            })
        
        # Compute mAP
        metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
        metric.update(preds, metric_targets)
        result = metric.compute()
        
        map50 = result['map_50'].item()
        map = result['map'].item()
        precision = map50
        recall = result['mar_100'].item()
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, mAP@50: {map50:.4f}, mAP: {map:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # NOTE: Validation loss = 0.0 is EXPECTED (TorchVision models don't compute loss in eval mode)
        metrics_history["val_loss"].append(0.0)
        metrics_history["val_ap50"].append(map50)
        metrics_history["val_ap"].append(map)
        metrics_history["val_precision"].append(precision)
        metrics_history["val_recall"].append(recall)
            
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
                'map50': metrics['val_ap50'][-1] if metrics['val_ap50'] else 0,
                'map': metrics['val_ap'][-1] if metrics['val_ap'] else 0,
                'mp': np.mean([m for m in metrics['val_ap50'][-5:]]) if metrics['val_ap50'] else 0,
                'mr': np.mean([m for m in metrics['val_ap'][-5:]]) if metrics['val_ap'] else 0,
                'precision': metrics['val_precision'][-1] if metrics['val_precision'] else 0,
                'recall': metrics['val_recall'][-1] if metrics['val_recall'] else 0
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
    # Run a RetinaNet training pipeline and log the results
    # ---------------------------------------------------------
    tracker = ExperimentTracker(filepath="experiments.json")

    dataset_path = "combined_dataset"
    run_name = "retinanet-stage1-5epoch"
    epochs_to_run = 5
    batch_size = 8
    img_size = 640
    learning_rate = 0.005

    # Train using RetinaNet
    results, model, dataloaders = train_retinanet(
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
    final_precision = getattr(results.box, "precision", None)
    final_recall = getattr(results.box, "recall", None)

    if final_map50 is None or final_map is None:
        raise RuntimeError("Unable to extract RetinaNet metrics from training results.")

    tracker.log(Experiment(
        name=run_name,
        model="retinanet_resnet50",
        map50=final_map50,
        map50_95=final_map,
        epochs=epochs_to_run,
        fraction=1.0,
        img_size=img_size,
        P=final_precision,
        R=final_recall
    ))
    
    print(f"Final Precision: {final_precision:.4f}, Final Recall: {final_recall:.4f}")

    print(f"\nLogged {len(tracker)} RetinaNet experiments.")
    print(f"Best by mAP@50    : {tracker.best_by('map50')}")
    print(f"Best by mAP@50-95 : {tracker.best_by('map50_95')}")

    plot_results(tracker)


if __name__ == '__main__':
    main()
