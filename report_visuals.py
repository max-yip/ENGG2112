import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn


def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    return [x1, y1, x2, y2]


def load_yolo_labels(label_path, img_w, img_h):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            xyxy = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
            boxes.append({"class": cls, "xyxy": xyxy, "conf": None})
    return boxes


def predict_yolo(image_path, model_path, imgsz=960, conf=0.2): # Adjusted conf to 0.05 to match torchvision
    model = YOLO(str(model_path))
    results = model(str(image_path), imgsz=imgsz, conf=conf)
    if len(results) == 0:
        return []
    boxes = []
    res = results[0]
    if hasattr(res, "boxes") and res.boxes is not None:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        for cls, box, score in zip(classes, xyxy, confs):
            boxes.append({"class": int(cls), "xyxy": [float(box[0]), float(box[1]), float(box[2]), float(box[3])], "conf": float(score)})
    return boxes


def load_torchvision_model(model_path, model_type, device):
    if model_type == "faster_rcnn":
        model = fasterrcnn_resnet50_fpn(num_classes=2)
    elif model_type == "retinanet":
        model = retinanet_resnet50_fpn(num_classes=2)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_torchvision(image_path, model_path, model_type, img_size=640, score_thresh=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_torchvision_model(model_path, model_type, device)
    image = torchvision.io.read_image(str(image_path))
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] == 4:
        image = image[:3, :, :]
    image = image.float() / 255.0
    image = F.resize(image, [img_size, img_size], antialias=True)
    image = image.to(device)
    with torch.no_grad():
        prediction = model([image])[0]

    boxes = []
    img = load_image(image_path)
    orig_h, orig_w = img.shape[:2]
    ratio_h = orig_h / img_size
    ratio_w = orig_w / img_size

    pred_boxes = prediction.get("boxes")
    pred_scores = prediction.get("scores")
    pred_labels = prediction.get("labels")
    if pred_boxes is None or pred_scores is None:
        return boxes

    keep = pred_scores >= score_thresh
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    pred_labels = pred_labels[keep]

    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        x1, y1, x2, y2 = box.tolist()
        x1 *= ratio_w
        x2 *= ratio_w
        y1 *= ratio_h
        y2 *= ratio_h
        boxes.append({"class": int(label.item()), "xyxy": [x1, y1, x2, y2], "conf": float(score.item())})
    return boxes


def find_run_dir(base_runs_dir, model_name):
    base = Path(base_runs_dir)
    if not base.exists():
        return None
    for p in base.iterdir():
        if p.is_dir() and model_name in p.name:
            return p
    # fallback: exact name
    cand = base / model_name
    if cand.exists():
        return cand
    # last resort: try substring match
    for p in base.iterdir():
        if p.is_dir() and any(tok in p.name for tok in model_name.split("_")):
            return p
    return None


def load_predictions_for_image(run_dir, image_stem, img_w, img_h):
    possible_dirs = [run_dir / "labels", run_dir / "predict" / "labels", run_dir / "results" / "labels"]
    for d in possible_dirs:
        if d.exists():
            p = d / f"{image_stem}.txt"
            if p.exists():
                return parse_pred_file(p, img_w, img_h)
    for f in run_dir.glob("**/*.txt"):
        if f.name == f"{image_stem}.txt":
            return parse_pred_file(f, img_w, img_h)
    return []


def draw_boxes(ax, img, boxes, color=(1, 0, 0), label_prefix=""):
    ax.imshow(img)
    h, w = img.shape[:2]
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        label = None
        
        # Format label strictly as value with no prefix
        if b.get("conf") is not None:
            label = f"{b.get('conf'):.2f}"
        elif b.get("class") is not None:
            if label_prefix.startswith("GT") and b["class"] == 0:
                label = None
            else:
                label = f"{label_prefix}{b.get('class')}"
                
        if label:
            # Removed background color, added max(..., 10) to prevent top clipping
            ax.text(x1, max(y1 - 4, 10), label, color=color, fontsize=9, weight='bold')
    ax.axis('off')


def visualize(image_path, models, output_dir=None):
    image_path = Path(image_path)
    image = load_image(image_path)
    img_h, img_w = image.shape[:2]
    stem = image_path.stem

    repo_root = Path(__file__).parent
    gt_labels = repo_root / "combined_dataset" / "labels" / "test" / f"{stem}.txt"
    gt_boxes = load_yolo_labels(gt_labels, img_w, img_h)

    nplots = 1 + len(models)
    ncols = 2
    nrows = (nplots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    draw_boxes(axes[0], image, gt_boxes, color=(0, 1, 0), label_prefix="GT:")
    model_title_fontsize = plt.rcParams.get('font.size', 10) * 2
    axes[0].set_title("Ground Truth", fontsize=model_title_fontsize)

    for empty_ax in axes[nplots:]:
        empty_ax.axis('off')

    for i, model in enumerate(models, start=1):
        label, model_type, model_path, img_size = model
        if not Path(model_path).exists():
            title = f"{label} (missing weights)"
            boxes = []
        else:
            if model_type == "yolo":
                boxes = predict_yolo(image_path, model_path, imgsz=img_size)
            else:
                boxes = predict_torchvision(image_path, model_path, model_type, img_size=img_size)
            title = label

        draw_boxes(axes[i], image, gt_boxes, color=(0, 1, 0), label_prefix="GT:")
        draw_boxes(axes[i], image, boxes, color=(1, 0, 0), label_prefix="P:")
        # Double font size for model titles
        axes[i].set_title(title, fontsize=model_title_fontsize)

    plt.tight_layout()
    if output_dir is None:
        output_dir = repo_root
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"report_visuals_{stem}.png"
    plt.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved visualization to {out_file}")


def main():
    examples = [
        r"C:\Users\maxyi\Documents\ENGG2112\combined_dataset\images\test\c2a_fire_image0062_0.jpg",
        r"C:\Users\maxyi\Documents\ENGG2112\combined_dataset\images\test\vis_0000347_01728_d_0000051.jpg",
    ]
    models = [
        ("YOLO26n-960", "yolo", Path("runs/detect/yolo26n-960px/weights/best.pt"), 960),
        ("RetinaNet-stage1-10epoch", "retinanet", Path("runs/detect/retinanet-stage1-10epoch/model.pt"), 640),
        ("FasterRCNN-10epoch", "faster_rcnn", Path("runs/detect/faster_rcnn_10epochs/model.pt"), 640),
    ]

    for example_path in examples:
        visualize(example_path, models, output_dir=Path(__file__).parent)


if __name__ == "__main__":
    main()