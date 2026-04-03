import os
import yaml
import kagglehub
import cv2
from ultralytics import YOLO
import shutil
from tqdm import tqdm

# Dynamically locate the dataset
print("Locating dataset...")
c2a_path = kagglehub.dataset_download("rgbnihal/c2a-dataset")
visdrone_path = kagglehub.dataset_download("banuprasadb/visdrone-dataset")

# Build paths dynamically (OS-independent)
C2A_TRAIN_IMG = os.path.join(c2a_path, "C2A_Dataset", "new_dataset3", "train", "images")
C2A_TRAIN_LABEL = os.path.join(c2a_path, "C2A_Dataset", "new_dataset3", "train", "labels")

C2A_VAL_IMG = os.path.join(c2a_path, "C2A_Dataset", "new_dataset3", "val", "images")
C2A_VAL_LABEL = os.path.join(c2a_path, "C2A_Dataset", "new_dataset3", "val", "labels")

VIS_TRAIN_IMG = os.path.join(visdrone_path, "VisDrone_Dataset", "VisDrone2019-DET-train", "images")
VIS_TRAIN_LABEL = os.path.join(visdrone_path, "VisDrone_Dataset", "VisDrone2019-DET-train", "labels")
VIS_VAL_IMG = os.path.join(visdrone_path, "VisDrone_Dataset", "VisDrone2019-DET-val", "images")
VIS_VAL_LABEL = os.path.join(visdrone_path, "VisDrone_Dataset", "VisDrone2019-DET-val", "labels")

OUTPUT = "combined_dataset"


for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT, "labels", split), exist_ok=True)



def convert_visdrone(txt_path, img_w, img_h):
    yolo_lines = []

    with open(txt_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split(",")

            if len(parts) < 6:
                continue

            x, y, w, h = map(float, parts[:4])
            category = int(parts[5])

            # 只保留人
            if category not in [1, 2]:
                continue

            x_c = (x + w / 2) / img_w
            y_c = (y + h / 2) / img_h
            w /= img_w
            h /= img_h

            if w <= 0 or h <= 0:
                continue

            yolo_lines.append(f"0 {x_c} {y_c} {w} {h}")

    return yolo_lines



def process_c2a(img_dir, label_dir, split):
    for file in tqdm(os.listdir(img_dir), desc=f"C2A {split}"):
        if not file.endswith((".jpg", ".png")):
            continue

        name = os.path.splitext(file)[0]

        img_src = os.path.join(img_dir, file)
        label_src = os.path.join(label_dir, name + ".txt")

        new_name = f"c2a_{name}"

        img_dst = os.path.join(OUTPUT, "images", split, new_name + ".jpg")
        label_dst = os.path.join(OUTPUT, "labels", split, new_name + ".txt")

        shutil.copy(img_src, img_dst)

        if os.path.exists(label_src):
            with open(label_src, "r") as f:
                lines = f.readlines()

            new_lines = []
            for l in lines:
                parts = l.strip().split()
                if len(parts) != 5:
                    continue
                parts[0] = "0"
                new_lines.append(" ".join(parts))

            with open(label_dst, "w") as f:
                f.write("\n".join(new_lines))
        else:
            open(label_dst, "w").close()



def process_vis(img_dir, label_dir, split):
    for file in tqdm(os.listdir(img_dir), desc=f"VisDrone {split}"):
        if not file.endswith((".jpg", ".png")):
            continue

        name = os.path.splitext(file)[0]

        img_path = os.path.join(img_dir, file)
        txt_path = os.path.join(label_dir, name + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        yolo_lines = []
        if os.path.exists(txt_path):
            yolo_lines = convert_visdrone(txt_path, w, h)

        if len(yolo_lines) == 0:
            continue

        new_name = f"vis_{name}"

        img_dst = os.path.join(OUTPUT, "images", split, new_name + ".jpg")
        label_dst = os.path.join(OUTPUT, "labels", split, new_name + ".txt")

        shutil.copy(img_path, img_dst)

        with open(label_dst, "w") as f:
            f.write("\n".join(yolo_lines))


process_c2a(C2A_TRAIN_IMG, C2A_TRAIN_LABEL, "train")
process_c2a(C2A_VAL_IMG, C2A_VAL_LABEL, "val")

process_vis(VIS_TRAIN_IMG, VIS_TRAIN_LABEL, "train")
process_vis(VIS_VAL_IMG, VIS_VAL_LABEL, "val")

print("✅ DONE")