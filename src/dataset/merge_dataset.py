import os
import kagglehub
import cv2
import shutil
from tqdm import tqdm

# Download datasets
c2a_path = kagglehub.dataset_download("rgbnihal/c2a-dataset")
visdrone_path = kagglehub.dataset_download("banuprasadb/visdrone-dataset")

# C2A Paths
C2A_TRAIN_IMG = os.path.join(c2a_path, "C2A_Dataset", "new_dataset3", "train", "images")
C2A_TRAIN_LABEL = os.path.join(c2a_path, "C2A_Dataset", "new_dataset3", "train", "labels")
C2A_VAL_IMG = os.path.join(c2a_path, "C2A_Dataset", "new_dataset3", "val", "images")
C2A_VAL_LABEL = os.path.join(c2a_path, "C2A_Dataset", "new_dataset3", "val", "labels")
C2A_TEST_IMG = os.path.join(c2a_path, "C2A_Dataset", "new_dataset3", "test", "images")
C2A_TEST_LABEL = os.path.join(c2a_path, "C2A_Dataset", "new_dataset3", "test", "labels")

# VisDrone Paths
VIS_TRAIN_IMG = os.path.join(visdrone_path, "VisDrone_Dataset", "VisDrone2019-DET-train", "images")
VIS_TRAIN_LABEL = os.path.join(visdrone_path, "VisDrone_Dataset", "VisDrone2019-DET-train", "labels")
VIS_VAL_IMG = os.path.join(visdrone_path, "VisDrone_Dataset", "VisDrone2019-DET-val", "images")
VIS_VAL_LABEL = os.path.join(visdrone_path, "VisDrone_Dataset", "VisDrone2019-DET-val", "labels")
VIS_TEST_IMG = os.path.join(visdrone_path, "VisDrone_Dataset", "VisDrone2019-DET-test-dev", "images")
VIS_TEST_LABEL = os.path.join(visdrone_path, "VisDrone_Dataset", "VisDrone2019-DET-test-dev", "labels")

OUTPUT = "combined_dataset"

def convert_visdrone(txt_path, img_w, img_h):
    yolo_lines = []
    with open(txt_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue

            x, y, w, h = map(float, parts[:4])
            category = int(parts[5])

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
    count = 0
    if not os.path.exists(img_dir):
        return count
        
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
        count += 1

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
            
    return count

def process_vis(img_dir, label_dir, split):
    count = 0
    if not os.path.exists(img_dir):
        return count
        
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
        count += 1

        yolo_lines = []
        if os.path.exists(txt_path):
            yolo_lines = convert_visdrone(txt_path, w, h)

        new_name = f"vis_{name}"
        img_dst = os.path.join(OUTPUT, "images", split, new_name + ".jpg")
        label_dst = os.path.join(OUTPUT, "labels", split, new_name + ".txt")

        shutil.copy(img_path, img_dst)

        with open(label_dst, "w") as f:
            f.write("\n".join(yolo_lines))
            
    return count

def merge_datasets():
    # --- NEW CHECK ADDED HERE ---
    if os.path.exists(OUTPUT):
        print(f"The directory '{OUTPUT}' already exists.")
        print("Skipping merge. If you need to rebuild the dataset, please delete the folder first.")
        return 
    # ----------------------------

    print("Locating dataset...")
    
    # Build directories for train, val, AND test
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT, "labels", split), exist_ok=True)

    # Process and tally C2A
    c2a_t = process_c2a(C2A_TRAIN_IMG, C2A_TRAIN_LABEL, "train")
    c2a_v = process_c2a(C2A_VAL_IMG, C2A_VAL_LABEL, "val")
    c2a_te = process_c2a(C2A_TEST_IMG, C2A_TEST_LABEL, "test")

    # Process and tally VisDrone
    vis_t = process_vis(VIS_TRAIN_IMG, VIS_TRAIN_LABEL, "train")
    vis_v = process_vis(VIS_VAL_IMG, VIS_VAL_LABEL, "val")
    vis_te = process_vis(VIS_TEST_IMG, VIS_TEST_LABEL, "test")

    # Print the final tally
    print("\n✅ DONE!")
    print("-" * 30)
    print(f"C2A Files       -> Train: {c2a_t} | Val: {c2a_v} | Test: {c2a_te}")
    print(f"VisDrone Files  -> Train: {vis_t} | Val: {vis_v} | Test: {vis_te}")
    print("-" * 30)
    print(f"TOTALS          -> Train: {c2a_t + vis_t} | Val: {c2a_v + vis_v} | Test: {c2a_te + vis_te}")
    
    # Print background proportions
def print_background_proportions():
    """Print the proportions of background images (0 humans) in all splits."""
    print("\n--- Background Image Proportions (0 humans) ---")
    for split in ["train", "val", "test"]:
        labels_dir = os.path.join(OUTPUT, "labels", split)
        if not os.path.exists(labels_dir):
            continue
            
        total_images = 0
        background_images = 0
        
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue
                
            total_images += 1
            label_path = os.path.join(labels_dir, label_file)
            
            # Check if file is empty or contains only whitespace
            with open(label_path, 'r') as f:
                content = f.read().strip()
                if not content:  # Empty file = background image
                    background_images += 1
        
        if total_images > 0:
            proportion = background_images / total_images
            print(f"{split.capitalize()}: {background_images}/{total_images} ({proportion:.1%}) background images")
        else:
            print(f"{split.capitalize()}: No images found")


if __name__ == '__main__':
    merge_datasets()
    print_background_proportions()