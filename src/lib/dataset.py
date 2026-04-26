import os
import kagglehub
import cv2
import shutil
import random
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
    """Filter pre-converted VisDrone YOLO annotations.
    
    The Kaggle dataset is already in YOLO format (spaces, not commas).
    VisDrone category mapping:
    0 = pedestrian (person)
    1 = people
    """
    yolo_lines = []
    with open(txt_path, "r") as f:
        for line in f.readlines():
            # 1. Split by spaces instead of commas
            parts = line.strip().split()
            
            # 2. YOLO format has exactly 5 parts
            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            # 3. Include only pedestrian (0) and people (1)
            if class_id in [0, 1]:
                # Force class ID to 0 to seamlessly merge with C2A
                parts[0] = "0"
                yolo_lines.append(" ".join(parts))
                
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

def print_background_proportions():
    """Print the proportions of background images (0 humans) in all splits.
    
    Returns:
        float: The proportion of background images in the training set.
    """
    print("\n--- Background Image Proportions (0 humans) ---")
    train_proportion = 0
    
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
            
            # Store train proportion for later use
            if split == "train":
                train_proportion = proportion
        else:
            print(f"{split.capitalize()}: No images found")
    
    return train_proportion

def reduce_background_images_in_train(curr_proportion, target_percentage=10):
    """Reduce the amount of background images in the training set to a target percentage.
    
    Moves excess background images to a separate 'background_dataset' directory.
    
    Args:
        curr_proportion (float): Current proportion of background images in training set.
        target_percentage (int): Target percentage of background images to keep (default 10%).
    """
    dataset_path = "combined_dataset"
    background_dataset_path = "background_dataset"
    train_labels_dir = os.path.join(dataset_path, "labels", "train")
    train_images_dir = os.path.join(dataset_path, "images", "train")
    
    # Safe check: if already at or below target percentage, return immediately
    if curr_proportion <= target_percentage / 100:
        print(f"\n--- Background Images Already at Target ---")
        print(f"Current proportion: {curr_proportion:.1%}")
        print(f"Target proportion: {target_percentage/100:.1%}")
        print("No reduction needed. Returning immediately.")
        return
    
    if not os.path.exists(train_labels_dir):
        print("Training set not found!")
        return
    
    # Create background dataset directory structure
    bg_labels_dir = os.path.join(background_dataset_path, "labels", "background")
    bg_images_dir = os.path.join(background_dataset_path, "images", "background")
    os.makedirs(bg_labels_dir, exist_ok=True)
    os.makedirs(bg_images_dir, exist_ok=True)
    
    # Identify background images (empty label files)
    background_images = []
    total_images = 0
    
    for label_file in os.listdir(train_labels_dir):
        if not label_file.endswith('.txt'):
            continue
        
        total_images += 1
        label_path = os.path.join(train_labels_dir, label_file)
        
        # Check if file is empty or contains only whitespace
        with open(label_path, 'r') as f:
            content = f.read().strip()
            if not content:  # Background image
                background_images.append(label_file)
    
    if not background_images:
        print("No background images found in train set.")
        return
    
    # Calculate how many background images to remove
    target_count = max(1, int(len(background_images) * target_percentage / 100))
    images_to_remove = background_images[target_count:]
    
    print(f"\n--- Reducing Background Images in Training Set ---")
    print(f"Total background images: {len(background_images)}")
    print(f"Target percentage: {target_percentage}%")
    print(f"Images to keep: {target_count}")
    print(f"Images to remove: {len(images_to_remove)}")
    
    # Move excess background images to background_dataset
    for label_file in tqdm(images_to_remove, desc="Moving background images"):
        # Get corresponding image filename
        image_file = os.path.splitext(label_file)[0] + ".jpg"
        
        # Move label
        label_src = os.path.join(train_labels_dir, label_file)
        label_dst = os.path.join(bg_labels_dir, label_file)
        if os.path.exists(label_src):
            shutil.move(label_src, label_dst)
        
        # Move image
        image_src = os.path.join(train_images_dir, image_file)
        image_dst = os.path.join(bg_images_dir, image_file)
        if os.path.exists(image_src):
            shutil.move(image_src, image_dst)
    
    print(f"✅ Moved {len(images_to_remove)} background images to 'background_dataset'.")

def verify_dataset():
    """Checks if the required YOLO directory structure exists."""

    dataset_path = "combined_dataset"
    
    print(f"Dataset active directory: {dataset_path}")
    assert os.path.exists(os.path.join(dataset_path, "images", "train")), "Train images missing!"
    assert os.path.exists(os.path.join(dataset_path, "labels", "train")), "Train labels missing!"
    assert os.path.exists(os.path.join(dataset_path, "images", "val")), "Validation images missing!"
    assert os.path.exists(os.path.join(dataset_path, "labels", "val")), "Validation labels missing!"
    print("✅ Dataset verified!")

if __name__ == '__main__':
    merge_datasets()
    train_proportion = print_background_proportions()
    reduce_background_images_in_train(train_proportion)
    verify_dataset()