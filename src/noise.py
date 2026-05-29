import cv2 as cv
import numpy as np
import random as rand
from pathlib import Path
import shutil
from tqdm import tqdm

def rescaleImage(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv.resize(image, (width, height), interpolation=cv.INTER_AREA)

def drawRectangle(image: np.ndarray, label: str, a: int, b: int, x: int, y: int):
    cv.rectangle(image, (a,b), (x, y), (0,0,255), 2)
    cv.putText(image, label, (a, b-4), cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
    return image

def addMotionBlur(image):
    # Create a matrix of random size, this is the kernel
    kernelHeight = rand.randint(2, 4)
    kernelWidth = rand.randint(2, 4)
    kernel = np.zeros((kernelHeight, kernelWidth), dtype=np.float32)
    
    # Put 1s across kernel, must go from one side to another either horizontal or vertical
    if kernelWidth >= kernelHeight:        
        a, x = 0, kernelWidth
        b, y = rand.randint(0, kernelHeight-1), rand.randint(0, kernelWidth-1)
    else:
        a, x = rand.randint(0, kernelHeight-1), rand.randint(0, kernelWidth-1)
        b, y = 0, kernelHeight
    kernel = cv.line(kernel, pt1=(a, b), pt2=(x, y), color=(1.0,), thickness=1)    
    kernel /= kernel.sum()

    motionBluredImage = cv.filter2D(image, -1, kernel)    
    return motionBluredImage

def addRotationMotionBlur(image):
    output = np.zeros_like(image, dtype=np.float32)
    num_steps = 20
    angle_range = rand.randint(-1, 1) # total degrees of rotation
    width = image.shape[1]
    height = image.shape[0]

    for i in range(num_steps):
        angle = (i / num_steps) * angle_range - angle_range / 2
        M = cv.getRotationMatrix2D((width//2, height//2), angle, scale=1.0) 
        rotated = cv.warpAffine(image, M, (width, height)) 
        output += rotated.astype(np.float32) 

    output /= num_steps
    rotationMotionBluredImage = output.astype(np.uint8) 
    return rotationMotionBluredImage

def addSmoke(image):
    width = image.shape[1]
    height = image.shape[0]

    # Ensure we have minimum space for smoke region (at least 50px, adaptive to image size)
    min_offset = min(50, max(10, width // 4, height // 4))

    x1 = rand.randint(0, max(0, width - min_offset - 1))
    y1 = rand.randint(0, max(0, height - min_offset - 1))
    
    # Ensure x2 and y2 are valid and create a region of at least min_offset size
    x2 = rand.randint(x1 + min_offset, width)
    y2 = rand.randint(y1 + min_offset, height)

    bluredImage = cv.GaussianBlur(image, (5,5), sigmaX=0)

    bluredImage = cv.cvtColor(bluredImage, cv.COLOR_BGR2HSV).astype(np.float32)
    bluredImage[:, :, 1] *= 0.92  # reduce the saturation
    np.clip(bluredImage, 0, 255, out=bluredImage)  # clip the whole array in place

    output = cv.cvtColor(bluredImage.astype(np.uint8), cv.COLOR_HSV2BGR)

    smokeImage = image.copy()
    smokeImage[y1:y2, x1:x2] = output[y1:y2, x1:x2]    
    return smokeImage

def filterImage(image, mBlur=False, rBlur=False, smoke=False, sharpen=False):
    """
    Applies selected filters and returns the modified image.
    """
    newImage = image.copy()

    if mBlur:
        newImage = addMotionBlur(newImage)
    if rBlur:
        newImage = addRotationMotionBlur(newImage)
    if smoke:
        newImage = addSmoke(newImage)
    elif sharpen:
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        newImage = cv.filter2D(newImage, -1, kernel)

    return newImage


def augment_test_dataset(input_dataset_dir: str, output_dataset_dir: str):
    """
    Reads all images in the test split, applies random augmentations, 
    and saves them to a new directory while copying the labels.
    """
    input_dir = Path(input_dataset_dir)
    output_dir = Path(output_dataset_dir)

    # Define paths based on standard YOLO format
    img_in_dir = input_dir / "images" / "test"
    lbl_in_dir = input_dir / "labels" / "test"
    
    img_out_dir = output_dir / "images" / "test"
    lbl_out_dir = output_dir / "labels" / "test"

    # Create output directories if they don't exist
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    # Gather all image files
    image_files = list(img_in_dir.glob("*.jpg")) + list(img_in_dir.glob("*.png"))
    
    if not image_files:
        print(f"No images found in {img_in_dir}!")
        return

    print(f"Processing {len(image_files)} images from the test dataset...")

    for img_path in tqdm(image_files, desc="Augmenting Images"):
        # 1. Read image
        image = cv.imread(str(img_path))
        if image is None:
            continue
            
        # 2. Randomly decide which filters to apply to simulate varying real-world noise
        apply_mBlur = rand.choice([True, False])
        apply_rBlur = rand.choice([True, False])
        apply_smoke = rand.choice([True, False])
        # Only sharpen if smoke isn't applied (matching your original elif logic)
        apply_sharpen = not apply_smoke and rand.choice([True, False])

        # 3. Apply the filters
        augmented_img = filterImage(
            image, 
            mBlur=apply_mBlur, 
            rBlur=apply_rBlur, 
            smoke=apply_smoke, 
            sharpen=apply_sharpen
        )

        # 4. Save the new image to the output directory
        out_img_path = img_out_dir / img_path.name
        cv.imwrite(str(out_img_path), augmented_img)

        # 5. Copy the corresponding label file (if it exists)
        # Because we didn't translate or squish the image, bounding boxes remain identically placed!
        lbl_path = lbl_in_dir / (img_path.stem + ".txt")
        out_lbl_path = lbl_out_dir / (img_path.stem + ".txt")
        
        if lbl_path.exists():
            shutil.copy(str(lbl_path), str(out_lbl_path))

    print(f"\n✅ Augmented dataset successfully created at: {output_dataset_dir}")
    print("You can now evaluate your models on this new test set to check robustness!")


if __name__ == '__main__':
    # Define your existing dataset and where you want the corrupted dataset to go
    INPUT_DATASET = "combined_dataset"
    OUTPUT_DATASET = "combined_dataset_corrupted"
    
    augment_test_dataset(INPUT_DATASET, OUTPUT_DATASET)