import os
import shutil
import random
import yaml
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

# Config
RAW_DATA_DIR = os.path.join(os.getcwd(), "data")
PROCESSED_DATA_DIR = os.path.join(os.getcwd(), "data_processed")
CLASSES = ["broke", "good", "circle", "lose", "uncovered"]

# Augmentation Configuration
AUGMENT_TIMES = 2  # Generate 2 augmented images for each original image
ENABLE_AUGMENTATION = True

def get_augment_pipeline():
    """
    Define the augmentation pipeline.
    Includes geometric transformations and pixel-level enhancements.
    """
    return A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomScale(scale_limit=0.1, p=0.5),
        
        # Pixel-level enhancements (Image Enhancement)
        A.OneOf([
            A.RandomBrightnessContrast(p=0.8),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5), # Contrast Limited Adaptive Histogram Equalization
            A.HueSaturationValue(p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
        ], p=0.8),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

def read_yolo_label(label_path):
    """Reads YOLO format label file."""
    bboxes = []
    class_labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    bboxes.append([x, y, w, h])
                    class_labels.append(cls)
    return bboxes, class_labels

def save_yolo_label(label_path, bboxes, class_labels):
    """Saves YOLO format label file."""
    with open(label_path, 'w') as f:
        for bbox, cls in zip(bboxes, class_labels):
            # Ensure normalized coordinates are within [0, 1]
            x, y, w, h = bbox
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def prepare_data():
    """
    Organizes data into standard YOLO structure with Augmentation.
    """
    print("Scanning for data...")
    
    # 1. Collect all image-label pairs
    all_samples = []
    labels_dir = os.path.join(RAW_DATA_DIR, "labels")
    
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        if "labels" in root: continue 
        
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                img_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                label_name = base_name + ".txt"
                
                label_path = os.path.join(labels_dir, label_name)
                if not os.path.exists(label_path):
                    label_path = os.path.join(root, label_name)
                
                if os.path.exists(label_path):
                    all_samples.append((img_path, label_path))
    
    print(f"Found {len(all_samples)} valid image-label pairs.")
    if len(all_samples) == 0:
        raise ValueError("No data found! Please check data/ directory structure.")
    
    # 2. Split Train/Val
    random.seed(42)
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    # 3. Process and Augment
    if os.path.exists(PROCESSED_DATA_DIR):
        print("Cleaning up old processed data...")
        shutil.rmtree(PROCESSED_DATA_DIR)
    
    transform = get_augment_pipeline()
    
    for phase, samples in [("train", train_samples), ("val", val_samples)]:
        img_dest_dir = os.path.join(PROCESSED_DATA_DIR, phase, "images")
        lbl_dest_dir = os.path.join(PROCESSED_DATA_DIR, phase, "labels")
        os.makedirs(img_dest_dir, exist_ok=True)
        os.makedirs(lbl_dest_dir, exist_ok=True)
        
        for img_src, lbl_src in tqdm(samples, desc=f"Processing {phase}"):
            # --- Process Original ---
            base_name = os.path.splitext(os.path.basename(img_src))[0]
            ext = os.path.splitext(img_src)[1]
            
            # Copy original image
            shutil.copy(img_src, os.path.join(img_dest_dir, f"{base_name}{ext}"))
            # Copy original label
            shutil.copy(lbl_src, os.path.join(lbl_dest_dir, f"{base_name}.txt"))
            
            # --- Augmentation (Only for Train set) ---
            if phase == "train" and ENABLE_AUGMENTATION:
                # Read image
                image = cv2.imread(img_src)
                if image is None: continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Read label
                bboxes, class_labels = read_yolo_label(lbl_src)
                
                if not bboxes: continue # Skip augmentation if no bboxes (background images)
                
                for i in range(AUGMENT_TIMES):
                    try:
                        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                        aug_img = augmented['image']
                        aug_bboxes = augmented['bboxes']
                        aug_labels = augmented['class_labels']
                        
                        if len(aug_bboxes) == 0: continue # Skip if augmentation removed all boxes
                        
                        # Save augmented image
                        aug_filename = f"{base_name}_aug_{i}"
                        aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(img_dest_dir, f"{aug_filename}{ext}"), aug_img_bgr)
                        
                        # Save augmented label
                        save_yolo_label(os.path.join(lbl_dest_dir, f"{aug_filename}.txt"), aug_bboxes, aug_labels)
                    except Exception as e:
                        # print(f"Augmentation failed for {base_name}: {e}")
                        pass

    # 4. Create data.yaml
    yaml_content = {
        "path": PROCESSED_DATA_DIR,
        "train": "train/images",
        "val": "val/images",
        "names": {i: name for i, name in enumerate(CLASSES)}
    }
    
    yaml_path = os.path.join(PROCESSED_DATA_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"Data preparation complete. Config saved to {yaml_path}")
    return yaml_path

if __name__ == "__main__":
    prepare_data()
