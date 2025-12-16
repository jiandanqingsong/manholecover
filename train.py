from ultralytics import YOLO
import os

import torch

def train_yolov8(data_yaml_path, epochs=100, img_size=640):
    """
    Trains YOLOv8 model using standard Ultralytics pipeline.
    """
    print(f"Starting training with config: {data_yaml_path}")
    
    # Load model
    # Using yolov8s.pt (Small) for better performance on RTX 4060 Laptop
    # yolov8n (Nano) is fastest, yolov8s (Small) is balanced, yolov8m (Medium) is heavier.
    model_path = "yolov8s.pt"
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found locally, it will be downloaded.")
    
    model = YOLO(model_path) 
    
    # Determine device
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Train
    # Using standard arguments for best results
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=16, # Batch 16 is usually fine for 8GB VRAM with YOLOv8s. If OOM, try 8.
        patience=20, # Early stopping
        save=True,
        device=device,
        project="runs/train",
        name="manhole_yolov8",
        exist_ok=True, # Overwrite existing experiment
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        cos_lr=True, # Cosine annealing
        val=True, # Validate during training
        amp=True, # Mixed precision
        workers=4 # Adjust based on CPU cores. 4 is usually safe.
    )
    
    print("Training complete.")
    print(f"Best model saved to {results.save_dir}/weights/best.pt")
    return model

if __name__ == "__main__":
    # For testing standalone
    if os.path.exists("data_processed/data.yaml"):
        train_yolov8("data_processed/data.yaml", epochs=1)
