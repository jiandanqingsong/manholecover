from ultralytics import YOLO
import os
import argparse

def evaluate_model(data_yaml, split, model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using 'python main.py'")
        return

    if not os.path.exists(data_yaml):
        print(f"Error: Data config not found at {data_yaml}")
        return

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    print(f"Starting evaluation on '{split}' set from {data_yaml}...")
    
    try:
        # Run validation
        metrics = model.val(
            data=data_yaml,
            split=split,           # Evaluate on specified split (train, val, test)
            project="runs/val",    # Save results to runs/val/
            name="evaluation",     # Save results to runs/val/evaluation/
            exist_ok=True,         # Overwrite if exists
            plots=True             # Generate plots (Confusion Matrix, PR curve, etc.)
        )
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("Note: Ensure your data.yaml contains the specified split and that the dataset has labels.")
        return

    # Print key metrics
    print("\n===== Evaluation Results =====")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    # Calculate and print TP, FP, FN for each class
    print("\n===== Class-wise Statistics (TP, FP, FN) =====")
    try:
        cm = metrics.confusion_matrix.matrix
        names = metrics.names
        
        print(f"{'Class':<15} | {'TP':<8} | {'FP':<8} | {'FN':<8}")
        print("-" * 46)
        
        for i, name in names.items():
            tp = int(cm[i, i])
            fp = int(cm[:, i].sum() - tp)
            fn = int(cm[i, :].sum() - tp)
            print(f"{name:<15} | {tp:<8} | {fp:<8} | {fn:<8}")
            
    except Exception as e:
        print(f"Could not extract detailed statistics: {e}")
    
    print("\n===== Plots Saved =====")
    print(f"Detailed plots (P_curve, R_curve, PR_curve, F1_curve, confusion_matrix) are saved in:")
    print(f"-> {metrics.save_dir}")
    print("Please check this folder to view the curves.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model on a dataset.")
    
    parser.add_argument("--data", type=str, default="data_processed/data.yaml", 
                        help="Path to data.yaml file (default: data_processed/data.yaml)")
    
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Dataset split to evaluate on (default: val)")
    
    parser.add_argument("--model", type=str, default="runs/train/manhole_yolov8/weights/best.pt",
                        help="Path to trained model weights")

    args = parser.parse_args()
    
    evaluate_model(args.data, args.split, args.model)
