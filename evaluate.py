from ultralytics import YOLO
import os

def evaluate_model():
    # Path to the best trained model
    # Adjust this path if your training run name is different
    model_path = "runs/train/manhole_yolov8/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using 'python main.py'")
        # Fallback for demonstration if user hasn't trained yet but wants to see code run
        # model_path = "yolov8s.pt" 
        return

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # Run validation
    # This will generate P, R, PR, F1 curves and calculate mAP automatically
    print("Starting evaluation on validation set...")
    # data argument points to the yaml file created by pre.py
    metrics = model.val(
        data="data_processed/data.yaml",
        split="val",           # Evaluate on validation set
        project="runs/val",    # Save results to runs/val/
        name="evaluation",     # Save results to runs/val/evaluation/
        exist_ok=True,         # Overwrite if exists
        plots=True             # Generate plots (Confusion Matrix, PR curve, etc.)
    )

    # Print key metrics
    print("\n===== Evaluation Results =====")
    # metrics.box.map50 is mAP@0.5
    # metrics.box.map is mAP@0.5:0.95
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    # Calculate and print TP, FP, FN for each class
    print("\n===== Class-wise Statistics (TP, FP, FN) =====")
    try:
        # Access confusion matrix
        # metrics.confusion_matrix is a ConfusionMatrix object
        # .matrix is the numpy array of shape (nc+1, nc+1)
        # Rows: True Class, Columns: Predicted Class
        cm = metrics.confusion_matrix.matrix
        
        # Class names
        names = metrics.names
        
        # Header
        print(f"{'Class':<15} | {'TP':<8} | {'FP':<8} | {'FN':<8}")
        print("-" * 46)
        
        # Iterate over classes
        for i, name in names.items():
            # TP: Predicted i, Actual i
            tp = int(cm[i, i])
            
            # FP: Predicted i, Actual NOT i (Sum of column i - TP)
            # This includes background predicted as class i
            fp = int(cm[:, i].sum() - tp)
            
            # FN: Actual i, Predicted NOT i (Sum of row i - TP)
            # This includes class i predicted as background or other classes
            fn = int(cm[i, :].sum() - tp)
            
            print(f"{name:<15} | {tp:<8} | {fp:<8} | {fn:<8}")
            
    except Exception as e:
        print(f"Could not extract detailed statistics: {e}")
    
    print("\n===== Plots Saved =====")
    print(f"Detailed plots (P_curve, R_curve, PR_curve, F1_curve, confusion_matrix) are saved in:")
    print(f"-> {metrics.save_dir}")
    print("Please check this folder to view the curves.")

if __name__ == "__main__":
    evaluate_model()
