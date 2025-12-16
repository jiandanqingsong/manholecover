import os
from pre import prepare_data
from train import train_yolov8

def main():
    print("===== Manhole Cover Defect Detection System =====")
    
    # Step 1: Prepare Data
    print("\n===== 1. Data Preparation =====")
    try:
        data_yaml = prepare_data()
    except Exception as e:
        print(f"Error during data preparation: {e}")
        return

    # Step 2: Train Model
    print("\n===== 2. Training YOLOv8 =====")
    # You can adjust epochs here. 100 is good for production, 1-5 for quick test.
    try:
        train_yolov8(data_yaml, epochs=100)
    except Exception as e:
        print(f"Error during training: {e}")
        return

    print("\n===== Pipeline Complete =====")
    print("You can now use process.py to run inference on new images.")

if __name__ == "__main__":
    main()
