from ultralytics import YOLO
import cv2
import os
import sys

class ManholeDetector:
    def __init__(self, model_path="runs/train/manhole_yolov8/weights/best.pt"):
        # Check if trained model exists
        if not os.path.exists(model_path):
            print(f"Warning: Trained model not found at {model_path}.")
            # Fallback to yolov8s.pt (Small) as per performance optimization
            model_path = "yolov8s.pt"
            print(f"Using base model: {model_path}")
            
        self.model = YOLO(model_path)
        
    def predict(self, img_path, conf_thres=0.25, save_path="inference_result.jpg"):
        if not os.path.exists(img_path):
            print(f"Error: Image {img_path} not found.")
            return None
            
        print(f"Processing {img_path}...")
        results = self.model(img_path, conf=conf_thres)[0]
        
        # Visualize
        res_plotted = results.plot()
        cv2.imwrite(save_path, res_plotted)
        
        # Print info
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            conf = float(box.conf[0])
            print(f"  Found {cls_name}: {conf:.2f}")
            
        return results

if __name__ == "__main__":
    # Example usage:
    # python process.py                     -> Processes 'test' folder
    # python process.py path/to/folder      -> Processes specified folder
    # python process.py path/to/image.jpg   -> Processes single image
    
    detector = ManholeDetector()
    
    # Determine target (file or folder)
    target = "test" # Default folder
    if len(sys.argv) > 1:
        target = sys.argv[1]
        
    if os.path.isfile(target):
        # Process single image
        print(f"--- Processing Single Image: {target} ---")
        detector.predict(target, save_path="inference_result.jpg")
        print("Result saved to inference_result.jpg")
        
    elif os.path.isdir(target):
        # Process folder
        output_dir = "inference_results"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"--- Processing Folder: {target} ---")
        print(f"Results will be saved to: {output_dir}/")
        
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        count = 0
        
        for filename in os.listdir(target):
            if filename.lower().endswith(valid_exts):
                img_path = os.path.join(target, filename)
                save_path = os.path.join(output_dir, f"res_{filename}")
                
                detector.predict(img_path, save_path=save_path)
                count += 1
                
        if count == 0:
            print(f"No images found in {target}")
        else:
            print(f"Done. Processed {count} images.")
            
    else:
        print(f"Error: Target '{target}' not found.")
