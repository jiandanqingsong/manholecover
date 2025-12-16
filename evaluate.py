import os
import yaml
import glob
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
from tqdm import tqdm

# ================= Helper Functions =================

def xywhn2xyxy(x, w, h):
    """Convert normalized xywh to pixel xyxy"""
    # x: [class, x_c, y_c, w, h]
    # w, h: image width, height
    labels = []
    for box in x:
        cls = int(box[0])
        xc, yc, bw, bh = box[1], box[2], box[3], box[4]
        
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h
        labels.append([cls, x1, y1, x2, y2])
    return np.array(labels)

def compute_iou(box1, box2):
    """
    Calculate IoU between two boxes [x1, y1, x2, y2]
    """
    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / (union_area + 1e-6)

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves. """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Area under curve
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

# ================= Main Evaluation Logic =================

def evaluate_custom(model_path="runs/train/manhole_yolov8/weights/best.pt", 
                   data_yaml="data_processed/data.yaml",
                   iou_thres=0.5,
                   conf_thres_stats=0.25):
    
    print(f"Loading model: {model_path}")
    if not os.path.exists(model_path):
        print("Model not found. Using yolov8s.pt for demo.")
        model_path = "yolov8s.pt"
    
    model = YOLO(model_path)
    
    # 1. Parse Data Config
    with open(data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)
        
    val_dir = os.path.join(data_cfg['path'], data_cfg['val'])
    class_names = data_cfg['names']
    
    # Get all validation images
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    img_files = []
    for ext in img_exts:
        img_files.extend(glob.glob(os.path.join(val_dir, f"*{ext}")))
    
    print(f"Found {len(img_files)} validation images.")
    
    # Store predictions and ground truths
    # Structure: stats[class_id] = {'tp': [], 'conf': [], 'num_gt': 0}
    stats = {i: {'tp': [], 'conf': [], 'num_gt': 0} for i in range(len(class_names))}
    
    # 2. Inference Loop
    print("Running inference and matching...")
    for img_path in tqdm(img_files):
        # Load Image
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # Load Label
        label_path = img_path.replace("images", "labels").rsplit('.', 1)[0] + ".txt"
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                l = [list(map(float, x.split())) for x in f.read().strip().splitlines() if x.strip()]
                if l:
                    gt_boxes = xywhn2xyxy(l, w, h)
        
        # Update GT counts
        for gb in gt_boxes:
            cls_id = int(gb[0])
            if cls_id in stats:
                stats[cls_id]['num_gt'] += 1
        
        # Predict
        results = model(img, verbose=False, conf=0.001)[0] # Low conf to get PR curve
        preds = results.boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, cls]
        
        # Match Predictions to GT
        # For each class
        for cls_id in stats.keys():
            cls_preds = preds[preds[:, 5] == cls_id]
            cls_gts = gt_boxes[gt_boxes[:, 0] == cls_id] if len(gt_boxes) > 0 else np.empty((0, 5))
            
            # Sort predictions by confidence descending
            cls_preds = cls_preds[np.argsort(-cls_preds[:, 4])]
            
            detected_gt = [False] * len(cls_gts)
            
            for pred in cls_preds:
                pred_box = pred[:4]
                conf = pred[4]
                
                # Find best matching GT
                best_iou = 0
                best_gt_idx = -1
                
                for i, gt in enumerate(cls_gts):
                    if detected_gt[i]: continue # Already matched
                    iou = compute_iou(pred_box, gt[1:])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                if best_iou > iou_thres:
                    stats[cls_id]['tp'].append(1)
                    detected_gt[best_gt_idx] = True
                else:
                    stats[cls_id]['tp'].append(0) # FP
                
                stats[cls_id]['conf'].append(conf)

    # 3. Calculate Metrics & Plot
    print("\n" + "="*60)
    print(f"{'Class':<15} | {'Images':<8} | {'Targets':<8} | {'P':<8} | {'R':<8} | {'mAP@50':<8}")
    print("-" * 60)
    
    mAPs = []
    
    # Setup Plot
    plt.figure(figsize=(10, 8))
    plt.title(f'Precision-Recall Curve (IoU={iou_thres})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    
    for cls_id, cls_name in class_names.items():
        data = stats[cls_id]
        num_gt = data['num_gt']
        
        if num_gt == 0:
            continue
            
        tp_list = np.array(data['tp'])
        conf_list = np.array(data['conf'])
        
        if len(tp_list) == 0:
            print(f"{cls_name:<15} | {len(img_files):<8} | {num_gt:<8} | {0.0:<8} | {0.0:<8} | {0.0:<8}")
            continue
            
        # Cumulative TP and FP
        cum_tp = np.cumsum(tp_list)
        cum_fp = np.cumsum(1 - tp_list)
        
        # Precision and Recall curve
        recalls = cum_tp / (num_gt + 1e-16)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-16)
        
        # Average Precision
        ap = compute_ap(recalls, precisions)
        mAPs.append(ap)
        
        # Plot Curve
        plt.plot(recalls, precisions, label=f'{cls_name} (AP={ap:.2f})', linewidth=2)
        
        # --- Statistics at specific confidence threshold ---
        # Find index closest to conf_thres_stats
        # Since conf_list is descending, we find the last index where conf >= thres
        valid_indices = np.where(conf_list >= conf_thres_stats)[0]
        
        if len(valid_indices) > 0:
            idx = valid_indices[-1]
            tp_val = cum_tp[idx]
            fp_val = cum_fp[idx]
            fn_val = num_gt - tp_val
            
            p_at_thres = precisions[idx]
            r_at_thres = recalls[idx]
        else:
            tp_val = 0
            fp_val = 0
            fn_val = num_gt
            p_at_thres = 0.0
            r_at_thres = 0.0
            
        print(f"{cls_name:<15} | {len(img_files):<8} | {num_gt:<8} | {p_at_thres:.3f}    | {r_at_thres:.3f}    | {ap:.3f}")
        print(f"   > Stats @ Conf={conf_thres_stats}: TP={int(tp_val)}, FP={int(fp_val)}, FN={int(fn_val)}")

    print("-" * 60)
    print(f"mAP@50 (Mean): {np.mean(mAPs):.4f}")
    
    plt.legend()
    save_path = 'PR_Curve_Custom.png'
    plt.savefig(save_path)
    print(f"\nP-R Curve saved to {save_path}")
    # plt.show() # Uncomment if running in GUI environment

if __name__ == "__main__":
    evaluate_custom()
