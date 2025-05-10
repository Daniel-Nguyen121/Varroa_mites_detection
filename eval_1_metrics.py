import numpy as np
import os
from eval_1_utils import debug_print

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def calculate_ap(recall, precision):
    """Calculate AP using 11-point interpolation"""
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.
    return ap

def calculate_ap_range(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_thresholds):
    """Calculate AP over multiple IoU thresholds"""
    aps = []
    for iou_threshold in iou_thresholds:
        # Recalculate metrics with current IoU threshold
        metrics = calculate_metrics(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold)
        ap = metrics['ap']
        aps.append(ap)
    return aps

def calculate_metrics(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.25):
    """Calculate precision, recall, F1-score, and AP"""
    metrics = {}
    
    # Sort predictions by confidence
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[sorted_indices]
    pred_labels = pred_labels[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    
    # Initialize arrays for precision-recall curve
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    
    # Match predictions to ground truth
    gt_matched = np.zeros(len(gt_boxes))
    
    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(gt_boxes):
            if gt_matched[j] == 0:  # If ground truth box not matched yet
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp[i] = 1
            gt_matched[best_gt_idx] = 1
        else:
            fp[i] = 1
    
    # Calculate precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len(gt_boxes) if len(gt_boxes) > 0 else np.zeros_like(tp_cumsum)
    
    # Calculate F1-score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-16)
    
    # Calculate AP
    ap = calculate_ap(recall, precision)
    
    metrics['precision'] = precision[-1] if len(precision) > 0 else 0
    metrics['recall'] = recall[-1] if len(recall) > 0 else 0
    metrics['f1_score'] = f1_score[-1] if len(f1_score) > 0 else 0
    metrics['ap'] = ap
    
    return metrics

def save_metrics_to_file(metrics, out_dir):
    """Save metrics to a text file"""
    with open(os.path.join(out_dir, 'detailed_metrics.txt'), 'w') as f:
        f.write("Detailed Evaluation Metrics\n")
        f.write("=========================\n\n")
        
        # Overall metrics
        f.write("Overall Metrics:\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-score: {metrics['f1_score']:.4f}\n")
        f.write(f"AP@50: {metrics['ap_50']:.4f}\n")
        f.write(f"AP@50-95: {metrics['ap_50_95']:.4f}\n")
        f.write(f"FPS: {metrics['fps']:.2f}\n\n")
        
        # Timing metrics
        f.write("Timing Metrics (seconds):\n")
        f.write(f"Average Preprocessing Time: {metrics['avg_preprocess_time']:.4f}\n")
        f.write(f"Average Inference Time: {metrics['avg_inference_time']:.4f}\n")
        f.write(f"Average Postprocessing Time: {metrics['avg_postprocess_time']:.4f}\n")
        f.write(f"Average Total Time: {metrics['avg_total_time']:.4f}\n") 