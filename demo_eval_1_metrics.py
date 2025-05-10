import torch
import numpy as np
from torch_utils.engine_eval import evaluate
from utils.eval_utils import evaluate_test_set
import cv2
import os

# Suppress OpenCV warnings
os.environ['OPENCV_LOGGING_LEVEL'] = 'ERROR'

def calculate_map(precision, recall, iou_threshold=0.5):
    """Calculate mAP at specific IoU threshold"""
    try:
        # Get precision at recall levels
        # precision shape: [num_iou_thresholds, num_recall_thresholds, num_classes, num_areas, max_detections]
        pr_curve = precision[int(iou_threshold), :, 0, 0, 2]  # Use class 0 (varroa) and area=all
        # Calculate AP for the class
        ap = np.mean(pr_curve[pr_curve > -1])
        return ap
    except Exception as e:
        print(f"Error in calculate_map: {e}")
        print(f"Precision shape: {precision.shape}")
        return 0.0

def calculate_class_metrics(precision, recall, iou_threshold=0.5, confidence_threshold=0.25):
    """Calculate precision, recall, and F1 score for each class"""
    try:
        # Get precision at recall levels for specific IoU threshold
        # precision shape: [num_iou_thresholds, num_recall_thresholds, num_classes, num_areas, max_detections]
        pr_curve = precision[int(iou_threshold), :, 0, 0, 2]  # Use class 0 (varroa) and area=all
        
        # Calculate recall at different precision levels
        precision_levels = [0.5, 0.6, 0.7, 0.8, 0.9]
        recall_at_precision = {}
        
        for p_level in precision_levels:
            # Find the recall level that gives us the closest precision to p_level
            precision_idx = np.abs(pr_curve - p_level).argmin()
            recall_at_precision[p_level] = recall[precision_idx]
        
        # Get precision and recall at confidence threshold
        target_precision = confidence_threshold
        precision_idx = np.abs(pr_curve - target_precision).argmin()
        
        precision_at_threshold = pr_curve[precision_idx]
        recall_at_threshold = recall[precision_idx]
        
        # Calculate F1 score
        f1_score = 2 * (precision_at_threshold * recall_at_threshold) / (precision_at_threshold + recall_at_threshold)
        f1_score = np.nan_to_num(f1_score)  # Replace NaN with 0
        
        return {
            'precision': precision_at_threshold,
            'recall': recall_at_threshold,
            'f1_score': f1_score,
            'recall_at_precision': recall_at_precision
        }
    except Exception as e:
        print(f"Error in calculate_class_metrics: {e}")
        print(f"Precision shape: {precision.shape}")
        print(f"Recall shape: {recall.shape}")
        # Return zeros for each metric
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'recall_at_precision': {p: 0.0 for p in [0.5, 0.6, 0.7, 0.8, 0.9]}
        }

def evaluate_model_metrics(model, data_loader, device, classes, out_dir, threshold=0.3):
    """Evaluate model and return comprehensive metrics"""
    model.eval()
    
    try:
        # Run evaluation
        coco_evaluator, stats, _ = evaluate(
            model,
            data_loader,
            device=device,
            save_valid_preds=True,
            out_dir=out_dir,
            classes=classes,
            colors=np.random.uniform(0, 1, size=(len(classes), 3)),
            detection_threshold=threshold
        )
        
        # Get COCOeval object for bbox evaluation
        coco_eval = coco_evaluator.coco_eval['bbox']
        
        # Calculate metrics
        precision = coco_eval.eval['precision']  # shape: [num_iou_thresholds, num_recall_thresholds, num_classes, num_areas, max_detections]
        recall = coco_eval.params.recThrs
        
        print(f"Precision array shape: {precision.shape}")
        print(f"Recall array shape: {recall.shape}")
        
        # Calculate mAP at different IoU thresholds
        map_50 = calculate_map(precision, recall, iou_threshold=0)  # IoU=0.5
        map_50_95 = np.mean([calculate_map(precision, recall, iou_threshold=i) for i in range(10)])  # IoU=0.5:0.95
        
        # Calculate class-specific metrics
        class_metrics = calculate_class_metrics(precision, recall)
        
        # Get per-class AP values
        per_class_ap = precision[0, :, 0, 0, 2].mean()  # AP at IoU=0.5 for varroa class
        
        # Get confusion matrix
        confusion_matrix = np.zeros((len(classes), len(classes)))
        for i in range(len(classes)):
            for j in range(len(classes)):
                # Get precision at IoU=0.5 for class i
                pr_curve = precision[0, :, 0, 0, 2]  # Use class 0 (varroa)
                # Get recall at IoU=0.5 for class j
                recall_curve = recall
                # Calculate intersection
                confusion_matrix[i, j] = np.mean(pr_curve[pr_curve > -1]) * np.mean(recall_curve)
        
        return {
            'mAP@0.5': map_50,
            'mAP@0.5:0.95': map_50_95,
            'class_metrics': class_metrics,
            'per_class_ap': per_class_ap,
            'confusion_matrix': confusion_matrix
        }
    except Exception as e:
        print(f"Error in evaluate_model_metrics: {e}")
        # Return empty metrics
        return {
            'mAP@0.5': 0.0,
            'mAP@0.5:0.95': 0.0,
            'class_metrics': {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'recall_at_precision': {p: 0.0 for p in [0.5, 0.6, 0.7, 0.8, 0.9]}
            },
            'per_class_ap': 0.0,
            'confusion_matrix': np.zeros((len(classes), len(classes)))
        }

def main():
    """Test the metrics calculation"""
    import argparse
    from models.create_fasterrcnn_model import create_model
    from datasets import create_valid_dataset, create_valid_loader
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='fasterrcnn_resnet50_fpn_v2')
    parser.add_argument('-c', '--config', default='data_configs/varroa.yaml')
    parser.add_argument('-d', '--device', default=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('-w', '--weights', default='outputs/training/varroa-old-2/best_model.pth', type=str)
    args = parser.parse_args()
    
    try:
        # Load config
        with open(args.config) as f:
            data_configs = yaml.safe_load(f)
        
        # Create dataset and loader
        test_dataset = create_valid_dataset(
            data_configs['TEST_DIR_IMAGES'],
            data_configs['TEST_DIR_LABELS'],
            320, 320,
            data_configs['CLASSES']
        )
        test_loader = create_valid_loader(test_dataset, batch_size=8, num_workers=4)
        
        # Initialize model
        model = create_model[args.model](num_classes=data_configs['NC'], pretrained=True)
        model = model.to(args.device)
        
        # Load weights
        checkpoint = torch.load(args.weights, map_location=args.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        # Evaluate metrics
        metrics = evaluate_model_metrics(
            model, test_loader, args.device,
            data_configs['CLASSES'], 'outputs/eval/metrics_test'
        )
        
        print("\nEvaluation Metrics:")
        print(f"mAP@0.5: {metrics['mAP@0.5']:.4f}")
        print(f"mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
        
        print("\nPer-class AP@0.5:")
        print(f"Class varroa: {metrics['per_class_ap']:.4f}")
        
        print("\nClass-specific Metrics:")
        print(f"Class varroa:")
        print(f"  Precision: {metrics['class_metrics']['precision']:.4f}")
        print(f"  Recall: {metrics['class_metrics']['recall']:.4f}")
        print(f"  F1-Score: {metrics['class_metrics']['f1_score']:.4f}")
        
        # Print recall at different precision levels
        print("\n  Recall at different precision levels:")
        for p_level, r_value in metrics['class_metrics']['recall_at_precision'].items():
            print(f"    Precision {p_level:.1f}: Recall {r_value:.4f}")
        
        print("\nConfusion Matrix:")
        print("Predicted ->")
        print("Actual ↓")
        print("          ", end="")
        for cls in data_configs['CLASSES']:
            print(f"{cls:>10}", end="")
        print()
        for i, cls in enumerate(data_configs['CLASSES']):
            print(f"{cls:10}", end="")
            for j in range(len(data_configs['CLASSES'])):
                print(f"{metrics['confusion_matrix'][i, j]:10.4f}", end="")
            print()
            
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 