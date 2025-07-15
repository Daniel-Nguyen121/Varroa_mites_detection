import os
import time
import json
import torch
import yaml
import numpy as np
from pathlib import Path
from ptflops import get_model_complexity_info

# Import from existing codebase
from datasets import create_valid_dataset, create_valid_loader
from models.create_fasterrcnn_model import create_model
from torch_utils.engine_eval import evaluate

# ─── A. CONFIG ────────────────────────────────────────────────────────────────

# Load configuration from YAML file
#CONFIG_PATH = "data_configs/varroa.yaml"
#MODEL_PATH = "outputs/training/fol_1_train_200_ep/best_model.pth"
CONFIG_PATH = "data_configs/varroa_1820.yaml"
MODEL_PATH = "outputs/training/fol_2_train_200_ep/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCORE_THR = 0.5
IOU_THR = 0.5
BATCH_SIZE = 1
NUM_WORKERS = 4
IMG_SIZE = 800

# ─── B. LOAD CONFIGURATION ───────────────────────────────────────────────────

def load_config():
    with open(CONFIG_PATH) as file:
        data_configs = yaml.safe_load(file)
    return data_configs

# ─── C. MODEL LOADING ────────────────────────────────────────────────────────

def load_model(data_configs):
    """Load model using the same approach as train.py"""
    NUM_CLASSES = data_configs['NC']
    CLASSES = data_configs['CLASSES']
    
    print('Loading pretrained weights...')
    
    # Load the pretrained checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE) 
    ckpt_state_dict = checkpoint['model_state_dict']
    
    # Get the number of classes from the loaded checkpoint
    old_classes = ckpt_state_dict['roi_heads.box_predictor.cls_score.weight'].shape[0]

    # Build the new model with number of classes same as checkpoint
    build_model = create_model['fasterrcnn_resnet50_fpn_v2']
    model = build_model(num_classes=old_classes)
    
    # Load weights
    model.load_state_dict(ckpt_state_dict)

    # IMPORTANT: Don't modify the classifier heads if the number of classes matches
    # The model was already trained with the correct number of classes
    if old_classes != NUM_CLASSES:
        print(f"Warning: Model was trained with {old_classes} classes but current dataset has {NUM_CLASSES} classes")
        # Only modify if there's a mismatch
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
            in_features=in_features, out_features=NUM_CLASSES, bias=True
        )
        model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
            in_features=in_features, out_features=NUM_CLASSES*4, bias=True
        )
    else:
        print(f"Model loaded successfully with {NUM_CLASSES} classes")
    
    model = model.to(DEVICE)
    model.eval()
    
    return model, CLASSES, NUM_CLASSES

# ─── D. DATASET CREATION ────────────────────────────────────────────────────

def create_test_dataset(data_configs, CLASSES):
    """Create test dataset using existing functions"""
    TEST_DIR_IMAGES = data_configs['TEST_DIR_IMAGES']
    TEST_DIR_LABELS = data_configs['TEST_DIR_LABELS']
    
    test_dataset = create_valid_dataset(
        TEST_DIR_IMAGES, TEST_DIR_LABELS, 
        IMG_SIZE, IMG_SIZE, CLASSES
    )
    test_loader = create_valid_loader(test_dataset, BATCH_SIZE, NUM_WORKERS)
    
    return test_dataset, test_loader

# ─── E. EVALUATION ───────────────────────────────────────────────────────────

def evaluate_model(model, test_loader, CLASSES, DEVICE):
    """Evaluate model using existing evaluate function"""
    COLORS = np.random.uniform(0, 1, size=(len(CLASSES), 3))
    
    # Use the existing evaluate function from torch_utils.engine_eval
    coco_evaluator, stats, test_pred_image = evaluate(
        model, 
        test_loader, 
        device=DEVICE,
        save_valid_preds=False,  # Don't save prediction images for evaluation
        out_dir=None,
        classes=CLASSES,
        colors=COLORS,
        detection_threshold=SCORE_THR  # Use the score threshold for filtering
    )
    
    return coco_evaluator, stats

# ─── F. MANUAL METRICS CALCULATION ──────────────────────────────────────────

def calculate_manual_metrics(model, test_loader, DEVICE, SCORE_THR, IOU_THR):
    """Calculate precision, recall, F1 manually, matching both boxes and classes."""
    from torchvision.ops import box_iou
    
    tp = fp = fn = 0
    times = {"pre": 0.0, "inf": 0.0, "post": 0.0}
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = [image.to(DEVICE) for image in images]
            
            # Pre-processing (transform)
            t0 = time.time()
            images_transformed, _ = model.transform(images, None)
            times["pre"] += time.time() - t0
            
            # Inference (backbone + RPN + ROI heads)
            t1 = time.time()
            feats = model.backbone(images_transformed.tensors)
            props, _ = model.rpn(images_transformed, feats, None)
            dets, _ = model.roi_heads(feats, props, images_transformed.image_sizes, None)
            times["inf"] += time.time() - t1
            
            # Post-processing (NMS and filtering)
            t2 = time.time()
            outputs = model.transform.postprocess(dets, images_transformed.image_sizes, images_transformed.image_sizes)
            times["post"] += time.time() - t2
            
            # Calculate TP/FP/FN for each image
            for i, (output, target) in enumerate(zip(outputs, targets)):
                boxes = output["boxes"].cpu()
                scores = output["scores"].cpu()
                labels = output["labels"].cpu()
                gt_boxes = target["boxes"].cpu()
                gt_labels = target["labels"].cpu()
                
                # Debug: Print first few values to understand the data
                if batch_idx == 0 and i == 0:
                    print(f"DEBUG - Sample data:")
                    print(f"  Pred boxes shape: {boxes.shape}, sample: {boxes[:2]}")
                    print(f"  Pred scores shape: {scores.shape}, sample: {scores[:5]}")
                    print(f"  Pred labels shape: {labels.shape}, sample: {labels[:5]}")
                    print(f"  GT boxes shape: {gt_boxes.shape}, sample: {gt_boxes[:2]}")
                    print(f"  GT labels shape: {gt_labels.shape}, sample: {gt_labels[:5]}")
                    print(f"  Score threshold: {SCORE_THR}")
                    print(f"  Image sizes from transform: {images_transformed.image_sizes}")
                    print(f"  Original image shape: {images[0].shape}")
                
                # Filter by score threshold
                mask = scores >= SCORE_THR
                pred_boxes = boxes[mask]
                pred_labels = labels[mask]
                
                if pred_boxes.size(0) > 0 and gt_boxes.size(0) > 0:
                    matched_gt = set()
                    matched_pr = set()
                    
                    # Debug: Print matching details for first image
                    if batch_idx == 0 and i == 0:
                        print(f"DEBUG - Matching process:")
                        print(f"  Pred boxes after threshold: {pred_boxes.shape}")
                        print(f"  GT boxes: {gt_boxes.shape}")
                        print(f"  Pred boxes range: [{pred_boxes.min():.1f}, {pred_boxes.max():.1f}]")
                        print(f"  GT boxes range: [{gt_boxes.min():.1f}, {gt_boxes.max():.1f}]")
                    
                    for pred_idx, (p_box, p_label) in enumerate(zip(pred_boxes, pred_labels)):
                        ious = box_iou(p_box.unsqueeze(0), gt_boxes)  # shape: (1, N_gt)
                        iou_max, gt_idx = ious.max(dim=1)
                        gt_idx = gt_idx.item()
                        
                        # Debug: Print first few matches
                        if batch_idx == 0 and i == 0 and pred_idx < 3:
                            print(f"  Pred {pred_idx}: box={p_box}, label={p_label}, max_iou={iou_max.item():.3f}, gt_idx={gt_idx}")
                            if gt_boxes.size(0) > gt_idx:
                                print(f"    GT {gt_idx}: box={gt_boxes[gt_idx]}, label={gt_labels[gt_idx]}")
                        
                        if iou_max.item() >= IOU_THR and p_label == gt_labels[gt_idx] and gt_idx not in matched_gt:
                            tp += 1
                            matched_gt.add(gt_idx)
                            matched_pr.add(pred_idx)
                    
                    fp += pred_boxes.size(0) - len(matched_pr)
                    fn += gt_boxes.size(0) - len(matched_gt)
                else:
                    fp += pred_boxes.size(0)
                    fn += gt_boxes.size(0)
    
    # Debug prints
    print(f"DEBUG - TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"DEBUG - Total predictions: {tp + fp}, Total ground truth: {tp + fn}")
    
    return tp, fp, fn, times

# ─── G. MODEL COMPLEXITY ────────────────────────────────────────────────────

def calculate_model_complexity(model):
    """Calculate model layers, parameters, and GFLOPs"""
    # Count leaf modules as "layers"
    num_layers = sum(1 for m in model.modules() if len(list(m.children())) == 0)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate GFLOPs (for input size 3×IMG_SIZE×IMG_SIZE)
    macs, _ = get_model_complexity_info(
        model, (3, IMG_SIZE, IMG_SIZE), as_strings=False, print_per_layer_stat=False
    )
    gflops = macs / 1e9
    
    return num_layers, trainable_params, gflops

# ─── H. MAIN EVALUATION ─────────────────────────────────────────────────────

def main():
    print("Starting evaluation...")
    
    # Load configuration
    data_configs = load_config()
    
    # Load model
    model, CLASSES, NUM_CLASSES = load_model(data_configs)
    
    # Create test dataset
    test_dataset, test_loader = create_test_dataset(data_configs, CLASSES)
    print(f"Number of test samples: {len(test_dataset)}")
    
    # Evaluate using existing function
    coco_evaluator, stats = evaluate_model(model, test_loader, CLASSES, DEVICE)
    
    # Calculate manual metrics (simplified to avoid hanging)
    print("Calculating manual metrics...")
    tp, fp, fn, times = calculate_manual_metrics(model, test_loader, DEVICE, SCORE_THR, IOU_THR)
    
    # Calculate model complexity
    num_layers, trainable_params, gflops = calculate_model_complexity(model)
    
    # ─── I. COMPUTE FINAL METRICS ───────────────────────────────────────────
    
    # COCO AP metrics
    AP50_95 = stats[0]  # AP @ IoU=0.50:0.95
    AP50 = stats[1]     # AP @ IoU=0.50
    
    # Manual precision/recall/F1
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    # Timing metrics
    n = len(test_dataset)
    pre_ms = times["pre"] * 1e3 / n
    inf_ms = times["inf"] * 1e3 / n
    post_ms = times["post"] * 1e3 / n
    tot_ms = (times["pre"] + times["inf"] + times["post"]) * 1e3 / n
    fps = n / (times["pre"] + times["inf"] + times["post"])
    
    # ─── J. PRINT RESULTS ───────────────────────────────────────────────────
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AP@50: {AP50:.4f}")
    print(f"AP@[50:95]: {AP50_95:.4f}")
    print("-"*60)
    print(f"Preprocessing time: {pre_ms:.1f} ms")
    print(f"Inference time: {inf_ms:.1f} ms")
    print(f"Post-processing time: {post_ms:.1f} ms")
    print(f"Total time: {tot_ms:.1f} ms")
    print(f"FPS: {fps:.1f}")
    print("-"*60)
    print(f"Number of layers: {num_layers}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"GFLOPs: {gflops:.1f}")
    print("="*60)

if __name__ == "__main__":
    main()
