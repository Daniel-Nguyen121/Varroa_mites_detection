"""
Evaluation utilities for validation during training and final test set evaluation.
"""

import torch
import numpy as np
from torch_utils.engine_eval import evaluate
import os

def evaluate_during_training(model, data_loader, device, classes, out_dir):
    """Run evaluation on validation set during training"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        coco_evaluator, stats, _ = evaluate(
            model, 
            data_loader,
            device=device,
            save_valid_preds=True,
            out_dir=out_dir,
            classes=classes,
            colors=np.random.uniform(0, 1, size=(len(classes), 3))
        )
        
        # Calculate validation loss
        for batch_idx, (images, targets) in enumerate(data_loader):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            
            # Debug information
            print(f"\nBatch {batch_idx + 1} Loss Information:")
            print(f"Type of loss_dict: {type(loss_dict)}")
            
            if isinstance(loss_dict, dict):
                print("Loss dictionary keys:", loss_dict.keys())
                print("Loss dictionary values:", {k: v.item() for k, v in loss_dict.items()})
                losses = sum(loss for loss in loss_dict.values())
            else:  # If it's a list
                print("Loss list length:", len(loss_dict))
                print("Loss list values:", [loss.item() for loss in loss_dict])
                losses = sum(loss_dict)
            
            print(f"Total loss for batch: {losses.item():.4f}")
            total_loss += losses.item()
            num_batches += 1
            
    model.train()
    
    # Calculate average validation loss
    val_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"\nFinal Validation Loss: {val_loss:.4f}")
    
    # Return mAP@0.5:0.95, mAP@0.5, and validation loss
    return stats[0], stats[1], val_loss

def evaluate_test_set(model, data_loader, device, classes, out_dir, threshold=0.3):
    """Comprehensive evaluation on test set after training"""
    model.eval()
    
    # Initialize metrics
    test_stats = {
        'total_time': 0,
        'inference_times': [],
        'predictions': [],
        'ground_truths': []
    }
    
    with torch.no_grad():
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
    
    # Extract metrics
    mAP_50_95 = stats[0]  # mAP@0.5:0.95
    mAP_50 = stats[1]     # mAP@0.5
    precision = stats[2]   # Precision
    recall = stats[3]      # Recall
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Save metrics
    metrics = {
        'mAP@0.5:0.95': mAP_50_95,
        'mAP@0.5': mAP_50,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score
    }
    
    # Save metrics to file
    save_test_metrics(metrics, out_dir)
    
    return metrics

def save_test_metrics(metrics, out_dir):
    """Save test set metrics to a file"""
    metrics_file = os.path.join(out_dir, 'test_metrics.txt')
    
    with open(metrics_file, 'w') as f:
        f.write('Test Set Evaluation Metrics:\n')
        f.write('-' * 30 + '\n')
        for metric_name, value in metrics.items():
            f.write(f'{metric_name}: {value:.4f}\n')

def calculate_model_stats(model, input_size=(320, 320)):
    """Calculate and return model statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Create dummy input for FLOPS calculation
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    
    # Calculate approximate FLOPS
    def count_conv2d(m, x, y):
        x = x[0]
        cin = m.in_channels
        cout = m.out_channels
        kh, kw = m.kernel_size
        batch_size = x.size()[0]
        out_h = y.size(2)
        out_w = y.size(3)
        
        # ops = batch_size * out_h * out_w * cin * cout * kh * kw
        kernel_ops = kh * kw * cin
        bias_ops = 1 if m.bias is not None else 0
        ops_per_element = kernel_ops + bias_ops
        
        # total ops
        total_ops = batch_size * ops_per_element * out_h * out_w * cout
        
        return total_ops
    
    total_ops = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            total_ops += count_conv2d(m, (dummy_input,), 
                                    m(dummy_input))
    
    stats = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'approximate_gflops': total_ops / 1e9
    }
    
    return stats 