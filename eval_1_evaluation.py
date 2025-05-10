import torch
import os
import torch.amp as amp
from eval_1_utils import measure_timing, debug_print
from eval_1_metrics import calculate_metrics

@measure_timing
def preprocess_batch(batch, device):
    """Preprocess a batch of images"""
    images = batch['images'].to(device, non_blocking=True)
    return images

@measure_timing
def inference_batch(model, images, scaler):
    """Run model inference on a batch"""
    with amp.autocast(device_type='cuda'):
        outputs = model(images)
    return outputs

@measure_timing
def postprocess_batch(outputs, conf_thres, iou_thres):
    """Postprocess model outputs"""
    processed_outputs = []
    for output in outputs:
        pred_boxes = output['boxes'].cpu().numpy()
        pred_scores = output['scores'].cpu().numpy()
        pred_labels = output['labels'].cpu().numpy()
        
        mask = pred_scores >= conf_thres
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]
        pred_labels = pred_labels[mask]
        
        processed_outputs.append({
            'boxes': pred_boxes,
            'scores': pred_scores,
            'labels': pred_labels
        })
    return processed_outputs

@torch.no_grad()
def evaluate_batch(model, batch, device, conf_thres, iou_thres, scaler):
    """Evaluate a batch of images and return metrics"""
    # Measure preprocessing time
    images, preprocess_time = preprocess_batch(batch, device)
    gt_boxes = batch['gt_boxes']
    gt_labels = batch['gt_labels']
    img_paths = batch['img_paths']
    
    debug_print(f"Processing batch of {len(images)} images")
    
    # Measure inference time
    outputs, inference_time = inference_batch(model, images, scaler)
    
    # Measure postprocessing time
    processed_outputs, postprocess_time = postprocess_batch(outputs, conf_thres, iou_thres)
    
    batch_metrics = []
    for i, output in enumerate(processed_outputs):
        pred_boxes = output['boxes']
        pred_scores = output['scores']
        pred_labels = output['labels']
        
        gt_boxes_i = gt_boxes[i].numpy()
        gt_labels_i = gt_labels[i].numpy()
    
        metrics = calculate_metrics(pred_boxes, pred_scores, pred_labels, gt_boxes_i, gt_labels_i, iou_thres)
        metrics['image'] = os.path.basename(img_paths[i])
        metrics['preprocess_time'] = preprocess_time
        metrics['inference_time'] = inference_time
        metrics['postprocess_time'] = postprocess_time
        metrics['total_time'] = preprocess_time + inference_time + postprocess_time
        
        # Store predictions and ground truth for AP calculation
        metrics['pred_boxes'] = pred_boxes
        metrics['pred_scores'] = pred_scores
        metrics['pred_labels'] = pred_labels
        metrics['gt_boxes'] = gt_boxes_i
        metrics['gt_labels'] = gt_labels_i
        
        batch_metrics.append(metrics)
        
        debug_print(f"Processed image {i+1}/{len(processed_outputs)}")
        debug_print(f"Found {len(pred_boxes)} predictions")
        debug_print(f"Ground truth boxes: {len(gt_boxes_i)}")
    
    return batch_metrics 