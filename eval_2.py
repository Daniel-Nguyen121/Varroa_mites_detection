import torch
import argparse
import yaml
import os
import time
import numpy as np
from datasets import create_train_dataset, create_valid_dataset, create_train_loader, create_valid_loader
from models.create_fasterrcnn_model import create_model
from utils.metrics import MetricsTracker
from utils.eval_utils import evaluate_during_training, evaluate_test_set
from torch_utils.engine import train_one_epoch
from tqdm import tqdm
import thop
import torch.nn as nn
import cv2
from torchvision import transforms
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.cuda import Stream
import torch.amp as amp
import xml.etree.ElementTree as ET
import resource

def increase_file_limit():
    """Increase the system's file descriptor limit"""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f"Successfully increased file descriptor limit to {hard}")
    except ValueError as e:
        print(f"Warning: Could not increase file descriptor limit: {e}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='fasterrcnn_resnet50_fpn_v2')
    parser.add_argument('-c', '--config', default='data_configs/varroa.yaml')
    parser.add_argument('-d', '--device', default=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('-ims', '--img-size', default=320, type=int)
    parser.add_argument('-pn', '--project-name', default='eval_varroa_v1', type=str)
    parser.add_argument('-w', '--weights', default="/media/data4/home/vuhai/MOT_train/Varroa_detection/Detection_algos/fasterrcnn_resnet50_fpn_v2_varroa/outputs/training/varroa-old-2/best_model.pth", type=str)
    parser.add_argument('--conf-thres', type=float, default=0.65, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.25, help='NMS IoU threshold')
    return vars(parser.parse_args())

def load_coco_annotation(xml_path, img_width, img_height):
    """Load COCO format annotation from XML file"""
    boxes = []
    labels = []
    
    if not os.path.exists(xml_path):
        print(f"Warning: XML file not found: {xml_path}")
        return boxes, labels
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image size from XML
        size = root.find('size')
        if size is None:
            print(f"Warning: No size tag found in {xml_path}")
            return boxes, labels
            
        xml_width = int(size.find('width').text)
        xml_height = int(size.find('height').text)
    
        # Debug print
        print(f"\nProcessing XML: {xml_path}")
        print(f"Image size from XML: {xml_width}x{xml_height}")
        print(f"Target size: {img_width}x{img_height}")
        
        for obj in root.findall('object'):
            # Get class label
            class_name = obj.find('name').text
            # Convert class name to index (assuming class names are in order in your config)
            class_id = 1  # Default to 0 for varroa
            
            # Get bounding box
            bbox = obj.find('bndbox')
            if bbox is None:
                print(f"Warning: No bndbox found for object in {xml_path}")
                continue
                
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Debug print
            print(f"Found object: {class_name} at [{xmin}, {ymin}, {xmax}, {ymax}]")
            
            # Scale coordinates if image size is different
            if xml_width != img_width or xml_height != img_height:
                xmin = int(xmin * img_width / xml_width)
                ymin = int(ymin * img_height / xml_height)
                xmax = int(xmax * img_width / xml_width)
                ymax = int(ymax * img_height / xml_height)
                print(f"Scaled to: [{xmin}, {ymin}, {xmax}, {ymax}]")
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_id)
        
        print(f"Total objects found: {len(boxes)}")
        return boxes, labels
        
    except Exception as e:
        print(f"Error processing {xml_path}: {str(e)}")
        return boxes, labels

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

def calculate_ap_range(recall, precision, iou_thresholds):
    """Calculate AP over multiple IoU thresholds"""
    aps = []
    for iou_threshold in iou_thresholds:
        ap = calculate_ap(recall, precision)
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
    # print(f"Check gt_matched: {len(gt_matched)}")
    # print(f"\t - Check: {gt_matched}")
    for i, pred_box in enumerate(pred_boxes):
        print(f"i = {i}")

        best_iou = 0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(gt_boxes):
            # print(f"\t - j = {j}")
            # print(f"\t\t - gt_matched[j] = {gt_matched[j]}")
            if gt_matched[j] == 0:  # If ground truth box not matched yet
                iou = calculate_iou(pred_box, gt_box)
                # print(f"\t\t - iou = {iou}")
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
        # print(f"\t\t - best_iou = {best_iou}")
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp[i] = 1
            gt_matched[best_gt_idx] = 1
        else:
            fp[i] = 1
    
    # Calculate precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    #print(f"Check tp_cumsum: {len(tp_cumsum)}")
    #print(f"\t - Check: {tp_cumsum}")
    #print(f"Check fp_cumsum: {len(fp_cumsum)}")
    #print(f"\t - Check: {fp_cumsum}")
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len(gt_boxes) if len(gt_boxes) > 0 else np.zeros_like(tp_cumsum)
    
    # Calculate F1-score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-16)
    
    # Calculate AP
    ap = calculate_ap(recall, precision)

    # print(f"Check precision: {len(precision)}")
    # print(f"\t - Check: {precision}")
    
    metrics['precision'] = precision[-1] if len(precision) > 0 else 0
    metrics['recall'] = recall[-1] if len(recall) > 0 else 0
    metrics['f1_score'] = f1_score[-1] if len(f1_score) > 0 else 0
    metrics['ap'] = ap
    
    return metrics

class FastImageDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        # Pre-load all images and labels
        self.images = []
        self.gt_boxes = []
        self.gt_labels = []
        self.img_sizes = []
        
        print("Pre-loading dataset...")
        for img_path, label_path in tqdm(zip(image_paths, label_paths), total=len(image_paths)):
            # Load and preprocess image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = image.shape
            
            # Load ground truth from COCO XML
            xml_path = label_path.replace('.txt', '.xml')
            boxes, labels = load_coco_annotation(xml_path, img_w, img_h)
            
            if self.transform:
                image = self.transform(image)
            
            self.images.append(image)
            self.gt_boxes.append(torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32))
            self.gt_labels.append(torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long))
            self.img_sizes.append((img_h, img_w))
            
            # Debug print
            print(f"\nProcessed {img_path}")
            print(f"Image size: {img_w}x{img_h}")
            print(f"Number of boxes: {len(boxes)}")
            if len(boxes) > 0:
                print(f"First box: {boxes[0]}")
                print(f"First label: {labels[0]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'gt_boxes': self.gt_boxes[idx],
            'gt_labels': self.gt_labels[idx],
            'img_path': self.image_paths[idx],
            'img_size': self.img_sizes[idx]
        }

def collate_fn(batch):
    images = torch.stack([b['image'] for b in batch])
    return {
        'images': images,
        'gt_boxes': [b['gt_boxes'] for b in batch],
        'gt_labels': [b['gt_labels'] for b in batch],
        'img_paths': [b['img_path'] for b in batch],
        'img_sizes': [b['img_size'] for b in batch]
    }

def measure_timing(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    return wrapper

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
        batch_metrics.append(metrics)
    
    return batch_metrics

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

def main(args):
    # Increase file descriptor limit
    increase_file_limit()
    
    # Load config
    with open(args['config']) as f:
        data_configs = yaml.safe_load(f)
    
    # Create output directory
    out_dir = os.path.join('outputs', 'eval_1', args['project_name'])
    os.makedirs(out_dir, exist_ok=True)

    # Initialize model
    model_name = args['model']
    if model_name not in create_model:
        raise ValueError(f"Model {model_name} not found. Available models: {list(create_model.keys())}")
    
    print(f"Creating model: {model_name}")
    print(f"Number of classes: {data_configs['NC']}")
    
    model = create_model[model_name](num_classes=data_configs['NC'], pretrained=True)
    model = model.to(args['device'])

    # Enable automatic mixed precision with updated GradScaler
    scaler = amp.GradScaler('cuda')
    
    # Load weights
    print('Loading best model...')
    checkpoint = torch.load(args['weights'], map_location=args['device'])
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        state_dict = {k: v for k, v in state_dict.items() if 'total_ops' not in k and 'total_params' not in k}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(args['device'])
    model.eval()
    
    # Get test directory
    test_dir = data_configs['TEST_DIR_IMAGES']
    
    # Prepare dataset
    image_paths = []
    label_paths = []
    for filename in os.listdir(test_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(test_dir, filename))
            label_paths.append(os.path.join(test_dir, filename.rsplit('.', 1)[0] + '.xml'))  # Changed to .xml
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    
    # Create dataset with pre-loaded data
    dataset = FastImageDataset(image_paths, label_paths, transform=transform)
    
    # Create dataloader with optimized settings and reduced workers
    dataloader = DataLoader(
        dataset,
        batch_size=args['batch_size'],
        num_workers=min(4, args['workers']),  # Limit workers to prevent file descriptor issues
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Process batches
    print("Processing images...")
    total_time = 0
    total_images = 0
    all_metrics = []
    
    for batch in tqdm(dataloader):
        batch_metrics = evaluate_batch(
            model, batch, args['device'],
            args['conf_thres'], args['iou_thres'],
            scaler
        )
        all_metrics.extend(batch_metrics)
        total_images += len(batch['images'])
    
    # Convert results to DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Calculate average metrics
    avg_metrics = df.mean(numeric_only=True)
    
    # Calculate FPS
    fps = total_images / total_time
    
    # Calculate AP@50 and AP@50-95
    ap_50 = calculate_ap_range(avg_metrics['recall'], avg_metrics['precision'], [0.5])[0]
    ap_50_95 = calculate_ap_range(avg_metrics['recall'], avg_metrics['precision'], 
                                 np.linspace(0.5, 0.95, 10))[0]
    
    # Prepare final metrics
    final_metrics = {
        'precision': avg_metrics['precision'],
        'recall': avg_metrics['recall'],
        'f1_score': avg_metrics['f1_score'],
        'ap_50': ap_50,
        'ap_50_95': ap_50_95,
        'fps': fps,
        'avg_preprocess_time': avg_metrics['preprocess_time'],
        'avg_inference_time': avg_metrics['inference_time'],
        'avg_postprocess_time': avg_metrics['postprocess_time'],
        'avg_total_time': avg_metrics['total_time']
    }
    
    # Save detailed metrics to file
    save_metrics_to_file(final_metrics, out_dir)
    
    # Print results
    print("\nAverage Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
    print(f"\nDetailed results saved to {os.path.join(out_dir, 'metrics.csv')}")
    print(f"Detailed metrics saved to {os.path.join(out_dir, 'detailed_metrics.txt')}")

if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    args = parse_opt()
    main(args) 