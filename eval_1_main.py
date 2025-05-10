import torch
import argparse
import yaml
import os
from eval_1_utils import increase_file_limit
from eval_1_dataset import FastImageDataset
from eval_1_metrics import calculate_metrics, calculate_ap_range, save_metrics_to_file
from eval_1_evaluation import evaluate_batch
from models.create_fasterrcnn_model import create_model
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import torch.amp as amp
import torch.multiprocessing as mp
import numpy as np
import time

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

    # Enable automatic mixed precision
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
            label_paths.append(os.path.join(test_dir, filename.rsplit('.', 1)[0] + '.xml'))
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    
    # Create dataset and dataloader
    dataset = FastImageDataset(image_paths, label_paths, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args['batch_size'],
        num_workers=min(4, args['workers']),
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Process batches
    print("Processing images...")
    total_time = 0
    total_images = 0
    all_metrics = []
    
    start_time = time.time()  # Add overall timing
    
    for batch in tqdm(dataloader):
        batch_metrics = evaluate_batch(
            model, batch, args['device'],
            args['conf_thres'], args['iou_thres'],
            scaler
        )
        all_metrics.extend(batch_metrics)
        total_images += len(batch['images'])
    
    end_time = time.time()  # End overall timing
    total_time = end_time - start_time
    
    # Convert results to DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Calculate average metrics
    avg_metrics = df.mean(numeric_only=True)
    
    # Calculate FPS
    fps = total_images / total_time if total_time > 0 else 0
    
    # Calculate AP@50 and AP@50-95
    # Get all predictions and ground truth
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []
    
    for metrics in all_metrics:
        if 'pred_boxes' in metrics:
            all_pred_boxes.extend(metrics['pred_boxes'])
            all_pred_scores.extend(metrics['pred_scores'])
            all_pred_labels.extend(metrics['pred_labels'])
            all_gt_boxes.extend(metrics['gt_boxes'])
            all_gt_labels.extend(metrics['gt_labels'])
    
    # Convert to numpy arrays
    all_pred_boxes = np.array(all_pred_boxes)
    all_pred_scores = np.array(all_pred_scores)
    all_pred_labels = np.array(all_pred_labels)
    all_gt_boxes = np.array(all_gt_boxes)
    all_gt_labels = np.array(all_gt_labels)
    
    # Calculate AP@50
    ap_50 = calculate_ap_range(all_pred_boxes, all_pred_scores, all_pred_labels, 
                             all_gt_boxes, all_gt_labels, [0.5])[0]
    
    # Calculate AP@50-95
    ap_50_95 = calculate_ap_range(all_pred_boxes, all_pred_scores, all_pred_labels,
                                all_gt_boxes, all_gt_labels, 
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