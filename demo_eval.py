import torch
import argparse
import yaml
import os
import time
from datasets import create_train_dataset, create_valid_dataset, create_train_loader, create_valid_loader
from models.create_fasterrcnn_model import create_model
from utils.metrics import MetricsTracker
from utils.eval_utils import evaluate_during_training, evaluate_test_set
from torch_utils.engine import train_one_epoch
from tqdm import tqdm
import thop
import torch.nn as nn

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='fasterrcnn_resnet50_fpn_v2')
    parser.add_argument('-c', '--config', default='data_configs/varroa.yaml')
    parser.add_argument('-d', '--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('-ims', '--img-size', default=320, type=int)
    parser.add_argument('-pn', '--project-name', default='eval_varroa_v1', type=str)
    parser.add_argument('-w', '--weights', default="/media/data4/home/vuhai/MOT_train/Varroa_detection/Detection_algos/fasterrcnn_resnet50_fpn_v2_varroa/outputs/training/varroa-old-2/best_model.pth", type=str)
    return vars(parser.parse_args())

def get_model_info(model, device, img_size):
    """Get model information including parameters, layers, and FLOPs"""
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count layers
    num_layers = len(list(model.children()))
    
    # Calculate FLOPs with valid bounding boxes
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    # Create valid bounding boxes (x1, y1, x2, y2) where x2 > x1 and y2 > y1
    x1 = torch.rand(1) * (img_size - 10)  # Leave some margin
    y1 = torch.rand(1) * (img_size - 10)
    x2 = x1 + torch.rand(1) * (img_size - x1 - 1)  # Ensure x2 > x1
    y2 = y1 + torch.rand(1) * (img_size - y1 - 1)  # Ensure y2 > y1
    
    dummy_target = [{
        'boxes': torch.tensor([[x1, y1, x2, y2]]).to(device),
        'labels': torch.tensor([1]).to(device)
    }]
    
    try:
        flops, params = thop.profile(model, inputs=(dummy_input, dummy_target))
        flops_g = flops / 1e9
    except Exception as e:
        print(f"Warning: Could not calculate FLOPs: {e}")
        flops_g = 0
    
    return {
        'Total Parameters': total_params,
        'Trainable Parameters': trainable_params,
        'Number of Layers': num_layers,
        'FLOPs (G)': flops_g
    }

def measure_fps(model, device, img_size, num_runs=100):
    """Measure FPS of the model"""
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    
    # Create valid bounding boxes
    x1 = torch.rand(1) * (img_size - 10)
    y1 = torch.rand(1) * (img_size - 10)
    x2 = x1 + torch.rand(1) * (img_size - x1 - 1)
    y2 = y1 + torch.rand(1) * (img_size - y1 - 1)
    
    dummy_target = [{
        'boxes': torch.tensor([[x1, y1, x2, y2]]).to(device),
        'labels': torch.tensor([1]).to(device)
    }]
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input, dummy_target)
    
    # Measure FPS
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input, dummy_target)
    end_time = time.time()
    
    fps = num_runs / (end_time - start_time)
    return fps

def main(args):
    # Load config
    with open(args['config']) as f:
        data_configs = yaml.safe_load(f)
    
    # Create output directory
    out_dir = os.path.join('outputs', 'eval', args['project_name'])
    os.makedirs(out_dir, exist_ok=True)

    test_dataset = create_valid_dataset(
        data_configs['TEST_DIR_IMAGES'],
        data_configs['TEST_DIR_LABELS'],
        args['img_size'], args['img_size'],
        data_configs['CLASSES']
    )
    
    test_loader = create_valid_loader(test_dataset, args['batch_size'], args['workers'])

    print(f"Test samples: {len(test_dataset)}")

    # Initialize model
    model = create_model[args['model']](num_classes=data_configs['NC'], pretrained=True)
    model = model.to(args['device'])

    # Print model information
    model_info = get_model_info(model, args['device'], args['img_size'])
    print("\nModel Information:")
    for key, value in model_info.items():
        print(f"{key}: {value}")

    # Initialize optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Initialize metrics tracker
    metrics = MetricsTracker(out_dir)


    print('Loading best model...')
    start_time = time.time()

    checkpoint = torch.load(args['weights'], map_location=args['device'])
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    print(f"Model name: {checkpoint['model_name']}")

    # Load state dict
    if 'model_state_dict' in checkpoint:
        # Clean up the state dict by removing thop-related keys
        state_dict = checkpoint['model_state_dict']
        # Remove keys that contain 'total_ops' or 'total_params'
        state_dict = {k: v for k, v in state_dict.items() if 'total_ops' not in k and 'total_params' not in k}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    # Move model to device and set to eval mode
    model = model.to(args['device'])
    model.eval()
    print("Model loaded successfully!")
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")

    
    # Evaluate on test set
    print('\nEvaluating on test set...')
    test_metrics = evaluate_test_set(
        model, test_loader, args['device'],
        data_configs['CLASSES'], out_dir,
        threshold=0.9
    )
    
    # Measure FPS
    fps = measure_fps(model, args['device'], args['img_size'])
    
    print('\nTest Set Results:')
    for metric_name, value in test_metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print(f"\nPerformance Metrics:")
    print(f"FPS: {fps:.2f}")
    for key, value in model_info.items():
        print(f"{key}: {value}")

if __name__ == '__main__':
    args = parse_opt()
    main(args) 