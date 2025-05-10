import torch
import time
import numpy as np
import thop
from utils.eval_utils import calculate_model_stats

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

def get_model_info(model, device, img_size):
    """Get model information including parameters, layers, and FLOPs"""
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count layers
    num_layers = len(list(model.children()))
    
    # Calculate FLOPs
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    x1 = torch.rand(1) * (img_size - 10)
    y1 = torch.rand(1) * (img_size - 10)
    x2 = x1 + torch.rand(1) * (img_size - x1 - 1)
    y2 = y1 + torch.rand(1) * (img_size - y1 - 1)
    
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

def evaluate_model_performance(model, device, img_size=320):
    """Evaluate model performance metrics"""
    # Measure FPS
    fps = measure_fps(model, device, img_size)
    
    # Get model info
    model_info = get_model_info(model, device, img_size)
    
    # Calculate model stats
    model_stats = calculate_model_stats(model, (img_size, img_size))
    
    return {
        'FPS': fps,
        'Model Info': model_info,
        'Model Stats': model_stats
    }

def main():
    """Test performance evaluation"""
    import argparse
    from models.create_fasterrcnn_model import create_model
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='fasterrcnn_resnet50_fpn_v2')
    parser.add_argument('-c', '--config', default='data_configs/varroa.yaml')
    parser.add_argument('-d', '--device', default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('-w', '--weights', required=True, type=str)
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        data_configs = yaml.safe_load(f)
    
    # Initialize model
    model = create_model[args.model](num_classes=data_configs['NC'], pretrained=True)
    model = model.to(args.device)
    
    # Load weights
    checkpoint = torch.load(args.weights, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    # Evaluate performance
    performance = evaluate_model_performance(model, args.device)
    
    print("\nPerformance Metrics:")
    print(f"FPS: {performance['FPS']:.2f}")
    print("\nModel Information:")
    for key, value in performance['Model Info'].items():
        print(f"{key}: {value}")
    print("\nModel Statistics:")
    for key, value in performance['Model Stats'].items():
        print(f"{key}: {value}")

if __name__ == '__main__':
    main() 