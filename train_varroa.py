"""
USAGE:
# Train with default settings (FasterRCNN ResNet50 FPN V2):
python train_varroa.py --config data_configs/varroa.yaml --epochs 200 --batch-size 8

# Train with specific settings:
python train_varroa.py --model fasterrcnn_resnet50_fpn_v2 --epochs 200 --config data_configs/varroa.yaml \
    --batch-size 8 --img-size 320 --device cuda:0 --project-name varroa_train_v1

# Resume training:
python train_varroa.py --config data_configs/varroa.yaml --resume-training \
    --weights outputs/training/varroa_train_v1/last_model_state.pth
"""

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
    parser.add_argument('-d', '--device', default=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('-e', '--epochs', default=2, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('-ims', '--img-size', default=320, type=int)
    parser.add_argument('-pn', '--project-name', default='varroa_train_v1', type=str)
    parser.add_argument('-w', '--weights', default=None, type=str)
    parser.add_argument('-r', '--resume-training', action='store_true')
    parser.add_argument('-ca', '--cosine-annealing', action='store_true')
    return vars(parser.parse_args())

def train_epoch(model, optimizer, train_loader, device, epoch, loss_hist, scheduler=None):
    """Train one epoch with progress bar"""
    model.train()
    
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}', 
                       leave=False)
    
    for images, targets in progress_bar:
        # Move images and targets to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if scheduler:
            scheduler.step()
            
        # Update progress bar
        loss_value = losses.item()
        loss_hist.send(loss_value)
        progress_bar.set_postfix({'loss': f'{loss_value:.4f}'})
    
    progress_bar.close()
    return loss_hist.value, loss_dict

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
    out_dir = os.path.join('outputs', 'training', args['project_name'])
    os.makedirs(out_dir, exist_ok=True)

    # Create datasets and dataloaders
    train_dataset = create_train_dataset(
        data_configs['TRAIN_DIR_IMAGES'], 
        data_configs['TRAIN_DIR_LABELS'],
        args['img_size'], args['img_size'], 
        data_configs['CLASSES']
    )
    valid_dataset = create_valid_dataset(
        data_configs['VALID_DIR_IMAGES'],
        data_configs['VALID_DIR_LABELS'],
        args['img_size'], args['img_size'],
        data_configs['CLASSES']
    )
    test_dataset = create_valid_dataset(
        data_configs['TEST_DIR_IMAGES'],
        data_configs['TEST_DIR_LABELS'],
        args['img_size'], args['img_size'],
        data_configs['CLASSES']
    )
    
    train_loader = create_train_loader(train_dataset, args['batch_size'], args['workers'])
    valid_loader = create_valid_loader(valid_dataset, args['batch_size'], args['workers'])
    test_loader = create_valid_loader(test_dataset, args['batch_size'], args['workers'])

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
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
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True)

    # Initialize scheduler
    scheduler = None
    if args['cosine_annealing']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args['epochs']+10, T_mult=1
        )

    # Initialize metrics tracker
    metrics = MetricsTracker(out_dir)

    # Resume training if requested
    start_epoch = 0
    if args['resume_training'] and args['weights']:
        print('Loading checkpoint and resuming training...')
        checkpoint = torch.load(args['weights'], map_location=args['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        metrics.load_state(checkpoint)

    # Training loop
    print('\nStarting training...')
    print('------------------')
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(start_epoch, args['epochs']), 
                     desc='Training Progress', position=0)
    
    for epoch in epoch_pbar:
        # Train one epoch
        epoch_loss, loss_dict = train_epoch(
            model, optimizer, train_loader, 
            args['device'], epoch, metrics.train_loss_hist,
            scheduler=scheduler
        )
        
        # Extract component losses
        component_losses = (
            loss_dict['loss_classifier'].item(),
            loss_dict['loss_box_reg'].item(),
            loss_dict['loss_objectness'].item(),
            loss_dict['loss_rpn_box_reg'].item()
        )
        
        # Evaluate on validation set
        val_metrics = evaluate_during_training(
            model, valid_loader, args['device'],
            data_configs['CLASSES'], out_dir
        )
        
        # Update and save metrics
        is_best = metrics.update(epoch_loss, val_metrics, epoch, component_losses)
        metrics.save_model(model, optimizer, epoch, out_dir, data_configs, args['model'], is_best)
        metrics.plot_metrics()
        
        # Update progress bar description
        desc = f"Epoch {epoch+1}/{args['epochs']}"
        if val_metrics:
            desc += f" | Loss: {epoch_loss:.4f} | mAP@0.5: {val_metrics[1]:.4f}"
        epoch_pbar.set_description(desc)
    
    epoch_pbar.close()
    print('\nTraining completed!')
    
    # Evaluate on test set
    print('\nEvaluating on test set...')
    test_metrics = evaluate_test_set(
        model, test_loader, args['device'],
        data_configs['CLASSES'], out_dir
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