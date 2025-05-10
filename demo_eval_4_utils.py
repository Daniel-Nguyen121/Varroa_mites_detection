import torch
import yaml
import os
from models.create_fasterrcnn_model import create_model
from datasets import create_valid_dataset, create_valid_loader

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path) as f:
        return yaml.safe_load(f)

def setup_model(config_path, model_name, weights_path, device):
    """Setup model with weights"""
    # Load config
    config = load_config(config_path)
    
    # Initialize model
    model = create_model[model_name](num_classes=config['NC'], pretrained=True)
    model = model.to(device)
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    return model, config

def setup_data_loader(config, batch_size=8, num_workers=4):
    """Setup data loader for evaluation"""
    test_dataset = create_valid_dataset(
        config['TEST_DIR_IMAGES'],
        config['TEST_DIR_LABELS'],
        320, 320,
        config['CLASSES']
    )
    return create_valid_loader(test_dataset, batch_size, num_workers)

def create_output_dirs(base_dir):
    """Create output directories for evaluation results"""
    dirs = {
        'metrics': os.path.join(base_dir, 'metrics'),
        'visualizations': os.path.join(base_dir, 'visualizations'),
        'performance': os.path.join(base_dir, 'performance')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def save_results(results, save_path):
    """Save evaluation results to file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        for key, value in results.items():
            if isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")

def main():
    """Test utility functions"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='fasterrcnn_resnet50_fpn_v2')
    parser.add_argument('-c', '--config', default='data_configs/varroa.yaml')
    parser.add_argument('-d', '--device', default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('-w', '--weights', required=True, type=str)
    args = parser.parse_args()
    
    # Test utility functions
    model, config = setup_model(args.config, args.model, args.weights, args.device)
    data_loader = setup_data_loader(config)
    output_dirs = create_output_dirs('outputs/eval')
    
    print("Model and data loader setup successful!")
    print("\nOutput directories created:")
    for name, path in output_dirs.items():
        print(f"{name}: {path}")
    
    # Test saving results
    test_results = {
        'test_metric': 0.95,
        'nested_results': {
            'precision': 0.92,
            'recall': 0.88
        }
    }
    save_results(test_results, os.path.join(output_dirs['metrics'], 'test_results.txt'))
    print("\nTest results saved successfully!")

if __name__ == '__main__':
    main() 