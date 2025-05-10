import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_tensor
import os

def plot_metrics(metrics, save_path):
    """Plot and save metrics visualization"""
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot mAP values
    map_values = [metrics['mAP@0.5'], metrics['mAP@0.5:0.95']]
    ax1.bar(['mAP@0.5', 'mAP@0.5:0.95'], map_values)
    ax1.set_title('mAP Metrics')
    ax1.set_ylim(0, 1)
    
    # Plot class metrics
    class_metrics = metrics['class_metrics']
    x = np.arange(len(class_metrics['precision']))
    width = 0.25
    
    ax2.bar(x - width, class_metrics['precision'], width, label='Precision')
    ax2.bar(x, class_metrics['recall'], width, label='Recall')
    ax2.bar(x + width, class_metrics['f1_score'], width, label='F1-Score')
    ax2.set_title('Class-specific Metrics')
    ax2.set_xticks(x)
    ax2.set_ylim(0, 1)
    ax2.legend()
    
    # Plot precision-recall curve
    ax3.plot(class_metrics['recall'], class_metrics['precision'])
    ax3.set_title('Precision-Recall Curve')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # Plot confusion matrix (if available)
    if 'confusion_matrix' in metrics:
        ax4.imshow(metrics['confusion_matrix'], cmap='Blues')
        ax4.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_detections(image, boxes, scores, labels, class_names, threshold=0.3):
    """Visualize detection results on an image"""
    # Convert image to tensor if it's not already
    if not isinstance(image, torch.Tensor):
        image = to_tensor(image)
    
    # Filter detections based on threshold
    keep = scores >= threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    # Draw bounding boxes
    image_with_boxes = draw_bounding_boxes(
        image,
        boxes,
        [f"{class_names[l]} {s:.2f}" for l, s in zip(labels, scores)],
        colors=['red' if l == 1 else 'blue' for l in labels]  # Red for varroa, blue for background
    )
    
    return image_with_boxes

def save_detection_results(images, outputs, save_dir, class_names, threshold=0.3):
    """Save detection results as images"""
    os.makedirs(save_dir, exist_ok=True)
    
    for i, (image, output) in enumerate(zip(images, outputs)):
        # Convert image to numpy array if it's a tensor
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().transpose(1, 2, 0)
            image = (image * 255).astype(np.uint8)
        
        # Visualize detections
        image_with_boxes = visualize_detections(
            image,
            output['boxes'],
            output['scores'],
            output['labels'],
            class_names,
            threshold
        )
        
        # Convert to numpy array and save
        if isinstance(image_with_boxes, torch.Tensor):
            image_with_boxes = image_with_boxes.cpu().numpy().transpose(1, 2, 0)
            image_with_boxes = (image_with_boxes * 255).astype(np.uint8)
        
        cv2.imwrite(os.path.join(save_dir, f'detection_{i}.jpg'), cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))

def main():
    """Test visualization functions"""
    import argparse
    from models.create_fasterrcnn_model import create_model
    from datasets import create_valid_dataset, create_valid_loader
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
    
    # Create dataset and loader
    test_dataset = create_valid_dataset(
        data_configs['TEST_DIR_IMAGES'],
        data_configs['TEST_DIR_LABELS'],
        320, 320,
        data_configs['CLASSES']
    )
    test_loader = create_valid_loader(test_dataset, batch_size=1, num_workers=4)
    
    # Initialize model
    model = create_model[args.model](num_classes=data_configs['NC'], pretrained=True)
    model = model.to(args.device)
    
    # Load weights
    checkpoint = torch.load(args.weights, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    # Get a batch of images
    images, _ = next(iter(test_loader))
    images = list(img.to(args.device) for img in images)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    
    # Save detection results
    save_detection_results(
        images,
        outputs,
        'outputs/eval/visualizations',
        data_configs['CLASSES']
    )
    
    print("Visualization results saved to outputs/eval/visualizations/")

if __name__ == '__main__':
    main() 