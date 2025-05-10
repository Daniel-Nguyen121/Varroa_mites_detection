"""
USAGE

# Training with Faster RCNN ResNet50 FPN model without mosaic or any other augmentation:
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --config data_configs/voc.yaml --no-mosaic --batch-size 4

# Training on ResNet50 FPN with custom project folder name with mosaic augmentation (ON by default):
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --config data_configs/voc.yaml --project-name resnet50fpn_voc --batch-size 4

# Training on ResNet50 FPN with custom project folder name with mosaic augmentation (ON by default) and added training augmentations:
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --use-train-aug --config data_configs/voc.yaml --project-name resnet50fpn_voc --batch-size 4

# Evaluate on test set
python eval_varroa.py --weights outputs/training/varroa_faster-r-cnn-2/best_model.pth --config data_configs/eval_varroa.yaml
"""

from torch_utils.engine_eval import evaluate
from datasets import create_valid_dataset, create_valid_loader
from models.create_fasterrcnn_model import create_model
from utils.general_eval import set_training_dir, save_mAP
from utils.logging import set_log, coco_log
import torch
import argparse
import yaml
import numpy as np
import sys
import time
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F

torch.multiprocessing.set_sharing_strategy('file_system')

# For same annotation colors each time.
np.random.seed(42)

def calculate_model_complexity(model, input_size=(320, 320)):
    """Calculate number of layers, parameters, and GFLOPs"""
    total_layers = 0
    total_params = 0
    trainable_params = 0
    
    # Count layers and parameters
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total_layers += 1
            total_params += sum(p.numel() for p in module.parameters())
            trainable_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    # Calculate GFLOPs using a dummy input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(next(model.parameters()).device)
    
    def count_operations(module, input, output):
        if isinstance(module, nn.Conv2d):
            # For Conv2d: FLOPs = 2 * Cin * Cout * H * W * K * K
            out_h = output.shape[2]
            out_w = output.shape[3]
            kernel_ops = module.kernel_size[0] * module.kernel_size[1]
            bias_ops = 1 if module.bias is not None else 0
            ops_per_element = kernel_ops * module.in_channels // module.groups + bias_ops
            total_ops = ops_per_element * module.out_channels * out_h * out_w
            return total_ops
        elif isinstance(module, nn.Linear):
            # For Linear: FLOPs = 2 * in_features * out_features
            total_ops = 2 * module.in_features * module.out_features
            return total_ops
        return 0

    total_ops = 0
    hooks = []
    
    def add_hooks(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(
                lambda m, i, o: setattr(m, 'total_ops', count_operations(m, i, o))))
    
    model.eval()
    model.apply(add_hooks)
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    for module in model.modules():
        if hasattr(module, 'total_ops'):
            total_ops += module.total_ops
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    gflops = total_ops / 1e9
    
    return total_layers, total_params, trainable_params, gflops

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='fasterrcnn_resnet50_fpn_v2')
    parser.add_argument('-c', '--config', default="data_configs/eval_varroa.yaml")
    parser.add_argument('-d', '--device', default=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('-b', '--batch-size', dest='batch_size', default=1, type=int)
    parser.add_argument('-ims', '--img-size', dest='img_size', default=320, type=int)
    parser.add_argument('-pn', '--project-name', default='eval_varroa_faster-r-cnn-3', type=str)
    parser.add_argument('-w', '--weights', default='/media/data4/home/vuhai/MOT_train/Varroa_detection/Detection_algos/fasterrcnn_resnet50_fpn_v2_varroa/outputs/training/res_7/best_model.pth', type=str)
    parser.add_argument('-th', '--threshold', default=0.7, type=float, help='detection threshold')
    args = vars(parser.parse_args())
    return args

def main(args):
    with open(args['config']) as file:
        data_configs = yaml.safe_load(file)

    TEST_DIR_IMAGES = data_configs['TEST_DIR_IMAGES']
    TEST_DIR_LABELS = data_configs['TEST_DIR_LABELS']
    CLASSES = data_configs['CLASSES']
    NUM_CLASSES = data_configs['NC']
    NUM_WORKERS = args['workers']
    DEVICE = args['device']
    BATCH_SIZE = args['batch_size']
    IMAGE_WIDTH = args['img_size']
    IMAGE_HEIGHT = args['img_size']
    OUT_DIR = set_training_dir(args['project_name'])
    COLORS = np.random.uniform(0, 1, size=(len(CLASSES), 3))
    DETECTION_THRESHOLD = args['threshold']
    set_log(OUT_DIR)

    print('Loading model...')
    checkpoint = torch.load(args['weights'], map_location=DEVICE)
    ckpt_state_dict = checkpoint['model_state_dict']
    old_classes = ckpt_state_dict['roi_heads.box_predictor.cls_score.weight'].shape[0]

    build_model = create_model[args['model']]
    model = build_model(num_classes=old_classes)
    model.load_state_dict(ckpt_state_dict)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, NUM_CLASSES)
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features, NUM_CLASSES * 4)

    model = model.to(DEVICE)
    model.eval()

    # Calculate model complexity
    total_layers, total_params, trainable_params, gflops = calculate_model_complexity(
        model, input_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
    )
    print(f"\nModel Complexity:")
    print(f"Total Layers: {total_layers}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Approximate GFLOPs: {gflops:.2f}")

    print('\n[INFO] Running evaluation on test set...')
    test_dataset = create_valid_dataset(
        TEST_DIR_IMAGES, TEST_DIR_LABELS,
        IMAGE_WIDTH, IMAGE_HEIGHT, CLASSES
    )
    test_loader = create_valid_loader(test_dataset, BATCH_SIZE, NUM_WORKERS)
    
    print(f"Number of test samples: {len(test_dataset)}")
    
    # Initialize timing variables
    inference_times = []
    total_time_start = time.time()

    # Run evaluation with timing
    with torch.no_grad():
        coco_evaluator, stats, test_pred_image = evaluate(
            model,
            test_loader,
            device=DEVICE,
            save_valid_preds=True,
            out_dir=OUT_DIR,
            classes=CLASSES,
            colors=COLORS,
            detection_threshold=DETECTION_THRESHOLD
        )
        total_time = time.time() - total_time_start

    # Calculate metrics
    mAP_50 = stats[1]  # mAP@0.5
    mAP_50_95 = stats[0]  # mAP@0.5:0.95
    precision = stats[2]  # Precision
    recall = stats[3]  # Recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate average inference time and FPS
    avg_inference_time = total_time / len(test_dataset)
    fps = len(test_dataset) / total_time

    # Save mAP plots
    save_mAP(OUT_DIR, [mAP_50], [mAP_50_95])
    
    # Log the evaluation metrics
    coco_log(OUT_DIR, stats)

    # Print detailed metrics
    print("\nDetailed Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"mAP@50: {mAP_50:.4f}")
    print(f"mAP@50:95: {mAP_50_95:.4f}")
    print(f"\nPerformance Metrics:")
    print(f"Average Inference Time: {avg_inference_time*1000:.2f} ms")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"FPS: {fps:.2f}")
    print(f"GFLOPs: {gflops:.2f}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Save metrics to a file
    with open(f"{OUT_DIR}/metrics.txt", "w") as f:
        f.write("Model Complexity:\n")
        f.write(f"Total Layers: {total_layers}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"GFLOPs: {gflops:.2f}\n\n")
        
        f.write("Evaluation Metrics:\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1_score:.4f}\n")
        f.write(f"mAP@50: {mAP_50:.4f}\n")
        f.write(f"mAP@50:95: {mAP_50_95:.4f}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"Average Inference Time: {avg_inference_time*1000:.2f} ms\n")
        f.write(f"Total Time: {total_time:.2f} seconds\n")
        f.write(f"FPS: {fps:.2f}\n")

    print('\n[INFO] Test evaluation complete. Results saved to:', f"{OUT_DIR}/metrics.txt")

if __name__ == '__main__':
    args = parse_opt()
    main(args)
