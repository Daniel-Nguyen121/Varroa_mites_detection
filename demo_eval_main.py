import argparse
import os
from demo_eval_1_metrics import evaluate_model_metrics
from demo_eval_2_visualization import plot_metrics, save_detection_results
from demo_eval_3_performance import evaluate_model_performance
from demo_eval_4_utils import setup_model, setup_data_loader, create_output_dirs, save_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='fasterrcnn_resnet50_fpn_v2')
    parser.add_argument('-c', '--config', default='data_configs/varroa.yaml')
    parser.add_argument('-d', '--device', default='cuda:1' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-w', '--weights', default='outputs/training/varroa-old-2/best_model.pth', type=str)
    parser.add_argument('-o', '--output-dir', default='outputs/eval')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int)
    args = parser.parse_args()
    
    # Convert device string to torch device
    device = torch.device(args.device)
    
    # Create output directories
    output_dirs = create_output_dirs(args.output_dir)
    
    # Setup model and data loader
    model, config = setup_model(args.config, args.model, args.weights, device)
    data_loader = setup_data_loader(config, args.batch_size, args.workers)
    
    print("Starting evaluation...")
    
    # 1. Evaluate metrics
    print("\nEvaluating model metrics...")
    metrics = evaluate_model_metrics(
        model, data_loader, device,
        config['CLASSES'], output_dirs['metrics']
    )
    save_results(metrics, os.path.join(output_dirs['metrics'], 'metrics.txt'))
    plot_metrics(metrics, os.path.join(output_dirs['metrics'], 'metrics_plot.png'))
    
    # 2. Generate visualizations
    print("\nGenerating detection visualizations...")
    # Get a batch of images
    images, _ = next(iter(data_loader))
    images = list(img.to(device) for img in images)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    
    # Save detection results
    save_detection_results(
        images, outputs,
        output_dirs['visualizations'],
        config['CLASSES']
    )
    
    # 3. Evaluate performance
    print("\nEvaluating model performance...")
    performance = evaluate_model_performance(model, device)
    save_results(performance, os.path.join(output_dirs['performance'], 'performance.txt'))
    
    print("\nEvaluation completed!")
    print(f"Results saved to: {args.output_dir}")
    print("\nSummary of results:")
    print(f"mAP@0.5: {metrics['mAP@0.5']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
    print(f"FPS: {performance['FPS']:.2f}")

if __name__ == '__main__':
    main() 