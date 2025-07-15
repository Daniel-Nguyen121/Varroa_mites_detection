# Varroa Detection using Faster R-CNN

This repository contains a complete implementation of Faster R-CNN (ResNet50 FPN v2) for Varroa mite detection in bee colonies. The project supports training, evaluation, and inference on both images and videos.

## 📋 Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements a Faster R-CNN object detection model specifically designed for detecting Varroa mites in bee colony images. The model uses ResNet50 FPN v2 as the backbone and is trained to detect Varroa mites with high precision.

**Key Features:**
- Faster R-CNN with ResNet50 FPN v2 backbone
- COCO format dataset support
- Comprehensive training and evaluation pipeline
- Real-time inference on images and videos
- Detailed metrics and visualization

## 🔧 Requirements

### System Requirements
- Python 3.7+
- CUDA-compatible GPU (recommended for training)
- At least 8GB RAM
- 2GB+ free disk space for dataset and model storage

### Python Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- PyTorch & TorchVision
- OpenCV
- NumPy
- Matplotlib
- PyYAML
- Albumentations
- COCO API (pycocotools)

**Optional Dependencies:**
- Weights & Biases (wandb) for experiment tracking
- TensorBoard for training visualization

## 📦 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd fasterrcnn_resnet50_fpn_v2
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"
```

## 📊 Data Preparation

### 1. Download Dataset

Download the Varroa detection dataset from:
```
https://demo.data (Contact us to get the dataset)
```

The dataset should be in COCO format with the following structure:
```
dataset/
├── train/
│   ├── images/
│   └── annotations.json
├── val/
│   ├── images/
│   └── annotations.json
└── test/
    ├── images/
    └── annotations.json
```

### 2. Convert YOLO to COCO Format (if needed)

If your data is in YOLO format, use the conversion script:

```bash
python /mnt/disk2/home/comvis/TungND/Detect-Track-MO/Varroa_detection/Varroa_detect/convert_yolo2coco.py
```

**Important:** Before running the conversion script, you need to modify the code in the "Change code here" block:

```python
# Change code here
# Uncomment/comment the appropriate lines based on your image format:

# For .jpg images:
remove_image = os.path.join(os.sep.join(image_root), image_name+'.jpg')
possible_xml_name = os.path.join(self.labels_path, image_name.split('.jpg')[0]+'.xml')

# For .png images:
# remove_image = os.path.join(os.sep.join(image_root), image_name+'.png')
# possible_xml_name = os.path.join(self.labels_path, image_name.split('.png')[0]+'.xml')
#-----End-----#
```

### 3. Dataset Structure

After conversion, ensure your dataset follows this structure:
```
your_dataset/
├── train/
│   ├── image1.jpg
│   ├── image1.xml
│   ├── image2.jpg
│   └── image2.xml
├── val/
│   ├── image3.jpg
│   ├── image3.xml
│   ├── image4.jpg
│   └── image4.xml
└── test/
    ├── image5.jpg
    ├── image5.xml
    ├── image6.jpg
    └── image6.xml
```

## ⚙️ Configuration

### 1. Data Configuration Files

The project uses YAML configuration files to define dataset paths and parameters. Two example configurations are provided:

- `data_configs/varroa.yaml` - For standard dataset
- `data_configs/varroa_1820.yaml` - For extended dataset

### 2. Configuration Parameters

Edit the configuration file to match your dataset paths:

```yaml
# data_configs/varroa.yaml
TRAIN_DIR_IMAGES: "/path/to/your/dataset/train/"
TRAIN_DIR_LABELS: "/path/to/your/dataset/train/"
VALID_DIR_IMAGES: "/path/to/your/dataset/val/"
VALID_DIR_LABELS: "/path/to/your/dataset/val/"
TEST_DIR_IMAGES: "/path/to/your/dataset/test/"
TEST_DIR_LABELS: "/path/to/your/dataset/test/"

# Class names (background class + object classes)
CLASSES: [
    '__background__',
    'varroa'
]

# Number of classes (object classes + 1 for background)
NC: 2

# Whether to save validation predictions during training
SAVE_VALID_PREDICTION_IMAGES: True
```

## 🚀 Training

### 1. Basic Training

Train the model with default settings:

```bash
python train.py \
    --model fasterrcnn_resnet50_fpn_v2 \
    --config data_configs/varroa.yaml \
    --epochs 200 \
    --batch-size 8 \
    --img-size 800
```

### 2. Advanced Training Options

**Training with custom project name:**
```bash
python train.py \
    --model fasterrcnn_resnet50_fpn_v2 \
    --config data_configs/varroa.yaml \
    --epochs 200 \
    --batch-size 8 \
    --project-name varroa_detection_v1 \
    --img-size 800
```

**Training with additional augmentations:**
```bash
python train.py \
    --model fasterrcnn_resnet50_fpn_v2 \
    --config data_configs/varroa.yaml \
    --epochs 200 \
    --batch-size 8 \
    --use-train-aug \
    --project-name varroa_detection_aug
```

**Training without mosaic augmentation:**
```bash
python train.py \
    --model fasterrcnn_resnet50_fpn_v2 \
    --config data_configs/varroa.yaml \
    --epochs 200 \
    --batch-size 8 \
    --no-mosaic \
    --project-name varroa_detection_no_mosaic
```

**Training with cosine annealing scheduler:**
```bash
python train.py \
    --model fasterrcnn_resnet50_fpn_v2 \
    --config data_configs/varroa.yaml \
    --epochs 200 \
    --batch-size 8 \
    --cosine-annealing \
    --project-name varroa_detection_cosine
```

### 3. Training Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--model` | Model architecture | `fasterrcnn_resnet50_fpn_v2` | `fasterrcnn_resnet50_fpn`, `fasterrcnn_resnet50_fpn_v2` |
| `--config` | Data configuration file | `data_configs/varroa.yaml` | Path to YAML config |
| `--epochs` | Number of training epochs | `200` | Integer |
| `--batch-size` | Batch size | `8` | Integer |
| `--img-size` | Input image size | `800` | Integer |
| `--project-name` | Output directory name | Auto-generated | String |
| `--use-train-aug` | Enable additional augmentations | False | Flag |
| `--no-mosaic` | Disable mosaic augmentation | False | Flag |
| `--cosine-annealing` | Use cosine annealing scheduler | False | Flag |
| `--weights` | Path to pretrained weights | None | Path |
| `--resume-training` | Resume from checkpoint | False | Flag |

### 4. Training Output

Training outputs are saved in `outputs/training/[project_name]/`:

- `best_model.pth` - Best model weights
- `last_model.pth` - Last epoch weights
- `train_loss.png` - Training loss plot
- `train_loss_epoch.png` - Epoch-wise loss plot
- `mAP.png` - Mean Average Precision plot
- `loss_cls.png` - Classification loss plot
- `loss_bbox_reg.png` - Bounding box regression loss plot
- `loss_obj.png` - Objectness loss plot
- `loss_rpn_bbox.png` - RPN bounding box loss plot

## 📈 Evaluation

### 1. Model Evaluation

Evaluate a trained model:

```bash
python eval_faster.py
```

**Configuration in eval_faster.py:**
```python
# Change these paths in eval_faster.py
CONFIG_PATH = "data_configs/varroa.yaml"
MODEL_PATH = "outputs/training/your_project/best_model.pth"
```

### 2. Evaluation Metrics

The evaluation provides comprehensive metrics:

**Detection Metrics:**
- Precision, Recall, F1-Score
- AP@50 (Average Precision at IoU=0.5)
- AP@[50:95] (Average Precision at IoU=0.5:0.95)

**Performance Metrics:**
- Inference time per image
- FPS (Frames Per Second)
- Pre-processing, inference, and post-processing times

**Model Complexity:**
- Number of layers
- Trainable parameters
- GFLOPs (computational complexity)

### 3. Evaluation Output

The evaluation script outputs detailed results:

```
============================================================
EVALUATION RESULTS
============================================================
Precision: 0.9234
Recall: 0.8956
F1-Score: 0.9093
AP@50: 0.9123
AP@[50:95]: 0.6789
------------------------------------------------------------
Preprocessing time: 15.2 ms
Inference time: 45.8 ms
Post-processing time: 8.1 ms
Total time: 69.1 ms
FPS: 14.5
------------------------------------------------------------
Number of layers: 284
Trainable parameters: 41,177,026
GFLOPs: 45.2
============================================================
```

## 🔍 Inference

### 1. Image Inference

Run inference on single images or directories:

```bash
python inference.py \
    --input path/to/image.jpg \
    --weights outputs/training/your_project/best_model.pth \
    --config data_configs/varroa.yaml \
    --threshold 0.5 \
    --show-image
```

**Inference Parameters:**
- `--input`: Path to image or directory
- `--weights`: Path to trained model weights
- `--config`: Data configuration file
- `--threshold`: Detection confidence threshold (default: 0.3)
- `--show-image`: Display results in real-time
- `--mpl-show`: Show results using matplotlib

### 2. Video Inference

Run inference on video files:

```bash
python inference_video.py \
    --input path/to/video.mp4 \
    --weights outputs/training/your_project/best_model.pth \
    --config data_configs/varroa.yaml \
    --threshold 0.5 \
    --show-image
```

**Video Inference Features:**
- Real-time processing with FPS display
- Output video with detection annotations
- Configurable detection threshold
- Support for various video formats

### 3. Pretrained Models

Download pretrained model weights from:
```
[Link_weights](https://drive.google.com/drive/folders/1JrC8919cBcAYNlSjJwo6TjnzFt-BYpez?usp=sharing)
```

## 🏗️ Model Architecture

### Faster R-CNN with ResNet50 FPN v2

The model architecture consists of:

1. **Backbone**: ResNet50 with Feature Pyramid Network (FPN)
2. **Region Proposal Network (RPN)**: Generates region proposals
3. **RoI Heads**: Classifies proposals and refines bounding boxes
4. **Detection Head**: Final classification and regression layers

**Model Specifications:**
- Input size: 800×800 pixels
- Number of classes: 2 (background + varroa)
- Total parameters: ~41M
- GFLOPs: ~45.2

### Model Variants

Two model variants are supported:

1. **fasterrcnn_resnet50_fpn**: Standard Faster R-CNN
2. **fasterrcnn_resnet50_fpn_v2**: Enhanced version with improved performance

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python train.py --batch-size 4

# Reduce image size
python train.py --img-size 640
```

**2. Dataset Loading Errors**
- Verify dataset paths in configuration file
- Ensure images and annotations are paired correctly
- Check file permissions

**3. Model Loading Errors**
- Verify model weights path
- Ensure model architecture matches weights
- Check CUDA compatibility

**4. Training Convergence Issues**
- Try different learning rates
- Enable cosine annealing scheduler
- Adjust augmentation settings
- Increase training epochs

### Performance Optimization

**For Training:**
- Use GPU with sufficient VRAM (8GB+ recommended)
- Adjust batch size based on available memory
- Use mixed precision training if available

**For Inference:**
- Use GPU for real-time performance
- Adjust detection threshold for speed/accuracy trade-off
- Consider model quantization for deployment

### Debug Mode

Enable debug output by modifying the evaluation script:

```python
# In eval_faster.py, enable debug prints
print(f"DEBUG - Sample data:")
print(f"  Pred boxes shape: {boxes.shape}")
print(f"  GT boxes shape: {gt_boxes.shape}")
```

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [TorchVision Models](https://pytorch.org/vision/stable/models.html)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)

**Note:** This project is specifically designed for Varroa mite detection in bee colonies. For other object detection tasks, modify the configuration files and class definitions accordingly. 