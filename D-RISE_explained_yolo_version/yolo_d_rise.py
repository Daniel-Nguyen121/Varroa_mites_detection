import os
import torch
#import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from torchvision import transforms
import yaml
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
import math
#=============================#

def load_model(model_path):
    """Load the YOLO version > 5 model"""
    print("Loading model...")
    try:
        # First try loading with the latest ultralytics version
        model = YOLO(model_path)
    except AttributeError as e:
        if "C3k2" in str(e):
            print("Warning: Model architecture mismatch. Attempting to load with legacy support...")
            # Try loading with legacy support
            model = YOLO(model_path, task='detect')
        else:
            raise e
    
    print("Model loaded successfully")
    model.cuda()
    # Set the model to evaluation mode
    model.eval()
    return model

def preprocess_image(image_path):
    """Preprocess image for model input"""
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
        
    height, width = image.shape[:2]
    image = cv2.resize(image, ((width // 32) * 32, (height // 32) * 32))        #Used for original dataset
    #image = cv2.resize(image, (640, 640))                                       #Used for new dataset
    #image = cv2.resize(image, (320, 320))                                       #Used for original dataset
    rgb_image = image.copy()

    #image = np.float32(image) / 255
    #transform = transforms.ToTensor()
    #tensor = transform(image).unsqueeze(0)

    #return image, tensor, rgb_image
    return rgb_image
    
def parse_detections_v2(results, conf_score):
    boxes, colors, names, scores = [], [], [], []

    # YOLOv8 returns a Boxes object with attributes xyxy, conf, and cls.
    # Convert these tensors to numpy arrays (if they are torch tensors)
    xyxy = results[0].boxes.xyxy.cpu().numpy()  # shape: (n, 4)
    confidences = results[0].boxes.conf.cpu().numpy()  # shape: (n,)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)  # shape: (n,)

    # The names dictionary is available as an attribute of the result
    names_dict = results[0].names  # e.g., {0: 'varroa'}

    for i in range(xyxy.shape[0]):
        confidence = confidences[i]
        if confidence < conf_score:
            continue

        # Convert each coordinate to integer
        xmin, ymin, xmax, ymax = map(int, xyxy[i])
        category = classes[i]
        name = names_dict.get(category, str(category))

        # Use a predefined COLORS list or dictionary (make sure COLORS is defined)
        color = [255,0,0]

        boxes.append((xmin, ymin, xmax, ymax))
        colors.append(color)
        names.append(name)
        scores.append(round(float(confidence), 4))

    return boxes, colors, names, scores

def load_ground_truth(label_path, img_size):
    """Load ground truth annotations from YOLO format label file"""
    boxes = []
    labels = []

    if not os.path.exists(label_path):
        return boxes, labels

    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())

            # Convert YOLO format to COCO format
            x1 = (x_center - width/2) * img_size[0]
            y1 = (y_center - height/2) * img_size[1]
            x2 = (x_center + width/2) * img_size[0]
            y2 = (y_center + height/2) * img_size[1]

            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            labels.append(int(class_id))

    return boxes, labels

def get_yolo_boxes(label_path, img_size):
    """Load ground truth annotations from YOLO format label file"""
    boxes = []
    labels = []

    if not os.path.exists(label_path):
        return boxes, labels

    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height, score = map(float, line.strip().split())

            # Convert YOLO format to COCO format
            x1 = (x_center - width/2) * img_size[0]
            y1 = (y_center - height/2) * img_size[1]
            x2 = (x_center + width/2) * img_size[0]
            y2 = (y_center + height/2) * img_size[1]

            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            labels.append(int(class_id))

    return boxes, labels


def generate_mask(image_size, grid_size, prob_thresh):
    image_w, image_h = image_size
    grid_w, grid_h = grid_size
    cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
    up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h

    mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) <
            prob_thresh).astype(np.float32)
    mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
    offset_w = np.random.randint(0, cell_w)
    offset_h = np.random.randint(0, cell_h)
    mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
    return mask

def mask_image(image, mask):
    masked = ((image.astype(np.float32) / 255 * np.dstack([mask] * 3)) *
              255).astype(np.uint8)
    return masked

def iou(box1, box2):
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)
    tl = np.vstack([box1[:2], box2[:2]]).max(axis=0)
    br = np.vstack([box1[2:], box2[2:]]).min(axis=0)
    intersection = np.prod(br - tl) * np.all(tl < br).astype(float)
    area1 = np.prod(box1[2:] - box1[:2])
    area2 = np.prod(box2[2:] - box2[:2])
    return intersection / (area1 + area2 - intersection)

def generate_saliency_map(image, model,
                           target_box,
                           prob_thresh=0.65,
                           grid_size=(16, 16),
                           n_masks=5000,
                           seed=0):
    np.random.seed(seed)
    image_h, image_w = image.shape[:2]
    res = np.zeros((image_h, image_w), dtype=np.float32)
    total_score = 0.0
    
    for _ in tqdm(range(n_masks)):
        mask = generate_mask(image_size=(image_w, image_h),
                             grid_size=grid_size,
                             prob_thresh=prob_thresh)
        masked = mask_image(image, mask)
        out = model([masked])  # Use masked image instead of original

        boxes, colors, names, scores = parse_detections_v2(out, 0.2)  # Lower confidence threshold

        # Find best matching box and its score
        best_score = 0.0
        for box, score in zip(boxes, scores):
            iou_score = iou(target_box, box)
            combined_score = iou_score * score
            best_score = max(best_score, combined_score)

        # Accumulate the mask weighted by the score
        res += mask * best_score
        total_score += best_score

    # Normalize by the total score
    if total_score > 0:
        res = res / total_score
    
    return res

def main(image_path, model, save_path, prob_thresh=0.7,grid_size=(8, 8),n_masks=5000):
    np.random.seed(0)

    rgb_image = preprocess_image(image_path)
    
    # Load ground truth
    label_path = image_path.split('.')[0] + '.txt'
    image_name = os.path.basename(image_path)
    image_par = os.path.dirname(image_path)
    compare_path = os.path.join(image_par,'detect_' + image_name.split('.')[0] + '.txt')
    
    gt_boxes, gt_labels = load_ground_truth(label_path, (rgb_image.shape[1], rgb_image.shape[0]))
    cmr_boxes, _ = get_yolo_boxes(compare_path, (rgb_image.shape[1], rgb_image.shape[0]))

    #Get detection boxes
    #parse_detections_v2(results, conf_score)
    
    # Create a combined saliency map
    combined_saliency = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.float32)
    
    print(f"Processing {len(gt_boxes)} bounding boxes...")
    for i, gt_box in enumerate(gt_boxes):
        print(f"Processing box {i+1}/{len(gt_boxes)}")
        
        # Generate saliency map with optimized parameters
        saliency_map = generate_saliency_map(
                          rgb_image, model,
                          gt_box,
                          prob_thresh=prob_thresh,
                          grid_size=grid_size,
                          n_masks=n_masks,
                          seed=1)

        # Add to combined saliency map (take maximum value at each pixel)
        combined_saliency = np.maximum(combined_saliency, saliency_map)
        
        # Clear memory after each iteration
        gc.collect()
        torch.cuda.empty_cache()

    # Normalize the combined saliency map
    combined_saliency = (combined_saliency - combined_saliency.min()) / (combined_saliency.max() - combined_saliency.min() + 1e-8)
    
    # Apply Gaussian blur to smooth the combined heatmap
    combined_saliency = cv2.GaussianBlur(combined_saliency, (5, 5), 0)
    
    # Convert to uint8 for color mapping
    combined_saliency = (combined_saliency * 255).astype(np.uint8)
    
    # Apply color map (using COLORMAP_HOT for hotter colors)
    heat_map = cv2.applyColorMap(combined_saliency, cv2.COLORMAP_JET)
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB)

    # Create visualization
    image_with_bbox = rgb_image.copy()
    
    # Draw all bounding boxes
    for gt_box in gt_boxes:
        cv2.rectangle(image_with_bbox, tuple(gt_box[:2]), tuple(gt_box[2:]), (0, 0, 255), 2)
    
    # Draw all detection bounding boxes
    for box in cmr_boxes:
        cv2.rectangle(image_with_bbox, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)

    try:
        '''
        # Create figure with better visualization
        plt.figure(figsize=(5, 4))
        
        # Original image with all bboxes
        plt.subplot(121)
        plt.imshow(image_with_bbox[:, :, ::-1])
        plt.axis('off')
        
        # Overlay
        plt.subplot(122)
        plt.imshow(rgb_image[:, :, ::-1])
        plt.imshow(heat_map, alpha=0.5)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'heatmap_combined_{image_name}'), bbox_inches='tight', transparent=True)
        plt.close('all')  # Close all figures
        '''
        fig, axes = plt.subplots(
            1, 2,
            figsize=(10, 8),
        )
        axes[0].imshow(image_with_bbox[:, :, ::-1])
        axes[0].axis('off')

        axes[1].imshow(rgb_image[:, :, ::-1])
        axes[1].imshow(heat_map, alpha=0.5)
        axes[1].axis('off')

        fig.subplots_adjust(wspace=10)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'heatmap_combined_{image_name}'), bbox_inches='tight', transparent=True)
        
    except Exception as e:
        print(f"Error during plotting: {e}")
    finally:
        # Ensure cleanup
        plt.close('all')
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":

    # Initialize the model    
    src_pretrained_yolo_dir = "/mnt/disk2/home/tungnd/MOT_bee/Detection_varroa/Varroa_detection/pretrained_models"
    
    data_dir = "/mnt/disk2/home/tungnd/MOT_bee/Detection_varroa/Data_viss/"
    data_old = os.path.join(data_dir, 'Fol_1')
    data_new = os.path.join(data_dir, 'Fol_2')

    OUT_DIR = '/mnt/disk2/home/tungnd/MOT_bee/Detection_varroa/D-RISE_explained/outputs_vis_drise'
    
    out_old = os.path.join(OUT_DIR, 'Fol_1')
    out_new = os.path.join(OUT_DIR, 'Fol_2')
    os.makedirs(out_old, exist_ok=True)
    os.makedirs(out_new, exist_ok=True)
    
    prob_thresh=0.5
    grid_size=(8, 8)
    n_masks=5000

    for model_name in os.listdir(src_pretrained_yolo_dir):
        if '1820' in model_name:
            print('Skip')
            if '12x' in model_name:
                model_path = os.path.join(src_pretrained_yolo_dir, model_name)
                model = load_model(model_path)
                save_path = os.path.join(out_new, model_name.split('.')[0])
                os.makedirs(save_path, exist_ok=True)

                for img_name in os.listdir(data_new):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(data_new, img_name)
                        main(img_path, model, save_path, prob_thresh,grid_size,n_masks)
        else:
            if '12x' in model_name:
                model_path = os.path.join(src_pretrained_yolo_dir, model_name)
                model = load_model(model_path)
                save_path = os.path.join(out_old, model_name.split('.')[0])
                os.makedirs(save_path, exist_ok=True)

                for img_name in os.listdir(data_old):
                    if img_name.endswith('.png'):
                        img_path = os.path.join(data_old, img_name)
                        main(img_path, model, save_path, prob_thresh,grid_size,n_masks)
                print('Skip')
            else:
                print('Skip')

    
    
    
    
   

    # model = load_model(v12)
    # img_path = os.path.join(data_old, 'id_8025.png')  #8042 8201 1196
    # for img_name in os.listdir(data_old):
    #     if img_name.endswith('.png'):
    #         img_path = os.path.join(data_old, img_name)
    #         main(img_path, model, prob_thresh=0.5,grid_size=(8, 8),n_masks=5000)