import torch
from torch.utils.data import Dataset
import cv2
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from eval_1_utils import debug_print

def load_coco_annotation(xml_path, img_width, img_height):
    """Load COCO format annotation from XML file"""
    boxes = []
    labels = []
    
    if not os.path.exists(xml_path):
        debug_print(f"XML file not found: {xml_path}")
        return boxes, labels
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image size from XML
        size = root.find('size')
        if size is None:
            debug_print(f"No size tag found in {xml_path}")
            return boxes, labels
            
        xml_width = int(size.find('width').text)
        xml_height = int(size.find('height').text)
        
        debug_print(f"Processing XML: {xml_path}")
        debug_print(f"Image size from XML: {xml_width}x{xml_height}")
        debug_print(f"Target size: {img_width}x{img_height}")
        
        for obj in root.findall('object'):
            # Get class label
            class_name = obj.find('name').text
            # Convert class name to index (assuming class names are in order in your config)
            class_id = 1  # Default to 1 for varroa
            
            # Get bounding box
            bbox = obj.find('bndbox')
            if bbox is None:
                debug_print(f"No bndbox found for object in {xml_path}")
                continue
                
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            debug_print(f"Found object: {class_name} at [{xmin}, {ymin}, {xmax}, {ymax}]")
            
            # Scale coordinates if image size is different
            if xml_width != img_width or xml_height != img_height:
                xmin = int(xmin * img_width / xml_width)
                ymin = int(ymin * img_height / xml_height)
                xmax = int(xmax * img_width / xml_width)
                ymax = int(ymax * img_height / xml_height)
                debug_print(f"Scaled to: [{xmin}, {ymin}, {xmax}, {ymax}]")
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_id)
        
        debug_print(f"Total objects found: {len(boxes)}")
        return boxes, labels
        
    except Exception as e:
        debug_print(f"Error processing {xml_path}: {str(e)}")
        return boxes, labels

class FastImageDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        # Pre-load all images and labels
        self.images = []
        self.gt_boxes = []
        self.gt_labels = []
        self.img_sizes = []
        
        debug_print("Pre-loading dataset...")
        for img_path, label_path in tqdm(zip(image_paths, label_paths), total=len(image_paths)):
            # Load and preprocess image
            image = cv2.imread(img_path)
            if image is None:
                debug_print(f"Could not read image {img_path}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = image.shape
            
            # Load ground truth from COCO XML
            boxes, labels = load_coco_annotation(label_path, img_w, img_h)
            
            if self.transform:
                image = self.transform(image)
            
            self.images.append(image)
            self.gt_boxes.append(torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32))
            self.gt_labels.append(torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long))
            self.img_sizes.append((img_h, img_w))
            
            debug_print(f"Processed {img_path}")
            debug_print(f"Image size: {img_w}x{img_h}")
            debug_print(f"Number of boxes: {len(boxes)}")
            if len(boxes) > 0:
                debug_print(f"First box: {boxes[0]}")
                debug_print(f"First label: {labels[0]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'gt_boxes': self.gt_boxes[idx],
            'gt_labels': self.gt_labels[idx],
            'img_path': self.image_paths[idx],
            'img_size': self.img_sizes[idx]
        }
    
    def collate_fn(self, batch):
        images = torch.stack([b['image'] for b in batch])
        return {
            'images': images,
            'gt_boxes': [b['gt_boxes'] for b in batch],
            'gt_labels': [b['gt_labels'] for b in batch],
            'img_paths': [b['img_path'] for b in batch],
            'img_sizes': [b['img_size'] for b in batch]
        } 