import os
import json
import argparse
from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from ultralytics import YOLO
from ultralytics.data import YOLODataset
from ultralytics.utils import LOGGER
import tempfile
import yaml
from pathlib import Path

CLASS_NAMES = [
    "car",
    "traffic sign", 
    "traffic light",
    "person",
    "truck",
    "bus",
    "bike",
    "rider",
    "motor",
    "train"
]

CLASS_DICT = {name: i for i, name in enumerate(CLASS_NAMES)}

class JSONYOLODataset(YOLODataset):
    """Custom YOLO Dataset that loads JSON labels on-the-fly"""
    
    def __init__(self, img_path, json_path, img_width=1280, img_height=720, augment=True, prefix=""):
        self.img_path = img_path
        self.json_path = json_path
        self.img_width = img_width
        self.img_height = img_height
        
        self.im_files = self._get_img_files()
        self.json_files = self._get_json_files()

        self.matched_pairs = self._match_img_json_pairs()
        
        self.im_files = [pair['img'] for pair in self.matched_pairs]
        
        self.prefix = prefix
        
        super(YOLODataset, self).__init__()
        
        self.augment = augment
        self.rect = False
        self.mosaic_border = [0, 0]  
        self.stride = 32

        self.labels = self._load_json_labels()
        
        LOGGER.info(f'{prefix}Loaded {len(self.im_files)} images with {sum(len(l["cls"]) for l in self.labels)} labels from JSON')
    
    def _get_img_files(self):
        """Get all image files from the images directory"""
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif')
        img_files = []
        
        for ext in image_extensions:
            img_files.extend(glob(os.path.join(self.img_path, ext)))
            img_files.extend(glob(os.path.join(self.img_path, ext.upper())))
        
        return sorted(img_files)
    
    def _get_json_files(self):
        """Get all JSON files from the labels directory"""
        json_files = glob(os.path.join(self.json_path, '*.json'))
        return sorted(json_files)
    
    def _match_img_json_pairs(self):
        """Match image files with corresponding JSON files"""
        pairs = []
        
        for img_file in self.im_files:
            img_basename = os.path.splitext(os.path.basename(img_file))[0]
            
            json_file = None
            for jf in self.json_files:
                json_basename = os.path.splitext(os.path.basename(jf))[0]
                if json_basename == img_basename:
                    json_file = jf
                    break
            
            if json_file:
                pairs.append({'img': img_file, 'json': json_file})
            else:
                print(f"Warning: No JSON found for image {img_basename}")
        
        print(f"Matched {len(pairs)} image-JSON pairs out of {len(self.im_files)} images")
        return pairs
    
    def _parse_json_to_yolo(self, json_file):
        """Parse JSON file and convert to YOLO format"""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            yolo_labels = []
            
            frames = data.get("frames", [data])
            
            for frame in frames:
                objects = frame.get("objects", [])
                
                for obj in objects:
                    category = obj.get("category", "")
                    if category in CLASS_DICT and "box2d" in obj:
                        cls_id = CLASS_DICT[category]
                        box = obj["box2d"]
                        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                        x_center = ((x1 + x2) / 2) / self.img_width
                        y_center = ((y1 + y2) / 2) / self.img_height
                        width = (x2 - x1) / self.img_width
                        height = (y2 - y1) / self.img_height
                        if width > 0 and height > 0 and 0 <= x_center <= 1 and 0 <= y_center <= 1:
                            yolo_labels.append([cls_id, x_center, y_center, width, height])
                        else:
                            print(f"Invalid box in {json_file}: {box}")
            
            return yolo_labels
            
        except Exception as e:
            print(f"Error parsing {json_file}: {e}")
            return []
    
    def _load_json_labels(self):
        """Load labels from JSON files"""
        labels = []
        
        for pair in self.matched_pairs:
            img_file = pair['img']
            json_file = pair['json']
            
            label_dict = {
                'im_file': img_file,
                'shape': None,
                'cls': np.array([]),
                'bboxes': np.array([]).reshape(0, 4),
                'segments': [],
                'keypoints': None,
                'normalized': True,
                'bbox_format': 'xywh'
            }
            
            yolo_labels = self._parse_json_to_yolo(json_file)
            
            if yolo_labels:
                labels_array = np.array(yolo_labels, dtype=np.float32)
                label_dict['cls'] = labels_array[:, 0:1]
                label_dict['bboxes'] = labels_array[:, 1:5]
            
            labels.append(label_dict)
        
        return labels

def load_json_yolo_data(base_path=".", splits=['train', 'val', 'test'], img_width=1280, img_height=720):
    """Load data from images/ and labels/ directory structure with JSON labels"""
    datasets = {}
    
    for split in splits:
        img_path = os.path.join(base_path, 'images', split)
        json_path = os.path.join(base_path, 'labels', split)
        
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping {split}")
            continue
            
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, skipping {split}")
            continue
        
        img_files = glob(os.path.join(img_path, '*.*'))
        img_files = [f for f in img_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        json_files = glob(os.path.join(json_path, '*.json'))
        
        print(f"Found {len(img_files)} images and {len(json_files)} JSON files in {split}")
        
        if img_files and json_files:
            datasets[split] = {
                'img_path': img_path,
                'json_path': json_path
            }
    
    return datasets

class CustomYOLOTrainer:
    """Custom trainer that uses JSON-based dataset"""
    
    def __init__(self, model_path, class_names):
        self.model = YOLO(model_path)
        self.class_names = class_names
        
    def create_temp_yaml(self):
        """Create temporary dataset.yaml"""
        temp_dir = tempfile.mkdtemp()
        yaml_path = os.path.join(temp_dir, 'dataset.yaml')
        
        config = {
            'path': os.path.abspath('.'),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test', 
            'names': self.class_names
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f)
            
        return yaml_path
    
    def train(self, train_dataset, val_dataset=None, **kwargs):
        """Train with custom JSON datasets"""
        temp_yaml = self.create_temp_yaml()
        
        try:
            original_trainer_class = self.model.trainer_class
            
            class CustomTrainer(original_trainer_class):
                def get_dataset(self, dataset_path, mode='train', batch=None):
                    """Override to return our custom datasets"""
                    if mode == 'train':
                        return train_dataset
                    elif mode == 'val' and val_dataset is not None:
                        return val_dataset
                    else:
                        return train_dataset
            self.model.trainer_class = CustomTrainer
            results = self.model.train(data=temp_yaml, **kwargs)
            
            return results
            
        except Exception as e:
            print(f"Custom training failed: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to temporary YOLO format conversion...")
            return self._fallback_training(train_dataset, val_dataset, temp_yaml, **kwargs)
    
    def _fallback_training(self, train_dataset, val_dataset, temp_yaml, **kwargs):
        """Fallback method that creates temporary YOLO format files"""
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            for split, dataset in [('train', train_dataset), ('val', val_dataset)]:
                if dataset is None:
                    continue
                    
                img_dir = os.path.join(temp_dir, split, 'images')
                label_dir = os.path.join(temp_dir, split, 'labels')
                os.makedirs(img_dir, exist_ok=True)
                os.makedirs(label_dir, exist_ok=True)
                for i, label_dict in enumerate(dataset.labels):
                    img_file = label_dict['im_file']
                    img_basename = os.path.basename(img_file)
                    shutil.copy2(img_file, os.path.join(img_dir, img_basename))
                    label_basename = os.path.splitext(img_basename)[0] + '.txt'
                    label_file = os.path.join(label_dir, label_basename)
                    
                    with open(label_file, 'w') as f:
                        if len(label_dict['cls']) > 0:
                            for j in range(len(label_dict['cls'])):
                                cls_id = int(label_dict['cls'][j, 0])
                                bbox = label_dict['bboxes'][j]
                                f.write(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            config = {
                'path': temp_dir,
                'train': 'train/images',
                'val': 'val/images',
                'names': self.class_names
            }
            
            with open(temp_yaml, 'w') as f:
                yaml.dump(config, f)
            results = self.model.train(data=temp_yaml, **kwargs)
            
            return results
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="YOLO Training with JSON Labels")
    parser.add_argument("--data-dir", type=str, default=".", 
                       help="Root directory containing images/ and labels/ folders")
    parser.add_argument("--img-width", type=int, default=1280,
                       help="Original image width for normalization")
    parser.add_argument("--img-height", type=int, default=720,
                       help="Original image height for normalization")
    parser.add_argument("--imgsz", type=int, default=960, 
                       help="Target image size for training")
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, 
                       help="Batch size")
    parser.add_argument("--model", type=str, default="yolo11x.pt", 
                       help="Pretrained YOLO model")
    parser.add_argument("--device", type=str, default="0", 
                       help="GPU device ID")
    parser.add_argument("--project", type=str, default="runs/detect",
                       help="Project directory")
    parser.add_argument("--name", type=str, default="json_training",
                       help="Experiment name")
    parser.add_argument("--workers", type=int, default=8,
                       help="Number of dataloader workers")
    
    args = parser.parse_args()

    print("YOLO JSON Label Training")
    print("===========================")
    print(f"Data directory: {os.path.abspath(args.data_dir)}")
    print(f"Model: {args.model}")
    print(f"Image size: {args.img_width}x{args.img_height} -> {args.imgsz}")
    print(f"Classes: {len(CLASS_NAMES)}")

    data_paths = load_json_yolo_data(
        args.data_dir, 
        splits=['train', 'val', 'test'],
        img_width=args.img_width,
        img_height=args.img_height
    )
    
    if 'train' not in data_paths:
        raise ValueError("No training data found! Make sure you have images/train/ and labels/train/ directories")
    
    print("\nCreating JSON-based datasets...")
    
    train_dataset = JSONYOLODataset(
        data_paths['train']['img_path'],
        data_paths['train']['json_path'],
        img_width=args.img_width,
        img_height=args.img_height,
        augment=True, 
        prefix="train: "
    )
    
    val_dataset = None
    if 'val' in data_paths:
        val_dataset = JSONYOLODataset(
            data_paths['val']['img_path'],
            data_paths['val']['json_path'],
            img_width=args.img_width,
            img_height=args.img_height,
            augment=False, 
            prefix="val: "
        )

    border = [-args.imgsz // 2, -args.imgsz // 2]
    train_dataset.mosaic_border = border
    if val_dataset:
        val_dataset.mosaic_border = border
    
    print(f"✓ Training dataset: {len(train_dataset)} images")
    if val_dataset:
        print(f"✓ Validation dataset: {len(val_dataset)} images")

    print(f"\n🚀 Starting YOLO11x training...")
    
    trainer = CustomYOLOTrainer(args.model, CLASS_NAMES)
    
    results = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        verbose=True,
        workers=args.workers,
        patience=50,
        save=True,
        save_period=10
    )
    
    print("\n🎉 Training completed!")
    print(f"Results saved to: {args.project}/{args.name}/")
    print(f"Best model: {args.project}/{args.name}/weights/best.pt")

    if 'test' in data_paths:
        print("\n🔍 Testing on test set...")
        test_dataset = JSONYOLODataset(
            data_paths['test']['img_path'],
            data_paths['test']['json_path'],
            img_width=args.img_width,
            img_height=args.img_height,
            augment=False,
            prefix="test: "
        )
        
        best_model = YOLO(f"{args.project}/{args.name}/weights/best.pt")
        

        temp_test_yaml = trainer.create_temp_yaml()
        metrics = best_model.val(data=temp_test_yaml, split='test')
        
        print(f"Test mAP50-95: {metrics.box.map:.4f}")
        print(f"Test mAP50: {metrics.box.map50:.4f}")

if __name__ == "__main__":
    main()