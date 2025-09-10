import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm

DETECTION_CLASSES = [
    "bus", "traffic light", "traffic sign", "person", "bike",
    "truck", "motor", "car", "train", "rider"
]

def load_labels_from_directory(labels_dir: Path) -> pd.DataFrame:
    all_data: List[Dict[str, Any]] = []
    json_files = sorted(list(labels_dir.glob("*.json")))

    print(f"Parsing labels from {len(json_files)} files in {labels_dir}...")
    
    for file_path in tqdm(json_files, desc="Processing JSON files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_name = data.get('name', 'unknown').replace('.jpg', '')
        attributes = data.get('attributes', {})
        weather = attributes.get('weather', 'undefined')
        scene = attributes.get('scene', 'undefined')
        timeofday = attributes.get('timeofday', 'undefined')

        frames = data.get('frames', [])
        if not frames:
            continue
        
        objects = frames[0].get('objects', [])
        for obj in objects:
            category = obj.get("category")
            box2d = obj.get("box2d")

            if category in DETECTION_CLASSES and box2d:
                label_attributes = obj.get("attributes", {})
                record = {
                    'image_name': image_name,
                    'category': category,
                    'weather': weather,
                    'scene': scene,
                    'timeofday': timeofday,
                    'x1': box2d['x1'],
                    'y1': box2d['y1'],
                    'x2': box2d['x2'],
                    'y2': box2d['y2'],
                    'occluded': label_attributes.get('occluded', False),
                    'truncated': label_attributes.get('truncated', False),
                }
                all_data.append(record)

    df = pd.DataFrame(all_data)
    df['width'] = df['x2'] - df['x1']
    df['height'] = df['y2'] - df['y1']
    df['area'] = df['width'] * df['height']
    return df
