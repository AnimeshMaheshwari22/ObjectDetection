"""
This script prepares a YOLO-compatible dataset from raw zips of images and labels.

Steps performed:
1. Downloads and extracts images (train, val, test) from --img_zip
2. Downloads and extracts labels (train, val, test) from --label_zip
3. Creates the structure:
       dataset/
         train/images
         train/labels
         val/images
         val/labels
4. Converts JSON annotations into YOLO txt format
5. Writes a dataset.yaml ready for YOLO training

How to run:
    python data_prep_yolo.py \
        --img_zip LINK1 \
        --label_zip LINK2 \
        --out_root dataset

After running, your dataset will be in the `dataset` folder.

How to train with YOLO:
    python train.py \
        --data dataset/dataset.yaml \
        --epochs 100 \
        --img 640 \
        --batch 16 \
        --weights yolov11n.pt
"""
import os
import argparse
import zipfile
import requests
import shutil
import json

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
IMG_WIDTH = 1280
IMG_HEIGHT = 720

def download_and_extract(url, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    local_zip = os.path.join(out_dir, "temp.zip")
    with requests.get(url, stream=True) as r:
        with open(local_zip, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    with zipfile.ZipFile(local_zip, "r") as zip_ref:
        zip_ref.extractall(out_dir)
    os.remove(local_zip)

def convert_to_yolo(box, img_w=IMG_WIDTH, img_h=IMG_HEIGHT):
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height

def process_json(json_path, output_dir):
    with open(json_path, "r") as f:
        data = json.load(f)
    yolo_lines = []
    for frame in data.get("frames", []):
        for obj in frame.get("objects", []):
            category = obj.get("category")
            if category not in CLASS_DICT:
                continue
            if "box2d" not in obj:
                continue
            cls_id = CLASS_DICT[category]
            x_center, y_center, width, height = convert_to_yolo(obj["box2d"])
            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    if yolo_lines:
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        txt_path = os.path.join(output_dir, base_name + ".txt")
        os.makedirs(output_dir, exist_ok=True)
        with open(txt_path, "w") as out_f:
            out_f.write("\n".join(yolo_lines))

def prepare_dataset(img_root, label_root, out_root):
    for split in ["train", "val"]:
        img_dir = os.path.join(img_root, split)
        lbl_dir = os.path.join(label_root, split)
        out_img = os.path.join(out_root, split, "images")
        out_lbl = os.path.join(out_root, split, "labels")
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)
        for file in os.listdir(img_dir):
            src = os.path.join(img_dir, file)
            dst = os.path.join(out_img, file)
            shutil.copy(src, dst)
        for file in os.listdir(lbl_dir):
            if file.endswith(".json"):
                process_json(os.path.join(lbl_dir, file), out_lbl)

def write_yaml(out_root):
    yaml_path = os.path.join(out_root, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {os.path.join(out_root, 'train')}\n")
        f.write(f"val: {os.path.join(out_root, 'val')}\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n\n")
        f.write(f"names: {CLASS_NAMES}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_zip", type=str, help="URL to image zip")
    parser.add_argument("--label_zip", type=str, help="URL to label zip")
    parser.add_argument("--out_root", type=str, default="dataset")
    args = parser.parse_args()
    img_root = os.path.join(args.out_root, "images_raw")
    lbl_root = os.path.join(args.out_root, "labels_raw")
    download_and_extract(args.img_zip, img_root)
    download_and_extract(args.label_zip, lbl_root)
    prepare_dataset(img_root, lbl_root, args.out_root)
    write_yaml(args.out_root)

if __name__ == "__main__":
    main()
