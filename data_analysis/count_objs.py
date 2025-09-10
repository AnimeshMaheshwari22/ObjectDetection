import argparse
import json
from pathlib import Path
from collections import Counter

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
    "train",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir_label", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val"], required=True)
    return parser.parse_args()

def count_instances(labels_dir: Path) -> Counter:
    counter = Counter()
    json_files = sorted(labels_dir.glob("*.json"))
    for file_path in json_files:
        with open(file_path, "r") as f:
            data = json.load(f)
        frames = data.get("frames", [])
        if not frames:
            continue
        objects = frames[0].get("objects", [])
        for obj in objects:
            category = obj.get("category")
            if category in CLASS_NAMES:
                counter[category] += 1
    return counter

def main():
    args = parse_args()
    labels_dir = Path(args.data_dir_label) / args.split
    if not labels_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {labels_dir}")
    counts = count_instances(labels_dir)
    print(f"\nObject counts for {args.split} split:\n")
    for cls in CLASS_NAMES:
        print(f"{cls:15s}: {counts.get(cls, 0)}")

if __name__ == "__main__":
    main()
