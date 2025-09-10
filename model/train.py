import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo11x.pt")
    parser.add_argument("--data", type=str, default="dataset.yaml")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        batch=args.batch,
        epochs=args.epochs,
        workers=args.workers,
        imgsz=args.imgsz,
        device=args.device
    )

if __name__ == "__main__":
    main()
