import os
import cv2
import torch
import numpy as np
import argparse
from ultralytics import YOLO

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

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)
    inter = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2g - x1g + 1) * (y2g - y1g + 1)
    union = box1_area + box2_area - inter
    return inter / union if union > 0 else 0

def compute_ap(rec, prec):
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = np.max(prec[rec >= t]) if np.sum(rec >= t) != 0 else 0
        ap += p / 11.0
    return ap

def compute_ar(rec):
    return rec[-1] if len(rec) > 0 else 0

def evaluate_map(gt_dict, pred_dict, num_classes, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = [0.5]
    aps_all = {thr: [] for thr in iou_thresholds}
    ars_all = {thr: [] for thr in iou_thresholds}
    for cls in range(num_classes):
        aps_thr, ars_thr = [], []
        for thr in iou_thresholds:
            cls_preds = []
            npos = 0
            for img, gts in gt_dict.items():
                gt_cls = [g for g in gts if g[0] == cls]
                npos += len(gt_cls)
                preds = [p for p in pred_dict.get(img, []) if p[0] == cls]
                for p in preds:
                    cls_preds.append((img, p[1], p[2:]))
            if len(cls_preds) == 0 or npos == 0:
                aps_thr.append(0)
                ars_thr.append(0)
                continue
            cls_preds = sorted(cls_preds, key=lambda x: x[1], reverse=True)
            tp, fp = np.zeros(len(cls_preds)), np.zeros(len(cls_preds))
            matched = {img: np.zeros(len([g for g in gt_dict[img] if g[0] == cls])) for img in gt_dict}
            for i, (img, conf, box_pred) in enumerate(cls_preds):
                gt_boxes = [g[1:] for g in gt_dict[img] if g[0] == cls]
                if len(gt_boxes) == 0:
                    fp[i] = 1
                    continue
                ious = [compute_iou(box_pred, gt) for gt in gt_boxes]
                max_iou, j = max(ious), np.argmax(ious)
                if max_iou >= thr and matched[img][j] == 0:
                    tp[i], matched[img][j] = 1, 1
                else:
                    fp[i] = 1
            tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
            recall = tp_cum / (npos + 1e-6)
            precision = tp_cum / (tp_cum + fp_cum + 1e-6)
            aps_thr.append(compute_ap(recall, precision))
            ars_thr.append(compute_ar(recall))
        for i, thr in enumerate(iou_thresholds):
            aps_all[thr].append(aps_thr[i])
            ars_all[thr].append(ars_thr[i])
    return aps_all, ars_all

def main(args):
    model = YOLO(args.weights)
    gt_dict, pred_dict = {}, {}
    if args.visualize:
        os.makedirs(args.vis_dir, exist_ok=True)
    for img_file in os.listdir(args.test_images):
        if not img_file.endswith((".jpg", ".png")):
            continue
        img_path = os.path.join(args.test_images, img_file)
        label_path = os.path.join(args.test_labels, os.path.splitext(img_file)[0] + ".txt")
        gts = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls, x_c, y_c, w, h = map(float, line.strip().split())
                    cls = int(cls)
                    img = cv2.imread(img_path)
                    h_img, w_img = img.shape[:2]
                    x1 = int((x_c - w / 2) * w_img)
                    y1 = int((y_c - h / 2) * h_img)
                    x2 = int((x_c + w / 2) * w_img)
                    y2 = int((y_c + h / 2) * h_img)
                    gts.append((cls, x1, y1, x2, y2))
        gt_dict[img_file] = gts
        results = model(img_path, verbose=False)[0]
        preds = []
        for box in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = box
            cls = int(cls)
            preds.append((cls, conf, x1, y1, x2, y2))
        pred_dict[img_file] = preds
        if args.visualize:
            img = cv2.imread(img_path)
            for cls, x1, y1, x2, y2 in gts:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"GT-{CLASS_NAMES[cls]}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            for cls, conf, x1, y1, x2, y2 in preds:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(img, f"{CLASS_NAMES[cls]} {conf:.2f}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            save_path = os.path.join(args.vis_dir, img_file)
            cv2.imwrite(save_path, img)
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps_all, ars_all = evaluate_map(gt_dict, pred_dict, num_classes=len(CLASS_NAMES),
                                    iou_thresholds=iou_thresholds)
    mAP50 = np.mean(aps_all[0.5])
    print(f"\nmAP@0.5: {mAP50:.4f}")
    mAP5095 = np.mean([np.mean(aps_all[thr]) for thr in iou_thresholds])
    print(f"mAP@0.5:0.95: {mAP5095:.4f}\n")
    for i, name in enumerate(CLASS_NAMES):
        ap50 = aps_all[0.5][i]
        ap5095 = np.mean([aps_all[thr][i] for thr in iou_thresholds])
        ar50 = ars_all[0.5][i]
        print(f"Class {i} ({name}): AP50={ap50:.4f}, AP50-95={ap5095:.4f}, AR50={ar50:.4f}")
    print("\n⚠️ Safety-Critical Classes:")
    for cls_id in [3, 7]:
        ap50 = aps_all[0.5][cls_id]
        ap5095 = np.mean([aps_all[t][cls_id] for t in iou_thresholds])
        ar50 = ars_all[0.5][cls_id]
        print(f"{CLASS_NAMES[cls_id]} (cls={cls_id}): AP50={ap50:.4f}, AP50-95={ap5095:.4f}, AR50={ar50:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLO .pt model")
    parser.add_argument("--test_images", type=str, required=True, help="Folder with test images")
    parser.add_argument("--test_labels", type=str, required=True, help="Folder with test labels (.txt YOLO format)")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization of predictions vs GT")
    parser.add_argument("--vis_dir", type=str, default="vis_results", help="Folder to save visualization outputs")
    args = parser.parse_args()
    main(args)
