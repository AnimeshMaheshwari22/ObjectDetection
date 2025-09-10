# Object Detection on BDD100k Dataset

A comprehensive object detection pipeline using YOLOv11 on the BDD100k dataset for autonomous driving applications.

## Project Overview

This project implements a complete object detection system trained on the BDD100k dataset, focusing on 10 critical classes for autonomous driving: car, traffic sign, traffic light, person, truck, bus, bike, rider, motor, and train.

**Detailed report:** https://docs.google.com/document/d/1lXhbLTICkC20brzPqk1mvWR8aHfhu5q_t8xg7iI3hS0/edit?usp=sharing

**Code Repository:** https://github.com/AnimeshMaheshwari22/ObjectDetection

>Note: For all the images and visualizations please refer to the project report at: https://docs.google.com/document/d/1lXhbLTICkC20brzPqk1mvWR8aHfhu5q_t8xg7iI3hS0/edit?usp=sharing

## Dataset Analysis
# BDD100k Object Detection Analysis Dashboard

This dashboard allows interactive analysis of the BDD100k dataset for object detection. It visualizes class distributions, bounding box statistics, conditional distributions, and interesting samples in the dataset.

## 1. Download the Dataset

1. Go to the [BDD100k dataset page](https://bdd-data.berkeley.edu/).
2. Download the following:
   - **Images (100k, ~5.3GB)**
   - **Labels (107 MB)**
3. Extract the zip files to a local folder structure like this:

```
BDD100k/
├── images/
│   ├── train/img/
│   └── val/img/
└── labels/
    ├── train/
    └── val/
```

Only the detection classes (light, signs, person, car, etc.) are required. Semantic segmentation data (lanes, drivable areas) is not needed.

## 2. Build the Docker Image

Open a terminal, navigate to the root of the repository (where `Dockerfile` is located), and run:

```bash
cd data_analysis
docker build -t bdd-dashboard .
```

This packages the app and all dependencies into a self-contained Docker image.

## 3. Run the Dashboard

Mount the dataset folders and launch the container.

For the training split:

```bash
docker run -p 8501:8501 \
    -v /path/to/labels:/data/labels \
    -v /path/to/images:/data/images \
    bdd-dashboard \
    --data_dir_labels /data/labels \
    --data_dir_img /data/images \
    --split train
```

For the validation split:

```bash
docker run -p 8501:8501 \
    -v /path/to/labels:/data/labels \
    -v /path/to/images:/data/images \
    bdd-dashboard \
    --data_dir_labels /data/labels \
    --data_dir_img /data/images \
    --split val
```

**Notes:**
- Replace `/path/to/labels` and `/path/to/images` with the actual local paths.
- The `--split` flag selects which part of the dataset to analyze (train or val).

## 4. Access the Dashboard

Open a browser and go to:

```
http://localhost:8501
```

Use the sidebar to:
- Select the dataset split.
- Navigate between Overall Distribution, Detailed Analysis, and Interesting Samples.
- Explore class distributions, bounding box sizes, environmental condition statistics, and annotated sample images.
### Dataset Statistics
- **Total Samples:** 100,000 images
- **Train Split:** 70,000 samples
- **Test Split:** 20,000 samples
- **Validation Split:** 10,000 samples


### Key Findings

Please refer to the reprot for the same

## Model Architecture

### YOLOv11 Selection Rationale

YOLOv11 was selected as the primary model based on several critical factors:

- **Real-Time Performance:** Achieves 1.2-8.5ms inference times (100+ FPS)
- **Single-Stage Architecture:** Eliminates region proposal overhead
- **Hardware Compatibility:** Optimized for automotive edge computing
- **Resource Efficiency:** Suitable for deployment on constrained systems

### Architecture Components

#### Backbone Network
Enhanced Cross Stage Partial architecture with improved C2f modules:
- Input: 640×640×3
- Progressive downsampling through 5 stages
- Feature extraction from 64 to 1024 channels


#### Neck - Feature Pyramid Network
Path Aggregation Network combining multi-scale features:
- Top-down semantic feature fusion
- Bottom-up detail aggregation
- Bidirectional information flow

#### Detection Head - Anchor-Free Design
Direct prediction approach without pre-defined anchors:
- Classification head for object classes
- Regression head for bounding box coordinates
- Objectness head for confidence scores

### Key Architectural Innovations

- **Enhanced C2f Modules:** Improved gradient flow and reduced computational redundancy
- **Dynamic Label Assignment:** Quality-based positive/negative sample assignment
- **Multi-Scale Detection:** Three detection scales (20×20, 40×40, 80×80)
- **Advanced Data Augmentation:** Mosaic, MixUp, and Copy-paste techniques

### Loss Function
Composite loss balancing multiple objectives:
- Classification Loss (Binary Cross-Entropy)
- Regression Loss (Complete IoU-based)
- Objectness Loss (Confidence scoring)

## Training Configuration

### Infrastructure
- **GPU:** NVIDIA A10
- **Training Time:** 13 hours total
- **Training Samples:** 70,000
- **Validation Samples:** 10,000

### Hyperparameters
- **Image Size:** 960×960
- **Batch Size:** 8
- **Epochs:** 10
- **Optimization:** Automatic Mixed Precision (AMP)
- **Weight Averaging:** Exponential Moving Average (EMA)


## Results and Evaluation

### Evaluation Metrics

#### Mean Average Precision (mAP)
Comprehensive performance measure across all classes:
- Balances precision and recall
- Prevents bias toward common classes
- Single-value overall performance indicator

#### Average Recall per Class
Diagnostic tool for identifying class-specific weaknesses:
- Measures detection completeness per class
- Enables targeted improvements
- Critical for safety-critical applications

### Test Set Performance

| Metric | Score |
|--------|-------|
| mAP@0.5 | 0.4732 |
| mAP@0.5:0.95 | 0.3033 |

### Detailed Class Performance (Test Set)

| Class | AP50 | AP50-95 | AR50 |
|-------|------|---------|------|
| car | 0.7014 | 0.4637 | 0.7554 |
| traffic sign | 0.5873 | 0.3644 | 0.6633 |
| traffic light | 0.5580 | 0.2656 | 0.6202 |
| person | 0.5844 | 0.3424 | 0.6084 |
| truck | 0.5032 | 0.4078 | 0.5977 |
| bus | 0.5011 | 0.4126 | 0.5633 |
| bike | 0.4651 | 0.2658 | 0.5325 |
| rider | 0.4282 | 0.2710 | 0.4985 |
| motor | 0.4036 | 0.2398 | 0.4637 |
| train | 0.0000 | 0.0000 | 0.0000 |


### Validation Set Performance

| Metric | Score |
|--------|-------|
| mAP@0.5 | 0.4748 |
| mAP@0.5:0.95 | 0.3019 |

### Detailed Class Performance (Validation Set)

| Class | AP50 | AP50-95 | AR50 |
|-------|------|---------|------|
| car | 0.7025 | 0.4656 | 0.7551 |
| traffic sign | 0.5891 | 0.3647 | 0.6697 |
| traffic light | 0.5608 | 0.2568 | 0.6278 |
| person | 0.5846 | 0.3408 | 0.6123 |
| truck | 0.5022 | 0.4075 | 0.5978 |
| bus | 0.5155 | 0.4209 | 0.5855 |
| bike | 0.4667 | 0.2663 | 0.5194 |
| rider | 0.4245 | 0.2725 | 0.4730 |
| motor | 0.4022 | 0.2242 | 0.4248 |
| train | 0.0000 | 0.0000 | 0.0000 |

## Prediction Samples

## Future Improvements

### Model Enhancement
- Extended training beyond 10 epochs for improved performance
- Implementation of class balancing techniques for rare classes
- Advanced augmentation strategies for underrepresented classes

### Data Augmentation
- Increased training samples for sparse labels (train, motor, rider, bike)
- Synthetic data generation for rare scenarios
- Weather and lighting condition augmentation


## Installation and Usage

### Requirements
```
ultralytics
torch
torchvision
opencv-python
matplotlib
numpy
streamlit
```

### Training

# Setup and Training Instructions

## 1. Prepare Dataset
Run the dataset preparation script. It downloads images and labels, converts JSON annotations to YOLO format, and creates the correct directory structure with a `dataset.yaml`.

```bash
python data_prep_yolo.py \
    --img_zip LINK1 \
    --label_zip LINK2 \
    --out_root dataset
```

After running, you will have:

```
dataset/
  train/
    images/
    labels/
  val/
    images/
    labels/
  dataset.yaml
```

## 2. Download Pretrained Weights
Download the YOLOv11x pretrained checkpoint:

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11x.pt
```

## 3. Train the Model
Run the training script with your desired settings:

```bash
python train.py \
    --weights yolov11x.pt \
    --data dataset/dataset.yaml \
    --batch 8 \
    --epochs 10 \
    --workers 1 \
    --imgsz 960 \
    --device 0
```

## 4. Outputs
Training results, weights, and training plots will be saved automatically inside the `runs` directory.


### Inference, visualization and metrics
```bash
cd evaluation
python predict_and_eval.py \
    --weights runs/train/exp/weights/best.pt \
    --test_images data/test/images \
    --test_labels data/test/labels

python predict_and_eval.py \
    --weights runs/train/exp/weights/best.pt \
    --test_images data/test/images \
    --test_labels data/test/labels \
    --visualize \
    --vis_dir results_vis
```

## Improvements on Model and Data Aspects

# Model Improvements
* The model has only trained for 10 epochs and it is performing well. With enough training, it will be able to perform extremely well on the given test set.

# Data Improvements
* There need to be more training instance samples added for sparse labels like train, motor, rider, bike.


## Acknowledgments

- BDD100k dataset creators
- Ultralytics YOLOv11 pretrained weights on COCO
- Berkeley DeepDrive research group