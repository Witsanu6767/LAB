# BDD100K YOLOv8 Object Detection System

An end-to-end object detection pipeline based on **YOLOv8 (Ultralytics)** and **OpenCV**, customized and trained on the **BDD100K** dataset. Unlike crop classification approaches, this system directly performs bounding box detection and class recognition simultaneously in a single pass.

## 📌 Overview

This project provides a complete workflow for training, evaluating, and testing a YOLOv8 real-time object detector:

1. **Model Training (`train_2.py`)**: Fine-tunes a pretrained YOLOv8 Nano model (`yolov8n.pt`) on a dataset defined in `dataset/data.yaml` (BDD100K format).
2. **Evaluation (`evaluate_2.py`)**: Runs evaluation on the test dataset split and computes key object detection metrics including Precision, Recall, mAP50, and mAP50-95.
3. **Random Image Testing (`image_test_2.py`)**: Randomly samples up to 5 test images from `dataset/test/images`, performs detection using the trained weights, displays annotated results interactively with OpenCV, and saves the outputs.
4. **Real-time Webcam Inference (`realtime_test_2.py`)**: Captures video frames from the webcam (`camera 0`), runs real-time YOLOv8 bounding box detection at confidence threshold $0.25$, and overlays annotated detections live.

## 📁 Directory Structure

```
├── dataset/
│   ├── data.yaml           # Dataset configuration (classes, split paths)
│   ├── train/              # Training images and YOLO annotations
│   └── test/
│       └── images/         # Test images directory
├── runs/
│   ├── bdd100k/            # Training outputs (weights, plots, results.csv)
│   ├── evaluation/         # Validation/evaluation metrics outputs
│   └── detect\runs\bdd100k
├── train_2.py              # YOLOv8 fine-tuning script
├── evaluate_2.py           # Model evaluation script (mAP, Precision, Recall)
├── image_test_2.py         # Batch testing script for sample images
└── realtime_test_2.py      # Real-time webcam object detection demo
```

## 🏗️ Model & Pipeline Specifications

* **Backbone Model**: YOLOv8 Nano (`yolov8n.pt`)
* **Input Resolution**:
  * Training / Evaluation / Image Test: $480 \times 480$
  * Real-time Detection: $640 \times 640$
* **Hardware Acceleration**: Enabled for GPU (`device=0`, `amp=True`)

## 🚀 Usage Guidelines

### 1. Training the Model (`train_2.py`)

Trains the YOLOv8 model using `dataset/data.yaml`:

```bash
python train_2.py
```

* **Default Parameters**:
  * `epochs`: `10`
  * `imgsz`: `480`
  * `batch`: `8`
  * `device`: GPU `0`
  * `amp`: `True` (Automatic Mixed Precision)
* **Outputs**: Saved in `runs/bdd100k/weights/best.pt` and `last.pt`.

### 2. Evaluating Model Performance (`evaluate_2.py`)

Evaluates the trained model (`best.pt`) on the test split:

```bash
python evaluate_2.py
```

* Computes and prints:
  * **Precision** (`metrics.box.mp`)
  * **Recall** (`metrics.box.mr`)
  * **mAP50** (`metrics.box.map50`)
  * **mAP50-95** (`metrics.box.map`)

### 3. Testing on Sample Images (`image_test_2.py`)

Randomly picks up to 5 images from `dataset/test/images/` and visualizes detections:

```bash
python image_test_2.py
```

* **Controls**: Press any key to advance to the next image. Press **Q** to quit.

### 4. Real-time Webcam Object Detection (`realtime_test_2.py`)

Runs live object detection using camera index `0`:

```bash
python realtime_test_2.py
```

* **Controls**: Press **Q** to exit the live window.

## 🛠️ Requirements

* Python 3.8+
* PyTorch (with CUDA support for `device=0`)
* Ultralytics (`pip install ultralytics`)
* OpenCV (`pip install opencv-python`)
