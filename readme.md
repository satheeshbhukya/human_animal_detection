# Human and Animal Detection & Classification System

## Objective

The goal of this project is to design a computer vision system that **detects and classifies humans and animals in images and videos** using a **two-stage deep learning pipeline**:

1. **Object Detection Model** – localizes objects in an image or video frame
2. **Classification Model** – classifies each detected object as **Human** or **Animal**

This separation improves modularity, interpretability, and flexibility. 

## Clone the Repository

```
git clone https://github.com/<your-username>/human_animal_detection.git
cd human_animal_detection
```

---

## Install Dependencies

Make sure Python 3.8+ is installed.

```
pip install -r requirements.txt
```

---

## Repository Structure

```
human_animal_detection/
│
├── models/
│   ├── detector_mobilenet.pth
│   └── classifier_resnet18.pth
│
├── scripts/
│   ├── dataset.py               # Detection dataset loader
│   ├── dataset_cls.py           # Classification dataset loader
│   ├── train_detector.py
│   ├── train_classifier.py
│   └── inference_video.py
│
├── Notebook/
│   └── Notebook.ipynb           # Main notebook (runs all scripts in Colab)
│
├── test_video/
│   ├── video.mp4                # Input video
│   └── output.mp4               # Output annotated video
│
├── requirements.txt
└── README.md
```

---

## Dataset

### Dataset Description

* **Dataset Name:** Google Open Images Dataset V7
* **Total Images Used:** ~5,000
* **Annotation Format:** COCO (converted from Open Images annotations)
* **Classes:** Humans and multiple animal categories

The **Google Open Images V7 dataset** is a large-scale, publicly available dataset containing high-quality bounding box annotations collected from real-world images.

---

---

### Dataset Importance and Justification

The **Google Open Images V7 dataset** was selected for the following reasons:

* **Avoids restricted datasets**
  The project explicitly avoids widely used datasets such as **COCO** and **ImageNet**, as required. Open Images V7 provides a strong alternative while remaining research-acceptable.

* **High-quality bounding box annotations**
  Accurate bounding boxes are essential for training robust object detection models.

* **Diverse and realistic data**
  Images include different environments, lighting conditions, object scales, and backgrounds, improving generalization.

* **Rich category coverage**
  Includes multiple human-related and animal-related categories, making it suitable for both detection and classification tasks.

* **Efficient reuse of annotations**
  The same dataset is reused for:

  * Object detection (localizing objects)
  * Binary classification (Human vs Animal)

---

### Dataset Usage in This Project

#### 1. Object Detection

* Full COCO annotations are used to train the detector.
* Each object instance is localized with bounding boxes.

#### 2. Classification (Human vs Animal)

* Category names are grouped using keyword-based logic:

  * **Human:** person, man, woman, boy, girl
  * **Animal:** dog, cat, horse, bird, elephant, etc.
* Each image is assigned a binary label:

  * `0 → Human`
  * `1 → Animal`

---

## Models

### 1. Object Detection Model

**Architecture:** Faster R-CNN with MobileNetV3 + FPN
**Framework:** TorchVision

#### Model Importance and Selection Rationale

* **Faster R-CNN** provides strong localization accuracy and is well-suited for precise object detection tasks.
* **MobileNetV3 backbone** is lightweight and computationally efficient, making the model faster and more scalable.
* **Feature Pyramid Network (FPN)** improves detection of objects at multiple scales.
* **YOLO was intentionally not used**, as per project requirements.

**Output:** Bounding boxes for detected objects.

---

### 2. Classification Model

**Architecture:** ResNet18
**Classes:**

* `0 → Human`
* `1 → Animal`

#### Model Importance and Selection Rationale

* **ResNet18** is a well-established convolutional neural network with residual connections that improve training stability.
* Lightweight architecture enables fast training and inference.
* Strong generalization performance even with limited data.
* Easy integration with the detection pipeline for cropped object classification.

**Input:** Cropped detected regions
**Output:** Human or Animal label

---

## Training

### Preprocessing

* Image resizing and normalization
* COCO annotations parsed for bounding boxes
* Keyword-based category grouping for classification
* Custom PyTorch `Dataset` classes for loading data

---

### Detector Training

* **Optimizer:** SGD
* **Loss:** Faster R-CNN multi-task loss
* **Epochs:** 2

**Training Results:**

```
Epoch 1/2, Loss: 1.6379
Epoch 2/2, Loss: 1.3786
```

Model saved to:

```
models/detector_mobilenet.pth
```

---

### Classifier Training

* **Optimizer:** Adam
* **Loss:** CrossEntropyLoss
* **Epochs:** 2

**Training Results:**

```
Epoch 1/2 | Loss: 0.2059 | Acc: 92.22%
Epoch 2/2 | Loss: 0.0357 | Acc: 99.07%
```

Model saved to:

```
models/classifier_resnet18.pth
```

---

## Inference Pipeline

### Video Processing

* Place input videos in:

```
test_video/
```

**Pipeline Steps:**

1. Read video frame-by-frame
2. Detect objects using Faster R-CNN
3. Crop detected regions
4. Classify each crop as Human or Animal
5. Draw bounding boxes and labels
6. Save annotated video

### Output

```
test_video/output.mp4
```

---

## How to Run

### Option 1: Google Colab (Recommended)

Open and run:

```
Notebook/Notebook.ipynb
```

This notebook executes all training and inference scripts.

---

### Option 2: Local Execution

Install dependencies:

```
pip install -r requirements.txt
```

Train detector:

```
!python scripts/train_detector.py
```

Train classifier:

```
!python scripts/train_classifier.py
```

Run inference:

```
!python scripts/inference_video.py
```

---

## Results Summary

| Model      | Metric              | Value      |
| ---------- | ------------------- | ---------- |
| Detector   | Final Training Loss | **1.3786** |
| Classifier | Accuracy            | **99.07%** |

---

## Conclusion

This project demonstrates a **robust and modular two-stage computer vision system** that:

* Uses a high-quality, non-restricted dataset
* Separates detection and classification for clarity
* Avoids YOLO as required
* Supports automated video inference
* Produces accurate and interpretable results

---

## Future Improvements

* Multi-class animal classification
* Temporal object tracking
* Real-time webcam inference
* Model optimization for edge devices

---

## Author

**Satheesh Bhukya**

---

## License

This project is intended for academic and research purposes.
