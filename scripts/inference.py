
# scripts/inference.py
# scripts/inference_video.py
import os
import torch
from torchvision import transforms as T
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
from dataset import HumanAnimalDataset  # for num_classes info if needed

# ---------------- Paths ---------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DETECTOR_PATH = "/content/drive/MyDrive/human_animal_detection/models/detector_mobilenet.pth"
CLASSIFIER_PATH = "/content/drive/MyDrive/human_animal_detection/models/classifier_resnet18.pth"
VIDEO_PATH = "/content/drive/MyDrive/human_animal_detection/test_video/video.mp4"
OUTPUT_PATH = "/content/drive/MyDrive/human_animal_detection/test_video/output.mp4"

# ---------------- Load Detector ---------------- #
dataset_info = HumanAnimalDataset("/content/drive/MyDrive/human_animal_detection/datasets/train")
num_classes = len(dataset_info.cat2label) + 1  # background + all categories

detector = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
in_features = detector.roi_heads.box_predictor.cls_score.in_features
detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

detector.load_state_dict(torch.load(DETECTOR_PATH, map_location=DEVICE))
detector.to(DEVICE)
detector.eval()

# ---------------- Load Classifier ---------------- #
classifier = models.resnet18(weights=None)
classifier.fc = torch.nn.Linear(classifier.fc.in_features, 2)  # 0: human, 1: animal
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
classifier.to(DEVICE)
classifier.eval()

# ---------------- Video Setup ---------------- #
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

transform = T.ToTensor()

# ---------------- Inference Loop ---------------- #
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_tensor = transform(frame).to(DEVICE)

    with torch.no_grad():
        outputs = detector([img_tensor])

    boxes = outputs[0]['boxes']
    scores = outputs[0]['scores']

    for box, score in zip(boxes, scores):
        if score < 0.5:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            continue
        crop_tensor = transform(crop).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = classifier(crop_tensor).argmax(dim=1).item()
        class_name = "Human" if pred == 0 else "Animal"
        cv2.putText(frame, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    out.write(frame)

cap.release()
out.release()
print("Output video saved to:", OUTPUT_PATH)

