Civil Infrastructure Defect Detection through Computer Vision and Image Recognition

This project presents a deep learning–based approach for automated civil infrastructure defect detection and analysis using computer vision and image recognition.

The project investigates three complementary approaches to defect detection: Image Classification, Pixel-Level Segmentation, and Object Detection. Each approach provides a different level of information about infrastructure defects, from identifying the defect category to precisely locating the defect at the pixel and object levels.

Image Classification

For image-level defect classification, ResNet50 is used to classify infrastructure images into predefined defect categories.

Model: ResNet50
Library: Torchvision
Dataset: 40,000 images
Number of Classes: 2
Annotation Type: Image-level labels
Output: Defect class
Evaluation Metrics

The classification model is evaluated using:

Accuracy
Precision
Recall
F1 Score
Pixel-Level Segmentation

For precise delineation of defect regions, a U-Net architecture with a ResNet34 encoder is used. The model performs pixel-level segmentation by identifying the pixels corresponding to the defect and generating a pixel-level defect mask.

Model: U-Net with ResNet34 Encoder
Framework: PyTorch
Dataset: 10,995 images
Number of Classes: 1
Annotation Type: Pixel-level masks
Output: Pixel-level defect mask
Evaluation Metrics

The segmentation model is evaluated using:

Intersection over Union (IoU)
Dice Score
Precision
Recall
Object Detection

For object-level defect detection, YOLOv8n, the nano variant of YOLOv8, is used. The model detects individual defects within an image and provides their location, predicted class, and confidence score.

Model: YOLOv8n
Library: Ultralytics
Dataset: 14,411 images
Number of Classes: 13
Annotation Type: Bounding boxes
Output: Bounding box + Class + Confidence
Evaluation Metrics

The object detection model is evaluated using:

Precision
Recall
mAP@0.5
mAP@0.5:0.95
Overall Detection Framework

The three approaches provide complementary forms of infrastructure defect analysis:

Image Classification determines what type of defect is present in an image.

Pixel-Level Segmentation determines which pixels correspond to the defect, providing a precise representation of the defect region.

Object Detection determines where the defect is located, what type of defect it is, and how confident the model is in its prediction.

Together, these approaches provide a multi-level computer vision framework for civil infrastructure defect analysis, covering image-level recognition, pixel-level delineation, and object-level localization.

The code consists of 7 parts each for all of the three models. The parts are :

cofig.py
dataset.py
model.py
train.py
evaluate.py
visualization.py
predict.py
