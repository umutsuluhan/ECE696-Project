# ECE 696 Project Source Code

# Dataset Conversion

- Download VisDrone dataset from: https://github.com/VisDrone/VisDrone-Dataset
- Run convert_yolo.py under Resnet_SNN_YOLO folder to convert dataset into YOLO format. Also adjust the paths according to your setup.

# Resnet/SNN Based YOLO Implementation

- Resnet_SNN_YOLO folder contains initial training pipeline code with two models, where backbone is Resnet or SNN and yolo detection heads are used

- File organization is as follows:
-- convert_yolo.py : Converts dataset to YOLO format.
-- dataset.py      : Pytorch dataset script used to load data for training and validation
-- knn_anchors.py  : Generates anchors using KNN clustering using training split
-- snn_model.py    : Contains SNN model data
-- snn_train.py    : Contains SNN training data
-- utils.py        : Contains helper functions

- Run train.py script for training and validation after converting dataset to YOLO format.


# SpikeYOLO/YOLO11x Training

- SpikeYOLO folder contains training code for SpikeYOLO and YOLO11x models
- Run train.py to train SpikeYOLO and yolo_train.py to train YOLO11x model