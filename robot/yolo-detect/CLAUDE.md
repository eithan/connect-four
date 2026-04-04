# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

This is a backup of YOLO object detection training outputs for a Connect Four board piece detector. It contains no runnable source code — only training artifacts from two YOLO runs (`detect/train` and `detect/val`).

## Contents

- `detect/train/weights/best.pt` — best model checkpoint from training
- `detect/train/weights/last.pt` — last model checkpoint from training
- `detect/train/args.yaml` — full training configuration (model: `yolo26s.pt`, dataset: `Connect4-1`, 100 epochs, imgsz 640, batch 16)
- `detect/train/results.csv` — per-epoch metrics
- Various `.png` files — F1/P/R/PR curves, confusion matrices, label and prediction visualizations

## Training Configuration (from args.yaml)

- **Model**: `yolo26s.pt` (pretrained)
- **Dataset**: `/content/datasets/Connect4-1/data.yaml` (Roboflow-style dataset, trained in Colab)
- **Task**: object detection
- **Epochs**: 100, patience 100
- **Image size**: 640, batch 16
- **Optimizer**: auto, lr0=0.01, lrf=0.01
- **Augmentation**: mosaic, randaugment, fliplr=0.5, erasing=0.4

## Using the Trained Weights

To run inference with the saved weights using Ultralytics:

```python
from ultralytics import YOLO
model = YOLO("detect/train/weights/best.pt")
results = model.predict(source="your_image.jpg", imgsz=640)
```
