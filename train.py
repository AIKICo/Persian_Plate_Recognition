from ultralytics import YOLO
import gc
import torch

# Load a model
model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="dataset", epochs=100, imgsz=640, batch=-1,cache=False, verbose=True,name='my_model')
