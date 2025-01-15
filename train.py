from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/home/mohammad/MySourceCodes/python/Persian_Plate_Recognition/dataset", epochs=100, imgsz=640)