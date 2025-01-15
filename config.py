from ultralytics import YOLO


def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model


charclassnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'AIN', 'B', 'D', 'Disabled', 'F', 'G', 'GH', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'S', 'SAD', 'SH', 'T', 'TA', 'TH', 'V', 'Y', 'Z']
