from ultralytics import YOLO


def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model


charclassnames = ['0','9','b','d','ein','ein','g','gh','h','n','s','1','malul','n','s','sad','t','ta','v','y','2'
                  ,'3','4','5','6','7','8']
