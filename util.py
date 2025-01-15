import cv2

from config import charclassnames


# Function to detect plates
def detect_plates(image, model):
    results = model(image)
    return results

# Function to detect characters in a plate region
def detect_characters(plate_region, model):
    plate_region = cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB)
    results = model(plate_region)
    detected_classes = []

    if results and results[0].boxes is not None:
        for box in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            detected_classes.append((int(cls), int(x1)))

    # Sort detected characters
    detected_classes.sort(key=lambda x: x[1])
    detected_characters = ''.join([charclassnames[cls] for cls, _ in detected_classes if cls < len(charclassnames)])
    
    return detected_characters
#     cap.release()
#     out.release()
