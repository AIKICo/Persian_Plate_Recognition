import cv2

from config import load_yolo_model
from util import detect_plates, detect_characters

# Load models
plate_model_path = "weights/best.pt"
ocr_model_path = "weights/yolov8n_char_new.pt"
plate_model = load_yolo_model(plate_model_path)
ocr_model = load_yolo_model(ocr_model_path)


def draw_text_with_background(image, text, position, font_scale=0.5, font_thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    x, y = position
    rectangle_pt1 = (x, y - text_size[1] - 5)
    rectangle_pt2 = (x + text_size[0] + 2, y)
    cv2.rectangle(image, rectangle_pt1, rectangle_pt2, (0, 0, 0), thickness=cv2.FILLED)
    cv2.putText(image, text, (x, y - 2), font, font_scale, (255, 255, 255), font_thickness)


if __name__ == '__main__':
    videoCap = cv2.VideoCapture(0)
    while True:
        ret, frame = videoCap.read()
        if not ret:
            continue
        cv2.imshow('frame', frame)
        results = detect_plates(frame, plate_model)

        for result in results:
            if result.boxes is not None:
                for box in result.boxes.data:
                    x1, y1, x2, y2, conf, cls = box.tolist()
                    plate_region = frame[int(y1):int(y2), int(x1):int(x2)]
                    detected_characters = detect_characters(plate_region, ocr_model)

                    # Draw bounding box and text with background for each plate
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    draw_text_with_background(frame, detected_characters, (int(x1), int(y1) - 10))
                    print(f'License Card Plate Detection: {detected_characters}')

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            videoCap.release()
            break
