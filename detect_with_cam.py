import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np

# Load YOLO model
model = YOLO(r"models/new_model.pt")

# Initialize video capture
# cap = cv2.VideoCapture('images/video2.mp4')
cap = cv2.VideoCapture(0)
# Set desired frame width and height
desired_width = 640
desired_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Initialize PaddleOCR once
ocr = PaddleOCR(use_angle_cls=True, lang="en")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    if not ret:
        break

    # Perform detection
    result = model.predict(frame, show=False)
    boxes = result[0].boxes

    if len(boxes) > 0:
        box = boxes[0]
        top_left_x = int(box.xyxy.tolist()[0][0])
        top_left_y = int(box.xyxy.tolist()[0][1])
        bot_right_x = int(box.xyxy.tolist()[0][2])
        bot_right_y = int(box.xyxy.tolist()[0][3])
        height = bot_right_y - top_left_y
        width = bot_right_x - top_left_x

        # Draw rectangle and crop image
        cv2.rectangle(frame, (top_left_x, top_left_y), (bot_right_x, bot_right_y), (0, 0, 255), 2)
        cropped_img = frame[top_left_y:bot_right_y, top_left_x:bot_right_x]

        # Convert image for OCR
        cropped_numpy_image_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

        # Perform OCR
        readed_text = ocr.ocr(cropped_numpy_image_rgb, cls=True)

        # Check if OCR detected text
        try:
            if width/height>3:
                print(f"License Plate Number:\n{readed_text[0][0][1][0]}")
                cv2.putText(frame, readed_text[0][0][1][0], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            else:
                print(f"License Plate Number:\n{readed_text[0][0][1][0]} \n{readed_text[0][1][1][0]} ")
                cv2.putText(frame, readed_text[0][0][1][0], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, readed_text[0][1][1][0], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        except (TypeError, IndexError):
            print("OCR returned None")
            continue

        # Display cropped image
        # cv2.imshow("plate", cropped_img)

    # Display the frame
    cv2.imshow("frame", frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
