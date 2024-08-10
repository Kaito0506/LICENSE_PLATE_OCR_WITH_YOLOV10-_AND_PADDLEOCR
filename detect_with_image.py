import cv2
# import support
from ultralytics import YOLO
from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy

image_path = r"images/plate2.jpg"

img = cv2.imread(image_path)
img = cv2.resize(img, (640, 480))


model = YOLO(r"models/new_model.pt")

result = model(img, show=False)
boxes = result[0].boxes
if len(boxes) >0 and boxes is not None:
        #### extract the bounding box of license plate ################
        box = boxes[0]
        top_left_x = int(box.xyxy.tolist()[0][0])
        top_left_y = int(box.xyxy.tolist()[0][1])
        bot_right_x = int(box.xyxy.tolist()[0][2])
        bot_right_y = int(box.xyxy.tolist()[0][3])
        ########### draw rectangle of license plate##########
        cv2.rectangle(img, (top_left_x, top_left_y), (bot_right_x, bot_right_y), (0, 0, 255), 2) 
        # crop the licenplate###
        cropped_img = img[top_left_y:bot_right_y, top_left_x:bot_right_x]
        
        ###################convert image to numpy array  ################################
        ocr = PaddleOCR(use_angle_cls=True, lang="en")
        cropped_numpy_image = numpy.array(cropped_img)
        cropped_numpy_image_rgb = cropped_numpy_image[:, :, ::-1].copy()
        ###### predict number ######
        readed_text = ocr.ocr(cropped_numpy_image_rgb, cls=True)
        #### define height to identify type of license plate
        height = bot_right_y - top_left_y
        width = bot_right_x - top_left_x
        print(f"Height: {height}, Width: {width}")
        if width/height>3:
                print(f"License Plate Number:\n{readed_text[0][0][1][0]}")
        else:
                print(f"License Plate Number:\n{readed_text[0][0][1][0]} \n{readed_text[0][1][1][0]} ")

        ##show image
        cv2.imshow("Licenplate", img)
        cv2.waitKey(7000)

        # print(f"License Plate Number:\n{readed_text[0][0][1][0]} \n{readed_text[0][1][1][0]} ")
        # print(readed_text)
