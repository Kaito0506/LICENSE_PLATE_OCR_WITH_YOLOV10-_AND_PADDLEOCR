from ultralytics import YOLO
import multiprocessing

def train_model():
    # Load the model.
    model = YOLO('yolov10s.pt')

    # Training.
    results = model.train(
       data=r'D:\DATA_FOLDER\HOC_TAP\NLCN\DUNG\OCR\TRAIN_MODEL\Dataset\data.yaml',
       imgsz=640,
       epochs=50,
       batch=16,
       name='license_plate_detector')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_model()
