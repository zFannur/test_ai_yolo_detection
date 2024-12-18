from ultralytics import YOLO
import cv2
import torch

print(torch.cuda.is_available())

model = YOLO('yolo11m-pose.pt')

cap = cv2.VideoCapture(2)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, conf=0.5)

        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv11", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
