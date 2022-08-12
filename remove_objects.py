import cv2
import numpy as np

file_path = '/home/dzhura/Documents/Computer Vision/Video/Pexels Videos 2053100.mp4'
cap = cv2.VideoCapture(file_path)
first_iter = True
result1 = None

while True:
    ret, frame = cap.read()
    if frame is None:
        break

    if first_iter:
       avg = np.float32(frame)
       first_iter = False

    cv2.accumulateWeighted(frame, avg, 0.005)
    result1 = cv2.convertScaleAbs(avg)

cv2.imwrite('/home/dzhura/Code/Yolov5_StrongSORT_OSNet/runs/test_frame_empty.png', result1)

