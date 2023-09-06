#
# Banafshe Bamdad
# Di Sep 05 07:40 CET
#
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from ultralytics import YOLO

source = '/media/banafshe/Banafshe_2TB/Datasets/TUM/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/10_frames'
contour_path = os.path.join(source, 'contours')
#os.makedirs(contour_path, exist_ok=True)

if os.path.exists(contour_path):
    shutil.rmtree(contour_path)

shutil.copytree(source, contour_path)

model_root = '/home/banafshe/YOLOv8/models/'
model = YOLO(model_root + 'segmentation/yolov8x-seg.pt')

results = model(source, stream=False, save=True, save_txt=True, save_conf=True, save_crop=True, show=True, classes=[0])

for result in results:
    path = result.path # The path to the image file.

    frame_name = os.path.basename(path)
    num_of_humans = result.masks.shape[0]
    for i in range(num_of_humans):
        contour = result.masks.xy[i]
        contour_txt = os.path.splitext(frame_name)[0] + "_H" + str(i + 1) + ".txt"
        contour_txt_path = os.path.join(contour_path, contour_txt)

        with open(contour_txt_path, "a") as file:
            for x, y in result.masks.xy[i]:
                file.write(f"{x}, {y}\n")

        # Draw the contour
        frame_copy = os.path.join(contour_path, frame_name)
        
        contour_img = cv2.imread(frame_copy)
        x_coords = []
        y_coords = []

        with open(contour_txt_path, 'r') as file:
            for line in file:
                values = line.strip().split(',')
                x_coords.append(float(values[0]))
                y_coords.append(float(values[1]))

        for x, y in zip(x_coords, y_coords):
            cv2.circle(contour_img, (int(x), int(y)), 1, (0, 255, 0), -1)

        full_path = os.path.join(contour_path, frame_name)
        cv2.imwrite(full_path, contour_img)





