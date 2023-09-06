import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import ultralytics

from pathlib import Path
from ultralytics import YOLO
#ultralytics.checks()

image_folder = '/home/banafshe/YOLOv8'

model = YOLO('yolov8n.pt')

print("\n--- Model Info. ---\n")
model.info()
print("\n--- End of model Info. ---\n")
for image_file in Path(image_folder).glob('*.png'):
    image = cv2.imread(str(image_file))
    results = model(image, show=True, save=True)

    #results = model(['2person.png', 'back.png', 'blur.png', 'sit.png'])

    for result in results:
        boxes = result.boxes
        masks = result.masks
        keypoints = result.keypoints
        probs = result.probs

        img_with_boxes = image.copy()
        for box in boxes:
            # print("Shape of 'boxes' tensor:", box.shape)
            # print("box.data--->", box.data, "<---")
            x1 = box.data[0, 0].item()
            y1 = box.data[0, 1].item()
            x2 = box.data[0, 2].item()
            y2 = box.data[0, 3].item()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        plt.title("YOLOv8 Person Detection")
        plt.axis('off')
        plt.show()

model.model.float()
model.model.cpu()
torch.cuda.empty_cache()


