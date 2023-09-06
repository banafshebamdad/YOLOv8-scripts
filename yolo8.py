#
# Banafshe Bamdad
# Di Sep 05 07:40 CET
#
import cv2
import datetime
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

image_width = 1256
image_height = 1257
img = Image.new("RGB", (image_width, image_height), "white")
draw = ImageDraw.Draw(img)
point_radius = 1

#ultralytics.checks()


#source = '/home/banafshe/YOLOv8/images/banafshe/'
source = '/media/banafshe/Banafshe_2TB/Datasets/TUM/Dynamic_Objects/rgbd_dataset_freiburg3_walking_xyz/rgb'
model_root = '/home/banafshe/YOLOv8/models/'
#model = YOLO(model_root + 'detection/yolov8x.pt')
model = YOLO(model_root + 'segmentation/yolov8x-seg.pt')

print("\n--- Model Info:")
model.info()
print("\n")

# Arguments: https://docs.ultralytics.com/modes/predict/#inference-arguments
#results = model(source, stream=False, save=True, save_txt=True, save_conf=True, save_crop=True, show=True, imgsz=(848, 480), conf=0.5, device='cpu', show_labels=True, show_conf=True, max_det=300, line_width=1, visualize=True, classes=[2, 26], boxes=True)
results = model(source, stream=False, save=True, save_txt=True, save_conf=True, save_crop=True, show=True, classes=[0])

# Results methods: https://docs.ultralytics.com/reference/engine/results/
# Results object attributes
# Attributs: https://docs.ultralytics.com/modes/predict/#inference-arguments
for result in results:
    orig_img = result.orig_img # The original image as a numpy array
    orig_shape = result.orig_shape # The original image shape in (height, width) format.
    boxes = result.boxes # A Boxes object containing the detection bounding boxes.
    masks = result.masks # A Masks object containing the detection masks.
    probs = result.probs # A Probs object containing probabilities of each class for classification task.
    keypoints = result.keypoints # A Keypoints object containing detected keypoints for each object.
    speed = result.speed # A dictionary of preprocess, inference, and postprocess speeds in milliseconds per image.
    names = result.names # A dictionary of class names.
    path = result.path # The path to the image file.

    #cur_date = datetime.datetime.now()
    #timestamp = cur_date.timestamp()

    #im_array = result.plot()  # plot a BGR numpy array of predictions
    #im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    #im.show()
    #im.save(str(timestamp) + '.png')
    #print("\n --- boxes.xyxy: ", boxes.xyxy, " --- ")
    #print(" --- masks: ", result.masks.data)
    #print(" --- masks.xy: ", result.masks.xy)
    #print(" --- masks.xy.size: ", len(result.masks.xy[0]))
    #print(" --- masks.xy.shape : ", (result.masks.xy[0]).shape)

    # Test if a specific point is inside or outside of a contour
    contour = result.masks.xy[0]
    my_x, my_y = 618, 212
    contour = np.array(contour, dtype=np.int32)
    point = (my_x, my_y)
    distance = cv2.pointPolygonTest(contour, point, measureDist=True)
    if distance > 0:
        print(f"Point ({my_x}, {my_y}) is inside the contour.")
    elif distance < 0:
        print(f"Point ({my_x}, {my_y}) is outside the contour.")
    else:
        print(f"Point ({my_x}, {my_y}) is on the contour.")


    for x, y in result.masks.xy[0]:
        x_int, y_int = int(x), int(y)
        draw.rectangle( [(x_int - point_radius, y_int - point_radius), (x_int + point_radius, y_int + point_radius),], fill="black", )
    print("\n --- masks.shape", masks.shape)

img.save(source + "point_image.png")
img.show()
# 
# Related links
#
# COCO dataset classes: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml


"""
COCO's Classes names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
"""
