from ultralytics import SAM

source = '/home/banafshe/YOLOv8/screenshot'
model_root = '/home/banafshe/YOLOv8/models/'

model = SAM(model_root + 'SAM/sam_b.pt')
model.info()

# Run inference with bboxes prompt
model(source, save=True, save_txt=True, show=True, bboxes=[439, 437, 524, 709])
#model(source, save=True, save_txt=True, show=True, points=[900, 370])
