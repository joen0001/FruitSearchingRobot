import torch

# Model
ckpt = 'network/scripts/model/best100.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom',path = ckpt)

# Image
im = 'lab_output/pred_0.png'

# Inference
results = model(im)

print(results.pandas().xyxy[0])