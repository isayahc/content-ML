from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolov7.pt")  # load a pretrained model (recommended for training)


# Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model

import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# results = model('https://ultralytics.com/images/zidane.jpg')
results = model("https://ultralytics.com/videos/bus.mp4")
results.print()

metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
results = model("https://ultralytics.com/videos/bus.mp4")
path = model.export(format="onnx")  # export the model to ONNX format

