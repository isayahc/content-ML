import cv2
import torch
from tqdm import tqdm

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture('video.mp4') # open video file
total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # get total number of frames
for i in tqdm(range(total_frame_count)): # loop over frames with tqdm
    cap.set(cv2.CAP_PROP_POS_FRAMES, i) # move to i-th frame
    ret, frame = cap.read() # read frame
    if ret: # if frame is valid
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) # convert to RGB (remove this line)
        results = model(frame) # pass to model
        results.print() # print results
    else: # if frame is invalid
        break # exit loop

cap.release() # release video