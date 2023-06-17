import cv2
import torch
import os
from tqdm import tqdm

# set flag
store_only_people = True  # set this flag to False if you want to store images without people

# load model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# create directory to store frames if not exist
output_dir = "./assets/output_frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture('video.mp4') # open video file
total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # get total number of frames

frame_skip_threshold = 5  # set this to the number of frames you want to skip
frame_counter = 0

for i in tqdm(range(total_frame_count)): # loop over frames with tqdm
    cap.set(cv2.CAP_PROP_POS_FRAMES, i) # move to i-th frame
    ret, frame = cap.read() # read frame
    if ret: # if frame is valid
        frame_counter += 1
        if frame_counter >= frame_skip_threshold:  # process frame only if counter has reached threshold
            results = model(frame) # pass to model
            # Select only person class detections
            person_detections = [x for x in results.xyxy[0] if x[-1] == 0.0]
            if person_detections: # Check if detections are not empty
                for *box, _, _, class_id in person_detections:
                    # check if bounding box has all the four coordinates
                    if len(box) == 4:
                        # draw the bounding box on the frame
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                if store_only_people:
                    # save the frame with people as image file
                    cv2.imwrite(os.path.join(output_dir, f"frame_{i}.jpg"), frame)
            else: # if no person detected
                if not store_only_people:
                    # save the frame without people as image file
                    cv2.imwrite(os.path.join(output_dir, f"frame_{i}.jpg"), frame)
            frame_counter = 0  # reset counter
    else: # if frame is invalid
        break # exit loop

cap.release() # release video
