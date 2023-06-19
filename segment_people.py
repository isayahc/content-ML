# Import modules
import cv2
import torch
import os
from tqdm import tqdm

# Load model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Check video file
video_path = 'video.mp4'
if not os.path.isfile(video_path):
    raise FileNotFoundError(f"The video file {video_path} does not exist.")

cap = cv2.VideoCapture(video_path)

# Check output directory
output_dir = "./assets/output_frames"
if not os.access(output_dir, os.W_OK):
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        raise PermissionError(f"Cannot write to the directory {output_dir}. Details: {str(e)}")

total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_skip_threshold = 25
frame_counter = 0
person_counter = 0

for i in tqdm(range(total_frame_count)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if ret:
        frame_counter += 1
        if frame_counter >= frame_skip_threshold:
            results = model(frame)
            person_detections = [x for x in results.xyxy[0] if x[-1] == 0.0]
            if person_detections:
                for *box, _, _, class_id in person_detections:
                    if len(box) == 4:
                        # Extract ROI
                        roi = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                        filename = os.path.join(output_dir, f"person_{person_counter}.jpg")
                        try:
                            # Save ROI as image
                            cv2.imwrite(filename, roi)
                            if os.path.isfile(filename):
                                print(f'Saved image {filename}')
                                person_counter += 1
                            else:
                                print(f'Failed to save image {filename}')
                        except Exception as e:
                            print(f'Error writing file {filename}: {str(e)}')
            frame_counter = 0
    else:
        print(f'Read invalid frame {i}')
        break

cap.release()
