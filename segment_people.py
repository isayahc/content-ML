# Import modules
import cv2
import torch
import os
from tqdm import tqdm
from typing import Dict, List
import numpy as np

def save_roi_images(model: torch.nn.Module, video_path: str, output_dir: str, frame_skip_threshold: int = 25) -> None:
    """
    This function performs person detection on a given video file using the provided model,
    then saves regions of interest (ROI) as images in the specified output directory.

    :param model: PyTorch model for person detection.
    :param video_path: Path to the input video file.
    :param output_dir: Path to the output directory where ROI images will be saved.
    :param frame_skip_threshold: The number of frames to skip between detections.
    """

    # check video file
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"The video file {video_path} does not exist.")

    cap = cv2.VideoCapture(video_path)

    # check output directory
    if not os.access(output_dir, os.W_OK):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            raise PermissionError(f"Cannot write to the directory {output_dir}. Details: {str(e)}")

    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
                    for detection in person_detections:
                        box = detection[:4]
                        box = [int(i) for i in box]
                        roi = frame[box[1]:box[3], box[0]:box[2]]
                        cv2.imwrite(os.path.join(output_dir, f"person_{person_counter}.jpg"), roi)
                        person_counter += 1
                frame_counter = 0
        else:
            break

    cap.release()

def get_roi_dict(model: torch.nn.Module, video_path: str, frame_skip_threshold: int = 25) -> Dict[str, List[np.array]]:
    """
    This function performs person detection on a given video file using the provided model,
    then returns a dictionary where for every frame it stores a list of ROI values.

    :param model: PyTorch model for person detection.
    :param video_path: Path to the input video file.
    :param frame_skip_threshold: The number of frames to skip between detections.
    :return: A dictionary where the keys are 'frame_n' (n is the frame number)
             and the values are lists of ROI values.
    """

    # check video file
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"The video file {video_path} does not exist.")

    cap = cv2.VideoCapture(video_path)

    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_counter = 0

    roi_dict = {}

    for i in tqdm(range(total_frame_count)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_counter += 1
            if frame_counter >= frame_skip_threshold:
                results = model(frame)
                person_detections = [x for x in results.xyxy[0] if x[-1] == 0.0]
                roi_list = []
                if person_detections:
                    for detection in person_detections:
                        box = detection[:4]
                        box = [int(i) for i in box]
                        roi = frame[box[1]:box[3], box[0]:box[2]]
                        roi_list.append(roi)
                roi_dict[f'frame_{i}'] = roi_list
                frame_counter = 0
        else:
            break

    cap.release()

    return roi_dict

def main() -> None:
    """
    The main function loads the model and performs person detection on the input video file,
    then either saves the ROI as images or gets a dictionary of ROI values for each frame in the video.
    """

    # load model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # video file
    video_path = 'video.mp4'
    # output directory
    output_dir = "./assets/output_frames"

    # check if output_dir exists, if not, create it.
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # function to perform person detection and save the ROIs as images
    # save_roi_images(model, video_path, output_dir)

    # function to return a dictionary of ROI values for each frame in a given video
    roi_dict = get_roi_dict(model, video_path)

    # print some information about the detected ROIs
    for key in roi_dict.keys():
        print(f"{key}: {len(roi_dict[key])} people detected")


if __name__ == "__main__":
    main()
