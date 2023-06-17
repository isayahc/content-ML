from deepface import DeepFace

import cv2
import os


img_path_1 = 'assets/output_frames/frame_299.jpg'
img_path_2 = 'assets/output_frames/frame_99.jpg'



os.environ['CUDA_VISIBLE_DEVICES'] = '' # hack way to force using CPU
from deepface import DeepFace

# Set the enforce_detection parameter to False to use the CPU
result = DeepFace.verify(img1_path=img_path_1, img2_path=img_path_2, enforce_detection=False)
x=0