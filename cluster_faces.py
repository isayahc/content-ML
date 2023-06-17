import numpy as np
import os
from sklearn.cluster import DBSCAN
from keras_facenet import FaceNet
import cv2
from mtcnn import MTCNN


embedder = FaceNet()
detector = MTCNN()

def get_faces(image):
    result = detector.detect_faces(image)
    
    # Extract bounding box from the first face
    faces = []
    for i in range(len(result)):
        bounding_box = result[i]['box']
        faces.append(image[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]])
    return faces

def cluster_faces(faces):
    embeddings = embedder.embeddings(faces)
    
    clustering_model = DBSCAN(metric='euclidean')
    clustering_model.fit(embeddings)
    
    unique_faces_indices = np.unique(clustering_model.labels_, return_index=True)[1]
    unique_faces = [faces[index] for index in sorted(unique_faces_indices)]
    
    return unique_faces

def process_directory(directory_path):
    faces = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Add/modify the image formats as per your needs
            img = cv2.imread(os.path.join(directory_path, filename))
            if img is not None:
                faces.extend(get_faces(img))
    
    unique_faces = cluster_faces(faces)
    return unique_faces

# Call the function on a directory
unique_faces = process_directory('/home/isayahc/projects/machine_learning/facial_recognition/deepface_inference/assets/output_frames')