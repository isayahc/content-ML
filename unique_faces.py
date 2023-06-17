import os
import glob
from tqdm import tqdm
import urllib.request
import cv2
import numpy as np
import os
import insightface
import shutil
from pytube import YouTube
from tqdm import tqdm

def calculate_frame_similarity(frame1, frame2):
    # Calculate the structural similarity index (SSIM) between two frames
    ssim = cv2.compareSSIM(frame1, frame2, multichannel=True)

    return ssim


def remove_similar_frames(frames, threshold):
    # Initialize a list to store unique frames
    unique_frames = []

    # Compare each frame with the previous frame
    for frame in frames:
        if not unique_frames or calculate_frame_similarity(frame, unique_frames[-1]) < threshold:
            unique_frames.append(frame)

    return unique_frames

def identify_unique_persons(input_image_dir, output_directory, similarity_threshold):
    # Load the pre-trained model for face detection
    model_detection = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    model_detection.prepare(ctx_id=0, det_size=(640, 640))

    # Load the pre-trained model for face recognition
    # model_recognition = insightface.model_zoo.get_model('arcface_r100_v1')
    # model_recognition = insightface.model_zoo.get_model('arcface_r100_v1', download=True, )
    model_recognition = insightface.model_zoo.get_model('/home/isayahc/projects/machine_learning/facial_recognition/deepface_inference/models/antelopev2/glintr100.onnx')

    # Load the face recognition model parameters
    model_recognition.prepare(ctx_id=0)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Initialize variables for tracking unique persons
    unique_persons = {}
    next_person_id = 0

    # List to store detected person IDs
    person_ids = []

    # List to store frames
    frames = []

    # List of all image files in input directory
    image_files = glob.glob(os.path.join(input_image_dir, '*'))

    for image_file in tqdm(image_files, desc='Processing images', unit='image'):
        # Read an image from the directory
        frame = cv2.imread(image_file)

        # Convert the frame to RGB color space
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform face detection
        faces = model_detection.get(frame_rgb)

        # Process each detected face
        for face in faces:
            # Extract the bounding box coordinates
            bbox = face.bbox.astype(int)

            # Crop the face region from the original frame
            face_img = frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            # Perform face recognition
            # embeddings = model_recognition.get_embedding(face_img)
            embeddings = model_recognition.get(face_img)
            face_embedding = model.get(detected_face, img)


            # Check if the face belongs to a known person
            matched = False
            for person_id, person_embeddings in unique_persons.items():
                # Calculate the distance between the embeddings
                distance = np.linalg.norm(person_embeddings - embeddings)

                # If the distance is below a threshold, consider it a match
                if distance < similarity_threshold:
                    matched = True
                    break

            if not matched:
                # Assign a new person ID
                person_id = next_person_id
                next_person_id += 1

                # Add the person's embeddings to the dictionary
                unique_persons[person_id] = embeddings

            # Draw the bounding box and person ID on the frame
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, str(person_id), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the frame in the frames list
        frames.append(frame)

    # Remove similar frames based on the threshold
    frames = remove_similar_frames(frames, similarity_threshold)

    # Save the frames as images in the output directory
    for idx, frame in enumerate(frames):
        frame_filename = f"frame_{idx + 1:05d}.jpg"
        frame_path = os.path.join(output_directory, frame_filename)
        cv2.imwrite(frame_path, frame)

    # Return the output directory path
    return output_directory

# Example usage
# output_directory = "/content/output_frames0"
# input_video_path = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
# output_video_path = "/content/output_video.mp4"

input_image_dir = "./assets/output_frames"
output_directory= "./assets/output_frames0"

similarity_threshold = 0.9
# output_directory = identify_unique_persons(input_video_path, output_video_path, output_directory, similarity_threshold)

output_directory = identify_unique_persons(input_image_dir,output_directory,similarity_threshold)

