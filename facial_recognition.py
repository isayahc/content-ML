!pip install pytube opencv-python face_recognition tqdm insightface onnxruntime-gpu
!pip install numpy==1.23 # as per https://github.com/deepinsight/insightface/issues/2251

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


def identify_unique_persons(input_video_path, output_video_path, output_directory, similarity_threshold):
    # Load the pre-trained model for face detection
    model_detection = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    model_detection.prepare(ctx_id=0, det_size=(640, 640))

    # Load the pre-trained model for face recognition
    model_recognition = insightface.model_zoo.get_model('arcface_r100_v1')

    # Load the face recognition model parameters
    model_recognition.prepare(ctx_id=0)

    # Download the input video
    youtube = YouTube(input_video_path)
    video = youtube.streams.filter(adaptive=True, file_extension='mp4').first()
    video.download(output_directory)
    input_video_file = os.path.join(output_directory, video.default_filename)

    # Open the input video
    cap = cv2.VideoCapture(input_video_file)

    # Get the video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Initialize variables for tracking unique persons
    unique_persons = {}
    next_person_id = 0

    # List to store detected person IDs
    person_ids = []

    # List to store frames
    frames = []

    for _ in tqdm(range(total_frames), desc='Processing frames', unit='frame'):
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

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
            embeddings = model_recognition.get_embedding(face_img)
            # embeddings = model_recognition.get(face_img)

            # Check if the face belongs to a known person
            matched = False
            for person_id, person_embeddings in unique_persons.items():
                # Calculate the distance between the embeddings
                distance = np.linalg.norm(person_embeddings - embeddings)

                # If the distance is below a threshold, consider it a match
                if distance < 0.6:
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

    # Write the frames to the output video file
    for frame in frames:
        out.write(frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()

    # Remove the downloaded input video file
    os.remove(input_video_file)

    # Return the output directory path
    return output_directory


# Example usage
output_directory = "/content/output_frames0"
input_video_path = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
output_video_path = "/content/output_video.mp4"
similarity_threshold = 0.9
output_directory = identify_unique_persons(input_video_path, output_video_path, output_directory, similarity_threshold)

# Zip the output directory
shutil.make_archive(output_directory, 'zip', output_directory)

# Print the output directory path
print("Output directory path:", output_directory)

# Download the zip file
from google.colab import files
files.download(output_directory + ".zip")
