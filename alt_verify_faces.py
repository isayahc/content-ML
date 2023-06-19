import os
import shutil
from deepface import DeepFace
from tqdm import tqdm

# Set the enforce_detection parameter to False to use the CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def process_directory(directory_path):
    image_paths = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if filename.endswith(".jpg") or filename.endswith(".png")]
    num_images = len(image_paths)

    # We use a simple list of lists as a clustering mechanism. Each sub-list represents a unique face.
    clusters = []

    # Compare every pair of images
    for i in tqdm(range(num_images), desc="Processing images", unit="image"):
        img_path_1 = image_paths[i]

        for j in range(i + 1, num_images):
            img_path_2 = image_paths[j]

            # Compare the two images
            result = DeepFace.verify(img1_path=img_path_1, img2_path=img_path_2, enforce_detection=False)

            # If the images contain the same face
            if result["verified"]:
                # Check if the face already exists in the clusters
                found_cluster = False
                for cluster in clusters:
                    if i in cluster:
                        cluster.append(j)
                        found_cluster = True
                        break

                # If the face is not found in existing clusters, create a new cluster
                if not found_cluster:
                    clusters.append([i, j])

    # Create a new directory for each cluster and copy the images to the corresponding directory
    for i, cluster in enumerate(clusters):
        os.makedirs(os.path.join(directory_path, f"person{i + 1}"), exist_ok=True)

        for j in cluster:
            shutil.copy(image_paths[j], os.path.join(directory_path, f"person{i + 1}"))

    return

# Call the function on a directory
process_directory('./assets/output_frames')
