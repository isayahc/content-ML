import os
from deepface import DeepFace

# Set the enforce_detection parameter to False to use the CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def process_directory(directory_path):
    image_paths = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if filename.endswith(".jpg") or filename.endswith(".png")]
    num_images = len(image_paths)
    
    # We use a simple list of lists as a clustering mechanism. Each sub-list represents a unique face.
    clusters = []

    for i in range(num_images):
        img1_path = image_paths[i]
        
        for j in range(i + 1, num_images):
            img2_path = image_paths[j]
            
            # Use DeepFace.verify() to check whether the two images are of the same person
            result = DeepFace.verify(img1_path=img1_path, img2_path=img2_path, enforce_detection=False)
            
            if result['verified']:  # If the two images are of the same person
                # Check if the face already exists in the clusters
                found_cluster = False
                for cluster in clusters:
                    if img1_path in cluster:
                        cluster.append(img2_path)
                        found_cluster = True
                        break

                # If the face is not found in existing clusters, create a new cluster
                if not found_cluster:
                    clusters.append([img1_path, img2_path])

    # Converting the clusters into the dictionary format you wanted
    faces_dict = {"person{}".format(i + 1): clusters[i] for i in range(len(clusters))}

    return faces_dict

# Call the function on a directory
# faces_dict = process_directory('path_to_your_directory')


# Call the function on a directory
faces_dict = process_directory('./assets/output_frames')
x=0
