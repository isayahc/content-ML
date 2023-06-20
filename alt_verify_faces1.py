import os
import shutil
import numpy as np
from deepface import DeepFace
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import hdbscan
from typing import List, Dict, Union

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Set the enforce_detection parameter to False to use the CPU

def get_deepface_embeddings(image_paths: List[str]) -> Dict[int, Dict[str, Union[np.ndarray, Dict[str, int]]]]:
    """
    Calculates DeepFace embeddings and facial area for all images in the provided list of image paths.

    Parameters:
        image_paths (List[str]): A list of image file paths.

    Returns:
        embeddings (Dict[int, Dict[str, Union[np.ndarray, Dict[str, int]]]]): 
            A dictionary mapping an integer index to another dictionary, which contains:
                - 'embedding': the DeepFace embedding for the image.
                - 'facial_area': a dictionary with 'x', 'y', 'w', 'h' representing the face's location and dimensions in the image.
    """
    embeddings = {}
    for i, img_path in enumerate(tqdm(image_paths, desc="Calculating embeddings", unit="image")):
        result = DeepFace.represent(img_path=img_path, model_name='Facenet', enforce_detection=False)
        print(f"Type: {type(result)}, Content: {result}")
        embeddings[i] = result
    return embeddings

def get_clusters(embeddings: Dict[int, np.ndarray], algorithm: str = 'DBSCAN', params: dict = None) -> Dict[int, int]:
    """
    Perform clustering on the DeepFace embeddings.

    Parameters:
        embeddings (Dict[int, np.ndarray]): A dictionary of DeepFace embeddings.
        algorithm (str): The clustering algorithm to use. Can be 'DBSCAN' or 'HDBSCAN'. Default is 'DBSCAN'.
        params (dict): The parameters to use for the clustering algorithm.

    Returns:
        A dictionary mapping each image index to a cluster label.
    """
    clustering_model = None
    if algorithm == 'DBSCAN':
        eps = params.get('eps', 0.5)
        min_samples = params.get('min_samples', 5)
        clustering_model = DBSCAN(eps=eps, min_samples=min_samples)
    elif algorithm == 'HDBSCAN':
        min_cluster_size = params.get('min_cluster_size', 5)
        clustering_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    else:
        raise ValueError("Invalid algorithm choice. Choose 'DBSCAN' or 'HDBSCAN'.")

    # embedding_list = [embedding for embedding in embeddings.values()]

    # Flatten list of embeddings
    embedding_list = [embedding for sublist in embeddings.values() for embedding in (sublist if isinstance(sublist, list) else [sublist])]
    embedding_list = [i['embedding'] for i in embedding_list]
    
    embedding_array = np.vstack(embedding_list)
    # embedding_array = np.vstack(embedding_list)
    labels = clustering_model.fit_predict(embedding_array)
    
    return {img_idx: cluster_label for img_idx, cluster_label in enumerate(labels)}

def process_directory(input_dir: str, output_dir: str, algorithm: str = 'DBSCAN', params: dict = None):
    """
    Process a directory of images. This involves calculating DeepFace embeddings for each image, performing clustering,
    and saving the images in clusters to the output directory.

    Parameters:
        input_dir (str): The input directory containing images.
        output_dir (str): The output directory to save the clusters of images.
        algorithm (str): The clustering algorithm to use. Can be 'DBSCAN' or 'HDBSCAN'. Default is 'DBSCAN'.
        params (dict): The parameters to use for the clustering algorithm.
    """
    image_paths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if filename.endswith(".jpg") or filename.endswith(".png")]
    embeddings = get_deepface_embeddings(image_paths)
    clusters = get_clusters(embeddings, algorithm, params)

    for img_idx in clusters.keys():
        cluster_dir = os.path.join(output_dir, f"person{clusters[img_idx] + 1}")
        os.makedirs(cluster_dir, exist_ok=True)
        shutil.copy(image_paths[img_idx], cluster_dir)

def main():
    """
    The main function of the script. It processes a directory of images twice, once with DBSCAN and once with HDBSCAN.
    """
    input_directory = "/home/isayahc/projects/machine_learning/facial_recognition/deepface_inference/assets/video1"
    output_directory = "/home/isayahc/projects/machine_learning/facial_recognition/deepface_inference/assets/video2"
    process_directory(input_directory, output_directory, 'DBSCAN', params={'eps': 0.3, 'min_samples': 5})
    process_directory(input_directory, output_directory, 'HDBSCAN', params={'min_cluster_size': 5})

if __name__ == '__main__':
    main()
