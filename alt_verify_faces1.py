import os
import shutil
import numpy as np
from deepface import DeepFace
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import hdbscan
from typing import List, Dict

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Set the enforce_detection parameter to False to use the CPU

def get_deepface_embeddings(image_paths: List[str]) -> Dict[int, np.ndarray]:
    embeddings = {}
    for i, img_path in enumerate(tqdm(image_paths, desc="Calculating embeddings", unit="image")):
        embeddings[i] = DeepFace.represent(img_path=img_path, model_name='Facenet', enforce_detection=False)
    return embeddings

def get_clusters(embeddings: Dict[int, np.ndarray], algorithm: str = 'DBSCAN', params: dict = None) -> Dict[int, int]:
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

    # Prepare data for clustering
    embedding_list = [embedding for embedding in embeddings.values()]
    # embedding_array = np.array(embedding_list)
    embedding_array = np.vstack(embedding_list)
    labels = clustering_model.fit_predict(embedding_array)
    
    return {img_idx: cluster_label for img_idx, cluster_label in enumerate(labels)}

def process_directory(input_dir: str, output_dir: str, algorithm: str = 'DBSCAN', params: dict = None):
    # Get all image paths in the directory
    image_paths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if filename.endswith(".jpg") or filename.endswith(".png")]

    # Calculate embeddings for all images
    embeddings = get_deepface_embeddings(image_paths)

    # Get clusters
    clusters = get_clusters(embeddings, algorithm, params)

    # Save images in clusters
    common_indices = set(image_paths).intersection(set(clusters.keys()))
    for img_idx in common_indices:
        cluster_dir = os.path.join(output_dir, f"person{clusters[img_idx] + 1}")
        os.makedirs(cluster_dir, exist_ok=True)
        shutil.copy(image_paths[img_idx], cluster_dir)

def main():
    input_directory = "/home/isayahc/projects/machine_learning/facial_recognition/deepface_inference/assets/video1"
    output_directory = "/home/isayahc/projects/machine_learning/facial_recognition/deepface_inference/assets/video2"
    process_directory(input_directory, output_directory, 'DBSCAN', params={'eps': 0.3, 'min_samples': 5})
    process_directory(input_directory, output_directory, 'HDBSCAN', params={'min_cluster_size': 5})

if __name__ == '__main__':
    main()





