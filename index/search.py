import numpy as np
import torch
from typing import List, Tuple
import logging
from config import config
from utils.logger import get_index_logger
from models.clip_model import CLIPRetrievalModel


logger = get_index_logger()


class SearchEngine:
    """Search engine for multimodal retrieval"""

    def __init__(self, index_builder, model: CLIPRetrievalModel = None):
        """
        Initialize the search engine

        Args:
            index_builder: FAISS index builder instance
            model: CLIP model instance (created if None)
        """
        self.index_builder = index_builder
        self.model = model if model is not None else CLIPRetrievalModel()

    def search_by_text(self, query_text: str, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for images based on text query

        Args:
            query_text: Text query
            k: Number of results to return

        Returns:
            Tuple of (distances, image_paths)
        """
        # Encode the query text
        query_embedding = self.model.encode_single_text(query_text)

        # Search in the index
        distances, indices = self.index_builder.search(query_embedding, k)

        # Assuming we have a mapping from indices to image paths
        # This should be passed or accessible from the index builder
        # For now, we'll return indices as is - caller needs to map them
        return distances[0], indices[0]

    def search_by_image(self, image_path: str, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for text descriptions based on image query

        Args:
            image_path: Path to the query image
            k: Number of results to return

        Returns:
            Tuple of (distances, text_descriptions_or_indices)
        """
        # Encode the query image
        query_embedding = self.model.encode_single_image(image_path)

        # Search in the index
        distances, indices = self.index_builder.search(query_embedding, k)

        # This method assumes we have a text index as well
        # For a complete implementation, we'd need both image and text indices
        return distances[0], indices[0]

    def get_top_k_images(self, query_text: str, k: int = 5, id_to_path: dict = None) -> List[Tuple[str, float]]:
        """
        Get top-k image paths for a text query

        Args:
            query_text: Text query
            k: Number of results to return
            id_to_path: Mapping from index to image path

        Returns:
            List of tuples (image_path, similarity_score)
        """
        distances, indices = self.search_by_text(query_text, k)

        if id_to_path is None:
            # Return indices if path mapping is not provided
            results = [(f"index_{idx}", float(dist)) for idx, dist in zip(indices, distances)]
        else:
            results = [(id_to_path.get(int(idx), f"unknown_{idx}"), float(dist)) for idx, dist in zip(indices, distances)]

        return results

    def get_top_k_texts(self, image_path: str, k: int = 5, id_to_text: dict = None) -> List[Tuple[str, float]]:
        """
        Get top-k text descriptions for an image query

        Args:
            image_path: Path to the query image
            k: Number of results to return
            id_to_text: Mapping from index to text description

        Returns:
            List of tuples (text_description, similarity_score)
        """
        # For image-to-text search, we'd need a text embedding index
        # This implementation assumes we have both image and text indices
        # Since we're mainly focusing on image retrieval, we'll return indices
        distances, indices = self.search_by_image(image_path, k)

        if id_to_text is None:
            # Return indices if text mapping is not provided
            results = [(f"index_{idx}", float(dist)) for idx, dist in zip(indices, distances)]
        else:
            results = [(id_to_text.get(int(idx), f"unknown_{idx}"), float(dist)) for idx, dist in zip(indices, distances)]

        return results

    def batch_search_by_text(self, query_texts: List[str], k: int = 5) -> List[List[Tuple[str, float]]]:
        """
        Perform batch search for multiple text queries

        Args:
            query_texts: List of text queries
            k: Number of results to return for each query

        Returns:
            List of lists of tuples (image_path, similarity_score) for each query
        """
        # Encode all query texts
        query_embeddings = []
        for text in query_texts:
            emb = self.model.encode_single_text(text)
            query_embeddings.append(emb)

        # Stack embeddings
        query_embeddings = torch.cat(query_embeddings, dim=0).numpy()

        # Search in the index
        distances, indices = self.index_builder.search(query_embeddings, k)

        # Format results
        batch_results = []
        for i in range(len(query_texts)):
            results = [(f"index_{idx}", float(dist)) for idx, dist in zip(indices[i], distances[i])]
            batch_results.append(results)

        return batch_results