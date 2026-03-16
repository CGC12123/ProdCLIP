import faiss
import numpy as np
import os
import torch
import logging
from config import config
from utils.logger import get_index_logger


logger = get_index_logger()


class FaissIndexBuilder:
    """FAISS index builder for efficient similarity search"""

    def __init__(self):
        self.index = None
        self.index_path = os.path.join(config.index.index_file)

    def build_index(self, embeddings):
        """
        Build FAISS index from embeddings

        Args:
            embeddings: Array of embeddings (each row is an embedding vector)

        Returns:
            FAISS index
        """
        num_embeddings, embedding_dim = embeddings.shape

        # Since embeddings are L2 normalized, we can use Inner Product (which is equivalent to cosine similarity)
        # FAISS uses Inner Product (dot product) for normalized vectors
        self.index = faiss.IndexFlatIP(embedding_dim)

        # Convert embeddings to float32 and add to index
        embeddings = embeddings.astype(np.float32)

        logger.info(f"Adding {num_embeddings} embeddings to index...")
        self.index.add(embeddings)

        logger.info(f"Index built successfully. Total vectors: {self.index.ntotal}")
        return self.index

    def save_index(self, index_path: str = None):
        """
        Save the FAISS index to disk

        Args:
            index_path: Path to save the index (uses config default if None)
        """
        if index_path is None:
            index_path = self.index_path

        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(self.index, index_path)
        logger.info("Index saved successfully")

    def load_index(self, index_path: str = None):
        """
        Load a FAISS index from disk

        Args:
            index_path: Path to load the index from (uses config default if None)

        Returns:
            Loaded FAISS index
        """
        if index_path is None:
            index_path = self.index_path

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")

        logger.info(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)
        logger.info(f"Index loaded successfully. Total vectors: {self.index.ntotal}")
        return self.index

    def search(self, query_embeddings, k):
        """
        Search for nearest neighbors in the index

        Args:
            query_embeddings: Query embeddings to search for
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None:
            raise ValueError("Index not built or loaded")

        # Normalize query embeddings if they aren't already
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        # Convert PyTorch tensors to NumPy arrays if needed
        if torch.is_tensor(query_embeddings):
            query_embeddings = query_embeddings.detach().cpu().numpy()

        query_embeddings = query_embeddings.astype(np.float32)

        # For L2 normalized vectors, FAISS Inner Product gives cosine similarity
        distances, indices = self.index.search(query_embeddings, k)

        return distances, indices

    def is_trained(self):
        """Check if the index is trained (for indices that require training)"""
        if self.index is None:
            return False
        return self.index.is_trained

    def get_ntotal(self):
        """Get the number of vectors in the index"""
        if self.index is None:
            return 0
        return self.index.ntotal