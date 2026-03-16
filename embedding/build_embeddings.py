import os
import numpy as np
import json
import torch
from tqdm import tqdm
import logging
from config import config
from utils.logger import get_embedding_logger
from dataset.dataset import get_data_loaders
from models.clip_model import CLIPRetrievalModel


logger = get_embedding_logger()


def build_image_embeddings(model: CLIPRetrievalModel = None, dataloader=None, dataset_type: str = 'train'):
    """
    Build image embeddings for the specified dataset type

    Args:
        model: CLIP model instance (created if None)
        dataloader: DataLoader for the training data (created if None)
        dataset_type: Type of dataset to use ('train', 'test', or 'val'). Default is 'train'.

    Returns:
        Tuple of (embeddings, id_to_path_mapping)
    """
    if model is None:
        model = CLIPRetrievalModel()

    if dataloader is None:
        dataloaders = get_data_loaders(model.processor)
        dataloader = dataloaders[dataset_type]  # Use specified dataset type for embeddings

    # Prepare cache directory
    os.makedirs(config.embedding.cache_dir, exist_ok=True)

    embeddings_cache_path = os.path.join(config.embedding.cache_dir, config.embedding.embeddings_file)
    id_to_path_cache_path = os.path.join(config.embedding.cache_dir, config.embedding.id_to_path_file)

    # Check if cached embeddings exist
    if os.path.exists(embeddings_cache_path) and os.path.exists(id_to_path_cache_path):
        logger.info(f"Loading cached image embeddings for {dataset_type}...")
        embeddings = np.load(embeddings_cache_path)
        with open(id_to_path_cache_path, 'r') as f:
            id_to_path = json.load(f)

        # Convert string keys back to integers
        id_to_path = {int(k): v for k, v in id_to_path.items()}

        logger.info(f"Loaded cached embeddings with shape: {embeddings.shape}")
        return embeddings, id_to_path

    logger.info(f"Building image embeddings for {dataset_type}...")

    all_embeddings = []
    id_to_path = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Encoding images")):
            # Extract image features
            image_embeddings = model.get_image_features_batch(batch)

            # Store embeddings
            all_embeddings.append(image_embeddings.numpy())

            # Map indices to image paths
            # Note: batch['image_path'] is the key in the dataset's returned batch,
            # even though the source column name is 'image'
            for i, img_path in enumerate(batch['image_path']):
                global_idx = batch_idx * config.model.batch_size + i
                id_to_path[global_idx] = img_path

    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)

    logger.info(f"Built embeddings with shape: {embeddings.shape}")

    # Save embeddings to cache
    logger.info(f"Saving embeddings to {embeddings_cache_path}")
    np.save(embeddings_cache_path, embeddings)

    # Save id_to_path mapping
    logger.info(f"Saving id_to_path mapping to {id_to_path_cache_path}")
    with open(id_to_path_cache_path, 'w') as f:
        json.dump(id_to_path, f)

    return embeddings, id_to_path


def load_embeddings():
    """
    Load precomputed embeddings from cache

    Returns:
        Tuple of (embeddings, id_to_path_mapping)
    """
    embeddings_cache_path = os.path.join(config.embedding.cache_dir, config.embedding.embeddings_file)
    id_to_path_cache_path = os.path.join(config.embedding.cache_dir, config.embedding.id_to_path_file)

    if not os.path.exists(embeddings_cache_path) or not os.path.exists(id_to_path_cache_path):
        raise FileNotFoundError(f"Embeddings cache not found. Expected: {embeddings_cache_path} and {id_to_path_cache_path}")

    logger.info("Loading cached image embeddings...")
    embeddings = np.load(embeddings_cache_path)

    with open(id_to_path_cache_path, 'r') as f:
        id_to_path = json.load(f)

    # Convert string keys back to integers
    id_to_path = {int(k): v for k, v in id_to_path.items()}

    logger.info(f"Loaded embeddings with shape: {embeddings.shape}")

    return embeddings, id_to_path


def compute_text_embeddings_for_dataset(model: CLIPRetrievalModel = None, dataloader=None, dataset_type: str = 'train'):
    """
    Compute text embeddings for the specified dataset type

    Args:
        model: CLIP model instance (created if None)
        dataloader: DataLoader for the specified dataset type (created if None)
        dataset_type: Type of dataset to use ('train', 'test', or 'val'). Default is 'train'.

    Returns:
        Tuple of (text_embeddings, id_to_text_mapping)
    """
    if model is None:
        model = CLIPRetrievalModel()

    if dataloader is None:
        dataloaders = get_data_loaders(model.processor)
        dataloader = dataloaders[dataset_type]  # Use specified dataset type for embeddings

    logger.info("Building text embeddings...")

    all_embeddings = []
    id_to_text = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Encoding texts")):
            # Extract text features
            text_embeddings = model.get_text_features_batch(batch)

            # Store embeddings
            all_embeddings.append(text_embeddings.numpy())

            # Map indices to text descriptions
            for i, caption in enumerate(batch['caption']):
                global_idx = batch_idx * config.model.batch_size + i
                id_to_text[global_idx] = caption

    # Concatenate all embeddings
    text_embeddings = np.vstack(all_embeddings)

    logger.info(f"Built text embeddings with shape: {text_embeddings.shape}")

    return text_embeddings, id_to_text