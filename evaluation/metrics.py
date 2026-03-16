import numpy as np
import torch
import os
from typing import Dict, List, Tuple
from collections import defaultdict
import logging
from config import config
from utils.logger import get_evaluation_logger


logger = get_evaluation_logger()


def compute_recall_at_k(similarity_matrix: np.ndarray, ground_truth: List[int], k_list: List[int]) -> Dict[int, float]:
    """
    Compute Recall@K for the given similarity matrix and ground truth

    Args:
        similarity_matrix: Matrix of shape (num_queries, num_reference_items) with similarity scores
        ground_truth: List of ground truth indices for each query
        k_list: List of K values to compute recall for

    Returns:
        Dictionary mapping K values to recall scores
    """
    num_queries = len(ground_truth)
    recalls = {k: 0.0 for k in k_list}

    for i, gt_idx in enumerate(ground_truth):
        # Get the similarity scores for this query
        scores = similarity_matrix[i]

        # Get top-K indices
        top_k_indices = np.argsort(scores)[::-1][:max(k_list)]

        # Compute recall for each K
        for k in k_list:
            if gt_idx in top_k_indices[:k]:
                recalls[k] += 1.0

    # Average over all queries
    for k in k_list:
        recalls[k] /= num_queries

    return recalls


def evaluate_text_to_image(
    model,
    image_embeddings: np.ndarray,
    test_data: List[Dict],
    k_list: List[int] = None
) -> Dict[int, float]:
    """
    Evaluate text-to-image retrieval performance

    Args:
        model: Model with encode_texts method
        image_embeddings: Precomputed image embeddings
        test_data: List of test samples with 'description' and ground truth indices
        k_list: List of K values to evaluate

    Returns:
        Dictionary mapping K values to recall scores
    """
    if k_list is None:
        k_list = config.evaluation.recall_at_k

    # Encode all test text queries
    text_embeddings = []
    ground_truth = []

    for i, sample in enumerate(test_data):
        # Encode the text description
        text_emb = model.encode_single_text(sample['description']).numpy()
        text_embeddings.append(text_emb)

        # Ground truth is the index of the corresponding image
        ground_truth.append(i)  # Assuming 1:1 correspondence for simplicity

    text_embeddings = np.vstack(text_embeddings)

    # Compute similarities between text and image embeddings
    # For L2 normalized embeddings, dot product gives cosine similarity
    similarity_matrix = np.dot(text_embeddings, image_embeddings.T)

    # Compute recall
    recalls = compute_recall_at_k(similarity_matrix, ground_truth, k_list)

    logger.info(f"Text-to-Image Retrieval Results: {recalls}")

    return recalls


def evaluate_image_to_text(
    model,
    text_embeddings: np.ndarray,
    test_data: List[Dict],
    k_list: List[int] = None
) -> Dict[int, float]:
    """
    Evaluate image-to-text retrieval performance

    Args:
        model: Model with encode_images method
        text_embeddings: Precomputed text embeddings
        test_data: List of test samples with 'image' and ground truth indices
        k_list: List of K values to evaluate

    Returns:
        Dictionary mapping K values to recall scores
    """
    if k_list is None:
        k_list = config.evaluation.recall_at_k

    # Encode all test images
    image_embeddings = []
    ground_truth = []

    for i, sample in enumerate(test_data):
        # Encode the image
        img_path = sample.get('image', '')
        if img_path:
            img_path = os.path.join(config.data.image_dir, img_path)
        img_emb = model.encode_single_image(img_path).numpy()
        image_embeddings.append(img_emb)

        # Ground truth is the index of the corresponding text
        ground_truth.append(i)  # Assuming 1:1 correspondence for simplicity

    image_embeddings = np.vstack(image_embeddings)

    # Compute similarities between image and text embeddings
    # For L2 normalized embeddings, dot product gives cosine similarity
    similarity_matrix = np.dot(image_embeddings, text_embeddings.T)

    # Compute recall
    recalls = compute_recall_at_k(similarity_matrix, ground_truth, k_list)

    logger.info(f"Image-to-Text Retrieval Results: {recalls}")

    return recalls


def evaluate_full_retrieval(
    model,
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    test_data: List[Dict],
    k_list: List[int] = None
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate both text-to-image and image-to-text retrieval performance

    Args:
        model: Model with encode_texts and encode_images methods
        image_embeddings: Precomputed image embeddings
        text_embeddings: Precomputed text embeddings
        test_data: List of test samples
        k_list: List of K values to evaluate

    Returns:
        Dictionary with 't2i' and 'i2t' keys mapping to recall dictionaries
    """
    if k_list is None:
        k_list = config.evaluation.recall_at_k

    results = {}

    # Text-to-Image evaluation
    logger.info("Evaluating Text-to-Image retrieval...")
    t2i_recalls = evaluate_text_to_image(
        model=model,
        image_embeddings=image_embeddings,
        test_data=test_data,
        k_list=k_list
    )
    results['t2i'] = t2i_recalls

    # Image-to-Text evaluation
    logger.info("Evaluating Image-to-Text retrieval...")
    i2t_recalls = evaluate_image_to_text(
        model=model,
        text_embeddings=text_embeddings,
        test_data=test_data,
        k_list=k_list
    )
    results['i2t'] = i2t_recalls

    # Print summary
    logger.info("Final Evaluation Results:")
    logger.info(f"Text-to-Image - R@1: {results['t2i'].get(1, 0):.4f}, R@5: {results['t2i'].get(5, 0):.4f}")
    logger.info(f"Image-to-Text - R@1: {results['i2t'].get(1, 0):.4f}, R@5: {results['i2t'].get(5, 0):.4f}")

    return results