import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Tuple
import logging
from config import config
from utils.logger import get_analysis_logger


logger = get_analysis_logger()


def plot_embedding_norm_distribution(embeddings: np.ndarray, output_path: str = None):
    """
    Plot the distribution of L2 norms of embeddings

    Args:
        embeddings: Embedding array of shape (N, D)
        output_path: Path to save the plot (uses config default if None)
    """
    if output_path is None:
        output_path = os.path.join(config.analysis.plots_output_dir, "embedding_norms.png")

    # Calculate L2 norms
    norms = np.linalg.norm(embeddings, axis=1)

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.histplot(norms, bins=50, kde=True)
    plt.title('Distribution of Embedding L2 Norms')
    plt.xlabel('L2 Norm')
    plt.ylabel('Frequency')

    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Saved embedding norm distribution plot to {output_path}")


def plot_similarity_distributions(
    positive_similarities: List[float],
    negative_similarities: List[float],
    output_path: str = None
):
    """
    Plot the distributions of positive and negative cosine similarities

    Args:
        positive_similarities: List of positive (matching) similarities
        negative_similarities: List of negative (non-matching) similarities
        output_path: Path to save the plot (uses config default if None)
    """
    if output_path is None:
        output_path = os.path.join(config.analysis.plots_output_dir, "similarity_distributions.png")

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot both distributions
    sns.histplot(positive_similarities, bins=50, alpha=0.7, label='Positive (Matching)', kde=True)
    sns.histplot(negative_similarities, bins=50, alpha=0.7, label='Negative (Non-Matching)', kde=True)

    plt.title('Distribution of Cosine Similarities\n(Positive: Matching Pairs, Negative: Random Pairs)')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.legend()

    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Saved similarity distribution plot to {output_path}")


def analyze_embeddings_and_similarities(
    model,
    embeddings: np.ndarray,
    text_list: List[str],
    image_paths: List[str],
    output_dir: str = None
):
    """
    Comprehensive embedding analysis including norms and similarity distributions

    Args:
        model: Model for encoding text and images
        embeddings: Image embeddings array
        text_list: List of text descriptions
        image_paths: List of image paths
        output_dir: Directory to save plots (uses config default if None)
    """
    if output_dir is None:
        output_dir = config.analysis.plots_output_dir

    # 1. Plot embedding norm distribution
    if config.analysis.plot_embedding_dist:
        plot_embedding_norm_distribution(embeddings)

    # 2. Plot similarity distributions
    if config.analysis.plot_similarity_dist and len(text_list) > 1:
        positive_similarities = []
        negative_similarities = []

        # Sample a subset for efficiency
        sample_size = min(100, len(text_list))
        sampled_indices = np.random.choice(len(text_list), sample_size, replace=False)

        for i in sampled_indices:
            # Positive similarity (matching text-image pair)
            text_emb = model.encode_single_text(text_list[i]).numpy()
            img_emb = embeddings[i:i+1]  # Take the corresponding image embedding
            pos_sim = float(np.dot(text_emb, img_emb.T)[0][0])
            positive_similarities.append(pos_sim)

            # Negative similarity (random pairing)
            rand_idx = np.random.choice([j for j in range(len(text_list)) if j != i])
            img_emb_rand = embeddings[rand_idx:rand_idx+1]
            neg_sim = float(np.dot(text_emb, img_emb_rand.T)[0][0])
            negative_similarities.append(neg_sim)

        plot_similarity_distributions(positive_similarities, negative_similarities)


def compute_embedding_statistics(embeddings: np.ndarray) -> dict:
    """
    Compute basic statistics of embeddings

    Args:
        embeddings: Embedding array of shape (N, D)

    Returns:
        Dictionary with statistics
    """
    stats = {
        'shape': embeddings.shape,
        'mean_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
        'std_norm': float(np.std(np.linalg.norm(embeddings, axis=1))),
        'min_norm': float(np.min(np.linalg.norm(embeddings, axis=1))),
        'max_norm': float(np.max(np.linalg.norm(embeddings, axis=1))),
        'mean_abs_value': float(np.mean(np.abs(embeddings))),
        'std_abs_value': float(np.std(np.abs(embeddings))),
    }

    logger.info(f"Embedding statistics: {stats}")
    return stats


def analyze_model_bias(model, embeddings: np.ndarray, top_k: int = 10):
    """
    Analyze potential bias in the retrieval model by checking if certain items
    are disproportionately retrieved

    Args:
        model: Model for encoding
        embeddings: Embedding array
        top_k: Number of top items to analyze
    """
    # Compute self-similarity matrix
    sim_matrix = np.dot(embeddings, embeddings.T)

    # Count how often each item appears in top-k for other items
    freq_counts = np.zeros(len(embeddings))

    for i in range(len(embeddings)):
        top_k_indices = np.argsort(sim_matrix[i])[::-1][1:top_k+1]  # Exclude self
        for idx in top_k_indices:
            freq_counts[idx] += 1

    # Get the most frequently retrieved items
    top_freq_indices = np.argsort(freq_counts)[::-1][:10]

    logger.info(f"Top {len(top_freq_indices)} most frequently retrieved items:")
    for i, idx in enumerate(top_freq_indices):
        logger.info(f"  {i+1}. Item {idx}: {freq_counts[idx]:.1f} times")

    return freq_counts