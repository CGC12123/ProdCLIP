import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_loss(image_features, text_features, temperature=0.07):
    """
    Compute contrastive loss for CLIP-style training

    Args:
        image_features: Image embeddings of shape (batch_size, embedding_dim)
        text_features: Text embeddings of shape (batch_size, embedding_dim)
        temperature: Temperature parameter for scaling logits

    Returns:
        Contrastive loss scalar
    """
    batch_size = image_features.size(0)

    # Normalize embeddings
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(image_features, text_features.T) / temperature

    # Create labels for cross entropy (diagonal positions)
    labels = torch.arange(batch_size).to(image_features.device)

    # Compute losses for image-to-text and text-to-image
    loss_i2t = F.cross_entropy(similarity_matrix, labels)
    loss_t2i = F.cross_entropy(similarity_matrix.T, labels)

    # Return average of both losses
    loss = (loss_i2t + loss_t2i) / 2

    return loss


def compute_metrics(similarity_matrix, k_values=[1, 5]):
    """
    Compute recall metrics for evaluation

    Args:
        similarity_matrix: Similarity matrix of shape (batch_size, batch_size)
        k_values: List of k values to compute recall for

    Returns:
        Dictionary with recall@k values
    """
    batch_size = similarity_matrix.size(0)
    labels = torch.arange(batch_size).to(similarity_matrix.device)

    # Get top-k predictions for each image
    _, top_k_indices = torch.topk(similarity_matrix, max(k_values), dim=1)

    metrics = {}
    for k in k_values:
        # Check if the correct text is in top-k for each image
        top_k_predictions = top_k_indices[:, :k]
        correct = (top_k_predictions == labels.unsqueeze(1)).any(dim=1).float()
        recall = correct.mean().item()
        metrics[f'recall@{k}'] = recall

    return metrics