"""Hard negative loss functions for CLIP training."""

import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import defaultdict


def hard_negative_contrastive_loss(
    image_features,
    text_features,
    categories=None,
    temperature=0.07,
    lambda_hard=0.1,
    hard_negative_type="batch"
):
    """
    Compute combined contrastive loss with hard negative mining.

    Args:
        image_features: Image embeddings of shape (batch_size, embedding_dim)
        text_features: Text embeddings of shape (batch_size, embedding_dim)
        categories: Category labels for each sample (used for category-level hard negatives)
        temperature: Temperature parameter for scaling logits
        lambda_hard: Weight for hard negative loss component
        hard_negative_type: Type of hard negative mining ("batch" or "category")

    Returns:
        Combined loss scalar
    """
    batch_size = image_features.size(0)

    # Normalize embeddings
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(image_features, text_features.T) / temperature

    # Standard CLIP loss (diagonal pairs)
    labels = torch.arange(batch_size).to(image_features.device)
    loss_i2t = F.cross_entropy(similarity_matrix, labels)
    loss_t2i = F.cross_entropy(similarity_matrix.T, labels)
    clip_loss = (loss_i2t + loss_t2i) / 2

    # Hard negative loss
    if hard_negative_type == "batch":
        hard_loss = _batch_hard_negative_loss(
            similarity_matrix,
            labels,
            image_features,
            text_features
        )
    elif hard_negative_type == "category" and categories is not None:
        hard_loss = _category_hard_negative_loss(
            image_features,
            text_features,
            categories,
            temperature
        )
    else:
        # If categories not provided for category-based hard negative, fall back to batch
        hard_loss = _batch_hard_negative_loss(
            similarity_matrix,
            labels,
            image_features,
            text_features
        )

    # Combine losses
    total_loss = clip_loss + lambda_hard * hard_loss

    return total_loss, clip_loss, hard_loss


def _batch_hard_negative_loss(similarity_matrix, labels, image_features, text_features):
    """
    Compute hard negative loss using negatives within the batch.

    This finds the hardest negatives in the batch for each positive pair.
    """
    batch_size = similarity_matrix.size(0)

    # For each image, exclude the corresponding text (positive) and find the hardest negative
    mask = torch.eye(batch_size, dtype=torch.bool, device=similarity_matrix.device)

    # Exclude positives to focus on negatives
    neg_similarities_img = similarity_matrix.masked_fill(mask, float('-inf'))
    neg_similarities_txt = similarity_matrix.T.masked_fill(mask, float('-inf'))

    # Find hardest negatives (highest similarity scores among negatives)
    hardest_neg_from_text = torch.max(neg_similarities_img, dim=1)[0]  # Hardest text for each image
    hardest_neg_from_img = torch.max(neg_similarities_txt, dim=1)[0]   # Hardest image for each text

    # Standard CLIP logits for positives
    pos_logits = torch.diag(similarity_matrix)

    # Compute margin-based hard negative loss
    # Ensure positive samples are more similar to each other than to hardest negatives
    margin = 0.1

    # Image-to-text hard negatives: pos_img_pos_txt vs pos_img_hardest_neg_txt
    i2t_hard_loss = torch.mean(torch.relu(hardest_neg_from_text - pos_logits + margin))

    # Text-to-image hard negatives: pos_txt_pos_img vs pos_txt_hardest_neg_img
    t2i_hard_loss = torch.mean(torch.relu(hardest_neg_from_img - pos_logits + margin))

    hard_loss = (i2t_hard_loss + t2i_hard_loss) / 2

    return hard_loss


def _category_hard_negative_loss(image_features, text_features, categories, temperature):
    """
    Compute hard negative loss using category information.

    Samples from the same category are treated as negatives for each other,
    which encourages the model to distinguish between fine-grained differences.
    """
    batch_size = image_features.size(0)

    # Convert categories to tensor if they are strings
    if isinstance(categories[0], str):
        unique_cats = list(set(categories))
        cat_to_idx = {cat: idx for idx, cat in enumerate(unique_cats)}
        cat_indices = torch.tensor([cat_to_idx[cat] for cat in categories], device=image_features.device)
    else:
        cat_indices = torch.tensor(categories, device=image_features.device)

    # Compute similarity matrix
    sim_matrix = torch.matmul(image_features, text_features.T) / temperature

    # Create mask for same-category pairs (these are our hard negatives)
    cat_mask = (cat_indices.unsqueeze(1) == cat_indices.unsqueeze(0)).float()

    # Exclude diagonal (same sample) and same-category pairs from being considered as positives
    identity_mask = torch.eye(batch_size, device=image_features.device)
    same_cat_excl_diag = cat_mask - identity_mask

    # Extract similarities for same-category pairs (hard negatives)
    same_category_similarities = sim_matrix * same_cat_excl_diag

    # For each sample, find the hardest negative from the same category
    same_category_similarities.masked_fill_(identity_mask.bool(), float('-inf'))

    # Get the maximum (hardest) negative similarity for each sample
    hardest_same_category = torch.max(same_category_similarities, dim=1)[0]

    # Get positive similarities (diagonal)
    pos_similarities = torch.diag(sim_matrix)

    # Margin-based loss to ensure positives are more similar than hard negatives
    margin = 0.1
    hard_loss_per_sample = torch.relu(hardest_same_category - pos_similarities + margin)

    # Only compute loss for samples that have same-category negatives
    has_hard_negatives = (same_cat_excl_diag.sum(dim=1) > 0).float()
    hard_loss = (hard_loss_per_sample * has_hard_negatives).sum() / (has_hard_negatives.sum() + 1e-8)

    return hard_loss


def compute_hard_negative_metrics(similarity_matrix, categories=None, k_values=[1, 5]):
    """
    Compute metrics that consider hard negatives based on category information.

    Args:
        similarity_matrix: Similarity matrix of shape (batch_size, batch_size)
        categories: Category labels for each sample
        k_values: List of k values to compute recall for

    Returns:
        Dictionary with recall metrics
    """
    batch_size = similarity_matrix.size(0)
    labels = torch.arange(batch_size).to(similarity_matrix.device)

    # Get top-k predictions for each image
    _, top_k_indices = torch.topk(similarity_matrix, max(k_values), dim=1)

    metrics = {}

    # Standard metrics
    for k in k_values:
        top_k_predictions = top_k_indices[:, :k]
        correct = (top_k_predictions == labels.unsqueeze(1)).any(dim=1).float()
        recall = correct.mean().item()
        metrics[f'recall@{k}'] = recall

    # If categories provided, also compute within-category metrics
    if categories is not None:
        # Convert categories to tensor if they are strings
        if isinstance(categories[0], str):
            unique_cats = list(set(categories))
            cat_to_idx = {cat: idx for idx, cat in enumerate(unique_cats)}
            cat_indices = torch.tensor([cat_to_idx[cat] for cat in categories], device=similarity_matrix.device)
        else:
            cat_indices = torch.tensor(categories, device=similarity_matrix.device)

        # Create mask for same-category pairs
        same_category_mask = (cat_indices.unsqueeze(1) == cat_indices.unsqueeze(0))

        for k in k_values:
            top_k_predictions = top_k_indices[:, :k]

            # Check if the correct text is in top-k for each image, excluding same-category samples
            correct_not_same_cat = []
            for i in range(batch_size):
                pred_indices = top_k_predictions[i]
                is_correct = torch.any(pred_indices == labels[i]).item()

                # Count if correct sample is in top-k AND not from the same category
                is_diff_category = torch.any(~same_category_mask[i, pred_indices]).item()

                correct_not_same_cat.append(is_correct and is_diff_category)

            recall_out_of_category = torch.tensor(correct_not_same_cat).float().mean().item()
            metrics[f'recall_out_of_category@{k}'] = recall_out_of_category

    return metrics