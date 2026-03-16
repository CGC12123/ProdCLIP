import torch
import torch.nn.functional as F


def embedding_distillation_loss(student_embeddings, teacher_embeddings):
    """
    Calculate the MSE loss between student and teacher embeddings.

    Args:
        student_embeddings: Student model embeddings (image or text)
        teacher_embeddings: Teacher model embeddings (image or text)

    Returns:
        MSE loss between embeddings
    """
    # Normalize embeddings before calculating loss
    student_norm = F.normalize(student_embeddings, p=2, dim=-1)
    teacher_norm = F.normalize(teacher_embeddings, p=2, dim=-1)

    # Calculate MSE loss between normalized embeddings
    loss = F.mse_loss(student_norm, teacher_norm.detach())
    return loss


def similarity_distillation_loss(student_image_emb, student_text_emb, teacher_image_emb, teacher_text_emb, temperature=1.0):
    """
    Calculate the similarity matrix distillation loss using KL divergence.

    Args:
        student_image_emb: Student model image embeddings
        student_text_emb: Student model text embeddings
        teacher_image_emb: Teacher model image embeddings
        teacher_text_emb: Teacher model text embeddings
        temperature: Temperature scaling factor for softmax

    Returns:
        KL divergence loss between teacher and student similarity matrices
    """
    # Normalize embeddings
    student_img_norm = F.normalize(student_image_emb, p=2, dim=-1)
    student_txt_norm = F.normalize(student_text_emb, p=2, dim=-1)
    teacher_img_norm = F.normalize(teacher_image_emb, p=2, dim=-1)
    teacher_txt_norm = F.normalize(teacher_text_emb, p=2, dim=-1)

    # Calculate similarity matrices
    teacher_sim_matrix = torch.matmul(teacher_img_norm, teacher_txt_norm.transpose(-2, -1)) / temperature
    student_sim_matrix = torch.matmul(student_img_norm, student_txt_norm.transpose(-2, -1)) / temperature

    # Apply softmax to get probability distributions
    teacher_probs = F.softmax(teacher_sim_matrix, dim=-1)
    student_log_probs = F.log_softmax(student_sim_matrix, dim=-1)

    # Calculate KL divergence loss
    kl_loss = F.kl_div(student_log_probs, teacher_probs.detach(), reduction='batchmean')
    return kl_loss


def logits_distillation_loss(student_image_emb, student_text_emb, teacher_image_emb, teacher_text_emb, temperature=1.0):
    """
    Calculate the logits distillation loss using KL divergence between similarity matrices.

    Args:
        student_image_emb: Student model image embeddings
        student_text_emb: Student model text embeddings
        teacher_image_emb: Teacher model image embeddings
        teacher_text_emb: Teacher model text embeddings
        temperature: Temperature scaling factor for logits

    Returns:
        KL divergence loss between teacher and student logits
    """
    # Normalize embeddings
    student_img_norm = F.normalize(student_image_emb, p=2, dim=-1)
    student_txt_norm = F.normalize(student_text_emb, p=2, dim=-1)
    teacher_img_norm = F.normalize(teacher_image_emb, p=2, dim=-1)
    teacher_txt_norm = F.normalize(teacher_text_emb, p=2, dim=-1)

    # Calculate logits (similarity matrices)
    teacher_logits = torch.matmul(teacher_img_norm, teacher_txt_norm.T) / temperature
    student_logits = torch.matmul(student_img_norm, student_txt_norm.T) / temperature

    # Calculate KL divergence loss
    loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean"
    )

    return loss


def combined_distillation_loss(
    student_image_emb, student_text_emb,
    teacher_image_emb, teacher_text_emb,
    clip_loss,
    distill_types=["embedding", "similarity"],  # New parameter to control which distillation methods to use
    lambda_embed=0.5,
    lambda_similarity=0.5,
    lambda_logits=0.5,
    temperature=1.0
):
    """
    Combine CLIP contrastive loss with various distillation losses.

    Args:
        student_image_emb: Student model image embeddings
        student_text_emb: Student model text embeddings
        teacher_image_emb: Teacher model image embeddings
        teacher_text_emb: Teacher model text embeddings
        clip_loss: Original CLIP contrastive loss
        distill_types: List of distillation methods to use ("embedding", "similarity", "logits")
        lambda_embed: Weight for embedding distillation loss
        lambda_similarity: Weight for similarity distillation loss
        lambda_logits: Weight for logits distillation loss
        temperature: Temperature for similarity calculation

    Returns:
        Total loss and individual components
    """
    total_loss = clip_loss
    embed_loss = 0.0
    sim_loss = 0.0
    logits_loss = 0.0

    # Calculate requested distillation losses
    if "embedding" in distill_types:
        img_embed_loss = embedding_distillation_loss(student_image_emb, teacher_image_emb)
        txt_embed_loss = embedding_distillation_loss(student_text_emb, teacher_text_emb)
        embed_loss = (img_embed_loss + txt_embed_loss) / 2.0
        total_loss += lambda_embed * embed_loss

    if "similarity" in distill_types:
        sim_loss = similarity_distillation_loss(
            student_image_emb, student_text_emb,
            teacher_image_emb, teacher_text_emb,
            temperature
        )
        total_loss += lambda_similarity * sim_loss

    if "logits" in distill_types:
        logits_loss = logits_distillation_loss(
            student_image_emb, student_text_emb,
            teacher_image_emb, teacher_text_emb,
            temperature
        )
        total_loss += lambda_logits * logits_loss

    return {
        'total_loss': total_loss,
        'clip_loss': clip_loss,
        'embed_loss': embed_loss if "embedding" in distill_types else 0.0,
        'sim_loss': sim_loss if "similarity" in distill_types else 0.0,
        'logits_loss': logits_loss if "logits" in distill_types else 0.0
    }