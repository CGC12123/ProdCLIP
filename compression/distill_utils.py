import torch
import os
from transformers import CLIPProcessor
from torch.utils.data import DataLoader
from utils.logger import get_compression_logger

logger = get_compression_logger()


def load_models_and_processor(teacher_model_name, student_model_name, device, teacher_use_lora=False, teacher_lora_path=None):
    """
    Load teacher and student models and processor.

    Args:
        teacher_model_name: Name or path of the teacher model
        student_model_name: Name or path of the student model
        device: Device to load models to
        teacher_use_lora: Whether to use LoRA adapter for teacher model
        teacher_lora_path: Path to LoRA adapter for teacher model

    Returns:
        teacher_model, student_model, processor
    """
    from transformers import CLIPModel

    # Load teacher model
    logger.info(f"Loading teacher model: {teacher_model_name}")

    if teacher_use_lora and teacher_lora_path:
        logger.info(f"Loading teacher model with LoRA adapter from: {teacher_lora_path}")

        # First load the base model
        teacher_model = CLIPModel.from_pretrained(teacher_model_name)

        # Then load the LoRA adapter if PEFT is available
        try:
            from peft import PeftModel
            teacher_model = PeftModel.from_pretrained(teacher_model, teacher_lora_path)
            logger.info("LoRA adapter loaded for teacher model")
        except ImportError:
            logger.warning("PEFT library not installed. Loading teacher model without LoRA adapter.")
            teacher_model = CLIPModel.from_pretrained(teacher_model_name)
    else:
        teacher_model = CLIPModel.from_pretrained(teacher_model_name)

    teacher_model.to(device)
    teacher_model.eval()  # Set to evaluation mode

    # Freeze teacher model parameters
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Load student model
    logger.info(f"Loading student model: {student_model_name}")
    student_model = CLIPModel.from_pretrained(student_model_name)
    student_model.to(device)
    student_model.train()  # Set to training mode

    # Load processor
    processor = CLIPProcessor.from_pretrained(teacher_model_name)

    logger.info("Successfully loaded teacher and student models")
    return teacher_model, student_model, processor


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint to {filepath} at epoch {epoch}")


def load_checkpoint(model, optimizer, filepath, device):
    """
    Load model checkpoint.

    Args:
        model: Model to load weights to
        optimizer: Optimizer to load state to
        filepath: Path to checkpoint file
        device: Device to load to

    Returns:
        epoch: Epoch from checkpoint
        loss: Loss from checkpoint
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    logger.info(f"Loaded checkpoint from {filepath}")
    return epoch, loss


def ensure_output_directory(output_dir):
    """
    Ensure output directory exists.

    Args:
        output_dir: Directory to create
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")


def log_training_step(epoch, step, total_steps, loss_dict, lr):
    """
    Log training step information.

    Args:
        epoch: Current epoch
        step: Current step
        total_steps: Total steps in epoch
        loss_dict: Dictionary containing loss values
        lr: Learning rate
    """
    logger.info(f"Epoch [{epoch}], Step [{step}/{total_steps}] "
          f"Loss: {loss_dict['total_loss']:.4f} "
          f"(CLIP: {loss_dict['clip_loss']:.4f}, "
          f"Embed: {loss_dict['embed_loss']:.4f}, "
          f"Sim: {loss_dict['sim_loss']:.4f}) "
          f"LR: {lr:.6f}")


def move_batch_to_device(batch, device):
    """
    Move batch to specified device.

    Args:
        batch: Input batch
        device: Target device

    Returns:
        Batch moved to device
    """
    if isinstance(batch, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, list):
        return [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
    else:
        return batch