import argparse
import torch
import os
import sys
from transformers import CLIPProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from compression.distill_utils import (
    load_models_and_processor,
    save_checkpoint,
    ensure_output_directory,
    log_training_step,
    move_batch_to_device
)
from compression.distill_trainer import DistillationTrainer
from utils.logger import get_compression_logger
from config import config
from dataset.dataset import load_data, split_data, MultimodalDataset, custom_collate_fn, get_data_loaders

logger = get_compression_logger()


def get_dataloader(dataset, batch_size, shuffle=True):
    """
    Create a dataloader from the dataset.
    This function assumes that the dataset follows the same structure as used in the original retrieval code.
    """
    # We're using a generic DataLoader here; in practice, you'd use your existing dataset
    # that returns batches with 'pixel_values', 'input_ids', and 'attention_mask'
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn  # Use the custom collate function from dataset module
    )
    return dataloader


def load_dataset(data_path):
    """
    Load dataset for training.
    This function should be adapted to use your existing dataset loading code.
    """
    # Load data using the existing dataset loading function
    df = load_data(data_path)

    # Get processor for the teacher model (used to initialize dataset)
    processor = CLIPProcessor.from_pretrained(config.distillation.teacher_model_name, use_fast=False)

    # Create dataset using the existing dataset class
    dataset = MultimodalDataset(df, processor)

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Train a student CLIP model using knowledge distillation from a teacher model.")

    parser.add_argument("--teacher_model", type=str,
                        default=config.distillation.teacher_model_name,
                        help="Path or name of the teacher model")
    parser.add_argument("--student_model", type=str,
                        default=config.distillation.student_model_name,
                        help="Path or name of the student model")
    parser.add_argument("--teacher_use_lora", action='store_true',
                        default=config.distillation.teacher_use_lora,
                        help="Whether to use LoRA adapter for teacher model")
    parser.add_argument("--teacher_lora_path", type=str,
                        default=config.distillation.teacher_lora_path,
                        help="Path to LoRA adapter for teacher model")
    parser.add_argument("--distill_types", nargs='+', type=str,
                        default=config.distillation.distill_types,
                        help="List of distillation methods to use (embedding, similarity, logits)")
    parser.add_argument("--batch_size", type=int,
                        default=config.distillation.batch_size,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float,
                        default=config.distillation.learning_rate,
                        help="Learning rate for training")
    parser.add_argument("--temperature", type=float,
                        default=config.distillation.temperature,
                        help="Temperature for similarity distillation")
    parser.add_argument("--lambda_embed", type=float,
                        default=config.distillation.lambda_embed,
                        help="Weight for embedding distillation loss")
    parser.add_argument("--lambda_similarity", type=float,
                        default=config.distillation.lambda_similarity,
                        help="Weight for similarity distillation loss")
    parser.add_argument("--lambda_logits", type=float,
                        default=config.distillation.lambda_logits,
                        help="Weight for logits distillation loss")
    parser.add_argument("--epochs", type=int,
                        default=config.distillation.epochs,
                        help="Number of epochs to train")
    parser.add_argument("--output_dir", type=str,
                        default=config.distillation.save_path,
                        help="Directory to save the trained student model")
    parser.add_argument("--checkpoint_interval", type=int,
                        default=config.distillation.checkpoint_interval,
                        help="Save checkpoints every N epochs")
    parser.add_argument("--resume_from_checkpoint", type=str,
                        default=config.distillation.resume_from_checkpoint,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--data_path", type=str,
                        default=config.data.data_path,
                        help="Path to training data")

    args = parser.parse_args()

    # Override config with command line args
    config.distillation.teacher_model_name = args.teacher_model
    config.distillation.student_model_name = args.student_model
    config.distillation.teacher_use_lora = args.teacher_use_lora
    config.distillation.teacher_lora_path = args.teacher_lora_path
    config.distillation.distill_types = args.distill_types
    config.distillation.batch_size = args.batch_size
    config.distillation.learning_rate = args.lr
    config.distillation.temperature = args.temperature
    config.distillation.lambda_embed = args.lambda_embed
    config.distillation.lambda_similarity = args.lambda_similarity
    config.distillation.lambda_logits = args.lambda_logits
    config.distillation.epochs = args.epochs
    config.distillation.save_path = args.output_dir
    config.distillation.checkpoint_interval = args.checkpoint_interval
    config.distillation.resume_from_checkpoint = args.resume_from_checkpoint

    # Determine device
    device = torch.device(config.distillation.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Ensure output directory exists
    ensure_output_directory(args.output_dir)
    logger.info(f"Output directory ensured: {args.output_dir}")

    # Log training configuration
    logger.info(f"Starting knowledge distillation with config:")
    logger.info(f"  Teacher model: {args.teacher_model}")
    logger.info(f"  Student model: {args.student_model}")
    logger.info(f"  Teacher use LoRA: {args.teacher_use_lora}")
    if args.teacher_use_lora:
        logger.info(f"  Teacher LoRA path: {args.teacher_lora_path}")
    logger.info(f"  Distillation types: {args.distill_types}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Lambda embed: {args.lambda_embed}")
    logger.info(f"  Lambda similarity: {args.lambda_similarity}")
    logger.info(f"  Lambda logits: {args.lambda_logits}")
    logger.info(f"  Epochs: {args.epochs}")

    # Load models and processor
    logger.info("Loading teacher and student models...")
    teacher_model, student_model, processor = load_models_and_processor(
        args.teacher_model, args.student_model, device,
        teacher_use_lora=args.teacher_use_lora,
        teacher_lora_path=args.teacher_lora_path
    )

    # Initialize trainer with processor
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        device=device,
        distill_types=args.distill_types,
        temperature=args.temperature,
        lambda_embed=args.lambda_embed,
        lambda_similarity=args.lambda_similarity,
        lambda_logits=args.lambda_logits,
        learning_rate=args.lr,
        processor=processor
    )

    # Load dataset
    logger.info("Loading dataset...")

    # Use get_data_loaders from dataset module to get train/val/test loaders
    data_loaders = get_data_loaders(processor)

    train_dataloader = data_loaders['train']
    logger.info(f"Created train dataloader with {len(train_dataloader)} batches per epoch")

    val_dataloader = data_loaders['val']
    logger.info(f"Created validation dataloader with {len(val_dataloader)} batches")

    global_step = 0
    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.epochs}")

        epoch_total_loss = 0
        epoch_clip_loss = 0
        epoch_embed_loss = 0
        epoch_sim_loss = 0
        epoch_logits_loss = 0  # Add this for the new logits loss

        # Create tqdm progress bar for the current epoch
        progress_bar = tqdm(enumerate(train_dataloader),
                            total=len(train_dataloader),
                            desc=f"Epoch {epoch+1}/{args.epochs}",
                            leave=False)

        for step, batch in progress_bar:
            # Move batch to device
            batch = move_batch_to_device(batch, device)

            # Perform training step
            loss_dict = trainer.train_step(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )

            # Accumulate losses for logging
            epoch_total_loss += loss_dict['total_loss'].item()
            epoch_clip_loss += loss_dict['clip_loss'].item()

            # Handle both old and new loss dictionaries
            if 'embed_loss' in loss_dict:
                embed_loss_val = loss_dict['embed_loss']
                epoch_embed_loss += embed_loss_val.item() if hasattr(embed_loss_val, 'item') else embed_loss_val
            if 'sim_loss' in loss_dict:
                sim_loss_val = loss_dict['sim_loss']
                epoch_sim_loss += sim_loss_val.item() if hasattr(sim_loss_val, 'item') else sim_loss_val
            if 'logits_loss' in loss_dict:
                logits_loss_val = loss_dict['logits_loss']
                epoch_logits_loss += logits_loss_val.item() if hasattr(logits_loss_val, 'item') else logits_loss_val

            # Update progress bar with current losses
            total_loss_item = loss_dict['total_loss'].item() if hasattr(loss_dict['total_loss'], 'item') else loss_dict['total_loss']
            clip_loss_item = loss_dict['clip_loss'].item() if hasattr(loss_dict['clip_loss'], 'item') else loss_dict['clip_loss']
            lr_value = trainer.get_current_lr()

            progress_bar.set_postfix({
                'total_loss': f"{total_loss_item:.4f}",
                'clip_loss': f"{clip_loss_item:.4f}",
                'lr': f"{lr_value:.2e}"
            })

            # Log training step periodically
            # if step % 10 == 0:  # Log every 10 steps
            #     log_training_step(
            #         epoch=epoch+1,
            #         step=step+1,
            #         total_steps=len(train_dataloader),
            #         loss_dict=loss_dict,
            #         lr=trainer.get_current_lr()
            #     )

            # global_step += 1

        # Calculate average losses for the epoch
        avg_total_loss = epoch_total_loss / len(train_dataloader)
        avg_clip_loss = epoch_clip_loss / len(train_dataloader)
        avg_embed_loss = epoch_embed_loss / len(train_dataloader) if epoch_embed_loss > 0 else 0
        avg_sim_loss = epoch_sim_loss / len(train_dataloader) if epoch_sim_loss > 0 else 0
        avg_logits_loss = epoch_logits_loss / len(train_dataloader) if epoch_logits_loss > 0 else 0

        logger.info(f"Epoch {epoch+1} completed. Average losses:")
        avg_total_loss_item = avg_total_loss.item() if hasattr(avg_total_loss, 'item') else avg_total_loss
        avg_clip_loss_item = avg_clip_loss.item() if hasattr(avg_clip_loss, 'item') else avg_clip_loss
        logger.info(f"  Total Loss: {avg_total_loss_item:.4f}")
        logger.info(f"  CLIP Loss: {avg_clip_loss_item:.4f}")
        if avg_embed_loss > 0:
            avg_embed_loss_item = avg_embed_loss.item() if hasattr(avg_embed_loss, 'item') else avg_embed_loss
            logger.info(f"  Embed Loss: {avg_embed_loss_item:.4f}")
        if avg_sim_loss > 0:
            avg_sim_loss_item = avg_sim_loss.item() if hasattr(avg_sim_loss, 'item') else avg_sim_loss
            logger.info(f"  Sim Loss: {avg_sim_loss_item:.4f}")
        if avg_logits_loss > 0:
            avg_logits_loss_item = avg_logits_loss.item() if hasattr(avg_logits_loss, 'item') else avg_logits_loss
            logger.info(f"  Logits Loss: {avg_logits_loss_item:.4f}")

        # Evaluate on validation set
        # val_metrics = trainer.evaluate(val_dataloader)
        # for metric, value in val_metrics.items():
        #     logger.info(f"  {metric}: {value:.4f}")


        # Save checkpoint if required
        if (epoch + 1) % args.checkpoint_interval == 0:
            epoch_model_path = os.path.join(args.output_dir, f"student_model_epoch_{epoch+1}")
            trainer.save_student_model(epoch_model_path)
            logger.info(f"Student model at epoch {epoch+1} saved at {epoch_model_path}")


    # Save final student model
    # logger.info("Saving final student model...")
    # final_model_path = os.path.join(args.output_dir, "final_student_model")
    # trainer.save_student_model(final_model_path)
    # logger.info(f"Final student model saved at {final_model_path}")

    logger.info("Training completed!")


if __name__ == "__main__":
    main()