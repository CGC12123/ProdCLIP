#!/usr/bin/env python3
"""
Hard Negative fine-tuning script for CLIP text encoder with LoRA
This script fine-tunes the CLIP model using LoRA with hard negative mining on your specific dataset.
"""

import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import logging

from config import config
from utils.seed import set_seed
from utils.logger import get_models_logger
from dataset.dataset import get_data_loaders
from training.hard_negative_loss import hard_negative_contrastive_loss
from training.hard_negative_trainer import CLIPLoRAHardNegativeTrainer


logger = get_models_logger()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP with LoRA using hard negative mining")
    parser.add_argument("--epochs", type=int,
                        default=config.training.epochs,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float,
                        default=config.training.learning_rate,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int,
                        default=config.training.batch_size,
                        help="Batch size")
    parser.add_argument("--temperature", type=float,
                        default=config.training.temperature,
                        help="Temperature for contrastive loss")
    parser.add_argument("--lambda_hard", type=float,
                        default=config.training.lambda_hard,
                        help="Weight for hard negative loss")
    parser.add_argument("--hard_negative_type", type=str,
                        default=config.training.hard_negative_type,
                        choices=["batch", "category"],
                        help="Type of hard negative mining")
    parser.add_argument("--save_path", type=str,
                        default=config.training.save_path,
                        help="Path to save the LoRA adapter")
    parser.add_argument("--save_every_n_epochs", type=int,
                        default=config.training.save_every_n_epochs,
                        help="Save checkpoint every n epochs")
    parser.add_argument("--load_checkpoint", type=str,
                        default=config.training.lora_path,
                        help="Path to load existing LoRA checkpoint for continued training")

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(42)

    # Update config if needed
    config.model.batch_size = args.batch_size
    config.training.lambda_hard = args.lambda_hard
    config.training.hard_negative_type = args.hard_negative_type

    logger.info("Starting CLIP LoRA fine-tuning with hard negative mining...")
    logger.info(f"Parameters: epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}")
    logger.info(f"Hard negative params: lambda_hard={args.lambda_hard}, type={args.hard_negative_type}")

    # Create trainer
    trainer = CLIPLoRAHardNegativeTrainer()

    # Load checkpoint if provided
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        logger.info(f"Loading existing LoRA checkpoint from {args.load_checkpoint}")
        trainer.load_checkpoint(args.load_checkpoint)

    # Start training
    trainer.train(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        temperature=args.temperature,
        lambda_hard=args.lambda_hard,
        hard_negative_type=args.hard_negative_type,
        save_path=args.save_path,
        save_every_n_epochs=args.save_every_n_epochs
    )

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()