#!/usr/bin/env python3
"""
LoRA fine-tuning script for CLIP text encoder
This script fine-tunes the CLIP model using LoRA on your specific dataset.
"""

import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import logging

from config import config
from utils.seed import set_seed
from utils.logger import get_models_logger
from dataset.dataset import get_data_loaders
from training.contrastive_loss import contrastive_loss, compute_metrics
from training.trainer import CLIPLoRATrainer


logger = get_models_logger()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP with LoRA")
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
    parser.add_argument("--save_path", type=str, 
                        default=config.training.save_path, 
                        help="Path to save the LoRA adapter")
    parser.add_argument("--save_every_n_epochs", type=int, 
                        default=config.training.save_every_n_epochs, 
                        help="Save checkpoint every n epochs")

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(42)

    # Update config if needed
    config.model.batch_size = args.batch_size

    logger.info("Starting CLIP LoRA fine-tuning...")
    logger.info(f"Parameters: epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}")

    # Create trainer and start training
    trainer = CLIPLoRATrainer()

    trainer.train(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        temperature=args.temperature,
        save_path=args.save_path,
        save_every_n_epochs=args.save_every_n_epochs
    )

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()