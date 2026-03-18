"""Hard negative trainer for CLIP + LoRA model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm
import os
import logging
from config import config
from utils.logger import get_models_logger
from dataset.dataset import get_data_loaders
from training.hard_negative_loss import hard_negative_contrastive_loss, compute_hard_negative_metrics
import numpy as np


logger = get_models_logger()


class CLIPLoRAHardNegativeTrainer:
    """Trainer for fine-tuning CLIP with LoRA using hard negative mining"""

    def __init__(self, model_name: str = None):
        """
        Initialize the trainer with CLIP model and LoRA configuration

        Args:
            model_name: Name or path of the CLIP model to load
        """
        if model_name is None:
            model_name = config.model.model_name

        self.device = torch.device(config.model.device if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading CLIP model: {model_name} on {self.device}")

        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)

        # Apply LoRA to text encoder
        self.apply_lora()

        # Move model to device
        self.model.to(self.device)

        # Set up training components
        self.optimizer = None
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        logger.info("CLIP model with LoRA loaded successfully")

    def apply_lora(self):
        """Apply LoRA to the text encoder of the CLIP model"""
        # Apply LoRA to the text encoder
        lora_config = LoraConfig(
            inference_mode=False,
            r=config.training.lora_r,
            lora_alpha=config.training.lora_alpha,
            lora_dropout=config.training.lora_dropout,
            target_modules=config.training.target_modules
        )

        # Freeze vision model first
        for param in self.model.vision_model.parameters():
            param.requires_grad = False

        # Apply LoRA to the text model specifically
        self.model.text_model = get_peft_model(self.model.text_model, lora_config)

        # After applying PEFT, make sure only LoRA parameters in text model are trainable
        for name, param in self.model.named_parameters():
            if "text_model" in name and "lora_" in name:
                param.requires_grad = True
            elif "text_model" in name and "lora_" not in name:
                param.requires_grad = False  # Keep text model frozen except LoRA
            elif "vision_model" in name:
                param.requires_grad = False  # Keep vision model frozen
            else:
                param.requires_grad = False  # Other parameters (like logit_scale)

        logger.info("LoRA applied to text encoder")
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        lora_params = sum(p.numel() for n, p in self.model.named_parameters() if 'lora_' in n)

        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"LoRA parameters: {lora_params:,}")
        logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")

    def prepare_optimizer(self, learning_rate: float = 1e-4):
        """
        Prepare optimizer for training (only LoRA parameters)

        Args:
            learning_rate: Learning rate for the optimizer
        """
        # Only optimize LoRA parameters
        lora_params = [p for n, p in self.model.named_parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(lora_params, lr=learning_rate)
        logger.info(f"Optimizer prepared with {len(lora_params)} trainable parameters")

    def train_epoch(self, dataloader, temperature: float = 0.07, lambda_hard: float = 0.1, hard_negative_type: str = "batch"):
        """
        Train the model for one epoch using hard negative contrastive loss.

        Args:
            dataloader: DataLoader for training data
            temperature: Temperature for contrastive loss
            lambda_hard: Weight for hard negative loss
            hard_negative_type: Type of hard negative mining ("batch" or "category")

        Returns:
            Average loss over the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_clip_loss = 0.0
        total_hard_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc="Training")

        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Extract category information if available
            categories = batch.get('category', None)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision if available
            if torch.cuda.is_available() and self.scaler is not None:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    # Image embeddings
                    image_features = self.model.get_image_features(pixel_values)

                    # Text embeddings
                    text_features = self.model.get_text_features(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                    image_features = F.normalize(image_features, dim=-1)
                    text_features = F.normalize(text_features, dim=-1)

                    # Compute hard negative contrastive loss
                    loss, clip_loss, hard_loss = hard_negative_contrastive_loss(
                        image_features=image_features,
                        text_features=text_features,
                        categories=categories,
                        temperature=temperature,
                        lambda_hard=lambda_hard,
                        hard_negative_type=hard_negative_type
                    )
            else:
                # Image embeddings
                image_features = self.model.get_image_features(pixel_values)

                # Text embeddings
                text_features = self.model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)

                # Compute hard negative contrastive loss
                loss, clip_loss, hard_loss = hard_negative_contrastive_loss(
                    image_features=image_features,
                    text_features=text_features,
                    categories=categories,
                    temperature=temperature,
                    lambda_hard=lambda_hard,
                    hard_negative_type=hard_negative_type
                )

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            total_clip_loss += clip_loss.item()
            total_hard_loss += hard_loss.item()
            num_batches += 1

            progress_bar.set_postfix({
                "total_loss": loss.item(),
                "avg_total_loss": total_loss / num_batches,
                "avg_clip_loss": total_clip_loss / num_batches,
                "avg_hard_loss": total_hard_loss / num_batches
            })

        avg_loss = total_loss / num_batches
        avg_clip_loss = total_clip_loss / num_batches
        avg_hard_loss = total_hard_loss / num_batches

        return avg_loss, avg_clip_loss, avg_hard_loss

    def evaluate(self, val_dataloader, hard_negative_type: str = "batch"):
        """
        Evaluate the model on validation set

        Args:
            val_dataloader: Validation DataLoader
            hard_negative_type: Type of hard negative evaluation ("batch" or "category")

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_image_features = []
        all_text_features = []
        all_categories = []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating"):
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Image embeddings
                image_features = self.model.get_image_features(pixel_values)

                # Text embeddings
                text_features = self.model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)

                all_image_features.append(image_features.cpu())
                all_text_features.append(text_features.cpu())

                # Collect categories if available
                if 'category' in batch:
                    all_categories.extend(batch['category'])

        # Concatenate all features
        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(all_image_features, all_text_features.T)

        # Determine categories for metrics computation
        categories_for_metrics = all_categories if all_categories else None

        # Compute metrics
        metrics = compute_hard_negative_metrics(
            similarity_matrix,
            categories=categories_for_metrics,
            k_values=[1, 5]
        )

        return metrics

    def train(self, num_epochs: int = 5, learning_rate: float = 1e-4, temperature: float = 0.07,
              lambda_hard: float = 0.1, hard_negative_type: str = "batch",
              save_path: str = "lora_adapter", save_every_n_epochs: int = 1):
        """
        Train the model with LoRA and hard negative mining

        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
            temperature: Temperature for contrastive loss
            lambda_hard: Weight for hard negative loss
            hard_negative_type: Type of hard negative mining ("batch" or "category")
            save_path: Path to save the trained LoRA adapter
            save_every_n_epochs: Save checkpoint every n epochs
        """
        # Prepare optimizer
        self.prepare_optimizer(learning_rate)

        # Get data loaders with category support
        data_loaders = get_data_loaders(self.processor, use_category=True)
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']

        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Using hard negative type: {hard_negative_type}, lambda_hard: {lambda_hard}")
        logger.info(f"Saving checkpoints every {save_every_n_epochs} epoch(s)...")

        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

            # Train for one epoch
            train_loss, train_clip_loss, train_hard_loss = self.train_epoch(
                train_loader,
                temperature,
                lambda_hard,
                hard_negative_type
            )
            logger.info(f"Epoch {epoch + 1} - Total Training Loss: {train_loss:.4f}, "
                       f"CLIP Loss: {train_clip_loss:.4f}, Hard Loss: {train_hard_loss:.4f}")

            # Evaluate on validation set
            val_metrics = self.evaluate(val_loader, hard_negative_type)
            logger.info(f"Epoch {epoch + 1} - Validation Metrics:")
            for metric, value in val_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")

            # Save checkpoint if it's the right epoch
            if (epoch + 1) % save_every_n_epochs == 0:
                checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch + 1}")
                self.save_checkpoint(checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}")

        # Save final model
        self.save_checkpoint(save_path)
        logger.info(f"Final model saved to {save_path}")

    def save_checkpoint(self, path: str):
        """
        Save the LoRA checkpoint

        Args:
            path: Path to save the checkpoint
        """
        os.makedirs(path, exist_ok=True)

        # Save LoRA adapter
        self.model.text_model.save_pretrained(path)

        # Also save the processor in the same directory for easy loading
        self.processor.save_pretrained(path)

    def load_checkpoint(self, path: str):
        """
        Load a LoRA checkpoint

        Args:
            path: Path to the checkpoint
        """
        # Load the PEFT model from the checkpoint
        self.model.text_model = PeftModel.from_pretrained(self.model.text_model, path)
        for name, param in self.model.named_parameters():
            if "text_model" in name and "lora_" in name:
                param.requires_grad = True
            elif "text_model" in name and "lora_" not in name:
                param.requires_grad = False  # Keep text model frozen except LoRA
            elif "vision_model" in name:
                param.requires_grad = False  # Keep vision model frozen
            else:
                param.requires_grad = False  # Other parameters (like logit_scale)
        logger.info(f"LoRA checkpoint loaded from {path}")


def main():
    """Main training function"""
    trainer = CLIPLoRAHardNegativeTrainer()

    # Train the model with hard negative parameters
    trainer.train(
        num_epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        temperature=config.training.temperature,
        lambda_hard=config.training.lambda_hard,
        hard_negative_type=config.training.hard_negative_type,
        save_path=config.training.save_path,
        save_every_n_epochs=config.training.save_every_n_epochs
    )


if __name__ == "__main__":
    main()