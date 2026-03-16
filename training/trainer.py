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
from training.contrastive_loss import contrastive_loss, compute_metrics
import numpy as np


logger = get_models_logger()


class CLIPLoRATrainer:
    """Trainer for fine-tuning CLIP with LoRA"""

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
        # Instead of trying to specify target modules with regex,
        # we will apply PEFT to the text model directly
        # First, let's define the LoRA config without target_modules initially
        lora_config = LoraConfig(
            # task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=config.training.lora_r,
            lora_alpha=config.training.lora_alpha,
            lora_dropout=config.training.lora_dropout,
            target_modules=config.training.target_modules
        )

        # Apply LoRA to the text model part of CLIP separately to avoid the conflict
        # Freeze vision model first
        for param in self.model.vision_model.parameters():
            param.requires_grad = False

        # Apply LoRA to the text model specifically
        # Since PEFT has trouble with the whole CLIP model, we need a different approach
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

    def train_epoch(self, dataloader, temperature: float = 0.07):
        """
        Train the model for one epoch using contrastive loss (LoRA-compatible).

        Args:
            dataloader: DataLoader for training data
            temperature: Temperature for contrastive loss

        Returns:
            Average loss over the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc="Training")

        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

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

                    # Compute contrastive loss
                    loss = contrastive_loss(
                        image_features=image_features,
                        text_features=text_features,
                        temperature=temperature
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

                # Compute contrastive loss
                loss = contrastive_loss(
                    image_features=image_features,
                    text_features=text_features,
                    temperature=temperature
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
            num_batches += 1
            progress_bar.set_postfix({"loss": loss.item(), "avg_loss": total_loss / num_batches})

        avg_loss = total_loss / num_batches
        return avg_loss

    def evaluate(self, val_dataloader):
        """
        Evaluate the model on validation set

        Args:
            val_dataloader: Validation DataLoader

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_image_features = []
        all_text_features = []

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

        # Concatenate all features
        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(all_image_features, all_text_features.T)

        # Compute metrics
        metrics = compute_metrics(similarity_matrix)

        return metrics

    def train(self, num_epochs: int = 5, learning_rate: float = 1e-4, temperature: float = 0.07,
              save_path: str = "lora_adapter", save_every_n_epochs: int = 1):
        """
        Train the model with LoRA

        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
            temperature: Temperature for contrastive loss
            save_path: Path to save the trained LoRA adapter
            save_every_n_epochs: Save checkpoint every n epochs
        """
        # Prepare optimizer
        self.prepare_optimizer(learning_rate)

        # Get data loaders
        data_loaders = get_data_loaders(self.processor)
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']

        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Saving checkpoints every {save_every_n_epochs} epoch(s)...")

        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

            # Train for one epoch
            train_loss = self.train_epoch(train_loader, temperature)
            logger.info(f"Epoch {epoch + 1} - Training Loss: {train_loss:.4f}")

            # Evaluate on validation set
            val_metrics = self.evaluate(val_loader)
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
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, path)
        logger.info(f"LoRA checkpoint loaded from {path}")


def main():
    """Main training function"""
    trainer = CLIPLoRATrainer()

    # Train the model
    trainer.train(
        num_epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        temperature=config.training.temperature,
        save_path=config.training.save_path,
        save_every_n_epochs=config.training.save_every_n_epochs
    )


if __name__ == "__main__":
    main()