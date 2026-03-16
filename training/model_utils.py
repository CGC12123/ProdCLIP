"""
Utility functions for loading fine-tuned CLIP models with LoRA
"""
import torch
from transformers import CLIPModel, CLIPProcessor
import logging
from config import config
from utils.logger import get_models_logger


logger = get_models_logger()


def load_fine_tuned_model(base_model_name: str, lora_path: str):
    """
    Load a base CLIP model with fine-tuned LoRA adapter

    NOTE: This function requires PEFT library to be installed to work with LoRA adapters.
    Install with: pip install peft

    Args:
        base_model_name: Name of the base CLIP model
        lora_path: Path to the LoRA adapter

    Returns:
        Fine-tuned CLIP model with LoRA applied
    """
    logger.info(f"Loading base model: {base_model_name}")

    # Load the base model
    base_model = CLIPModel.from_pretrained(base_model_name)

    # If PEFT is installed, load LoRA adapter
    try:
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter from: {lora_path}")

        # Load the LoRA adapter
        model = PeftModel.from_pretrained(base_model, lora_path)

        # Set to evaluation mode
        model.eval()

        logger.info("Fine-tuned model with LoRA loaded successfully")

        # Load processor from the LoRA path (should have been saved alongside)
        processor = CLIPProcessor.from_pretrained(lora_path)
    except ImportError:
        logger.warning("PEFT library not installed. Loading base model without LoRA adapter.")
        model = base_model
        processor = CLIPProcessor.from_pretrained(base_model_name)

    return model, processor


def merge_lora_weights(model, base_model_name: str):
    """
    Merge LoRA weights with the base model to create a standalone model

    Args:
        model: PEFT model with LoRA (requires PEFT library)
        base_model_name: Name of the base model

    Returns:
        CLIP model with LoRA weights merged
    """
    try:
        # Merge the LoRA layers with the base model
        merged_model = model.merge_and_unload()

        logger.info("LoRA weights merged successfully")

        return merged_model
    except AttributeError:
        logger.warning("Cannot merge LoRA weights - PEFT library or LoRA adapter not loaded.")
        return model