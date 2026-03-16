import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import logging
from config import config
from utils.logger import get_models_logger


logger = get_models_logger()


class CLIPRetrievalModel:
    """CLIP model wrapper for multimodal retrieval"""

    def __init__(self, model_name: str = None, lora_path: str = None):
        """
        Initialize the CLIP model

        Args:
            model_name: Name or path of the CLIP model to load
            lora_path: Path to LoRA adapter (optional)
        """
        if model_name is None:
            model_name = config.model.model_name

        self.device = torch.device(config.model.device if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading CLIP model: {model_name} on {self.device}")

        # Load model and processor
        if lora_path:
            # Import PEFT only when needed to avoid dependency issues
            try:
                from peft import PeftModel
                # Load base model first
                # base_model = CLIPModel.from_pretrained(model_name)
                self.model = CLIPModel.from_pretrained(model_name)
                # Load with LoRA adapter
                self.model.text_model = PeftModel.from_pretrained(
                    self.model.text_model,
                    lora_path
                )
                logger.info(f"Loaded base model with LoRA adapter from: {lora_path}")
            except ImportError:
                logger.warning("PEFT library not installed. Loading base model without LoRA adapter.")
                self.model = CLIPModel.from_pretrained(model_name)
        else:
            self.model = CLIPModel.from_pretrained(model_name)
            logger.info("Loaded base CLIP model without LoRA")

        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)

        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        logger.info("CLIP model loaded successfully")

    def encode_images(self, pixel_values):
        """
        Encode images to embeddings

        Args:
            pixel_values: Preprocessed image tensors

        Returns:
            Normalized image embeddings
        """
        with torch.no_grad():
            # Call get_image_features which should return pooled image features
            image_features = self.model.get_image_features(pixel_values.to(self.device))

            # L2 normalize the embeddings
            image_features = F.normalize(image_features, p=2, dim=-1)

        return image_features.cpu()

    def encode_texts(self, input_ids, attention_mask):
        """
        Encode texts to embeddings

        Args:
            input_ids: Tokenized text input IDs
            attention_mask: Attention mask for the text

        Returns:
            Normalized text embeddings
        """
        with torch.no_grad():
            # Call get_text_features which should return pooled text features
            text_features = self.model.get_text_features(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device)
            )

            # L2 normalize the embeddings
            text_features = F.normalize(text_features, p=2, dim=-1)

        return text_features.cpu()

    def compute_similarities(self, query_embeddings, reference_embeddings):
        """
        Compute cosine similarities between query and reference embeddings

        Args:
            query_embeddings: Query embeddings
            reference_embeddings: Reference embeddings

        Returns:
            Similarity matrix of shape (num_queries, num_references)
        """
        # Cosine similarity is computed as dot product for L2 normalized vectors
        similarities = torch.matmul(query_embeddings, reference_embeddings.t())
        return similarities

    def get_image_features_batch(self, batch):
        """
        Extract image features from a batch of data

        Args:
            batch: Batch of data containing pixel_values

        Returns:
            Normalized image embeddings
        """
        return self.encode_images(batch['pixel_values'])

    def get_text_features_batch(self, batch):
        """
        Extract text features from a batch of data

        Args:
            batch: Batch of data containing input_ids and attention_mask

        Returns:
            Normalized text embeddings
        """
        return self.encode_texts(batch['input_ids'], batch['attention_mask'])

    def encode_single_image(self, image_path):
        """
        Encode a single image from file path

        Args:
            image_path: Path to the image file

        Returns:
            Normalized image embedding
        """
        from PIL import Image
        image = Image.open(image_path).convert('RGB')

        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        pixel_values = inputs['pixel_values']

        return self.encode_images(pixel_values)

    def encode_single_text(self, text):
        """
        Encode a single text

        Args:
            text: Input text string

        Returns:
            Normalized text embedding
        """
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=config.model.max_length)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        return self.encode_texts(input_ids, attention_mask)