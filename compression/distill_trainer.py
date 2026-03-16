import torch
import torch.nn as nn
from transformers import CLIPModel
from compression.distill_loss import combined_distillation_loss
import torch.nn.functional as F
from utils.logger import get_compression_logger
from tqdm import tqdm

logger = get_compression_logger()


class DistillationTrainer:
    """
    A trainer class for performing knowledge distillation from a teacher CLIP model to a student CLIP model.
    """

    def __init__(
        self,
        teacher_model: CLIPModel,
        student_model: CLIPModel,
        device,
        distill_types=["embedding", "similarity"],
        temperature: float = 1.0,
        lambda_embed: float = 0.5,
        lambda_similarity: float = 0.5,
        lambda_logits: float = 0.5,
        learning_rate: float = 1e-5,
        processor=None
    ):
        """
        Initialize the DistillationTrainer.

        Args:
            teacher_model: Pre-trained teacher model (frozen)
            student_model: Student model to be trained
            device: Device to run the models on
            distill_types: List of distillation methods to use ("embedding", "similarity", "logits")
            temperature: Temperature for similarity distillation
            lambda_embed: Weight for embedding distillation loss
            lambda_similarity: Weight for similarity distillation loss
            lambda_logits: Weight for logits distillation loss
            learning_rate: Learning rate for optimizer
            processor: Processor to be saved with the model
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = device
        self.distill_types = distill_types
        self.temperature = temperature
        self.lambda_embed = lambda_embed
        self.lambda_similarity = lambda_similarity
        self.lambda_logits = lambda_logits
        self.processor = processor  # Store processor reference

        # Set teacher model to eval mode and freeze parameters
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Set student model to train mode
        self.student_model.train()

        # Determine embedding dimensions by getting sample outputs
        dummy_pixel_values = torch.randn(1, 3, 224, 224, device=device)
        dummy_input_ids = torch.randint(0, 1000, (1, 77), device=device)
        dummy_attention_mask = torch.ones(1, 77, dtype=torch.long, device=device)

        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                pixel_values=dummy_pixel_values,
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask
            )
            teacher_image_dim = teacher_outputs.image_embeds.shape[-1]
            teacher_text_dim = teacher_outputs.text_embeds.shape[-1]

            student_outputs = self.student_model(
                pixel_values=dummy_pixel_values,
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask
            )
            student_image_dim = student_outputs.image_embeds.shape[-1]
            student_text_dim = student_outputs.text_embeds.shape[-1]

        # Create projection layers to match teacher dimensions
        self.image_projection = nn.Linear(student_image_dim, teacher_image_dim).to(device)
        self.text_projection = nn.Linear(student_text_dim, teacher_text_dim).to(device)

        # Add projection layer parameters to optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.student_model.parameters()) +
            list(self.image_projection.parameters()) +
            list(self.text_projection.parameters()),
            lr=learning_rate
        )

        # Setup loss function
        self.clip_loss_fn = nn.CrossEntropyLoss()

        logger.info(f"Initialized DistillationTrainer with teacher and student models")
        logger.info(f"Teacher image embedding dim: {teacher_image_dim}, text embedding dim: {teacher_text_dim}")
        logger.info(f"Student image embedding dim: {student_image_dim}, text embedding dim: {student_text_dim}")
        logger.info(f"Image projection layer: {student_image_dim} -> {teacher_image_dim}")
        logger.info(f"Text projection layer: {student_text_dim} -> {teacher_text_dim}")
        logger.info(f"Using distillation types: {distill_types}")

    def compute_clip_loss(self, image_features, text_features):
        """
        Compute the original CLIP contrastive loss.

        Args:
            image_features: Image embeddings from the model
            text_features: Text embeddings from the model

        Returns:
            CLIP contrastive loss
        """
        # Normalize embeddings
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # Calculate logits
        logits_per_image = torch.matmul(image_features, text_features.t()) * 100  # CLIP uses 100 as scale
        logits_per_text = logits_per_image.t()

        # Create ground truth labels
        batch_size = image_features.size(0)
        ground_truth = torch.arange(batch_size, dtype=torch.long, device=self.device)

        # Calculate losses
        clip_loss = (self.clip_loss_fn(logits_per_image, ground_truth) +
                     self.clip_loss_fn(logits_per_text, ground_truth)) / 2

        return clip_loss

    def forward_pass(self, pixel_values, input_ids, attention_mask):
        """
        Perform a forward pass with both teacher and student models.

        Args:
            pixel_values: Image pixel values
            input_ids: Text input IDs
            attention_mask: Text attention mask

        Returns:
            Dictionary containing all necessary embeddings and computed losses
        """
        # Ensure inputs are on the correct device
        pixel_values = pixel_values.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Forward pass through teacher model with no gradient computation
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            teacher_image_emb = teacher_outputs.image_embeds
            teacher_text_emb = teacher_outputs.text_embeds

        # Forward pass through student model
        student_outputs = self.student_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        student_image_emb = student_outputs.image_embeds
        student_text_emb = student_outputs.text_embeds

        # Project student embeddings to match teacher embedding dimensions
        projected_student_image_emb = self.image_projection(student_image_emb)
        projected_student_text_emb = self.text_projection(student_text_emb)

        # Compute original CLIP loss using projected embeddings
        clip_loss = self.compute_clip_loss(projected_student_image_emb, projected_student_text_emb)

        # Compute distillation loss using projected student embeddings and original teacher embeddings
        loss_dict = combined_distillation_loss(
            student_image_emb=projected_student_image_emb,
            student_text_emb=projected_student_text_emb,
            teacher_image_emb=teacher_image_emb,
            teacher_text_emb=teacher_text_emb,
            clip_loss=clip_loss,
            distill_types=self.distill_types,
            lambda_embed=self.lambda_embed,
            lambda_similarity=self.lambda_similarity,
            lambda_logits=self.lambda_logits,
            temperature=self.temperature
        )

        return {
            'loss_dict': loss_dict,
            'student_image_emb': projected_student_image_emb,
            'student_text_emb': projected_student_text_emb,
            'teacher_image_emb': teacher_image_emb,
            'teacher_text_emb': teacher_text_emb
        }

    def train_step(self, pixel_values, input_ids, attention_mask):
        """
        Perform a single training step.

        Args:
            pixel_values: Image pixel values
            input_ids: Text input IDs
            attention_mask: Text attention mask

        Returns:
            Dictionary with loss values
        """
        self.optimizer.zero_grad()

        outputs = self.forward_pass(pixel_values, input_ids, attention_mask)
        total_loss = outputs['loss_dict']['total_loss']

        # Backpropagate
        total_loss.backward()

        # Update student model parameters
        self.optimizer.step()

        return outputs['loss_dict']

    def evaluate(self, dataloader):
        """
        Evaluate the student model on the given dataloader.

        Args:
            dataloader: Dataloader for evaluation

        Returns:
            Average loss values over the evaluation set
        """
        self.student_model.eval()

        total_loss = 0
        total_clip_loss = 0
        total_embed_loss = 0
        total_sim_loss = 0
        total_logits_loss = 0
        num_batches = 0

        with torch.no_grad():
            # Create tqdm progress bar for evaluation
            progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

            for batch in progress_bar:
                # Handle both dict-style and tuple-style batches
                if isinstance(batch, dict):
                    # Dict-style batch (as expected in normal operation)
                    pixel_values = batch.get('pixel_values', batch.get('images'))
                    input_ids = batch.get('input_ids')
                    attention_mask = batch.get('attention_mask')
                else:
                    # Tuple-style batch (for testing purposes)
                    pixel_values, input_ids, attention_mask = batch

                # Ensure we have the required inputs
                if pixel_values is None or input_ids is None:
                    continue

                outputs = self.forward_pass(pixel_values, input_ids, attention_mask)
                loss_dict = outputs['loss_dict']

                total_loss += loss_dict['total_loss'].item()
                total_clip_loss += loss_dict['clip_loss'].item()

                # Handle both old and new loss dictionaries
                if 'embed_loss' in loss_dict:
                    total_embed_loss += loss_dict['embed_loss'].item() if hasattr(loss_dict['embed_loss'], 'item') else loss_dict['embed_loss']
                if 'sim_loss' in loss_dict:
                    total_sim_loss += loss_dict['sim_loss'].item() if hasattr(loss_dict['sim_loss'], 'item') else loss_dict['sim_loss']
                if 'logits_loss' in loss_dict:
                    total_logits_loss += loss_dict['logits_loss'].item() if hasattr(loss_dict['logits_loss'], 'item') else loss_dict['logits_loss']

                num_batches += 1

                total_loss = loss_dict['total_loss'].item() if hasattr(loss_dict['total_loss'], 'item') else loss_dict['total_loss']
                clip_loss = loss_dict['clip_loss'].item() if hasattr(loss_dict['clip_loss'], 'item') else loss_dict['clip_loss']

                # Update progress bar with current metrics
                progress_bar.set_postfix({
                    'total_loss': f"{total_loss:.4f}",
                    'clip_loss': f"{clip_loss:.4f}"
                })

        self.student_model.train()  # Switch back to train mode

        result = {
            'total_loss': total_loss / num_batches if num_batches > 0 else 0,
            'clip_loss': total_clip_loss / num_batches if num_batches > 0 else 0,
        }

        # Only add loss components that exist in the dictionary
        if num_batches > 0:
            if total_embed_loss > 0:
                result['embed_loss'] = total_embed_loss / num_batches
            if total_sim_loss > 0:
                result['sim_loss'] = total_sim_loss / num_batches
            if total_logits_loss > 0:
                result['logits_loss'] = total_logits_loss / num_batches

        return result

    def save_student_model(self, save_path):
        """
        Save the trained student model and processor.

        Args:
            save_path: Path to save the model
        """
        self.student_model.save_pretrained(save_path)

        # Save processor if available
        if self.processor is not None:
            self.processor.save_pretrained(save_path)

        logger.info(f"Student model and processor saved to {save_path}")

    def update_learning_rate(self, new_lr):
        """
        Update the learning rate of the optimizer.

        Args:
            new_lr: New learning rate value
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        logger.debug(f"Updated learning rate to {new_lr}")

    def get_current_lr(self):
        """
        Get the current learning rate.

        Returns:
            Current learning rate value
        """
        return self.optimizer.param_groups[0]['lr']