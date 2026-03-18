# CLIP-based Product Image–Text Retrieval with LoRA and Knowledge Distillation

A comprehensive multimodal retrieval system built with Hugging Face's CLIP model for cross-modal search between images and text, featuring LoRA fine-tuning capabilities for domain-specific performance enhancement and knowledge distillation for model compression.

## Overview

This project implements a production-ready multimodal retrieval system that supports:

1. **Text-to-Image Retrieval**: Find relevant images based on text descriptions
2. **Image-to-Text Retrieval**: Find relevant text descriptions based on images
3. **Performance Evaluation**: Compute Recall@1 and Recall@5 metrics
4. **Embedding Analysis**: Analyze embedding distributions and similarity patterns
5. **Interactive CLI**: Command-line interface for real-time searching
6. **LoRA Fine-tuning**: Low-rank adaptation for domain-specific model improvement
7. **Knowledge Distillation**: Compress large teacher models to smaller, efficient student models
8. **Hard Negative Training**: Enhanced training with hard negative mining for improved discrimination

## Architecture

The system follows a modular architecture:

```
project/
├── config.py                 # Configuration management (includes LoRA & distillation config)
├── run_pipeline.py           # Main pipeline orchestrator (supports LoRA option)
├── dataset/                  # Data loading and preprocessing
│   ├── dataset.py
│   └── preprocessing.py
├── models/                   # CLIP model wrapper (with LoRA support)
│   └── clip_model.py
├── training/                 # Training scripts
│   ├── lora_finetune.py      # LoRA fine-tuning script
│   ├── train_hard_negative.py # Hard negative training script
│   ├── trainer.py            # Original LoRA trainer
│   ├── hard_negative_trainer.py # Hard negative trainer
│   ├── contrastive_loss.py   # Original contrastive loss
│   ├── hard_negative_loss.py # Hard negative loss functions
│   └── model_utils.py
├── compression/              # Knowledge distillation module
│   ├── distill_loss.py       # Distillation loss functions
│   ├── distill_trainer.py    # Distillation trainer class
│   ├── distill_utils.py      # Distillation utility functions
│   ├── run_distill.py        # Distillation training script
│   └── README.md             # Documentation for distillation module
├── embedding/               # Embedding computation and caching
│   └── build_embeddings.py
├── index/                   # FAISS indexing and search
│   ├── build_index.py
│   └── search.py
├── evaluation/              # Metrics computation
│   └── metrics.py
├── analysis/                # Embedding analysis and visualization
│   └── embedding_analysis.py
├── demo/                    # Interactive CLI (with LoRA support)
│   └── search_cli.py
├── utils/                   # Utility functions
│   ├── logger.py
│   └── seed.py
└── requirements.txt         # Dependencies
```


## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 0. Configuration

Key configurations are managed in `config.py`:

- `DataConfig`: Paths to data files and split ratios
- `ModelConfig`: Model name, batch size, and device settings
- `TrainingConfig`: Training parameters 
- `LoRAConfig`: LoRA-specific settings (use_lora, lora_path, etc.)
- `DistillationConfig`: Knowledge distillation settings (teacher_model, student_model, distill_types, lambda weights, etc.)
- `EmbeddingConfig`: Cache directories and file paths
- `IndexConfig`: FAISS index settings
- `EvaluationConfig`: Metrics to compute
- `AnalysisConfig`: Visualization settings

Command-line arguments take precedence over configuration file values.


### 1. Data Preparation

The system expects a CSV file with the following columns:

```csv
image,description,display name,category
image001.jpg,"A beautiful mountain landscape","Mountain View","Nature"
image002.jpg,"Red sports car racing","Speed Demon","Vehicle"
...
```

Configure the data path in `config.py`.

As a quick start example, you can use the product image dataset from Kaggle: [Mini Product Image and Text Dataset](https://www.kaggle.com/datasets/nirmalsankalana/mini-product-image-and-text-dataset?resource=download)

### 2. Quick Start with CLI

Search for images based on text query:

```bash
# Basic search
python demo/search_cli.py --query "a beautiful landscape with mountains" --topk 5

# Search with LoRA adapter
python demo/search_cli.py --query "a beautiful landscape with mountains" --use_lora --topk 5

# Image-to-image search
python demo/search_cli.py --image_query /path/to/image.jpg --use_lora --topk 5
```

### 3. Full Pipeline Execution

Run the complete pipeline (build index, evaluate, analyze):

```bash
# With base CLIP model
python run_pipeline.py --task full_pipeline

# With LoRA fine-tuned model
python run_pipeline.py --task full_pipeline --use_lora

# With specific LoRA adapter
python run_pipeline.py --task full_pipeline --use_lora --lora_path ./path/to/lora/adapter
```

### 4. Individual Components

Run specific tasks:

```bash
# Build FAISS index with LoRA
python run_pipeline.py --task build_index --use_lora

# Evaluate model performance with LoRA
python run_pipeline.py --task evaluate --use_lora

# Analyze embeddings with LoRA
python run_pipeline.py --task analyze --use_lora

# Run demo search with LoRA
python run_pipeline.py --task demo --use_lora
```

### 5. LoRA Fine-tuning

Fine-tune the model for your specific domain using LoRA:

```bash
# Basic LoRA fine-tuning
python training/lora_finetune.py

# Custom fine-tuning parameters
python training/lora_finetune.py \
    --epochs 10 \
    --lr 5e-5 \
    --batch_size 16 \
    --save_path ./my_lora_adapter
```


### 6. Hard Negative Training

Enhance your model with hard negative mining to improve fine-grained discrimination:

```bash
# Basic hard negative training
python training/train_hard_negative.py

# Custom hard negative training parameters
python training/train_hard_negative.py \
    --epochs 10 \
    --lr 5e-5 \
    --batch_size 16 \
    --lambda_hard 0.5 \
    --hard_negative_type batch \
    --save_path ./my_hard_negative_lora_adapter

# Use category-level hard negatives for fine-grained learning
python training/train_hard_negative.py \
    --epochs 10 \
    --lr 5e-5 \
    --lambda_hard 0.3 \
    --hard_negative_type category \
    --save_path ./my_category_hard_negative_adapter

# Load existing checkpoint and continue training with hard negatives
python training/train_hard_negative.py \
    --load_checkpoint ./outputs/lora_adapter_l14_r16/checkpoint_epoch_30 \
    --epochs 20 \
    --lambda_hard 0.2 \
    --save_path ./continued_hard_negative_training
```

The hard negative training offers two approaches:
- **Batch-level hard negatives**: Mines the most challenging negative samples within each training batch
- **Category-level hard negatives**: Uses same-category samples as hard negatives to improve fine-grained discrimination

### 7. Knowledge Distillation

Compress large teacher models to efficient student models with multiple distillation strategies:

```bash
# Basic knowledge distillation
python compression/run_distill.py --data_path path/to/your/data.csv

# Using LoRA-fine-tuned teacher model
python compression/run_distill.py \
    --teacher_model openai/clip-vit-large-patch14 \
    --teacher_use_lora \
    --teacher_lora_path ./outputs/lora_adapter \
    --student_model openai/clip-vit-base-patch32 \
    --epochs 10 \
    --lr 1e-5 \
    --batch_size 32 \
    --lambda_embed 0.3 \
    --lambda_similarity 0.3 \
    --lambda_logits 0.4 \
    --temperature 1.0 \
    --output_dir ./outputs/distilled_model \
    --data_path data/cleaned_data.csv

# Custom distillation parameters with selective methods
python compression/run_distill.py \
    --teacher_model openai/clip-vit-large-patch14 \
    --student_model openai/clip-vit-base-patch32 \
    --distill_types embedding similarity logits \
    --epochs 10 \
    --lr 1e-5 \
    --batch_size 32 \
    --lambda_embed 0.3 \
    --lambda_similarity 0.3 \
    --lambda_logits 0.4 \
    --temperature 2.0 \
    --output_dir ./outputs/distilled_model \
    --data_path data/cleaned_data.csv

# Only use logits distillation
python compression/run_distill.py \
    --distill_types logits \
    --lambda_logits 1.0 \
    --data_path data/cleaned_data.csv
```

The `--distill_types` argument accepts one or more of: `embedding`, `similarity`, `logits`. The system automatically handles dimension mismatches between teacher and student models.

## Details

### Hard Negative Training Details

Hard negative training enhances the model's ability to distinguish between similar but non-matching samples. Two strategies are implemented:

- **Batch-level Hard Negatives**: Identifies the most difficult negative samples within each training batch by selecting the highest similarity scores among incorrect pairs
- **Category-level Hard Negatives**: Uses samples from the same category as hard negatives, forcing the model to learn fine-grained distinctions between similar items
- **Composite Loss Function**: Combines traditional CLIP contrastive loss with hard negative loss: `Total Loss = CLIP Loss + λ × Hard Negative Loss`
- **Flexible Configuration**: Adjustable `lambda_hard` parameter controls the influence of hard negative loss, and `hard_negative_type` selects between batch or category strategies

### Knowledge Distillation Details

The knowledge distillation module allows compressing large teacher models (e.g., CLIP ViT-L/14) into efficient student models (e.g., CLIP ViT-B/32) while preserving performance. Our enhanced implementation includes multiple distillation strategies:

- **Teacher Model**: Frozen, used only for inference during training
- **Student Model**: Trained using combined objectives
- **Three Distillation Methods**:
  - **Embedding Distillation**: Aligns teacher and student embeddings using MSE loss
  - **Similarity Distillation**: Aligns similarity matrices using KL divergence with temperature scaling
  - **Logits Distillation**: Directly compares final similarity scores using KL divergence
- **Automatic Dimension Alignment**: Handles dimension mismatches between teacher and student models with learnable projection layers
- **Configurable Loss Weights**: Independent lambda weights for each distillation method (lambda_embed, lambda_similarity, lambda_logits)
- **Supports**: GPU acceleration, checkpoint saving, LoRA-integrated teachers, and resume training

The system automatically handles cases where teacher and student models have different embedding dimensions (e.g., 768-dim ViT-L/14 teacher vs 512-dim ViT-B/32 student) by creating learnable linear projection layers.

### LoRA Fine-tuning Details

LoRA (Low-Rank Adaptation) enables efficient fine-tuning of large models:

- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `out_proj` layers
- **Rank (r)**: Default 16
- **Alpha**: Default 32
- **Dropout**: Default 0.1

This approach significantly reduces trainable parameters while maintaining strong performance.

### Performance Considerations

- Embeddings are L2 normalized for efficient cosine similarity computation
- FAISS IndexFlatIP is used for exact similarity search
- Caching prevents redundant embedding computations
- Batch processing optimizes inference speed
- LoRA integration allows domain-specific model improvements without full retraining


## License

This project is open-source and available under the MIT license.