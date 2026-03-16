#!/usr/bin/env python3
"""
Main entry point for the multimodal retrieval system.
This script orchestrates the full pipeline:
1. Load and preprocess data
2. Build image embeddings
3. Create FAISS index
4. Evaluate the system
5. Generate analysis plots
"""

import os
import sys
import argparse
from typing import Dict, List
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from utils.seed import set_seed
from utils.logger import get_demo_logger
from dataset.dataset import get_data_loaders, load_data, split_data
from models.clip_model import CLIPRetrievalModel
from embedding.build_embeddings import build_image_embeddings, compute_text_embeddings_for_dataset
from index.build_index import FaissIndexBuilder
from index.search import SearchEngine
from evaluation.metrics import evaluate_full_retrieval
from analysis.embedding_analysis import analyze_embeddings_and_similarities, compute_embedding_statistics


logger = get_demo_logger()


def build_and_save_index(use_lora: bool = False, lora_path: str = None):
    """Build FAISS index and save it"""
    logger.info(f"Building and saving FAISS index... (LoRA: {use_lora})")

    # Initialize model
    model = CLIPRetrievalModel(lora_path=lora_path if use_lora else None)

    # Build image embeddings
    image_embeddings, id_to_path = build_image_embeddings(model, dataset_type='test')

    # Build FAISS index
    index_builder = FaissIndexBuilder()
    index_builder.build_index(image_embeddings)

    # Save index
    index_builder.save_index()

    logger.info("Index built and saved successfully.")
    return index_builder, image_embeddings, id_to_path


def run_evaluation(use_lora: bool = False, lora_path: str = None):
    """Run evaluation on test set"""
    logger.info(f"Starting evaluation... (LoRA: {use_lora})")

    # Initialize model
    model = CLIPRetrievalModel(lora_path=lora_path if use_lora else None)

    # Load data
    df = load_data()
    splits = split_data(df)
    test_data = splits['test'].to_dict('records')

    # Load embeddings
    image_embeddings, id_to_path = build_image_embeddings(model, dataset_type='test')
    text_embeddings, id_to_text = compute_text_embeddings_for_dataset(model, dataset_type='test')

    # Evaluate
    results = evaluate_full_retrieval(
        model=model,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        test_data=test_data
    )

    logger.info("Evaluation completed.")
    return results


def run_analysis(use_lora: bool = False, lora_path: str = None):
    """Run embedding analysis and generate plots"""
    logger.info(f"Running embedding analysis... (LoRA: {use_lora})")

    # Initialize model
    model = CLIPRetrievalModel(lora_path=lora_path if use_lora else None)

    # Load data and embeddings
    df = load_data()
    splits = split_data(df)
    train_data = splits['train'].to_dict('records')

    image_embeddings, id_to_path = build_image_embeddings(model, dataset_type='test')

    # Prepare text and image lists for analysis
    text_list = [item['description'] for item in train_data]
    image_paths = [os.path.join(config.data.image_dir, item['image']) for item in train_data]

    # Perform analysis
    analyze_embeddings_and_similarities(
        model=model,
        embeddings=image_embeddings,
        text_list=text_list,
        image_paths=image_paths
    )

    # Compute statistics
    stats = compute_embedding_statistics(image_embeddings)
    logger.info(f"Embedding statistics: {stats}")


def run_demo_search(use_lora: bool = False, lora_path: str = None):
    """Run a demonstration search"""
    logger.info(f"Running demo search... (LoRA: {use_lora})")

    # Initialize model
    model = CLIPRetrievalModel(lora_path=lora_path if use_lora else None)

    # Load or build index
    index_builder = FaissIndexBuilder()
    try:
        index_builder.load_index()
        logger.info("Loaded existing index")
    except FileNotFoundError:
        logger.info("Building index for demo...")
        _, id_to_path = build_image_embeddings(model, dataset_type='test')
        image_embeddings, _ = build_image_embeddings(model, dataset_type='test')
        index_builder.build_index(image_embeddings)
        index_builder.save_index()

    # Load id_to_path mapping
    _, id_to_path = build_image_embeddings(model, dataset_type='test')

    # Initialize search engine
    search_engine = SearchEngine(index_builder, model)

    # Example search
    query = "a beautiful landscape with mountains"
    results = search_engine.get_top_k_images(query, k=5, id_to_path=id_to_path)

    logger.info(f"\nDemo search results for query: '{query}'")
    for i, (image_path, score) in enumerate(results, 1):
        print(f"Top{i}: {image_path} (Score: {score:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Multimodal Retrieval System")
    parser.add_argument("--task", type=str, choices=['build_index', 'evaluate', 'analyze', 'demo', 'full_pipeline'],
                        default='full_pipeline', help="Task to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA fine-tuned model")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapter (defaults to config)")

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Use command line arguments if provided, otherwise use config
    use_lora = args.use_lora or config.lora.use_lora
    lora_path = args.lora_path if args.lora_path else config.lora.lora_path

    # Create necessary directories
    os.makedirs(config.embedding.cache_dir, exist_ok=True)
    os.makedirs(config.analysis.plots_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.index.index_file), exist_ok=True)

    if args.task == 'build_index':
        build_and_save_index(use_lora=use_lora, lora_path=lora_path)
    elif args.task == 'evaluate':
        run_evaluation(use_lora=use_lora, lora_path=lora_path)
    elif args.task == 'analyze':
        run_analysis(use_lora=use_lora, lora_path=lora_path)
    elif args.task == 'demo':
        run_demo_search(use_lora=use_lora, lora_path=lora_path)
    elif args.task == 'full_pipeline':
        logger.info("Starting full pipeline...")

        # Step 1: Build index
        index_builder, image_embeddings, id_to_path = build_and_save_index(use_lora=use_lora, lora_path=lora_path)

        # Step 2: Run evaluation
        eval_results = run_evaluation(use_lora=use_lora, lora_path=lora_path)

        # Step 3: Run analysis
        run_analysis(use_lora=use_lora, lora_path=lora_path)

        # Step 4: Run demo
        run_demo_search(use_lora=use_lora, lora_path=lora_path)

        logger.info("Full pipeline completed successfully!")


if __name__ == "__main__":
    main()