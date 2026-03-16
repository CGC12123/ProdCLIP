import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import config
from models.clip_model import CLIPRetrievalModel
from embedding.build_embeddings import load_embeddings, build_image_embeddings
from index.build_index import FaissIndexBuilder
from index.search import SearchEngine


def main():
    parser = argparse.ArgumentParser(description="CLI for multimodal search")
    parser.add_argument("--query", type=str, help="Text query for image search")
    parser.add_argument("--topk", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--image_query", type=str, help="Path to image for reverse search (optional)")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA fine-tuned model")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapter (defaults to config)")

    # Require at least one query type
    args = parser.parse_args()

    if not args.query and not args.image_query:
        print("Error: Please provide either --query or --image_query")
        parser.print_help()
        return

    # Use command line arguments if provided, otherwise use config
    use_lora = args.use_lora or config.lora.use_lora
    lora_path = args.lora_path if args.lora_path else config.lora.lora_path

    # Initialize model
    print(f"Loading model... (LoRA: {use_lora})")
    model = CLIPRetrievalModel(lora_path=lora_path if use_lora else None)

    # Load precomputed embeddings, or build them if they don't exist
    print("Loading embeddings...")
    try:
        embeddings, id_to_path = load_embeddings()
        print("Successfully loaded existing embeddings.")
    except FileNotFoundError:
        print("Embeddings not found. Building embeddings from data...")
        embeddings, id_to_path = build_image_embeddings(model, dataset_type='val')
        print("Embeddings built and loaded successfully.")

    # Initialize and load FAISS index
    print("Loading FAISS index...")
    index_builder = FaissIndexBuilder()
    try:
        # Try to load existing index
        index_builder.load_index()
        print("Successfully loaded existing index.")
    except FileNotFoundError:
        # If index doesn't exist, build it from the embeddings we just loaded/built
        print("Index not found. Building index from embeddings...")
        index_builder.build_index(embeddings)
        index_builder.save_index()
        print("Index built and saved successfully.")

    # Initialize search engine
    search_engine = SearchEngine(index_builder, model)

    if args.image_query:
        # Image-to-image search (finding similar images)
        if not os.path.exists(args.image_query):
            print(f"Image file not found: {args.image_query}")
            return

        print(f"Searching for images similar to image: {args.image_query}")

        # For image-to-image search, encode the query image and search in the image index
        query_embedding = model.encode_single_image(args.image_query)

        # Search in the index for similar images
        distances, indices = index_builder.search(query_embedding, args.topk + 1)  # +1 to potentially exclude the query itself

        # Map indices to image paths and exclude the query image if it's in the results
        results = []
        query_img_name = os.path.basename(args.image_query)

        for dist, idx in zip(distances[0], indices[0]):
            candidate_img_path = id_to_path.get(int(idx), f"index_{idx}")
            candidate_img_name = os.path.basename(candidate_img_path)

            # Skip if this is the same image as the query (exact match)
            if query_img_name != candidate_img_name or len(results) == 0:  # Include at least one if no others found
                results.append((candidate_img_path, float(dist)))

            if len(results) >= args.topk:
                break

        print(f"\nTop {args.topk} images similar to {args.image_query}:")
        for i, (image_path, score) in enumerate(results, 1):
            print(f"Top{i}: {image_path} (Score: {score:.4f})")
    else:
        # Text-to-image search
        print(f"Searching for images similar to query: '{args.query}'")
        results = search_engine.get_top_k_images(args.query, k=args.topk, id_to_path=id_to_path)

        print(f"\nTop {args.topk} images for query '{args.query}':")
        for i, (image_path, score) in enumerate(results, 1):
            print(f"Top{i}: {image_path} (Score: {score:.4f})")


if __name__ == "__main__":
    main()