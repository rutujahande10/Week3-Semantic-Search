
import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = Path("snippets.json")
EMBEDDINGS_FILE = Path("embeddings.npy")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_index():
    """Load snippets and their saved embeddings."""

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        snippets = json.load(file)

    embeddings = np.load(EMBEDDINGS_FILE)

    if len(snippets) != embeddings.shape[0]:
        raise ValueError("Snippet and embedding counts do not match.")

    return snippets, embeddings


def load_model():
    """Load the sentence transformer model."""
    return SentenceTransformer(MODEL_NAME)


def search_snippets(query, snippets, embeddings, model, top_n=3):
    """Find the most relevant snippets using cosine similarity."""

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if top_n < 1:
        raise ValueError("--top must be at least 1.")

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = cosine_similarity(query_embedding, embeddings)[0]

    ranked_indices = np.argsort(scores)[::-1][:top_n]

    results = []

    for index in ranked_indices:
        results.append({
            "snippet": snippets[index],
            "score": float(scores[index])
        })

    return results


def display_results(results):
    """Display ranked search results."""

    print("\n========== SEMANTIC SEARCH RESULTS ==========")

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Similarity Score: {result['score']:.4f}")
        print(f"Snippet: {result['snippet']}")

    print("\n=============================================")


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Search Tool"
    )

    parser.add_argument(
        "query",
        help="Enter your search question"
    )

    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Number of results (default: 3)"
    )

    args = parser.parse_args()

    try:
        snippets, embeddings = load_index()
        model = load_model()

        results = search_snippets(
            args.query,
            snippets,
            embeddings,
            model,
            args.top
        )

        display_results(results)

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
