
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_FILE = Path("snippets.json")
EMBEDDINGS_FILE = Path("embeddings.npy")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_snippets():
    """Load text snippets from JSON."""

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        snippets = json.load(file)

    if len(snippets) != 20:
        raise ValueError("Exactly 20 snippets are required.")

    return snippets


def generate_embeddings(snippets):
    """Generate numerical embeddings for snippets."""

    print("Loading Sentence Transformer model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")

    embeddings = model.encode(
        snippets,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings


def save_embeddings(embeddings):
    """Save embeddings into a NumPy file."""

    np.save(EMBEDDINGS_FILE, embeddings)

    print("Embeddings saved successfully!")


def build_index():
    """Build and save the embedding index."""

    snippets = load_snippets()

    embeddings = generate_embeddings(snippets)

    save_embeddings(embeddings)

    print("\nIndex created successfully!")
    print("Total snippets:", len(snippets))
    print("Embedding shape:", embeddings.shape)


if __name__ == "__main__":
    build_index()
