
# Week 3 - Semantic Search Tool

## Project Overview

This project implements a semantic search tool using Python and Sentence Transformers.

It converts text snippets and user queries into numerical embeddings and uses cosine similarity to identify the three most relevant snippets.

Unlike keyword-based search, semantic search understands the meaning and context of the query.

## Technologies Used

- Python
- Sentence Transformers
- NumPy
- Scikit-learn
- JSON
- Cosine Similarity

## Project Structure

```text
week3-semantic-search/
├── snippets.json
├── build_index.py
├── search.py
├── embeddings.npy
├── requirements.txt
├── .gitignore
└── README.md
```

## Features

1. Loads 20 text snippets from a JSON file.
2. Generates embeddings using all-MiniLM-L6-v2.
3. Saves embeddings in a NumPy file.
4. Accepts user queries through the command line.
5. Calculates cosine similarity.
6. Displays the top 3 relevant snippets.
7. Supports a custom number of results using --top.

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

## Build the Embedding Index

Run this command once to generate and save embeddings:

```bash
python build_index.py
```

## Run Semantic Search

Example:

```bash
python search.py "What are text embeddings?"
```

By default, the application displays the top 3 results.

To display 5 results:

```bash
python search.py "What are text embeddings?" --top 5
```

## Testing

Tested with different queries related to:

- Text embeddings
- Cosine similarity
- Natural Language Processing
- NumPy
- Semantic search
- Zero-shot and few-shot prompting

All tested queries returned relevant results.

## Embedding Model

Model: sentence-transformers/all-MiniLM-L6-v2

Embedding dimensions: 384

Total snippets: 20
