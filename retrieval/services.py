
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from django.conf import settings

CHROMA_PATH = Path(settings.BASE_DIR) / "chroma_db"
COLLECTION_NAME = "pdf_documents"

client = chromadb.PersistentClient(path=str(CHROMA_PATH))


def retrieve(query, k=3, threshold=None):
    query = query.strip()

    if not query:
        return []

    # Check whether the expected collection exists.
    collection_names = [
        item.name for item in client.list_collections()
    ]

    if COLLECTION_NAME not in collection_names:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' was not found. "
            f"Database path: {CHROMA_PATH}. "
            f"Available collections: {collection_names}. "
            "Run ingest_pdf.py first."
        )

    collection = client.get_collection(name=COLLECTION_NAME)

    # Load the model only when a search is requested.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []
    distances = results["distances"][0] or []

    retrieved = []

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        if threshold is not None and distance > threshold:
            continue

        retrieved.append({
            "text": document,
            "source": (metadata or {}).get(
                "source", "Unknown source"
            ),
            "score": float(distance),
        })

    return retrieved
