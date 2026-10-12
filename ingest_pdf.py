
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "retrieval" / "data" / "AI_Course.pdf"
DB_PATH = BASE_DIR / "chroma_db"

if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"PDF not found: {PDF_PATH}\n"
        "Put your PDF inside the data folder."
    )

# Extract PDF text
reader = PdfReader(str(PDF_PATH))
chunks = []
metadata = []

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    text = text.strip()

    # Simple character-based chunking with overlap
    chunk_size = 500
    overlap = 100
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)
            metadata.append({
                "source": PDF_PATH.name,
                "page": page_number,
            })

        if start + chunk_size >= len(text):
            break

        start += chunk_size - overlap

if not chunks:
    raise ValueError(
        "No text extracted. Your PDF may be scanned "
        "and require OCR."
    )

print(f"Extracted {len(chunks)} chunks.")

# Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks).tolist()

# Save everything to ChromaDB
client = chromadb.PersistentClient(path=str(DB_PATH))

collection = client.get_or_create_collection(
    name="pdf_documents"
)

# Avoid duplicate IDs if this script is rerun
existing = collection.get()
existing_ids = set(existing["ids"])

new_chunks = [
    (i, chunk, meta, embedding)
    for i, (chunk, meta, embedding) in enumerate(
        zip(chunks, metadata, embeddings)
    )
    if f"pdf_{i}" not in existing_ids
]

if new_chunks:
    collection.add(
        ids=[f"pdf_{i}" for i, _, _, _ in new_chunks],
        documents=[chunk for _, chunk, _, _ in new_chunks],
        metadatas=[meta for _, _, meta, _ in new_chunks],
        embeddings=[embedding for _, _, _, embedding in new_chunks],
    )

print(f"Collection: {collection.name}")
print(f"Total stored chunks: {collection.count()}")
print("PDF ingestion completed successfully.")
