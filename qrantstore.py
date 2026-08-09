"""
Combined semantic chunking + Qdrant storage helper.

This file merges the functionality from `sentencemodel.py` (semantic
chunking using a local HuggingFace embedding model) and the original
Qdrant storage module so a single script can chunk PDFs and store/query
embeddings. Configuration is loaded from a `.env` file when present.
"""

import logging
import os
import sys
import uuid
from pathlib import Path
from typing import List, Optional, Sequence

from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai


from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
import requests

# LlamaIndex imports used for semantic chunking
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment-configurable values (set these in a .env file)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "financeaethal1tkmsk")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

client = genai.Client(api_key=GEMINI_API_KEY)


def get_qdrant_client() -> QdrantClient:
    """Return a cloud Qdrant client.
    
    Requires QDRANT_URL and QDRANT_API_KEY to be set.
    """
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError(
            "QDRANT_URL and QDRANT_API_KEY must be set in environment variables"
        )
    print("connecting to Qdrant Cloud")
    print(QDRANT_URL)
    print(QDRANT_API_KEY)
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_collection(client: QdrantClient, collection_name: str = QDRANT_COLLECTION):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info(f"created collection '{collection_name}'")
    else:
        logger.info(f"collection '{collection_name}' already exists")


def store_chunks(
    chunk_texts: Sequence[str],
    metadatas: Optional[Sequence[dict]] = None,
    collection_name: str = QDRANT_COLLECTION,
    client: Optional[QdrantClient] = None,
) -> List[str]:
    if metadatas is not None and len(metadatas) != len(chunk_texts):
        raise ValueError("metadatas must be the same length as chunk_texts")
    print(collection_name)

    client = get_qdrant_client()
    ensure_collection(client, collection_name)

    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vectors = embedder.encode(list(chunk_texts), show_progress_bar=False).tolist()

    point_ids = [str(uuid.uuid4()) for _ in chunk_texts]
    points = [
        PointStruct(
            id=point_ids[i],
            vector=vectors[i],
            payload={"text": chunk_texts[i], **(metadatas[i] if metadatas else {})},
        )
        for i in range(len(chunk_texts))
    ]

    client.upsert(collection_name=collection_name, points=points)
    logger.info(f"stored {len(points)} chunk(s) in Qdrant collection '{collection_name}'")

    return point_ids


def search(
    query: str,
    top_k: int = 3,
    collection_name: str = QDRANT_COLLECTION,
    client: Optional[QdrantClient] = None,
) -> List[dict]:
    client = client or get_qdrant_client()
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_vector = embedder.encode(query).tolist()

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    ).points

    return [{"score": r.score, "payload": r.payload} for r in results]


def gemini_generate(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """Call a Gemini-like HTTP API to generate a completion for `prompt`.

    Requires `GEMINI_API_URL` and `GEMINI_API_KEY` to be set in the environment.
    The exact JSON response parsing is best-effort — adjust to match your
    provider's response shape.
    """
    if not GEMINI_API_URL or not GEMINI_API_KEY:
        raise RuntimeError("Set GEMINI_API_URL and GEMINI_API_KEY in .env to use Gemini")

    payload = {
        "model": "gemini-3.6-flash",
        "input": prompt,
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}

    resp = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Try several common response shapes
    if isinstance(data, dict):
        if "candidates" in data and data["candidates"]:
            c = data["candidates"][0]
            return c.get("content") or c.get("output") or str(c)
        if "output" in data and isinstance(data["output"], list) and data["output"]:
            out0 = data["output"][0]
            if isinstance(out0, dict):
                return out0.get("content") or out0.get("text") or str(out0)
        if "choices" in data and data["choices"]:
            return data["choices"][0].get("text")

    return str(data)


def gemini_generate_with_retrieval(query: str, top_k: int = 3, max_tokens: int = 512) -> str:
    """Retrieve relevant chunks from Qdrant and call Gemini with the context.

    This builds a prompt that includes the retrieved text snippets followed by
    the user question. Adjust the prompt template to taste.
    """
    results = search(query, top_k=top_k)
    context_texts = "\n\n---\n\n".join([r["payload"].get("text", "") for r in results])

    prompt = (
       f"""
You are an AI assistant.

Answer the user's question **only** using the provided context.

If the answer is not present in the context, reply:
"I couldn't find that information in the provided document."

Context:
{context_texts}

Question:
{query}

Answer:
""")

    return gemini_generate(prompt, max_tokens=max_tokens)


def load_text_from_pdf(file_path: str) -> List[str]:
    reader = PdfReader(file_path)
    texts = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            logger.warning(f"page {page_num}: no extractable text — skipped")
            continue
        texts.append(text)

    print(f"loaded {len(texts)} page(s) with text from '{file_path}'")
    return texts


DEFAULT_BREAKPOINT_PERCENTILE_THRESHOLD = 60


def semantic_chunk(
    texts: List[str],
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = DEFAULT_BREAKPOINT_PERCENTILE_THRESHOLD,
) -> List[BaseNode]:
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)

    splitter = SemanticSplitterNodeParser(
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        embed_model=embed_model,
    )

    documents = [Document(text=t) for t in texts]
    nodes = splitter.get_nodes_from_documents(documents)

    logger.info(
        f"semantic_chunk: {len(texts)} input text(s) -> {len(nodes)} chunk(s) "
        f"(threshold={breakpoint_percentile_threshold})"
    )
    return nodes


def chunk_pdf(
    file_path: str,
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = DEFAULT_BREAKPOINT_PERCENTILE_THRESHOLD,
) -> List[BaseNode]:
    texts = load_text_from_pdf(file_path)
    return semantic_chunk(
        texts,
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
    )


def chunk_pdf_to_payloads(
    file_path: str,
    document_name: Optional[str] = None,
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = DEFAULT_BREAKPOINT_PERCENTILE_THRESHOLD,
) -> List[dict]:
    """Chunk a PDF and return a list of payload dicts ready for Qdrant.

    Each payload follows the structure requested by the user:
    {
        "text": chunk_text,
        "document_name": "Investment_India.pdf",
        "page_number": 12,
        "chunk_id": 45,
        "source": "Investment_India.pdf",
        "section": "Tax Benefits"
    }

    Note: `section` is left empty because extracting semantic section names
    is out-of-scope for this helper — you can populate it before upload if
    you have section boundaries available.
    """
    texts = load_text_from_pdf(file_path)
    doc_name = document_name or Path(file_path).name

    payloads: List[dict] = []
    chunk_counter = 0

    # Chunk per-page to preserve page_number mapping
    for page_number, page_text in enumerate(texts, start=1):
        if not page_text:
            continue
        nodes = semantic_chunk(
            [page_text], buffer_size=buffer_size, breakpoint_percentile_threshold=breakpoint_percentile_threshold
        )

        for node in nodes:
            chunk_counter += 1
            payload = {
                "text": node.get_content(),
                "document_name": doc_name,
                "page_number": page_number,
                "chunk_id": chunk_counter,
                "source": doc_name,
                "section": "",
            }
            payloads.append(payload)

    logger.info(f"created {len(payloads)} payload(s) from '{file_path}'")
    return payloads


def upload_payloads(
    payloads: Sequence[dict],
    collection_name: str = QDRANT_COLLECTION,
    client: Optional[QdrantClient] = None,
) -> List[str]:
    """Upload prepared payload dicts to Qdrant.

    Each payload must contain a `text` field for embedding; remaining keys
    are stored as the point payload in Qdrant.
    Returns the list of point IDs created.
    """
    if not payloads:
        return []

    client = client or get_qdrant_client()
    ensure_collection(client, collection_name)

    texts = [p.get("text", "") for p in payloads]
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vectors = embedder.encode(texts, show_progress_bar=False).tolist()

    point_ids = [str(uuid.uuid4()) for _ in payloads]
    points = [
        PointStruct(id=point_ids[i], vector=vectors[i], payload=payloads[i])
        for i in range(len(payloads))
    ]

    client.upsert(collection_name=collection_name, points=points)
    logger.info(f"uploaded {len(points)} payload(s) to collection '{collection_name}'")
    return point_ids


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python qrantstore.py path/to/document.pdf "your question here"')
        sys.exit(1)

    file_path = "C:\Live Projects\FinancialRag\invInd.pdf"
    question = sys.argv[0]

    # Chunk the PDF
    nodes = chunk_pdf(file_path)
    chunk_texts = [node.get_content() for node in nodes]

    # Store in Qdrant
    store_chunks(chunk_texts)

    # Query
    results = search(question, top_k=3)
    print(f"\nTop matches for: '{question}'\n")
    for i, r in enumerate(results, start=1):
        print(f"--- Result {i} (score={r['score']:.4f}) ---")
        print(r["payload"]["text"][:500])
        print()