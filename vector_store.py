import chromadb
from sentence_transformers import SentenceTransformer

# 1. Load the embedding model locally (Runs on CPU, 100% free, very fast)
# 'all-MiniLM-L6-v2' outputs a 384-dimensional vector for each text chunk.
print("Loading embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Initialize ChromaDB with persistent storage on your disk
# Data will be saved in a folder named 'chroma_db'
chroma_client = chromadb.PersistentClient(path="./chroma_db")


def get_or_create_collection(collection_name: str = "pdf_knowledge"):
    """Gets an existing ChromaDB collection or creates a new one."""
    return chroma_client.get_or_create_collection(name=collection_name)


def reset_collection(collection_name: str = "pdf_knowledge"):
    """Clears previous collection when a new PDF is uploaded."""
    try:
        chroma_client.delete_collection(name=collection_name)
        print(f"Collection '{collection_name}' cleared successfully.")
    except Exception:
        pass

def store_chunks(chunks_data: list[dict], collection_name: str = "pdf_knowledge"):
    """
    Stores chunks along with page metadata in ChromaDB.
    chunks_data is: [{"text": "...", "page": 1}, ...]
    """
    collection = get_or_create_collection(collection_name)

    documents = [item["text"] for item in chunks_data]
    metadatas = [{"page": item["page"]} for item in chunks_data]
    ids = [f"chunk_{i}" for i in range(len(chunks_data))]

    # Generate embeddings
    embeddings = embedding_model.encode(documents, batch_size=32).tolist()
    
    # Store in ChromaDB in safe batches of 500 
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents = documents[i : i + batch_size],
            embeddings = embeddings[i : i + batch_size],
            metadatas = metadatas[i: i + batch_size],
            ids = ids[i : i + batch_size]
        )
    print(f"Stored {len(documents)} chunks with page breadcrumbs!")

def search_similar_chunks(query: str, n_results: int = 4, collection_name: str = "pdf_knowledge") -> list[dict]:
    """
    Returns list of matching chunks with their page numbers:
    [{"text": "...", "page": 2}, ...]
    """
    collection = get_or_create_collection(collection_name)
    query_vector = embedding_model.encode([query]).tolist()

    results = collection.query(query_embeddings=query_vector, n_results=n_results)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    matched = []
    for doc, meta in zip(docs, metas):
        matched.append({
            "text": doc,
            "page": meta.get("page", 1)
        })
    return matched 

def get_first_chunks(n: int = 4, collection_name: str = "pdf_knowledge") -> list[dict]:
    """
    Directly retrieves the first N chunks (Page 1 & 2) for guarented summaries!
    """
    collection = get_or_create_collection(collection_name)
    # get by ID: chunk_0, chunk_1, chunk_2 (The actual beginning of the PDF)
    ids_to_get = [f"chunk_{i}" for i in range(n)]
    try:
        results = collection.get(ids=ids_to_get)
        docs = results.get("documents",[])
        metas = results.get("metadatas", [])
        return [{"text": doc, "page": meta.get("page",1)} for doc, meta in zip(docs, metas)]
    except Exception:
        return []

def get_document_count(collection_name: str = "pdf_knowledge") -> int:
    """Returns how many chunks currently exists in the database."""
    try:
        collection = get_or_create_collection(collection_name)
        return collection.count()
    except Exception:
        return 0