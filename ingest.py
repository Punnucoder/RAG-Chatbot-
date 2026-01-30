import os
import uuid
from sentence_transformers import SentenceTransformer
import chromadb
from utils import load_file, chunk_text

MODEL_NAME = "all-MiniLM-L6-v2"


def get_chroma_client(persist_directory: str):
    # New Chroma way (auto persist)
    os.makedirs(persist_directory, exist_ok=True)
    return chromadb.PersistentClient(path=persist_directory)


def ingest_files(paths, persist_directory="c:/Users/Wifi/Desktop/Gen AI/rag/chroma_db"):
    model = SentenceTransformer(MODEL_NAME)

    client = get_chroma_client(persist_directory=persist_directory)
    coll = client.get_or_create_collection(name="documents")

    docs = []
    metadatas = []
    ids = []

    for path in paths:
        text = load_file(path)
        base = os.path.basename(path)

        chunks = chunk_text(text, chunk_size=350, overlap=60)


        for i, c in enumerate(chunks):
            docs.append(c)
            metadatas.append({"source": base, "chunk": i, "path": path})
            ids.append(str(uuid.uuid4()))

    if not docs:
        return {"inserted": 0}

    embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)
    embeddings_list = embeddings.tolist()  # ✅ direct nested list

    coll.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings_list
    )

    return {"inserted": len(docs)}
