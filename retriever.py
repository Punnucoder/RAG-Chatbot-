from sentence_transformers import SentenceTransformer
import chromadb

MODEL_NAME = "all-MiniLM-L6-v2"

# ✅ load once
_model = SentenceTransformer(MODEL_NAME)

def get_client(persist_directory):
    return chromadb.PersistentClient(path=persist_directory)

def retrieve(question, top_k=4, persist_directory="chroma_db"):
    client = get_client(persist_directory=persist_directory)
    coll = client.get_or_create_collection(name="documents")

    q_embedding = _model.encode([question], convert_to_numpy=True)[0].tolist()

    results = coll.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    out = []
    for d, m, dist in zip(docs, metas, dists):
        score = 1 / (1 + float(dist))
        out.append({
            "text": d,
            "meta": m,
            "distance": float(dist),
            "score": float(score)
        })

    return out
