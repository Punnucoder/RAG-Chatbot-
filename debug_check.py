import chromadb

CHROMA_DIR = "c:/Users/Wifi/Desktop/Gen AI/rag/chroma_db"

client = chromadb.PersistentClient(path=CHROMA_DIR)
coll = client.get_or_create_collection("documents")

print("Total docs in collection:", coll.count())
