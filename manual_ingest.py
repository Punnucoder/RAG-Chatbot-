import os
import sys
sys.path.append("src")

from ingest import ingest_files

PDF_DIR = r"C:\Users\Wifi\Desktop\Gen AI\rag\data\documents"

files = []
for f in os.listdir(PDF_DIR):
    if f.lower().endswith(".pdf"):
        files.append(os.path.join(PDF_DIR, f))

print("Total PDFs found:", len(files))

res = ingest_files(files, persist_directory="chroma_db")
print(res)
