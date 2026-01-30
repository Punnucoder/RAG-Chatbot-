import time
import os

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent / ".env")





import streamlit as st
from pathlib import Path

from ingest import ingest_files
from retriever import retrieve
from llm_adapter import generate_answer_ffill

# ✅ stop Chroma telemetry noise
os.environ["ANONYMIZED_TELEMETRY"] = "False"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "documents"
CHROMA_DIR = str(BASE_DIR / "chroma_db")

st.set_page_config(page_title="RAG Q&A Chatbot", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

st.title("RAG Q&A Chatbot — Streamlit")

# ---------------- Sidebar: Upload + Ingest ----------------
with st.sidebar:
    st.header("Upload / Ingest")
    uploaded = st.file_uploader(
        "Upload documents (PDF / DOCX / TXT)",
        accept_multiple_files=True
    )

    if st.button("Save & Ingest uploaded files"):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        for u in uploaded or []:
            dest = DATA_DIR / u.name
            with open(dest, "wb") as f:
                f.write(u.getbuffer())
            saved_paths.append(str(dest))

        if not saved_paths:
            st.warning("No file selected.")
        else:
            with st.spinner("Ingesting + indexing..."):
                res = ingest_files(saved_paths, persist_directory=CHROMA_DIR)
            st.success(f"Inserted {res.get('inserted', 0)} chunks into vector DB")

    st.markdown("---")
    st.markdown("### Existing documents")
    if DATA_DIR.exists():
        docs = [p.name for p in DATA_DIR.iterdir() if p.is_file()]
        for d in sorted(docs):
            st.write("📄", d)
    else:
        st.info("No documents yet.")


# ---------------- Core Q/A Logic ----------------
def ask_question(question: str, top_k: int = 3):
    start = time.time()

    results = retrieve(question, top_k=top_k, persist_directory=CHROMA_DIR)
    elapsed = time.time() - start

    # confidence (mean score)
    scores = [r.get("score", 0.0) for r in results] if results else []
    mean_score = float(sum(scores) / len(scores)) if scores else 0.0

    # ✅ stricter and cleaner rule
    if not results:
        answer = "Not found in documents."
        sources = []
        ctx = []
        confidence = 0.0
    else:
        ctx = results

        # ✅ IMPORTANT: generate_answer_ffill returns STRING (not dict)
        answer = generate_answer_ffill(question, ctx)

        # ✅ clean sources
        sources = []
        seen = set()
        for r in results:
            meta = r.get("meta", {}) or {}
            src = meta.get("source", "unknown")
            chunk = meta.get("chunk", 0)
            key = (src, chunk)
            if key not in seen:
                seen.add(key)
                sources.append({"source": src, "chunk": chunk})

        confidence = mean_score

    st.session_state.history.append({
        "q": question,
        "a": answer,
        "confidence": confidence,
        "time": elapsed,
        "sources": sources,
        "ctx": ctx
    })

    return answer, sources, confidence, elapsed, ctx


# ---------------- UI: Ask ----------------
with st.form(key="qa_form"):
    query = st.text_input("Ask a question about ingested documents:")
    top_k = st.slider("Top results to use", 1, 5, 3)
    submit = st.form_submit_button("Ask")

    if submit and query.strip():
        with st.spinner("Searching + generating answer..."):
            answer, sources, confidence, elapsed, ctx = ask_question(query, top_k=top_k)

        st.success("Answer generated ✅")

        st.markdown("## ✅ Answer")
        st.write(answer)

        st.markdown(f"**Confidence:** {confidence:.3f}  |  **Time:** {elapsed:.2f}s")

        if sources:
            st.markdown("### 📌 Sources")
            for s in sources:
                st.write(f"- {s['source']} (chunk {s['chunk']})")

        # ✅ hide messy context by default
        with st.expander("Show retrieved context (debug)"):
            for i, c in enumerate(ctx):
                meta = c.get("meta", {})
                st.markdown(
                    f"**{i+1}. {meta.get('source','unknown')} | chunk {meta.get('chunk',0)} | score {c.get('score',0):.3f}**"
                )
                st.write(c.get("text", ""))


# ---------------- Session History ----------------
st.markdown("---")
st.header("Session History")

for item in reversed(st.session_state.history[-10:]):  # last 10 only
    st.markdown(f"**Q:** {item['q']}")
    st.markdown(f"**A:** {item['a']}")
    st.markdown(f"**Confidence:** {item['confidence']:.3f} — **Time:** {item['time']:.2f}s")

    if item.get("sources"):
        st.markdown("**Sources:**")
        for s in item["sources"]:
            st.write(f"- {s['source']} (chunk {s['chunk']})")
    st.markdown("---")


# ---------------- Evaluation ----------------
st.markdown("## Evaluation")
if st.button("Run evaluation on tests/questions.txt"):
    eval_path = BASE_DIR / "tests" / "questions.txt"
    ans_path = BASE_DIR / "tests" / "answers.txt"

    if not eval_path.exists() or not ans_path.exists():
        st.error("tests/questions.txt or tests/answers.txt not found.")
    else:
        from evaluate import run_evaluation
        with st.spinner("Running evaluation..."):
            report = run_evaluation(str(eval_path), str(ans_path), persist_directory=CHROMA_DIR)

        st.json(report)
