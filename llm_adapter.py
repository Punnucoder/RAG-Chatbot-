import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"   # or "llama3"

def _build_prompt(question: str, contexts: list) -> str:
    prompt = (
        "You are a helpful assistant.\n"
        "Answer the question using ONLY the context below.\n"
        "If the answer is not in the context, say: Not found in documents.\n\n"
        "CONTEXT:\n"
    )

    for c in contexts:
        meta = c.get("meta", {})
        src = meta.get("source", "unknown")
        chunk = meta.get("chunk", 0)
        text = c.get("text", "")
        prompt += f"[{src} | chunk {chunk}]\n{text}\n\n"

    prompt += f"QUESTION:\n{question}\n"
    return prompt


def generate_answer_ffill(question: str, contexts: list) -> str:
    if not contexts:
        return "Not found in documents."

    prompt = _build_prompt(question, contexts)

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"LLM ERROR: {str(e)}"
