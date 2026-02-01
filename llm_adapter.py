import os

# disable telemetry from the SDK to avoid capture() errors
# MUST be set before importing the groq SDK
os.environ.setdefault("GROQ_DISABLE_TELEMETRY", "true")

from dotenv import load_dotenv
import streamlit as st
from groq import Groq

load_dotenv()  # load .env if present


def get_groq_client():
    """
    Return a Groq client. Reads API key from Streamlit secrets or environment.
    Do NOT perform a test API call on init (avoids early 401 and noisy errors).
    """
    api_key = None
    try:
        api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Set it in Streamlit secrets, as an environment variable, or in a .env file."
        )

    # common mistake: OpenAI keys start with "sk-"
    if api_key.startswith("sk-"):
        raise RuntimeError(
            "Detected an OpenAI-style key (starts with 'sk-'). Make sure you are using a GROQ API key, not an OpenAI key."
        )

    return Groq(api_key=api_key)


MODEL_NAME = "llama-3.1-8b-instant"


def _build_prompt(question: str, contexts: list) -> str:
    prompt = (
        "Answer the question using ONLY the context below.\n"
        "If the answer is not found, say: Not found in documents.\n\n"
        "CONTEXT:\n"
    )

    for c in contexts:
        text = c.get("text", "")
        prompt += text + "\n\n"

    prompt += f"QUESTION:\n{question}\n"
    return prompt


def generate_answer_ffill(question: str, contexts: list) -> str:
    if not contexts:
        return "Not found in documents."

    client = get_groq_client()
    prompt = _build_prompt(question, contexts)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        msg = str(e).lower()
        if "401" in msg or "invalid_api_key" in msg or "unauthorized" in msg:
            return (
                "LLM ERROR: GROQ API Key rejected (401). "
                "Verify GROQ_API_KEY is a valid GROQ key (not an OpenAI key), set it in Streamlit secrets or as an environment variable, "
                "and restart the app."
            )
        return f"LLM ERROR: {str(e)}"
