import time
from retriever import retrieve
from llm_adapter import generate_answer_ffill
from fuzzywuzzy import fuzz


def run_evaluation(questions_path: str, answers_path: str, persist_directory: str = "c:/Users/Wifi/Desktop/Gen AI/rag/chroma_db"):
    with open(questions_path, "r", encoding="utf-8", errors="ignore") as f:
        questions = [l.strip() for l in f.readlines() if l.strip()]
    with open(answers_path, "r", encoding="utf-8", errors="ignore") as f:

        answers = [l.strip() for l in f.readlines() if l.strip()]

    n = min(len(questions), len(answers), 20)
    report = {"total": n, "cases": []}
    correct = 0
    total_time = 0.0

    for i in range(n):
        q = questions[i]
        expected = answers[i]
        t0 = time.time()
        ctx = retrieve(q, top_k=4, persist_directory=persist_directory)
        gen = generate_answer_ffill(q, ctx)
        resp = gen.get("answer", "")
        t1 = time.time()
        elapsed = t1 - t0
        total_time += elapsed

        ratio = fuzz.token_set_ratio(expected, resp)
        is_correct = ratio >= 70
        if is_correct:
            correct += 1

        report["cases"].append({"q": q, "expected": expected, "response": resp, "time_s": elapsed, "score": ratio, "correct": is_correct})

    report["accuracy"] = correct / n if n else 0.0
    report["avg_response_time_s"] = total_time / n if n else 0.0
    return report
