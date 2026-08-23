import os, glob, json, requests, numpy as np

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3.2:3b"
CHUNK_SIZE = 500   # characters per chunk
SIM_THRESHOLD = 0.55  # below this -> abstain ("I don't know")

def embed(text):
    r = requests.post(f"{OLLAMA_URL}/api/embeddings",
                       json={"model": EMBED_MODEL, "prompt": text})
    return np.array(r.json()["embedding"])

def chunk_text(text, size=CHUNK_SIZE):
    words = text.split()
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(" ".join(cur)) > size:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def build_index(kb_dir="kb"):
    index = []  # list of (chunk_text, source_file, embedding)
    for path in glob.glob(f"{kb_dir}/*.txt"):
        text = open(path, encoding="utf-8").read()
        for chunk in chunk_text(text):
            index.append({
                "text": chunk,
                "source": os.path.basename(path),
                "embedding": embed(chunk)
            })
    return index

def retrieve(query, index, top_k=3):
    q_emb = embed(query)
    scored = [(cosine_sim(q_emb, item["embedding"]), item) for item in index]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

def generate(query, retrieved):
    top_score = retrieved[0][0]
    if top_score < SIM_THRESHOLD:
        return "I don't know — this isn't covered in the knowledge base.", [], top_score

    context = "\n\n".join([f"[Source: {r['source']}]\n{r['text']}" for _, r in retrieved])
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {query}

Answer:"""

    r = requests.post(f"{OLLAMA_URL}/api/generate",
                       json={"model": GEN_MODEL, "prompt": prompt, "stream": False})
    answer = r.json()["response"]
    sources = list(set(r["source"] for _, r in retrieved))
    return answer, sources, top_score

if __name__ == "__main__":
    print("Building index...")
    index = build_index()
    print(f"Indexed {len(index)} chunks.\n")

    test_questions = [
        "How many hours can I work per fortnight on a subclass 500 visa during semester?",
        "Can I work unlimited hours during scheduled course breaks?",
        "What is the age limit to apply for a subclass 485 visa?",
        "What's the difference between the Post-Vocational Education Work stream and Post-Higher Education Work stream?",
        "Do I need OSHC for my whole stay in Australia?",
        "How many days do I have to notify my provider of my address after arriving?",
        "Can I apply for a Skills in Demand 482 visa while holding a 485 visa?",
        "What happens if I exceed my work hour limit?",
        "What's the capital of France?",  # out-of-scope, should abstain
        "Can I bring my pet dog to Australia on a student visa?"  # out-of-scope
    ]

    results = []
    for q in test_questions:
        retrieved = retrieve(q, index)
        answer, sources, top_score = generate(q, retrieved)
        results.append({
            "question": q, "answer": answer,
            "sources": sources, "top_similarity": round(float(top_score), 3)
        })
        print(f"Q: {q}\nA: {answer}\nSources: {sources} | sim={top_score:.3f}\n{'-'*60}")

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)