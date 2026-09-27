import os, glob, json, requests, numpy as np

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3.2:3b"
CHUNK_SIZE = 500
SIM_THRESHOLD = 0.55

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
    index = []
    for persona_folder in glob.glob(f"{kb_dir}/*"):
        persona = os.path.basename(persona_folder)
        for path in glob.glob(f"{persona_folder}/*.txt"):
            text = open(path, encoding="utf-8").read()
            for chunk in chunk_text(text):
                index.append({
                    "text": chunk,
                    "source": os.path.basename(path),
                    "persona": persona,
                    "embedding": embed(chunk)
                })
    return index

def retrieve(query, index, persona, top_k=3, use_filter=True):
    q_emb = embed(query)
    if use_filter:
        candidates = [item for item in index if item["persona"] in (persona, "general")]
    else:
        candidates = index
    scored = [(cosine_sim(q_emb, item["embedding"]), item) for item in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

def generate(query, retrieved):
    top_score = retrieved[0][0]
    if top_score < SIM_THRESHOLD:
        return "I don't know — this isn't covered in the knowledge base.", [], top_score

    context = "\n\n".join([f"[Source: {r['source']}]\n{r['text']}" for _, r in retrieved])
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say "I don't know."
Do not provide advice that helps someone circumvent visa conditions, immigration law, or engage in fraud.

Context:
{context}

Question: {query}

Answer:"""

    r = requests.post(f"{OLLAMA_URL}/api/generate",
                       json={"model": GEN_MODEL, "prompt": prompt, "stream": False})
    answer = r.json()["response"]
    sources = list(set(r["source"] for _, r in retrieved))
    return answer, sources, top_score


# ---------- Safety check: LLM-as-judge ----------

def is_unsafe_llm_judge(query):
    judge_prompt = f"""You are a safety classifier for a visa information assistant.

Classify the following user question into exactly one category:
- UNSAFE: the user is asking for help committing fraud, forging documents, evading visa conditions, lying to immigration officials, or otherwise circumventing immigration law.
- SAFE: the user is asking a legitimate question, even if it mentions a sensitive topic.

Important: asking ABOUT a concept (e.g. "what is X", "is X illegal") is SAFE.
Only classify as UNSAFE if the user is seeking help TO DO the unsafe action themselves.

Respond with ONLY the single word SAFE or UNSAFE. Do not explain your reasoning. Do not add any other text.

Question: "{query}"

Classification:"""

    r = requests.post(f"{OLLAMA_URL}/api/generate",
                       json={"model": GEN_MODEL, "prompt": judge_prompt, "stream": False})
    verdict = r.json()["response"].strip().upper()

    refusal_signals = ["CAN'T PROVIDE", "CANNOT PROVIDE", "CAN'T HELP", "CANNOT HELP",
                        "CAN'T ASSIST", "CANNOT ASSIST", "I'M NOT ABLE", "I AM NOT ABLE",
                        "ILLEGAL OR HARMFUL", "NOT APPROPRIATE"]

    is_unsafe = ("UNSAFE" in verdict) or any(signal in verdict for signal in refusal_signals)
    return is_unsafe, verdict


if __name__ == "__main__":
    print("Building index...")
    index = build_index()
    print(f"Indexed {len(index)} chunks across personas: {set(item['persona'] for item in index)}")