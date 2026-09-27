import re
import streamlit as st
from rag_pipeline import build_index, retrieve, generate, is_unsafe_llm_judge

def redact_pii(text):
    text = re.sub(r'\b[A-Z]{1,2}\d{6,9}\b', '[REDACTED-ID]', text)
    text = re.sub(r'\bvisa\s+grant\s+number\s*:?\s*\d{6,12}\b', 'visa grant number [REDACTED-ID]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[REDACTED-DATE]', text)
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '[REDACTED-DATE]', text)
    return text

st.set_page_config(page_title="Visa Conditions Assistant", page_icon="🛂", layout="centered")

st.title("🛂 Visa Conditions Assistant")
st.caption("Ask me about your student or graduate visa conditions. I only answer from verified sources — I'll say \"I don't know\" if I'm not sure.")
st.caption("⚠️ Please don't share personal identifiers (passport numbers, visa grant numbers, date of birth) in your questions.")

@st.cache_resource
def load_index():
    return build_index()

if "index" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        st.session_state.index = load_index()

with st.sidebar:
    st.subheader("Your visa")
    persona = st.selectbox("Select your visa subclass", ["500 (Student)", "485 (Temporary Graduate)"])
    persona_key = persona.split()[0]

    st.divider()
    use_filter = st.toggle("Persona-filtered retrieval", value=True)
    st.divider()

    st.caption("🟢 Verified — high-confidence match")
    st.caption("🟡 Uncertain — weak match, double-check")
    st.caption("🔴 Not Found / Declined")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your visa conditions assistant. Select your visa type on the left, then ask away.", "meta": None}
    ]

for msg in st.session_state.messages:
    avatar = "🛂" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])
        if msg.get("meta"):
            st.caption(msg["meta"])

if query := st.chat_input("Type your question..."):
    original_query = query
    query = redact_pii(query)

    st.session_state.messages.append({"role": "user", "content": query, "meta": None})
    with st.chat_message("user", avatar="🧑"):
        st.write(query)
        if query != original_query:
            st.caption(f"🔒 Redacted: `{original_query}` → `{query}`")

    with st.chat_message("assistant", avatar="🛂"):
        with st.spinner("Checking..."):
            unsafe_flag, verdict = is_unsafe_llm_judge(query)
            st.caption(f"🔍 DEBUG raw verdict: `{verdict}`")
        if unsafe_flag:
            answer = ("I can't help with questions about circumventing visa conditions or immigration law. "
                      "If you're facing a difficult visa situation, please contact your institution's "
                      "international student office or a registered migration agent.")
            sources, score = [], 0.0
            meta = f"🔴 Declined — flagged as unsafe (LLM judge: {verdict})"
            st.write(answer)
            st.caption(meta)
        else:
            with st.spinner("Checking sources..."):
                retrieved = retrieve(query, st.session_state.index, persona_key, use_filter=use_filter)
                answer, sources, score = generate(query, retrieved)

            badge = "🟢 Verified" if score > 0.7 else "🟡 Uncertain" if score > 0.55 else "🔴 Not Found"
            meta = f"{badge}  ·  Sources: {', '.join(sources) if sources else 'none'}  ·  Similarity: {score:.2f}"
            st.write(answer)
            st.caption(meta)

    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})