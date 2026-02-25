import os
import time
import streamlit as st
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ------------------------
# Page config + small CSS
# ------------------------
st.set_page_config(page_title="Fordham RAG Assistant", page_icon="🎓", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem;}
      .stChatMessage {border-radius: 14px;}
      .small-muted {color: #6b7280; font-size: 0.9rem;}
      .answer-card {
        padding: 1rem 1.2rem;
        border: 1px solid rgba(49,51,63,0.2);
        border-radius: 14px;
        background-color: rgba(255,255,255,0.6);
        line-height: 1.6;
        font-size: 0.98rem;
      }
      .source-card {
        padding: 0.75rem 0.9rem;
        border: 1px solid rgba(49,51,63,0.15);
        border-radius: 12px;
        margin-bottom: 0.6rem;
        background-color: rgba(255,255,255,0.5);
      }
      .mono {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
      [data-testid="stChatMessage"] {margin-bottom: 1.25rem;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎓 Fordham RAG Assistant")
st.caption("Ask a question about Fordham University. Answers are generated using retrieved context from your document set.")

with st.expander("💡 Try asking things like…"):
    st.markdown("""
- What programs does Gabelli offer?
- How do I apply for financial aid?
- Where is Fordham located?
- What graduate programs are available?
""")

# ------------------------
# Sidebar controls
# ------------------------
with st.sidebar:
    st.header("⚙️ Retrieval Settings")
    k = st.slider("Retrieve top-k chunks", 3, 20, 8, 1)
    max_chunks = st.slider("Chunks sent to LLM", 2, 12, 6, 1)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    model_name = st.selectbox("LLM model", ["gpt-4o-mini", "gpt-4o"], index=0)
    st.divider()
    show_sources = st.checkbox("Show sources", value=True)
    show_debug = st.checkbox("Show debug scores", value=False)
    stream_answer = st.checkbox("Stream answer (typing effect)", value=True)

# ------------------------
# Load heavy stuff once
# ------------------------
@st.cache_resource
def load_models_and_data():
    chunks_df = pd.read_pickle("chunks_no_emb.pkl")
    embeddings = np.load("bge_embeddings.npy")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    emb_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Make sure it's set in your .env file.")
    client = OpenAI(api_key=api_key)

    return chunks_df, embeddings, emb_model, client

chunks_df, embeddings, emb_model, client = load_models_and_data()

# ------------------------
# Retrieve (semantic)
# ------------------------
def retrieve_semantic(question: str, k: int = 8):
    q_emb = emb_model.encode([question], normalize_embeddings=True)[0]
    scores = embeddings @ q_emb  # cosine similarity since normalized
    idx = np.argsort(scores)[-k:][::-1]
    out = chunks_df.iloc[idx].copy()
    out["score"] = scores[idx]
    return out.reset_index(drop=True)

# ------------------------
# Generate (with citations)
# ------------------------
def generate_answer(question: str, retrieved_df: pd.DataFrame, max_chunks: int, model: str, temperature: float):
    contexts = retrieved_df["text"].astype(str).tolist()[:max_chunks]

    context_block = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{c}" for i, c in enumerate(contexts)
    )

    system_prompt = (
        "You are a careful assistant answering questions about Fordham University. "
        "Use the provided context as your primary source of truth. "
        "If the context clearly contains relevant information, answer normally. "
        "Only say you lack information if the context truly does not contain the answer. "
        "Cite sources like [Source 2] when used."
    )

    user_prompt = f"""
Context:
{context_block}

Question:
{question}

Instructions:
- Answer clearly in 3–6 bullet points (or a short paragraph if better).
- Only use the context above.
- Include citations like [Source 1] where relevant.
- If context is insufficient, briefly explain what information is missing and suggest what the user could ask instead.

Answer:
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

def clean_answer_text(s: str) -> str:
    # Remove occasional leading bullet that looks awkward in the chat bubble
    s = s.strip()
    if s.startswith("•"):
        s = s.lstrip("•").strip()
    return s

# ------------------------
# Chat state
# ------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about Fordham (programs, campus, financial aid, admissions, etc.)."}
    ]

if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ------------------------
# Input + run
# ------------------------
question = st.chat_input("Type your question…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            retrieved = retrieve_semantic(question, k=k)
            st.session_state.last_retrieved = retrieved

            answer = generate_answer(
                question,
                retrieved,
                max_chunks=max_chunks,
                model=model_name,
                temperature=temperature,
            )

        answer = clean_answer_text(answer)

        st.markdown("### ✅ Answer")

        if stream_answer:
            placeholder = st.empty()
            typed = ""
            for char in answer:
                typed += char
                placeholder.markdown(
                    f"<div class='answer-card'>{typed}</div>",
                    unsafe_allow_html=True,
                )
                time.sleep(0.003)  # typing speed
            placeholder.markdown(
                f"<div class='answer-card'>{answer}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"<div class='answer-card'>{answer}</div>", unsafe_allow_html=True)

        confidence = float(retrieved["score"].iloc[0])
        st.markdown(
            f"<span class='small-muted'>🔎 Retrieval confidence: <b>{confidence:.3f}</b></span>",
            unsafe_allow_html=True,
        )

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ------------------------
# Sources (only after a question has been asked)
# ------------------------
if show_sources and st.session_state.last_retrieved is not None:
    n_sources = len(st.session_state.last_retrieved.head(max_chunks))
    with st.expander(f"📚 Sources ({n_sources})", expanded=False):
        for i, row in st.session_state.last_retrieved.head(max_chunks).iterrows():
            snippet = str(row["text"]).replace("\n", " ").strip()
            snippet = snippet[:240] + ("..." if len(snippet) > 240 else "")
            score_txt = (
                f"<span class='small-muted mono'>score={row['score']:.4f}</span>"
                if show_debug else ""
            )
            st.markdown(
                f"<div class='source-card'><b>[Source {i+1}]</b> {score_txt}<br/>{snippet}</div>",
                unsafe_allow_html=True
            )