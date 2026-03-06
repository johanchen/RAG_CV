import os
from typing import Dict, List, Tuple

import streamlit as st
from litellm import completion
from openai import OpenAI
from supabase import Client, create_client


st.set_page_config(page_title="Johan Chen Career Portfolio", layout="wide")

EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "openai/gpt-5-mini"
OPENROUTER_CHAT_MODEL = "openrouter/nvidia/nemotron-3-nano-30b-a3b:free"
SIMILARITY_THRESHOLD = 0.35


def apply_theme(theme_mode: str) -> None:
    if theme_mode == "Dark":
        bg = "#0E1117"
        panel = "#161B22"
        card = "#111827"
        text = "#E6EDF3"
        muted = "#9AA4B2"
        accent = "#22C55E"
        border = "#2A3441"
    else:
        bg = "#F6F8FB"
        panel = "#FFFFFF"
        card = "#F8FAFC"
        text = "#0F172A"
        muted = "#475569"
        accent = "#0EA5E9"
        border = "#D9E2EC"

    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {bg};
            --panel: {panel};
            --card: {card};
            --text: {text};
            --muted: {muted};
            --accent: {accent};
            --border: {border};
        }}
        .stApp {{
            background: radial-gradient(80% 120% at 100% 0%, rgba(14,165,233,0.12) 0%, transparent 45%), var(--bg);
            color: var(--text);
        }}
        .main-title {{
            padding: 1rem 1.2rem;
            border: 1px solid var(--border);
            border-radius: 14px;
            background: var(--panel);
            margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(2, 8, 23, 0.08);
        }}
        .main-title h1 {{
            margin: 0;
            font-size: 1.4rem;
            line-height: 1.3;
            color: var(--text);
        }}
        .main-title p {{
            margin: 0.35rem 0 0;
            color: var(--muted);
            font-size: 0.95rem;
        }}
        [data-testid="stChatMessage"] {{
            border: 1px solid var(--border);
            border-radius: 14px;
            background: var(--card);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _secret(name: str) -> str:
    value = st.secrets.get(name)
    if not value:
        raise ValueError(f"Missing required secret: {name}")
    return value


SUPABASE_URL = _secret("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = _secret("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = _secret("OPENAI_API_KEY")
GROQ_API_KEY = _secret("GROQ_API_KEY")
LINKEDIN_URL = _secret("LINKEDIN_URL")
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY


@st.cache_resource
def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


@st.cache_resource
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def trim_messages(messages: List[Dict[str, str]], max_turns: int = 10) -> List[Dict[str, str]]:
    return messages[-(max_turns * 2) :]


def embed_text(text: str) -> List[float]:
    client = get_openai_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def retrieve_relevant_chunks(query: str, top_k: int = 3) -> Tuple[List[Dict], List[Dict]]:
    query_embedding = embed_text(query)
    supabase = get_supabase_client()

    response = (
        supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_count": top_k,
            },
        )
        .execute()
    )

    raw_rows = response.data or []
    filtered_rows = [
        row for row in raw_rows if float(row.get("similarity", 0.0)) >= SIMILARITY_THRESHOLD
    ]
    return raw_rows, filtered_rows


def format_context(chunks: List[Dict]) -> str:
    blocks = []
    for row in chunks:
        filename = row.get("filename", "unknown")
        section = row.get("section") or "N/A"
        content = row.get("content", "")
        blocks.append(f"[Source: {filename}, Section: {section}]\n{content}")
    return "\n\n".join(blocks)


def stream_llm_response(model: str, messages: List[Dict[str, str]]):
    completion_kwargs = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    # Some models (for example gpt-5-mini) only allow default temperature.
    if not model.startswith("openai/gpt-5"):
        completion_kwargs["temperature"] = 0.2

    response = completion(**completion_kwargs)
    for chunk in response:
        try:
            yield chunk.choices[0].delta.content or ""
        except Exception:
            if isinstance(chunk, dict):
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                yield delta or ""
            else:
                yield ""


SYSTEM_PROMPT = (
    "You are a professional representative for Johan Chen, an IT/Cybersecurity Audit expert. "
    "Answer only using the provided retrieved context. "
    "Do not invent facts or use outside knowledge. "
    "If the answer is not present in the context, respond politely and ask the user to contact "
    f"Johan Chen directly via LinkedIn: {LINKEDIN_URL}."
)


with st.sidebar:
    st.header("Session")
    theme_mode = st.selectbox("Appearance", options=["Light", "Dark"], index=0)
    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_retrieval = []
        st.rerun()

    st.header("Model")
    model_choice = st.selectbox(
        "Choose response model",
        options=[OPENAI_CHAT_MODEL, OPENROUTER_CHAT_MODEL],
        index=0,
    )
    show_retrieval_debug = st.checkbox("Show retrieval debug", value=False)
    st.caption("Responses stream token-by-token via LiteLLM.")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_retrieval" not in st.session_state:
    st.session_state.last_retrieval = []

apply_theme(theme_mode)
st.markdown(
    """
    <div class="main-title">
      <h1>Johan Chen Career Portfolio Assistant</h1>
      <p>Professional, retrieval-grounded answers based on documented experience.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if user_input := st.chat_input("Ask about Johan's experience, projects, certifications, and skills..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages = trim_messages(st.session_state.messages, max_turns=10)

    with st.chat_message("user"):
        st.markdown(user_input)

    if model_choice.startswith("openrouter/") and not OPENROUTER_API_KEY:
        with st.chat_message("assistant"):
            st.error("Missing OPENROUTER_API_KEY in .streamlit/secrets.toml for the selected model.")
        st.stop()

    raw_chunks, filtered_chunks = retrieve_relevant_chunks(user_input, top_k=3)
    st.session_state.last_retrieval = raw_chunks

    if show_retrieval_debug:
        with st.sidebar:
            st.subheader("Last Retrieval Scores")
            if not raw_chunks:
                st.write("No rows returned by match_documents.")
            else:
                for row in raw_chunks:
                    st.write(
                        f"- {row.get('filename', 'unknown')} | "
                        f"{row.get('section') or 'N/A'} | "
                        f"similarity={float(row.get('similarity', 0.0)):.4f}"
                    )

    if not raw_chunks:
        fallback = (
            "I don't have specific information on that - please reach out to jctx directly via "
            f"LinkedIn: {LINKEDIN_URL}."
        )
        with st.chat_message("assistant"):
            st.markdown(fallback)
        st.session_state.messages.append({"role": "assistant", "content": fallback})
        st.session_state.messages = trim_messages(st.session_state.messages, max_turns=10)
    else:
        chunks = filtered_chunks if filtered_chunks else raw_chunks
        context_text = format_context(chunks)
        prior_messages = trim_messages(st.session_state.messages[:-1], max_turns=10)

        llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        llm_messages.extend(prior_messages)
        llm_messages.append(
            {
                "role": "user",
                "content": (
                    "Use only the context below to answer the user question.\n\n"
                    f"Context:\n{context_text}\n\n"
                    f"User question: {user_input}"
                ),
            }
        )

        with st.chat_message("assistant"):
            full_response = st.write_stream(stream_llm_response(model_choice, llm_messages))

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.messages = trim_messages(st.session_state.messages, max_turns=10)
