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
        bg = "#0F1117"
        card_user = "#23283A"
        card_assistant = "#181C2A"
        text = "#F1F5F9"
        text_secondary = "#CBD5E1"
        muted = "#94A3B8"
        accent = "#818CF8"
        accent_hover = "#A5B4FC"
        border = "#2D3348"
        input_bg = "#181C2A"
        sidebar_bg = "#111422"
        shadow = "rgba(0, 0, 0, 0.4)"
        link_color = "#93C5FD"
        header_gradient = "linear-gradient(135deg, #6366F1 0%, #4F46E5 50%, #7C3AED 100%)"
    else:
        bg = "#F8FAFC"
        card_user = "#EEF2FF"
        card_assistant = "#FFFFFF"
        text = "#1E293B"
        text_secondary = "#334155"
        muted = "#64748B"
        accent = "#6366F1"
        accent_hover = "#4F46E5"
        border = "#E2E8F0"
        input_bg = "#FFFFFF"
        sidebar_bg = "#F1F5F9"
        shadow = "rgba(0, 0, 0, 0.06)"
        link_color = "#4F46E5"
        header_gradient = "linear-gradient(135deg, #6366F1 0%, #4F46E5 50%, #7C3AED 100%)"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* ── Base ── */
        .stApp {{
            background: {bg} !important;
            color: {text} !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        .stApp .main .block-container {{
            color: {text} !important;
        }}

        /* ── Force ALL text colors ── */
        .stApp p, .stApp li, .stApp span, .stApp div,
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
            color: {text} !important;
        }}
        .stApp a {{
            color: {link_color} !important;
        }}

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {{
            background: {sidebar_bg} !important;
            border-right: 1px solid {border} !important;
        }}
        section[data-testid="stSidebar"] * {{
            color: {text} !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stButton"] > button {{
            background: {accent} !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.55rem 1rem;
            transition: background 0.2s ease, transform 0.1s ease;
        }}
        section[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {{
            background: {accent_hover} !important;
            transform: translateY(-1px);
        }}

        /* ── Header card ── */
        .main-title {{
            padding: 1.4rem 1.6rem;
            border-radius: 16px;
            background: {header_gradient};
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 24px rgba(99, 102, 241, 0.3);
            position: relative;
            overflow: hidden;
        }}
        .main-title::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            border-radius: 50%;
        }}
        .main-title h1 {{
            margin: 0;
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.3;
            color: #FFFFFF !important;
            position: relative;
        }}
        .main-title p {{
            margin: 0.4rem 0 0;
            color: rgba(255, 255, 255, 0.85) !important;
            font-size: 0.9rem;
            font-weight: 400;
            position: relative;
        }}

        /* ── Chat messages ── */
        [data-testid="stChatMessage"] {{
            border: 1px solid {border} !important;
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.6rem;
            box-shadow: 0 2px 8px {shadow};
            animation: fadeSlideIn 0.3s ease-out;
        }}
        @keyframes fadeSlideIn {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}

        /* User bubble */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
            background: {card_user} !important;
        }}
        /* Assistant bubble */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{
            background: {card_assistant} !important;
        }}

        /* Force readable text inside ALL chat bubbles */
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] span,
        [data-testid="stChatMessage"] div,
        [data-testid="stChatMessage"] h1,
        [data-testid="stChatMessage"] h2,
        [data-testid="stChatMessage"] h3,
        [data-testid="stChatMessage"] h4,
        [data-testid="stChatMessage"] strong,
        [data-testid="stChatMessage"] em {{
            color: {text} !important;
        }}
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {{
            color: {text} !important;
            line-height: 1.7;
            font-size: 0.95rem;
            font-weight: 400;
        }}
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {{
            color: {text_secondary} !important;
            line-height: 1.7;
            font-size: 0.95rem;
        }}
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {{
            color: {text} !important;
            font-weight: 600;
        }}
        [data-testid="stChatMessage"] a {{
            color: {link_color} !important;
            text-decoration: underline;
        }}

        /* ── Chat input ── */
        [data-testid="stChatInput"] textarea {{
            background: {input_bg} !important;
            border: 1.5px solid {border} !important;
            border-radius: 14px !important;
            color: {text} !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.95rem !important;
            padding: 0.8rem 1rem !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }}
        [data-testid="stChatInput"] textarea::placeholder {{
            color: {muted} !important;
        }}
        [data-testid="stChatInput"] textarea:focus {{
            border-color: {accent} !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
        }}
        [data-testid="stChatInput"] button {{
            background: {accent} !important;
            border-radius: 10px !important;
            color: #FFFFFF !important;
            transition: background 0.2s ease;
        }}
        [data-testid="stChatInput"] button:hover {{
            background: {accent_hover} !important;
        }}
        /* Fix the chat input container background */
        [data-testid="stChatInput"],
        [data-testid="stBottom"] {{
            background: {bg} !important;
        }}
        [data-testid="stBottom"] > div {{
            background: {bg} !important;
        }}

        /* ── Welcome state ── */
        .welcome-hint {{
            text-align: center;
            padding: 3rem 1rem;
        }}
        .welcome-hint .icon {{
            font-size: 2.5rem;
            margin-bottom: 0.8rem;
            opacity: 0.5;
        }}
        .welcome-hint h3 {{
            color: {text} !important;
            font-weight: 600;
            margin-bottom: 0.3rem;
            font-size: 1.1rem;
        }}
        .welcome-hint p {{
            color: {muted} !important;
            font-size: 0.9rem;
            max-width: 400px;
            margin: 0 auto;
            line-height: 1.5;
        }}

        /* ── Scrollbar ── */
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: {border}; border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {muted}; }}

        /* ── Hide Streamlit chrome ── */
        header[data-testid="stHeader"] {{ background: transparent !important; }}
        .stDeployButton {{ display: none !important; }}
        footer {{ display: none !important; }}
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
        delta = getattr(chunk, "choices", [None])[0]
        if delta is not None:
            yield getattr(delta.delta, "content", "") or ""


SYSTEM_PROMPT = (
    "You are a professional representative for Johan Chen, an IT/Cybersecurity Audit expert. "
    "Answer only using the provided retrieved context. "
    "Do not invent facts or use outside knowledge. "
    "If the answer is not present in the context, respond politely and ask the user to contact "
    f"Johan Chen directly via LinkedIn: {LINKEDIN_URL}."
)


with st.sidebar:
    st.markdown("#### Settings")
    theme_mode = st.selectbox("Appearance", options=["Light", "Dark"], index=0)
    model_choice = st.selectbox(
        "Response model",
        options=[OPENAI_CHAT_MODEL, OPENROUTER_CHAT_MODEL],
        index=0,
    )
    show_retrieval_debug = st.checkbox("Show retrieval debug", value=False)

    st.divider()
    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_retrieval = []
        st.rerun()

    st.caption("Responses stream token-by-token via LiteLLM.")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_retrieval" not in st.session_state:
    st.session_state.last_retrieval = []

apply_theme(theme_mode)
st.markdown(
    """
    <div class="main-title">
      <h1>Johan Chen &mdash; Career Portfolio Assistant</h1>
      <p>Ask me anything about Johan's experience, certifications, projects, and skills.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-hint">
          <div class="icon">&#128172;</div>
          <h3>How can I help you?</h3>
          <p>Try asking about work experience, technical skills, certifications, or past projects.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if user_input := st.chat_input("Ask about Johan's experience, projects, certifications, and skills..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    if model_choice.startswith("openrouter/") and not OPENROUTER_API_KEY:
        with st.chat_message("assistant"):
            st.error("Missing OPENROUTER_API_KEY in .streamlit/secrets.toml for the selected model.")
        st.stop()

    try:
        raw_chunks, filtered_chunks = retrieve_relevant_chunks(user_input, top_k=3)
    except Exception as exc:
        with st.chat_message("assistant"):
            st.error(f"Retrieval failed: {exc}")
        st.stop()

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
    else:
        chunks = filtered_chunks if filtered_chunks else raw_chunks
        context_text = format_context(chunks)

        # Build LLM messages from clean history (no RAG context leakage)
        history = trim_messages(st.session_state.messages[:-1], max_turns=10)
        llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        llm_messages.extend(history)
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

        # Store only the clean response (not the RAG-wrapped prompt)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.session_state.messages = trim_messages(st.session_state.messages, max_turns=10)
