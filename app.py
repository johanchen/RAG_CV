import os
from typing import Dict, List, Tuple

import streamlit as st
from litellm import completion
from openai import OpenAI
from supabase import Client, create_client


st.set_page_config(page_title="Johan Chen — Career Portfolio", layout="wide", page_icon="🔐")

EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "openai/gpt-5.1"
OPENROUTER_CHAT_MODEL = "openrouter/arcee-ai/trinity-large-preview:free"
SIMILARITY_THRESHOLD = 0.35

# Friendly display labels for the model selectbox
MODEL_LABELS = {
    OPENAI_CHAT_MODEL:      "GPT-5.1",
    OPENROUTER_CHAT_MODEL:  "Arcee Trinity",
}


def apply_theme(theme_mode: str) -> None:
    if theme_mode == "Dark":
        bg = "#07101F"
        surface = "#0C1828"
        card_user = "#0F1E35"
        card_assistant = "#0C1828"
        text = "#E4EAF6"
        text_secondary = "#9BADC8"
        muted = "#4E6280"
        accent = "#22D3EE"
        accent2 = "#F59E0B"
        border = "#1A2E47"
        input_bg = "#0C1828"
        sidebar_bg = "#060E1A"
        shadow = "rgba(0, 0, 0, 0.5)"
        link_color = "#38BDF8"
        dot_color = "rgba(34, 211, 238, 0.05)"
        badge_bg = "rgba(34, 211, 238, 0.1)"
        badge_border = "rgba(34, 211, 238, 0.3)"
        rag_chip_bg = "rgba(245, 158, 11, 0.1)"
        rag_chip_border = "rgba(245, 158, 11, 0.3)"
        rag_chip_text = "#F59E0B"
        header_bg = "linear-gradient(135deg, #0C2340 0%, #0A1A30 60%, #091525 100%)"
        header_border = "rgba(34, 211, 238, 0.18)"
        header_title_color = "#FFFFFF"
        header_sub_color = "rgba(255,255,255,0.55)"
        header_badge_bg = "rgba(34, 211, 238, 0.12)"
        header_badge_border = "rgba(34, 211, 238, 0.32)"
        header_badge_text = accent
        glow = "rgba(34, 211, 238, 0.08)"
        toolbar_bg = "rgba(7, 16, 31, 0.92)"
        select_border = "rgba(34, 211, 238, 0.4)"
        select_text = "#FFFFFF"
    else:
        bg = "#F2F5FA"
        surface = "#FFFFFF"
        card_user = "#E8F3FF"
        card_assistant = "#FFFFFF"
        text = "#0D1829"
        text_secondary = "#334E6B"
        muted = "#7A93AD"
        accent = "#0891B2"
        accent2 = "#D97706"
        border = "#C8D9E8"
        input_bg = "#FFFFFF"
        sidebar_bg = "#E8EFF8"
        shadow = "rgba(0, 0, 0, 0.06)"
        link_color = "#0891B2"
        dot_color = "rgba(8, 145, 178, 0.07)"
        badge_bg = "rgba(8, 145, 178, 0.08)"
        badge_border = "rgba(8, 145, 178, 0.25)"
        rag_chip_bg = "rgba(217, 119, 6, 0.08)"
        rag_chip_border = "rgba(217, 119, 6, 0.25)"
        rag_chip_text = "#B45309"
        header_bg = "linear-gradient(135deg, #FFFFFF 0%, #F0F6FF 55%, #E6F3FA 100%)"
        header_border = "rgba(8, 145, 178, 0.2)"
        header_title_color = "#0D1829"
        header_sub_color = "#5A7A9A"
        header_badge_bg = "#0C2340"
        header_badge_border = "rgba(12, 35, 64, 0.5)"
        header_badge_text = "#22D3EE"
        glow = "rgba(8, 145, 178, 0.05)"
        toolbar_bg = "rgba(242, 245, 250, 0.92)"
        select_border = "rgba(8, 145, 178, 0.5)"
        select_text = "#0D1829"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        /* ── CSS variables ── */
        :root {{
            --bg: {bg};
            --surface: {surface};
            --accent: {accent};
            --accent2: {accent2};
            --border: {border};
            --text: {text};
            --text-secondary: {text_secondary};
            --muted: {muted};
            --shadow: {shadow};
            --badge-bg: {badge_bg};
            --badge-border: {badge_border};
        }}

        /* ── Base ── */
        .stApp {{
            background: {bg} !important;
            color: {text} !important;
            font-family: 'Outfit', -apple-system, sans-serif;
        }}

        /* Dot-grid atmosphere */
        .stApp::before {{
            content: '';
            position: fixed;
            inset: 0;
            background-image: radial-gradient(circle, {dot_color} 1.2px, transparent 1.2px);
            background-size: 26px 26px;
            pointer-events: none;
            z-index: 0;
        }}

        /*
         * ── Block-container ──
         * padding-top is increased to compensate for the fixed toolbar (≈ 46px + gap).
         * Without this the header card would render beneath the floating toolbar.
         */
        .stApp .main .block-container {{
            color: {text} !important;
            position: relative;
            z-index: 1;
            padding-top: 1.2rem !important;
            max-width: 880px;
        }}

        /* ── Typography ── */
        .stApp p, .stApp li, .stApp span, .stApp div {{
            color: {text} !important;
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
            color: {text} !important;
            font-family: 'Syne', sans-serif;
        }}
        .stApp a {{
            color: {link_color} !important;
            text-decoration: none;
        }}
        .stApp a:hover {{ text-decoration: underline; }}

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {{
            background: {sidebar_bg} !important;
            border-right: 1px solid {border} !important;
        }}
        section[data-testid="stSidebar"] * {{
            color: {text} !important;
            font-family: 'Outfit', sans-serif !important;
        }}
        section[data-testid="stSidebar"] label {{
            font-size: 0.72rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.09em !important;
            text-transform: uppercase !important;
            color: {muted} !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stCheckbox"] label span {{
            font-size: 0.83rem !important;
            letter-spacing: 0 !important;
            text-transform: none !important;
        }}

        /* ═══════════════════════════════════════════════════════════════
         * FIXED TOP-RIGHT TOOLBAR
         *
         * How this works:
         *   apply_theme() calls st.markdown() to inject CSS — that call
         *   creates the FIRST div child of .block-container (height ≈ 0).
         *   The st.columns([3,1,1]) toolbar we render next becomes the
         *   SECOND child.  We target it with nth-child(2) and pin it to
         *   the viewport top-right with position:fixed + high z-index.
         *   padding-top on .block-container (above) compensates for the
         *   space the toolbar no longer occupies in the normal flow.
         * ═══════════════════════════════════════════════════════════════ */

        /* Pin the toolbar container — bottom-right, floating above chat input */
        .stApp .main .block-container > div:has(> [data-testid="stHorizontalBlock"]) {{
            position: fixed !important;
            bottom: 88px !important;
            right: 1.5rem !important;
            z-index: 999999 !important;
            width: 210px !important;
            padding: 0 !important;
            margin: 0 !important;
            background: {toolbar_bg} !important;
            backdrop-filter: blur(8px) !important;
            -webkit-backdrop-filter: blur(8px) !important;
            border-radius: 10px !important;
            border: 1px solid {border} !important;
            box-shadow: 0 2px 12px {shadow} !important;
        }}

        /* Collapse the inner horizontal block padding / gap */
        .stApp .main .block-container > div:has(> [data-testid="stHorizontalBlock"]) [data-testid="stHorizontalBlock"] {{
            gap: 0 !important;
            padding: 0.2rem 0.25rem !important;
            align-items: center !important;
        }}
        .stApp .main .block-container > div:has(> [data-testid="stHorizontalBlock"]) [data-testid="stColumn"] {{
            padding: 0 0.18rem !important;
            min-width: 0 !important;
        }}

        /* ── Toolbar: model selectbox ── */
        .stApp .main .block-container > div:has(> [data-testid="stHorizontalBlock"]) [data-testid="stSelectbox"] > label {{
            display: none !important;
        }}
        /* The select control itself */
        .stApp .main .block-container > div:has(> [data-testid="stHorizontalBlock"]) [data-baseweb="select"] > div {{
            background: transparent !important;
            border: 1.5px solid {select_border} !important;
            border-radius: 7px !important;
            height: 30px !important;
            min-height: 30px !important;
            padding: 0 0.35rem !important;
            cursor: pointer !important;
            transition: border-color 0.15s !important;
        }}
        .stApp .main .block-container > div:has(> [data-testid="stHorizontalBlock"]) [data-baseweb="select"] > div:hover {{
            border-color: {accent} !important;
        }}
        /* Selected value text */
        .stApp .main .block-container > div:has(> [data-testid="stHorizontalBlock"]) [data-baseweb="select"] [role="combobox"] span,
        .stApp .main .block-container > div:has(> [data-testid="stHorizontalBlock"]) [data-baseweb="select"] [role="combobox"] div {{
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.71rem !important;
            color: {select_text} !important;
            line-height: 30px !important;
        }}
        /* Dropdown chevron */
        .stApp .main .block-container > div:has(> [data-testid="stHorizontalBlock"]) [data-baseweb="select"] svg {{
            color: {muted} !important;
            width: 14px !important;
            height: 14px !important;
        }}

        /* ── Toolbar: icon buttons (theme + new chat) ── */
        .stApp .main .block-container > div:has(> [data-testid="stHorizontalBlock"]) [data-testid="stButton"] > button {{
            background: transparent !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            -webkit-tap-highlight-color: transparent !important;
            border-radius: 7px !important;
            color: {text} !important;
            font-size: 1.1rem !important;
            line-height: 1 !important;
            padding: 0 !important;
            height: 30px !important;
            width: 100% !important;
            min-height: 30px !important;
            transition: background 0.15s ease !important;
        }}
        .stApp .main .block-container > div:has(> [data-testid="stHorizontalBlock"]) [data-testid="stButton"] > button:hover {{
            background: {badge_bg} !important;
        }}
        .stApp .main .block-container > div:has(> [data-testid="stHorizontalBlock"]) [data-testid="stButton"] > button:focus,
        .stApp .main .block-container > div:has(> [data-testid="stHorizontalBlock"]) [data-testid="stButton"] > button:focus-visible,
        .stApp .main .block-container > div:has(> [data-testid="stHorizontalBlock"]) [data-testid="stButton"] > button:active {{
            outline: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }}

        /* ── Dropdown popup for model selectbox ── */
        [data-baseweb="popover"] ul[role="listbox"] {{
            background: #FFFFFF !important;
            border: 1px solid #C8D9E8 !important;
            border-radius: 8px !important;
            padding: 0.25rem !important;
        }}
        [data-baseweb="popover"] li[role="option"] {{
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.75rem !important;
            color: #0D1829 !important;
            background: transparent !important;
            border-radius: 5px !important;
            padding: 0.35rem 0.6rem !important;
        }}
        [data-baseweb="popover"] li[role="option"]:hover {{
            background: rgba(8, 145, 178, 0.10) !important;
        }}
        [data-baseweb="popover"] li[role="option"][aria-selected="true"] {{
            background: rgba(8, 145, 178, 0.12) !important;
            color: #0891B2 !important;
        }}

        /* ── Header card ── */
        .main-title {{
            padding: 1.6rem 2rem 1.4rem;
            border-radius: 16px;
            background: {header_bg};
            margin-bottom: 1.2rem;
            border: 1px solid {header_border};
            position: relative;
            overflow: hidden;
        }}
        .main-title::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent 0%, {accent} 40%, {accent2} 70%, transparent 100%);
        }}
        .main-title::after {{
            content: '';
            position: absolute;
            top: -100px; right: -80px;
            width: 300px; height: 300px;
            background: radial-gradient(circle, {glow} 0%, transparent 65%);
            border-radius: 50%;
            pointer-events: none;
        }}
        .main-title-inner {{
            display: flex;
            align-items: center;
            gap: 1.2rem;
            position: relative;
            z-index: 2;
        }}
        .main-title-badge {{
            width: 54px; height: 54px; min-width: 54px;
            border-radius: 12px;
            background: {header_badge_bg};
            border: 1px solid {header_badge_border};
            display: flex; align-items: center; justify-content: center;
            font-family: 'Syne', sans-serif !important;
            font-weight: 800; font-size: 1rem;
            color: {header_badge_text} !important;
            flex-shrink: 0; letter-spacing: 0.02em; line-height: 1;
        }}
        .main-title-badge,
        div.main-title-badge,
        .main-title .main-title-inner .main-title-badge {{
            color: {header_badge_text} !important;
        }}
        .main-title-text {{
            display: flex; flex-direction: column; justify-content: center;
        }}
        .main-title-text h1 {{
            margin: 0; padding: 0;
            font-family: 'Syne', sans-serif !important;
            font-size: 1.22rem !important; font-weight: 700 !important;
            color: {header_title_color} !important;
            letter-spacing: -0.01em; line-height: 1.25;
        }}
        .main-title-text p {{
            margin: 0.28rem 0 0; padding: 0;
            color: {header_sub_color} !important;
            font-size: 0.83rem !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 400; letter-spacing: 0.01em; line-height: 1.4;
        }}
        .main-title-intro {{
            margin-top: 1rem;
            padding-top: 0.9rem;
            border-top: 1px solid {header_border};
            font-family: 'Outfit', sans-serif !important;
            font-size: 0.84rem !important;
            color: {header_sub_color} !important;
            line-height: 1.65;
            position: relative; z-index: 2;
        }}
        .main-title-chips {{
            display: flex; gap: 0.45rem;
            margin-top: 1rem;
            position: relative; z-index: 2;
            flex-wrap: wrap;
        }}
        .chip {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem; font-weight: 500;
            padding: 0.22rem 0.65rem; border-radius: 4px;
            letter-spacing: 0.04em;
            display: inline-flex; align-items: center; gap: 0.3rem;
        }}
        .chip-cyan {{
            background: {badge_bg}; border: 1px solid {badge_border};
            color: {accent} !important;
        }}
        .chip-amber {{
            background: {rag_chip_bg}; border: 1px solid {rag_chip_border};
            color: {rag_chip_text} !important;
        }}

        /* ── Chat messages ── */
        [data-testid="stChatMessage"] {{
            border-radius: 12px;
            padding: 0.95rem 1.15rem;
            margin-bottom: 0.65rem;
            animation: msgIn 0.22s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
        }}
        @keyframes msgIn {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
            background: {card_user} !important;
            border: 1px solid {border} !important;
            border-left: 3px solid {accent} !important;
        }}
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{
            background: {card_assistant} !important;
            border: 1px solid {border} !important;
            border-left: 3px solid {accent2} !important;
            box-shadow: 0 2px 14px {shadow};
        }}
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] span,
        [data-testid="stChatMessage"] div,
        [data-testid="stChatMessage"] strong,
        [data-testid="stChatMessage"] em {{
            color: {text} !important;
        }}
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {{
            color: {text} !important; line-height: 1.78;
            font-size: 0.92rem; font-family: 'Outfit', sans-serif;
        }}
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {{
            color: {text_secondary} !important; line-height: 1.78; font-size: 0.92rem;
        }}
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {{
            color: {text} !important; font-weight: 600;
        }}
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem; background: {border} !important;
            padding: 0.12em 0.45em; border-radius: 4px; color: {accent} !important;
        }}
        [data-testid="stChatMessage"] a {{ color: {link_color} !important; }}

        /* ── Chat input ── */
        [data-testid="stChatInput"] textarea {{
            background: {input_bg} !important;
            border: 1.5px solid {border} !important;
            border-radius: 10px !important;
            color: {text} !important;
            font-family: 'Outfit', sans-serif !important;
            font-size: 0.92rem !important;
            padding: 0.85rem 1.1rem !important;
            transition: border-color 0.2s, box-shadow 0.2s;
        }}
        [data-testid="stChatInput"] textarea::placeholder {{ color: {muted} !important; }}
        [data-testid="stChatInput"] textarea:focus {{
            border-color: {accent} !important;
            box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.1) !important;
        }}
        [data-testid="stChatInput"] button {{
            display: flex !important; align-items: center !important;
            justify-content: center !important;
            background: {accent} !important; border: none !important;
            border-radius: 8px !important; color: #000000 !important;
            width: 34px !important; height: 34px !important;
            min-width: 34px !important; padding: 0 !important;
            margin: auto 0.35rem auto 0 !important; cursor: pointer !important;
            transition: opacity 0.2s ease, transform 0.15s ease !important;
            flex-shrink: 0 !important;
        }}
        [data-testid="stChatInput"] button:hover {{
            opacity: 0.82 !important; transform: scale(1.08) !important;
        }}
        [data-testid="stChatInput"] button svg {{
            fill: #000000 !important; width: 16px !important; height: 16px !important;
        }}
        [data-testid="stChatInput"], [data-testid="stBottom"] {{
            background: {bg} !important;
        }}
        [data-testid="stBottom"] > div {{ background: {bg} !important; }}

        /* ── Welcome state ── */
        .welcome-wrap {{
            text-align: center; padding: 3.5rem 1rem 2.5rem;
        }}
        .welcome-icon {{
            width: 62px; height: 62px; margin: 0 auto 1.2rem;
            border-radius: 16px; background: {badge_bg}; border: 1px solid {badge_border};
            display: flex; align-items: center; justify-content: center; font-size: 1.7rem;
        }}
        .welcome-wrap h3 {{
            color: {text} !important; font-family: 'Syne', sans-serif !important;
            font-weight: 700 !important; font-size: 1.08rem !important;
            margin-bottom: 0.45rem; letter-spacing: -0.01em;
        }}
        .welcome-wrap p {{
            color: {muted} !important; font-size: 0.87rem !important;
            max-width: 370px; margin: 0 auto; line-height: 1.65;
            font-family: 'Outfit', sans-serif !important;
        }}
        .suggestion-pills {{
            display: flex; flex-wrap: wrap; gap: 0.45rem;
            justify-content: center; margin-top: 1.4rem;
            max-width: 480px; margin-left: auto; margin-right: auto;
        }}
        .pill {{
            font-family: 'Outfit', sans-serif; font-size: 0.79rem;
            color: {text_secondary} !important; background: {surface};
            border: 1px solid {border}; border-radius: 20px; padding: 0.32rem 0.75rem;
        }}

        /* ── Scrollbar ── */
        ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: {border}; border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {muted}; }}

        /* ── Caption / footer text ── */
        .stApp .stCaption, .stApp .stCaption * {{
            color: {muted} !important; font-size: 0.76rem !important;
            font-family: 'JetBrains Mono', monospace !important;
        }}

        /* ── Hide Streamlit chrome ── */
        header[data-testid="stHeader"] {{ background: transparent !important; box-shadow: none !important; }}
        .stDeployButton {{ display: none !important; }}
        footer {{ display: none !important; }}
        #MainMenu {{ display: none !important; }}
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
EMAIL_ADDRESS = _secret("EMAIL_ADDRESS")
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
            {"query_embedding": query_embedding, "match_count": top_k},
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
        subsection = row.get("subsection") or ""
        content = row.get("content", "")
        if subsection and subsection not in ("N/A", section):
            location = f"{section} > {subsection}"
        else:
            location = section
        blocks.append(f"[Source: {filename}, Section: {location}]\n{content}")
    return "\n\n".join(blocks)


def stream_llm_response(model: str, messages: List[Dict[str, str]]):
    completion_kwargs = {"model": model, "messages": messages, "stream": True}
    if not model.startswith("openai/gpt-5"):
        completion_kwargs["temperature"] = 0.2
    response = completion(**completion_kwargs)
    for chunk in response:
        delta = getattr(chunk, "choices", [None])[0]
        if delta is not None:
            yield getattr(delta.delta, "content", "") or ""


SYSTEM_PROMPT = (
    "You are a professional assistant representative for Johan Chen, an IT/Cybersecurity expert. "
    "Answer only using the provided retrieved context. "
    "Do not invent facts or use outside knowledge. "
    "Always check the latest information of Johan. For example, between GM and VP, VP is the latest title."
    "If user is asking about professional certificate, you should refer to CISA, CISSP, CDPSE, and other related certifications."
    "If the user is asking to list down something, your output should be in the table format."
    f"If the answer is not present in the context, respond politely and ask the user to contact Johan Chen directly via LinkedIn: {LINKEDIN_URL}."
    f"If being asked about contact information of Johan Chen, only provide email address {EMAIL_ADDRESS} and LinkedIn URL, do not share mobile number."
)


# ── Session state ──────────────────────────────────────────────────────────────
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"
if "model_choice" not in st.session_state:
    st.session_state.model_choice = OPENAI_CHAT_MODEL
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_retrieval" not in st.session_state:
    st.session_state.last_retrieval = []

theme_mode = st.session_state.theme_mode

# ── Apply theme  ───────────────────────────────────────────────────────────────
# NOTE: this st.markdown call is the FIRST child of .block-container.
# The toolbar columns below become the SECOND child, which is what the
# nth-child(2) CSS rule above targets for position:fixed.
apply_theme(theme_mode)

# ── Sidebar (debug only) ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding: 0.8rem 0 0.6rem; font-family: JetBrains Mono, monospace; "
        "font-size: 0.63rem; letter-spacing: 0.1em; text-transform: uppercase; "
        "color: var(--muted); border-bottom: 1px solid var(--border); margin-bottom: 1rem;'>"
        "Debug</div>",
        unsafe_allow_html=True,
    )
    show_retrieval_debug = st.checkbox("Show retrieval debug", value=False)

# ── Fixed top-right toolbar: [Model ▾] [☀/🌙] [✏] ────────────────────────────
# This is the SECOND child of .block-container — pinned via nth-child(2) CSS above.
_col_model, _col_theme, _col_newchat = st.columns([3, 1, 1])

with _col_model:
    _model_options = list(MODEL_LABELS.keys())
    _current_idx = (
        _model_options.index(st.session_state.model_choice)
        if st.session_state.model_choice in _model_options else 0
    )
    _selected = st.selectbox(
        "Model",
        options=_model_options,
        format_func=lambda x: MODEL_LABELS[x],
        index=_current_idx,
        label_visibility="collapsed",
        key="model_select",
    )
    st.session_state.model_choice = _selected
    model_choice = _selected

with _col_theme:
    _theme_icon = "☀️" if theme_mode == "Dark" else "🌙"
    if st.button(_theme_icon, key="theme_toggle", help="Toggle light/dark mode",
                 use_container_width=True):
        st.session_state.theme_mode = "Light" if theme_mode == "Dark" else "Dark"
        st.rerun()

with _col_newchat:
    if st.button("✏️", key="new_chat_top", help="New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_retrieval = []
        st.rerun()

# ── Header card ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-title">
      <div class="main-title-inner">
        <div class="main-title-badge">JC</div>
        <div class="main-title-text">
          <h1>Johan Chen</h1>
          <p>IT &amp; Cybersecurity Audit Expert &mdash; Career Portfolio Assistant</p>
        </div>
      </div>
      <div class="main-title-intro">
        This AI assistant answers questions about Johan's professional background using his actual career documents.
        Ask about his work history, certifications (CISA, CISSP, CDPSE), technical skills, audit methodology, or past projects.
      </div>
      <div class="main-title-chips">
        <span class="chip chip-cyan">&#9632; RAG-Powered</span>
        <span class="chip chip-amber">&#10022; AI Assistant</span>
        <span class="chip chip-cyan">&#128274; Cybersecurity</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Welcome state ─────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-wrap">
          <div class="welcome-icon">&#128274;</div>
          <h3>What would you like to know?</h3>
          <p>Ask about work experience, certifications, technical skills, audit methodology, or past projects.</p>
          <div class="suggestion-pills">
            <span class="pill">Work experience</span>
            <span class="pill">Certifications</span>
            <span class="pill">Technical skills</span>
            <span class="pill">Past projects</span>
            <span class="pill">Audit expertise</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask about Johan's experience, projects, certifications, and skills..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    if model_choice.startswith("openrouter/") and not OPENROUTER_API_KEY:
        with st.chat_message("assistant"):
            st.error("Missing OPENROUTER_API_KEY in .streamlit/secrets.toml for the selected model.")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                raw_chunks, filtered_chunks = retrieve_relevant_chunks(user_input, top_k=10)
            except Exception as exc:
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
                        section = row.get("section") or "N/A"
                        subsection = row.get("subsection") or ""
                        location = (
                            f"{section} > {subsection}"
                            if subsection and subsection not in ("N/A", section)
                            else section
                        )
                        st.write(
                            f"- {row.get('filename', 'unknown')} | "
                            f"{location} | "
                            f"similarity={float(row.get('similarity', 0.0)):.4f}"
                        )

        if not raw_chunks:
            fallback = (
                "I don't have specific information on that - please reach out to Johan directly via "
                f"LinkedIn: {LINKEDIN_URL} or email: {EMAIL_ADDRESS}."
            )
            st.markdown(fallback)
            st.session_state.messages.append({"role": "assistant", "content": fallback})
        else:
            chunks = filtered_chunks if filtered_chunks else raw_chunks
            context_text = format_context(chunks)

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

            full_response = st.write_stream(stream_llm_response(model_choice, llm_messages))
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.session_state.messages = trim_messages(st.session_state.messages, max_turns=10)
