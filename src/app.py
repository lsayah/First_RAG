"""
FinReg – Assistant RAG réglementaire
Chatbot Streamlit premium, dark mode, branché sur rag.py.
"""

import streamlit as st
import sys
from pathlib import Path

# ─── Path : permet d'importer rag.py depuis le même dossier ─────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ─── Config page ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinReg · Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS global ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* Reset & base */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #0a0b0e !important;
    color: #e8e6e0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2d35; border-radius: 2px; }

/* Layout principal */
[data-testid="stMain"] {
    padding: 0 !important;
    background: #0a0b0e !important;
}
[data-testid="stMainBlockContainer"] {
    padding: 0 !important;
    max-width: 100% !important;
}
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Inputs Streamlit */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
.stTextInput input {
    background: #13151a !important;
    border: 1px solid #22252e !important;
    border-radius: 12px !important;
    color: #e8e6e0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.2rem !important;
    padding: 14px 18px !important;
    caret-color: #c8a96e !important;
    transition: border-color 0.2s ease !important;
    box-shadow: none !important;
}
[data-testid="stTextInput"] input:focus,
.stTextInput input:focus {
    border-color: #c8a96e !important;
    box-shadow: 0 0 0 3px rgba(200,169,110,0.1) !important;
    outline: none !important;
}

/* Boutons */
.stButton > button {
    background: #c8a96e !important;
    color: #0a0b0e !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 12px 22px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    background: #d4b87e !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(200,169,110,0.25) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* Bouton secondaire (clear) */
.btn-secondary > button {
    background: transparent !important;
    color: #6b7280 !important;
    border: 1px solid #22252e !important;
    font-size: 0.82rem !important;
    padding: 8px 16px !important;
}
.btn-secondary > button:hover {
    background: #13151a !important;
    color: #9ca3af !important;
    transform: none !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

# ─── État session ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_active" not in st.session_state:
    st.session_state.rag_active = True
if "sources_shown" not in st.session_state:
    st.session_state.sources_shown = set()

# ─── Import RAG réel ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag():
    """Charge les modèles RAG une seule fois (FAISS + SentenceTransformer)."""
    try:
        from rag import answer_question, build_context, get_faiss_index, get_embedding_model, get_metadata
        # Préchauffage des caches
        get_faiss_index()
        get_embedding_model()
        get_metadata()
        return answer_question, build_context
    except Exception as e:
        return None, str(e)


def query_rag(question: str) -> dict:
    """
    Appelle ton vrai RAG (rag.py → answer_question + chunks pour les sources).
    Retourne : {"answer": str, "sources": list[dict]}
    """
    answer_fn, build_ctx_fn = load_rag()

    # Si le chargement a échoué, build_ctx_fn contient le message d'erreur
    if answer_fn is None:
        return {
            "answer": f"❌ Impossible de charger le RAG : {build_ctx_fn}\n\nVérifiez que l'index FAISS est bien créé (`python src/main.py`) et que les variables d'environnement sont définies.",
            "sources": [],
        }

    try:
        # On appelle build_context pour récupérer les chunks (sources)
        from rag import build_context
        _, chunks = build_context(question)

        # Puis answer_question pour la réponse LLM
        answer, chunks = answer_fn(question)

        # Reformatage des chunks en sources pour l'UI
        sources = []
        for chunk in chunks:
            sources.append({
                "title": chunk.get("document", chunk.get("filename", "Document inconnu")),
                "ref":   chunk.get("filename", ""),
                "relevance": round(chunk.get("score", 0), 1),
                "first_phrase": chunk.get("first_phrase", ""),
            })

        return {"answer": answer, "sources": sources}

    except Exception as e:
        return {
            "answer": f"❌ Erreur lors de la requête RAG : {e}",
            "sources": [],
        }

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    position: sticky; top: 0; z-index: 100;
    background: rgba(10,11,14,0.92);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid #1a1d24;
    padding: 0 28px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
">
    <div style="display:flex;align-items:center;gap:14px;">
        <span style="font-size:1.2rem;">⚖️</span>
        <div>
            <div style="
                font-family:'Syne',sans-serif;
                font-size:1rem;
                font-weight:700;
                color:#e8e6e0;
                letter-spacing:0.02em;
                line-height:1.1;
            ">FinReg</div>
            <div style="
                font-size:0.65rem;
                color:#4b5563;
                text-transform:uppercase;
                letter-spacing:0.14em;
                font-weight:500;
            ">Assistant réglementaire</div>
        </div>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
        <div style="width:7px;height:7px;background:#34d399;border-radius:50%;box-shadow:0 0 8px rgba(52,211,153,0.5);"></div>
        <span style="font-size:0.75rem;color:#6b7280;font-weight:500;">RAG actif</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Zone de chat ────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    # Message de bienvenue si vide
    if not st.session_state.messages:
        st.markdown("""
        <div style="
            text-align:center;
            padding: 80px 20px 40px;
            max-width: 560px;
            margin: 0 auto;
        ">
            <div style="
                font-size:2.8rem;
                margin-bottom:20px;
                opacity:0.6;
            ">⚖️</div>
            <div style="
                font-family:'Syne',sans-serif;
                font-size:1.9rem;
                font-weight:700;
                color:#e8e6e0;
                margin-bottom:12px;
            ">Bonjour, comment puis-je vous aider ?</div>
            <div style="
                font-size:1.05rem;
                color:#4b5563;
                line-height:1.6;
                margin-bottom:36px;
            ">
                Je suis connecté à votre base documentaire réglementaire.<br>
                Posez-moi n'importe quelle question sur RGPD, MiFID, DORA, etc.
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:10px;justify-content:center;">
        """, unsafe_allow_html=True)
        
        suggestions = [
            "Droit à l'effacement RGPD",
            "Exigences DORA 2025",
            "Obligations MiFID II",
            "Conservation des données",
        ]
        cols = st.columns(len(suggestions))
        for i, s in enumerate(suggestions):
            with cols[i]:
                if st.button(s, key=f"sug_{i}"):
                    st.session_state.messages.append({"role": "user", "content": s})
                    with st.spinner(""):
                        result = query_rag(s)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })
                    st.rerun()
        
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Affichage des messages
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f"""
            <div style="
                display:flex;
                justify-content:flex-end;
                margin: 18px 28px 6px;
            ">
                <div style="
                    max-width: min(70%, 680px);
                    background: #c8a96e;
                    color: #0a0b0e;
                    border-radius: 18px 18px 4px 18px;
                    padding: 13px 18px;
                    font-size: 1rem;
                    line-height: 1.55;
                    font-weight: 450;
                ">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.markdown(f"""
            <div style="
                display:flex;
                justify-content:flex-start;
                align-items:flex-start;
                gap:12px;
                margin: 6px 28px 4px;
            ">
                <div style="
                    width:32px;height:32px;
                    background:#13151a;
                    border:1px solid #22252e;
                    border-radius:50%;
                    display:flex;align-items:center;justify-content:center;
                    font-size:0.85rem;
                    flex-shrink:0;
                    margin-top:2px;
                ">⚖️</div>
                <div style="max-width:min(72%,700px);">
                    <div style="
                        background:#13151a;
                        border:1px solid #1e2028;
                        border-radius:4px 18px 18px 18px;
                        padding:15px 20px;
                        font-size:1rem;
                        line-height:1.65;
                        color:#ddd8ce;
                    ">{msg['content']}</div>
            """, unsafe_allow_html=True)
            
            # Sources
            sources = msg.get("sources", [])
            if sources:
                src_key = f"src_{i}"
                show = st.session_state.sources_shown
                toggle_label = f"📎 {len(sources)} source{'s' if len(sources)>1 else ''}" if src_key not in show else "✕ Masquer"
                
                if st.button(toggle_label, key=f"btn_src_{i}"):
                    if src_key in show:
                        show.discard(src_key)
                    else:
                        show.add(src_key)
                    st.rerun()
                
                if src_key in show:
                    for src in sources:
                        first_phrase = src.get('first_phrase', '')
                        excerpt_html = f'<div style="font-size:0.78rem;color:#6b7280;margin-top:5px;font-style:italic;line-height:1.4;">« {first_phrase[:120]}… »</div>' if first_phrase else ''
                        st.markdown(f"""
                        <div style="
                            margin-top:6px;
                            background:#0e1016;
                            border:1px solid #1e2028;
                            border-left:3px solid #c8a96e;
                            border-radius:0 8px 8px 0;
                            padding:10px 14px;
                        ">
                            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                                <div style="flex:1;">
                                    <div style="font-size:0.9rem;font-weight:600;color:#c8c3bb;">{src['title']}</div>
                                    <div style="font-size:0.82rem;color:#4b5563;margin-top:2px;">{src['ref']}</div>
                                    {excerpt_html}
                                </div>
                                <div style="
                                    font-size:0.8rem;font-weight:700;
                                    color:#34d399;
                                    background:rgba(52,211,153,0.08);
                                    padding:3px 8px;border-radius:999px;
                                    white-space:nowrap;
                                    margin-left:12px;
                                    flex-shrink:0;
                                ">{src['relevance']}%</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)

# ─── Barre de saisie (sticky en bas) ────────────────────────────────────────
st.markdown("""
<div style="
    position:fixed; bottom:0; left:0; right:0;
    background: rgba(10,11,14,0.95);
    backdrop-filter:blur(20px);
    border-top:1px solid #1a1d24;
    padding:16px 28px 20px;
    z-index:100;
">
</div>
""", unsafe_allow_html=True)

# Zone input réelle
st.markdown('<div style="height:80px;"></div>', unsafe_allow_html=True)

with st.container():
    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"]:last-of-type {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(10,11,14,0.97);
        backdrop-filter: blur(20px);
        border-top: 1px solid #1a1d24;
        padding: 14px 24px 18px !important;
        z-index: 999;
        margin: 0 !important;
        gap: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col_input, col_send, col_clear = st.columns([8, 1, 1])
    
    with col_input:
        user_input = st.text_input(
            label="",
            placeholder="Posez votre question réglementaire…",
            key="chat_input",
            label_visibility="collapsed",
        )
    
    with col_send:
        send = st.button("→", key="send_btn", help="Envoyer")
    
    with col_clear:
        st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
        clear = st.button("✕", key="clear_btn", help="Effacer la conversation")
        st.markdown('</div>', unsafe_allow_html=True)

# ─── Logique d'envoi ─────────────────────────────────────────────────────────
if (send or user_input and user_input != st.session_state.get("_last_input", "")) and user_input.strip():
    st.session_state._last_input = user_input
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})
    
    with st.spinner("Recherche en cours…"):
        result = query_rag(user_input.strip())
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
    st.rerun()

if clear:
    st.session_state.messages = []
    st.session_state.sources_shown = set()
    st.session_state._last_input = ""
    st.rerun()