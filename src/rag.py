# ============================================================================
# 0. IMPORT & CONFIGURATION
# ============================================================================

import json
import os
import numpy as np
import faiss
import tiktoken
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from utils.embedding import get_embeddings
from utils.query_translate import expand_query, deduplicate_chunks          # ← AJOUT
from config import INDEX_PATH, METADATA_PATH, CONTEXT_PATH, EMBEDDING_MODEL, LLM_MODEL, MAX_CONTEXT_TOKENS, K_RESULTS

load_dotenv()

# ============================================================================
# CACHES GLOBAUX (pour éviter rechargements)
# ============================================================================
_embedding_model = None
_metadata_cache  = None
_faiss_index     = None


# ============================================================================
# FONCTIONS DE CACHE
# ============================================================================

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_metadata():
    global _metadata_cache
    if _metadata_cache is None:
        with open(METADATA_PATH, "r", encoding='utf-8') as f:
            _metadata_cache = json.load(f)

        saved_model = _metadata_cache.get("_config", {}).get("embedding_model")

        if saved_model and saved_model != EMBEDDING_MODEL:
            raise RuntimeError(
                f"❌ Modèle mismatch!\n"
                f"   Index créé avec : {saved_model}\n"
                f"   Config actuelle : {EMBEDDING_MODEL}\n"
                f"   → Supprimez index.faiss et metadata.json puis relancez"
            )

    return _metadata_cache


def get_faiss_index():
    global _faiss_index
    if _faiss_index is None:
        _faiss_index = faiss.read_index(str(INDEX_PATH))
    return _faiss_index


# ============================================================================
# 1. RÉCUPÉRATION CHUNKS DEPUIS FAISS  ← SEULE SECTION MODIFIÉE
# ============================================================================

def _search_single_query(index, metadata, sentence_transformer, query: str, k: int) -> list:
    """Recherche FAISS pour une query donnée. Retourne une liste de dicts."""
    embedded_q = get_embeddings(sentence_transformer, [query])
    embedded_q = np.array(embedded_q).astype('float32')

    distances, ids = index.search(embedded_q, k=k)

    results = []
    for idx, distance in zip(ids[0], distances[0]):
        if idx == -1:
            continue
        chunk_meta       = metadata[str(idx)]
        similarity_score = max(0, (2 - distance) / 2 * 100)
        results.append({
            'text':         chunk_meta.get('chunk', ''),
            'first_phrase': chunk_meta.get('first_phrase', 'N/A'),
            'source':       chunk_meta.get('source', 'Unknown'),
            'document':     chunk_meta.get('document', 'Unknown'),
            'filename':     chunk_meta.get('filename', 'Unknown'),
            'score':        similarity_score,
            '_score_raw':   float(-distance),   # pour deduplicate_chunks
        })
    return results


def retrieve_chunks(question, k=K_RESULTS):
    """
    Cherche k chunks similaires dans FAISS via bilingual query expansion.
    La question est cherchée en FR ET en EN simultanément.
    Les résultats sont fusionnés et dédupliqués avant retour.
    """
    try:
        index    = get_faiss_index()
        metadata = get_metadata()
    except FileNotFoundError:
        raise FileNotFoundError(
            "❌ Index FAISS ou métadonnées non trouvés!\n"
            "   Solution: Lancer 'python src/main.py'"
        )
    except Exception as e:
        raise RuntimeError(f"❌ Erreur lors du chargement: {e}")

    sentence_transformer = get_embedding_model()

    # ── Expansion bilingue 
    queries   = expand_query(question)
    chunks_fr = _search_single_query(index, metadata, sentence_transformer, queries["fr"], k)
    chunks_en = _search_single_query(index, metadata, sentence_transformer, queries["en"], k)

    print(f"   📦 Chunks FR : {len(chunks_fr)} | Chunks EN : {len(chunks_en)}")

    # ── Déduplication via Jaccard 
    tuples_fr = [(c['_score_raw'], c['text'], c) for c in chunks_fr]
    tuples_en = [(c['_score_raw'], c['text'], c) for c in chunks_en]

    merged = deduplicate_chunks(tuples_fr, tuples_en)   # → list of (score, text, meta_dict)
    merged = merged[:k]

    print(f"   ✅ Chunks après déduplication : {len(merged)}")

    # ── Reconversion au format attendu par build_context 
    return [meta for (_score, _text, meta) in merged]


# ============================================================================
# 2. COMPTAGE TOKENS
# ============================================================================

def count_tokens(text):
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        print(f"⚠️  Fallback token counting: {e}")
        return len(text) // 4


# ============================================================================
# 3. CONSTRUCTION CONTEXTE (sans compression)
# ============================================================================

def build_context(question, client=None):
    """Récupère k chunks et les assemble directement sans compression."""

    with open(CONTEXT_PATH, "r", encoding='utf-8') as f:
        context_template = f.read()

    chunks_with_meta = retrieve_chunks(question, k=K_RESULTS)

    chunks_text  = ""
    token_count  = 0
    chunks_final = []

    for chunk in chunks_with_meta:
        chunk_str = f"\n[Source: {chunk['filename']}]\n{chunk['text']}\n"
        tokens    = count_tokens(chunk_str)

        if token_count + tokens <= MAX_CONTEXT_TOKENS:
            chunks_text += chunk_str
            token_count += tokens
            chunks_final.append(chunk)
        else:
            break

    if not chunks_text:
        chunks_text = "Aucun extrait pertinent trouvé dans les documents."

    full_context = context_template.replace("{{Chuncks}}", chunks_text)
    return full_context, chunks_final


# ============================================================================
# 4. RÉPONSE GROQ + TOKEN TRACKING
# ============================================================================

def answer_question(question):

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    context, chunks_final = build_context(question)

    context_tokens  = count_tokens(context)
    question_tokens = count_tokens(question)

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": context},
            {"role": "user",   "content": question}
        ],
        model=LLM_MODEL
    )

    response = chat_completion.choices[0].message.content
    usage    = chat_completion.usage

    print("\n" + "="*50)
    print("📊 TOKEN STATISTICS")
    print("="*50)
    print(f"Context:     {context_tokens:4d} tokens")
    print(f"Question:    {question_tokens:4d} tokens")
    print(f"Total input: {usage.prompt_tokens:4d} tokens (Groq)")
    print(f"Response:    {usage.completion_tokens:4d} tokens (Groq)")
    print(f"Total:       {usage.total_tokens:4d} tokens")
    print(f"Chunks:      {len(chunks_final)}/{K_RESULTS}")
    print("="*50)

    print("\n📚 SOURCES UTILISÉES:")
    print("-"*70)
    if chunks_final:
        for i, chunk in enumerate(chunks_final, 1):
            print(f"\n[{i}] {chunk['filename']} (Pertinence: {chunk['score']:.1f}%)")
            print(f"    📝 {chunk['first_phrase']}")
    else:
        print("⚠️  Aucun chunk retenu.")
    print("\n" + "="*70 + "\n")

    return response, chunks_final


# ============================================================================
# 5. CLI INTERACTIVE
# ============================================================================

if __name__ == "__main__":
    print("\n💬 Assistant Réglementation Financière")
    print("Type 'exit' pour quitter\n")

    while True:
        question = input("❓ Question: ").strip()
        if question.lower() == 'exit':
            break

        try:
            response = answer_question(question)
            print(f"\n📝 Réponse:\n{response}\n")
        except Exception as e:
            print(f"❌ Erreur: {e}\n")