# ============================================================================
# 0. IMPORT & CONFIGURATION
# ============================================================================

import sys
from pathlib import Path

src_dir  = Path(__file__).parent
root_dir = src_dir.parent

sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(root_dir))

from config import DOCUMENTS_DIR, INDEX_PATH, PDF_DIR
from utils.extract_pdfs import extract_all_pdfs


# ============================================================================
# 1. Vérification PDF extraits
# ============================================================================

def check_documents():
    if DOCUMENTS_DIR.exists() and list(DOCUMENTS_DIR.glob("*.txt")):
        print("\n✅ Documents déjà extraits, skip extraction\n")
        return True
    print("\n📄 Documents manquants, extraction en cours...\n")
    return extract_all_pdfs()


# ============================================================================
# 2. Indexation (idempotent)
# ============================================================================

def indexation():
    if INDEX_PATH.exists():
        print("\n✅ Index FAISS déjà présent, skip indexation\n")
        return True

    print("\n📊 Création index FAISS...\n")

    try:
        from indexation import build_index
        return build_index()
    except ImportError:
        print("❌ indexation.py non trouvé")
        return False


# ============================================================================
# 3. RAG Chat — Bilingual Query Expansion
# ============================================================================

def chat():
    print("\n💬 Assistant Réglementation Financière\n")
    print("   Les documents restent dans leur langue d'origine.")
    print("   Votre question est automatiquement cherchée en FR + EN.\n")
    print("Type 'exit' pour quitter\n")

    try:
        from rag import answer_question
        while True:
            question = input("❓ Question: ").strip()
            if not question:
                continue
            if question.lower() == 'exit':
                break
            response = answer_question(question)
            print(f"\n📝 Réponse:\n{response}\n")
    except ImportError as e:
        print(f"❌ rag.py non trouvé : {e}")


# ============================================================================
# Main Exécution
# ============================================================================

if __name__ == "__main__":
    if not check_documents():
        print("❌ Erreur extraction PDFs")
        exit(1)

    # ✅ Plus de translate_documents() ici.
    # Les documents restent dans leur langue originale (FR ou EN).
    # La traduction se fait à la volée sur la question dans rag.py
    # via utils/query_expander.py → expand_query()

    if not indexation():
        print("❌ Erreur indexation")
        exit(1)

    chat()