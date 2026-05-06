# ============================================================================
# 0. IMPORT & CONFIGURATION
# ============================================================================

import sys
from pathlib import Path

# Fix imports src/ 
sys.path.insert(0, str(Path(__file__).parent))

from config import DOCUMENTS_DIR, INDEX_PATH, PDF_DIR

# Import la fonction d'extraction
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.extract_pdfs import extract_all_pdfs


# ============================================================================
# 1. Vérification PDF extraits
# ============================================================================

def check_documents():
    """Vérifie si les documents sont déjà extraits, sinon lance l'extraction"""
    
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
# 3. RAG Chat (N fois)
# ============================================================================

def chat():

    print("\n💬 Assistant Réglementation Financière\n")
    print("Type 'exit' pour quitter\n")
    
    try:
        from rag import answer_question
        while True:
            question = input("❓ Question: ").strip()
            if question.lower() == 'exit':
                break
            response = answer_question(question)
            print(f"\n📝 Réponse:\n{response}\n")
    except ImportError:
        print("❌ rag.py non trouvé")


# ============================================================================
# Main Exécution
# ============================================================================

if __name__ == "__main__":
    if not check_documents():
        print("❌ Erreur extraction PDFs")
        exit(1)
    
    if not indexation():
        print("❌ Erreur indexation")
        exit(1)
    
    chat()
