# ============================================================================
# 0. IMPORT & CONFIGURATION
# ============================================================================

import sys
from pathlib import Path

# Fix imports src/ - MUST BE BEFORE OTHER IMPORTS!
sys.path.insert(0, str(Path(__file__).parent))

from get_sources import fetch_mifid_ii, fetch_basel_iii, fetch_amf_guide
from config import DOCUMENTS_DIR, INDEX_PATH


# ============================================================================
# 1. Telechargement Sources (idempotent)
# ============================================================================

def fetch_sources():

    if DOCUMENTS_DIR.exists() and list(DOCUMENTS_DIR.glob("*.txt")):
        print("\n✅ Documents déjà présents, skip fetch\n")
        return True
    
    
    results = {
        "MiFID II": fetch_mifid_ii(),
        "Bâle III": fetch_basel_iii(),
        "AMF Guide": fetch_amf_guide()
    }
    
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {name}")
    
    return all(results.values())

print("\n🔍 Récupération sources financières OK\n")

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
    if not fetch_sources():
        print("❌ Erreur fetch")
        exit(1)
    
    if not indexation():
        print("❌ Erreur indexation")
        exit(1)
    
    chat()
