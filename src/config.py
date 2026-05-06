# ============================================================================
# CONFIGURATION CENTRALISÉE - Chemins absolus pour tout le projet
# ============================================================================

from pathlib import Path

# Répertoire racine du projet (src/ → parent = racine)
BASE_DIR = Path(__file__).parent.parent

# Répertoires
DOCUMENTS_DIR = BASE_DIR / "documents"
PDF_DIR = BASE_DIR / "DOCS" / "Sources"  

# Fichiers de données
INDEX_PATH = BASE_DIR / "index.faiss"
METADATA_PATH = BASE_DIR / "metadata.json"
CONTEXT_PATH = BASE_DIR / "context.txt"

# Modèles
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
# "distiluse-base-multilingual-cased-v2"

LLM_MODEL = "llama-3.3-70b-versatile"

# Modalité Chunks
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 50
K_RESULTS = 3

# Optimisation tokens
MAX_CONTEXT_TOKENS = 3000  # Réduit de 3000 pour économiser tokens

# Crée les dossiers s'ils n'existent pas
DOCUMENTS_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)

