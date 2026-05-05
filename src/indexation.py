# ============================================================================
# 0. IMPORT
# ============================================================================

import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from utils.embedding import get_embeddings


# ============================================================================
# 1. Extraction Métadonnées & Chunking
# ============================================================================

def extract_metadata(text):

    lines = text.split('\n')
    metadata = {}
    
    for line in lines[:10]:
        if line.startswith('[SOURCE:'):
            metadata['source'] = line.replace('[SOURCE:', '').replace(']', '').strip()
        elif line.startswith('[URL:'):
            metadata['url'] = line.replace('[URL:', '').replace(']', '').strip()
        elif line.startswith('[DOCUMENT:'):
            metadata['document'] = line.replace('[DOCUMENT:', '').replace(']', '').strip()
    
    return metadata


def chunk_text(text, chunk_size=500, overlap=50):

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    
    for para in paragraphs:
        words = para.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
    
    return chunks


def load_and_chunk_documents(docs_dir="documents"):

    all_chunks = []
    all_metadata = []
    
    docs_path = Path(docs_dir)
    for doc_file in sorted(docs_path.glob("*.txt")):
        print(f"📄 {doc_file.name}...", end=" ")
        
        text = doc_file.read_text(encoding='utf-8')
        metadata = extract_metadata(text)
        metadata['filename'] = doc_file.name
        
        chunks = chunk_text(text)
        print(f"{len(chunks)} chunks")
        
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append(metadata)
    
    return all_chunks, all_metadata


# ============================================================================
# 2. Création Index FAISS
# ============================================================================

def build_index():

    print("\n📊 INDEXATION FAISS\n")
    
    chunks, metadata = load_and_chunk_documents()
    
    if not chunks:
        print("❌ Aucun chunk trouvé!")
        return False
    
    print(f"\n✅ {len(chunks)} chunks créés")
    print("🔄 Embeddings...", end=" ")
    
    sentence_transformer = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    embeddings = get_embeddings(sentence_transformer, chunks)
    embeddings = np.array(embeddings).astype('float32')
    
    print(f"✅ ({embeddings.shape})")
    
    print("📦 Création index FAISS...", end=" ")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print("✅")
    
    print("💾 Sauvegarde...", end=" ")
    faiss.write_index(index, "index.faiss")
    
    metadata_dict = {i: meta for i, meta in enumerate(metadata)}
    with open("metadata.json", "w", encoding='utf-8') as f:
        json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
    
    print("✅")
    print(f"\n✨ Index créé: index.faiss + metadata.json\n")
    
    return True


if __name__ == "__main__":
    build_index()
