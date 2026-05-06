# ============================================================================
# 0. IMPORT & CONFIGURATION
# ============================================================================

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from utils.embedding import get_embeddings
from config import INDEX_PATH, METADATA_PATH, DOCUMENTS_DIR


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


def extract_first_phrase(text, max_length=150):
	"""Extrait la première phrase complète sans couper les mots"""
	
	text = text.strip()
	
	# Cherche la première phrase complète (finissant par . ! ? ou ;)
	for delimiter in ['. ', '! ', '? ', '; ']:
		if delimiter in text:
			first_phrase = text.split(delimiter)[0] + delimiter.strip()
			# Si ça fait moins de 200 caractères, prendre la phrase
			if len(first_phrase) < 200:
				return first_phrase.strip()
	
	# Sinon, prendre les max_length premiers caractères mais sans couper un mot
	if len(text) <= max_length:
		return text
	
	# Coupe à max_length sans casser de mot
	truncated = text[:max_length]
	# Cherche le dernier espace avant max_length
	last_space = truncated.rfind(' ')
	if last_space > 50:  # Au moins 50 caractères avant le dernier mot
		return truncated[:last_space] + "..."
	else:
		return truncated + "..."


def load_and_chunk_documents():
	"""Charge et chunke tous les documents du dossier DOCUMENTS_DIR"""
	
	all_chunks = []
	all_metadata = []
	
	for doc_file in sorted(DOCUMENTS_DIR.glob("*.txt")):
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
    faiss.write_index(index, str(INDEX_PATH))
    
    # Sauvegarde métadonnées + chunks complets + première phrase
    metadata_dict = {}
    for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
        # Extrait la première phrase intelligemment (sans couper les mots)
        first_phrase = extract_first_phrase(chunk, max_length=150)
        
        metadata_dict[i] = {
            **meta,  # source, document, filename
            'chunk': chunk,  # ✅ Complet (pour Groq via build_context)
            'first_phrase': first_phrase  # Résumé (pour l'affichage terminal)
        }
    
    with open(METADATA_PATH, "w", encoding='utf-8') as f:
        json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
    
    print("✅")
    print(f"\n✨ Index créé: index.faiss + metadata.json\n")
    
    return True


if __name__ == "__main__":
    build_index()
