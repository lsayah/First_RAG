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
from config import INDEX_PATH, METADATA_PATH, CONTEXT_PATH, EMBEDDING_MODEL, LLM_MODEL, MAX_CONTEXT_TOKENS

load_dotenv()

# ============================================================================
# CACHES GLOBAUX (pour éviter rechargements)
# ============================================================================
_embedding_model = None
_metadata_cache = None


# ============================================================================
# FONCTIONS DE CACHE
# ============================================================================

def get_embedding_model():
	"""Retourne le modèle embedding en cache (évite rechargement)"""
	global _embedding_model
	if _embedding_model is None:
		_embedding_model = SentenceTransformer(EMBEDDING_MODEL)
	return _embedding_model


def get_metadata():
	"""Retourne metadata.json en cache (évite I/O disk répété)"""
	global _metadata_cache
	if _metadata_cache is None:
		with open(METADATA_PATH, "r", encoding='utf-8') as f:
			_metadata_cache = json.load(f)
	return _metadata_cache


# ============================================================================
# 1. RÉCUPÉRATION CHUNKS DEPUIS FAISS
# ============================================================================

def retrieve_chunks(question, k=3):
	"""Cherche k chunks similaires dans FAISS"""
	
	# Charge index + metadata (avec gestion d'erreur)
	try:
		index = faiss.read_index(str(INDEX_PATH))
		metadata = get_metadata()  
	except FileNotFoundError:
		raise FileNotFoundError(
			"❌ Index FAISS ou métadonnées non trouvés!\n"
			"   Solution: Lancer 'python src/main.py'"
		)
	except Exception as e:
		raise RuntimeError(f"❌ Erreur lors du chargement: {e}")
	
	# Embed question 
	sentence_transformer = get_embedding_model()
	embedded_q = get_embeddings(sentence_transformer, [question])
	embedded_q = np.array(embedded_q).astype('float32')
	
	# Cherche dans FAISS
	distances, ids = index.search(embedded_q, k=k)
	ids = ids[0]  # Récupère les indices
	distances = distances[0]  # Récupère les distances (scores)
	
	# Récupère chunks + métadonnées + scores
	chunks_with_meta = []
	for idx, distance in zip(ids, distances):
		chunk_meta = metadata[str(idx)]
		# Convertir distance en score de similarité (0-100, plus haut = mieux)
		similarity_score = max(0, 100 - (distance * 10))
		
		chunks_with_meta.append({
			'text': chunk_meta.get('chunk', ''),  # Chunk complet pour Groq
			'first_phrase': chunk_meta.get('first_phrase', 'N/A'),  # Pour l'affichage
			'source': chunk_meta.get('source', 'Unknown'),
			'document': chunk_meta.get('document', 'Unknown'),
			'filename': chunk_meta.get('filename', 'Unknown'),
			'score': similarity_score
		})
	
	return chunks_with_meta


# ============================================================================
# 2. LIMITATION TOKENS + CONSTRUCTION CONTEXT
# ============================================================================

def count_tokens(text):

	try:
		encoding = tiktoken.get_encoding("cl100k_base")  
		tokens = encoding.encode(text)
		return len(tokens)
	except Exception as e:
		# Si tiktoken échoue, utilise estimation 
		print(f"⚠️  Fallback token counting: {e}")
		return len(text) // 4


def build_context(question):
	"""Construit context.txt avec chunks + sources (limité en tokens)"""
	
	# Charge template context
	with open(CONTEXT_PATH, "r", encoding='utf-8') as f:
		context_template = f.read()
	
	# Récupère chunks
	chunks_with_meta = retrieve_chunks(question, k=3)
	
	# Construit string chunks avec sources
	chunks_text = ""
	token_count = 0
	
	for chunk in chunks_with_meta:
		chunk_str = f"""
[Source: {chunk['filename']} - Document: {chunk['document']}]
{chunk['text']}
"""
		tokens = count_tokens(chunk_str)
		
		# Ajoute chunk si token limit pas atteint
		if token_count + tokens <= MAX_CONTEXT_TOKENS:
			chunks_text += chunk_str
			token_count += tokens
		else:
			break
	
	# Remplace placeholder
	full_context = context_template.replace("{{Chuncks}}", chunks_text)
	
	return full_context


# ============================================================================
# 3. RÉPONSE GROQ + TOKEN TRACKING
# ============================================================================

def answer_question(question):
	
	client = Groq(api_key=os.environ["GROQ_API_KEY"])
	
	# Récupère chunks (pour afficher sources)
	chunks_raw = retrieve_chunks(question, k=3)
	
	# Construit context avec chunks
	context = build_context(question)
	
	# Compte tokens avant appel Groq
	context_tokens = count_tokens(context)
	question_tokens = count_tokens(question)
	total_input_tokens = context_tokens + question_tokens
	
	# Appel Groq (récupère aussi usage stats)
	chat_completion = client.chat.completions.create(
		messages=[
			{"role": "system", "content": context},
			{"role": "user", "content": question}
		],
		model=LLM_MODEL
	)
	
	response = chat_completion.choices[0].message.content
	
	# Récupère stats réels de Groq
	usage = chat_completion.usage
	
	# Affiche stats
	print("\n" + "="*50)
	print("📊 TOKEN STATISTICS")
	print("="*50)
	print(f"Context:         {context_tokens:4d} tokens")
	print(f"Question:        {question_tokens:4d} tokens")
	print(f"Total input:     {usage.prompt_tokens:4d} tokens (Groq)")
	print(f"Response:        {usage.completion_tokens:4d} tokens (Groq)")
	print(f"Total:           {usage.total_tokens:4d} tokens")
	print("="*50)
	
	# Affiche les sources utilisées avec détails
	print("\n📚 SOURCES UTILISÉES:")
	print("-"*70)
	for i, chunk in enumerate(chunks_raw, 1):
		# Affiche le nom du fichier + score
		score = chunk.get('score', 0)
		print(f"\n[{i}] {chunk['filename']} (Pertinence: {score:.1f}%)")
		
		# Affiche la première phrase (depuis metadata)
		first_phrase = chunk.get('first_phrase', 'N/A')
		print(f"    📝 {first_phrase}")
	print("\n" + "="*70 + "\n")
	
	return response


# ============================================================================
# 4. CLI INTERACTIVE
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
