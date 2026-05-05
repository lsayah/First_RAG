# ============================================================================
# 0. IMPORT
# ============================================================================

import json
import os
import numpy as np
import faiss
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from utils.embedding import get_embeddings

load_dotenv()


# ============================================================================
# 1. RÉCUPÉRATION CHUNKS DEPUIS FAISS
# ============================================================================

def retrieve_chunks(question, k=3):
	"""Cherche k chunks similaires dans FAISS"""
	
	# Charge index FAISS + metadata
	index = faiss.read_index("index.faiss")
	with open("metadata.json", "r", encoding='utf-8') as f:
		metadata = json.load(f)
	
	# Embed question
	sentence_transformer = SentenceTransformer("distiluse-base-multilingual-cased-v2")
	embedded_q = get_embeddings(sentence_transformer, [question])
	embedded_q = np.array(embedded_q).astype('float32')
	
	# Cherche dans FAISS
	distances, ids = index.search(embedded_q, k=k)
	ids = ids[0]  # Récupère les indices
	
	# Récupère chunks + métadonnées
	chunks_with_meta = []
	for idx in ids:
		chunk_meta = metadata[str(idx)]
		chunks_with_meta.append({
			'text': chunk_meta.get('chunk', ''),
			'source': chunk_meta.get('source', 'Unknown'),
			'document': chunk_meta.get('document', 'Unknown'),
			'filename': chunk_meta.get('filename', 'Unknown')
		})
	
	return chunks_with_meta


# ============================================================================
# 2. LIMITATION TOKENS + CONSTRUCTION CONTEXT
# ============================================================================

def count_tokens(text):
	"""Estimation tokens (1 token ≈ 4 caractères)"""
	return len(text) // 4


def build_context(question):
	"""Construit context.txt avec chunks + sources (limité en tokens)"""
	
	# Charge template context
	with open("context.txt", "r", encoding='utf-8') as f:
		context_template = f.read()
	
	# Récupère chunks
	chunks_with_meta = retrieve_chunks(question, k=3)
	
	# Construit string chunks avec sources
	chunks_text = ""
	token_count = 0
	max_tokens = 1500
	
	for chunk in chunks_with_meta:
		chunk_str = f"""
[{chunk['source']} - {chunk['document']}]
{chunk['text']}
[Source: {chunk['filename']}]
"""
		tokens = count_tokens(chunk_str)
		
		# Ajoute chunk si token limit pas atteint
		if token_count + tokens <= max_tokens:
			chunks_text += chunk_str
			token_count += tokens
		else:
			break
	
	# Remplace placeholder
	full_context = context_template.replace("{{Chuncks}}", chunks_text)
	
	return full_context


# ============================================================================
# 3. RÉPONSE GROQ
# ============================================================================

def answer_question(question):
	
	client = Groq(api_key=os.environ["GROQ_API_KEY"])
	
	# Construit context avec chunks
	context = build_context(question)
	
	# Appel Groq
	chat_completion = client.chat.completions.create(
		messages=[
			{"role": "system", "content": context},
			{"role": "user", "content": question}
		],
		model="llama-3.3-70b-versatile"
	)
	
	return chat_completion.choices[0].message.content


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
