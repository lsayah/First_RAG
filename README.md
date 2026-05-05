# 📚 RAG Réglementation Financière

Système RAG from scratch (sans LangChain/LlamaIndex) pour répondre sur la régulation financière : MiFID II, Bâle III, AMF.

---

## ⚡ Quick Start

```bash
# 1. Installer dépendances
pip install -r requirements.txt

# 2. Créer .env
echo "GROQ_API_KEY=gsk_..." > .env

# 3. Lancer
python src/main.py
```

**Clé Groq :** https://console.groq.com/

---

## 🏗️ Architecture

```
Documents (3 fichiers)
      ↓
indexation.py → 25 chunks + embeddings
      ↓
index.faiss + metadata.json (persiste)
      ↓
rag.py → FAISS search + Groq + citations
      ↓
Réponse avec [Source: document]
```

---

## 📂 Structure

```
src/
├─ config.py           (Centralized paths)
├─ main.py             (Orchestrateur)
├─ rag.py              (RAG + Groq + chat)
├─ indexation.py       (FAISS creation)
├─ get_sources.py      (Scrapers - static)
└─ utils/
   └─ embedding.py     (Embeddings batch)

documents/             (3 fichiers .txt)
├─ mifid_ii.txt
├─ basel_iii.txt
└─ amf_guide.txt

index.faiss            (Generated)
metadata.json          (Generated)
context.txt            (System prompt)
```

---

## ✅ Checklist TP

- [x] From scratch (pas LangChain)
- [x] FAISS IndexFlatL2 + persistent
- [x] sentence-transformers (384 dims)
- [x] Groq llama-3.3-70b
- [x] Citations [Source: X]
- [x] Refusal (out of context)
- [x] Token limiting (1500 max)
- [x] Token statistics (display)
- [x] Metadata tracking
- [x] Error handling

---

## 📊 Features

| Feature             | Détail                                        |
| ------------------- | --------------------------------------------- |
| **Token Counting**  | Tiktoken precise (cl100k_base)                |
| **Token Display**   | Stats per query (context, question, response) |
| **Chunk Limit**     | 1500 tokens max before Groq                   |
| **Top-K Retrieval** | 3 chunks by L2 distance                       |
| **Idempotence**     | Skip fetch/index if exists                    |

---

## 🧪 Usage

```bash
# Lancer
python src/main.py

# Exemple interaction
❓ Question: Qu'est-ce que MiFID II ?
📊 TOKEN STATISTICS
Context:   339 tokens
Question:   12 tokens
Response:  232 tokens (Groq)
━━━━━━━━━━━━━━━━━━
Total:     618 tokens

📝 Réponse: MiFID II (2014/65/UE) est une directive...
           [Source: mifid_ii.txt]
⚠️ Disclaimer appliqué

❓ Question: exit
```

---

## 🔧 Configuration

Tous les chemins centralisés dans `src/config.py` :

- `INDEX_PATH`: Chemin index.faiss
- `METADATA_PATH`: Chemin metadata.json
- `DOCUMENTS_DIR`: Dossier documents/
- `CONTEXT_PATH`: context.txt (system prompt)

---

## 🛠️ Troubleshooting

| Erreur                   | Solution                                           |
| ------------------------ | -------------------------------------------------- |
| `Module not found`       | `pip install -r requirements.txt`                  |
| `GROQ_API_KEY error`     | Créer `.env` avec clé valide                       |
| `index.faiss not found`  | Supprime `index.faiss` et `metadata.json`, relance |
| `Slow first run (3 min)` | Télécharge modèle embedding (normal)               |

---

## 📋 Tech Stack

- **Chunking :** 500 mots, overlap 50
- **Embeddings :** distiluse-base-multilingual (384 dims)
- **Index :** FAISS IndexFlatL2
- **LLM :** Groq llama-3.3-70b-versatile
- **Tokenizer :** tiktoken cl100k_base

---

**Status :** ✅ Working (Mai 2026)
