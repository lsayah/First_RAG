# 📚 RAG Réglementation Financière

Système RAG from scratch (sans LangChain/LlamaIndex) pour répondre sur la régulation financière : MiFID II, Bâle III, AMF.


## Adaptation du Projet

Ce projet a été orienté vers un cas d'usage de finance réglementaire, avec l'idée de travailler à partir de plusieurs sources et dans plusieurs langues (francais et anglais).

Concrètement, le corpus rassemble plusieurs documents réglementaires au lieu de s'appuyer sur une seule base, ce qui permet d'avoir des réponses plus riches et plus fiables. L'indexation et la recherche utilisent aussi des modèles capables de traiter du français et de l'anglais, ce qui était important pour les documents du projet.

J'ai aussi gardé un fonctionnement RAG assez contrôlé : le système récupère uniquement les chunks les plus pertinents, limite le contexte envoyé au modèle et affiche les sources retrouvées. L'application Streamlit sert ensuite d'interface simple pour poser une question et consulter rapidement les passages utilisés.

L'objectif était surtout de rendre le prototype plus crédible pour un usage réglementaire et aussi me servir au niveau perso. 

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
Documents source (TXT / PDF)
      ↓
indexation.py → chunking (400 mots, overlap 50) + embeddings
      ↓
index.faiss + metadata.json (chunk, first_phrase, source)
      ↓
rag.py → FAISS search + Groq + citations + limite contexte
      ↓
Réponse avec sources affichées dans l'UI Streamlit
```

---

## 📂 Structure

```
src/
├─ config.py           (Chemins, modèles, limites)
├─ main.py             (Orchestrateur)
├─ rag.py              (RAG + Groq + chat)
├─ indexation.py       (Extraction + chunking + FAISS)
└─ utils/
      └─ embedding.py     (Embeddings batch)

DOCS/Sources/          (sources PDF / textes)
├─ Bâle 3 Final.txt
├─ Gouvernance BCBS 2015.txt
└─ autres sources du corpus

index.faiss            (Generated)
metadata.json          (Generated)
context.txt            (System prompt)
src/app.py             (Interface Streamlit)
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
| **Token Counting**  | Tiktoken précis (cl100k_base) avec fallback   |
| **Token Display**   | Stats par requête (contexte, question, réponse)|
| **Chunk Limit**     | 3000 tokens max avant Groq                    |
| **Top-K Retrieval** | 3 chunks by L2 distance                       |
| **Idempotence**     | Skip indexation si `index.faiss` existe       |
| **UI**              | Chat Streamlit avec sidebar, header, sources   |

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
- `PDF_DIR`: Dossier `DOCS/Sources/`
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

- **Chunking :** 400 mots, overlap 50
- **Embeddings :** paraphrase-multilingual-mpnet-base-v2
- **Index :** FAISS IndexFlatL2
- **LLM :** Groq llama-3.3-70b-versatile
- **Tokenizer :** tiktoken cl100k_base
- **Interface :** Streamlit

---

**Status :** ✅ Working (Mai 2026)
