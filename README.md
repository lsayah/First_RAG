# RAG Réglementation Financière - TP Noté

## 📋 Vue d'Ensemble

Système **RAG (Retrieval-Augmented Generation)** construit from scratch en Python, sans LangChain ni LlamaIndex.

**Sujet :** Assistant spécialisé en réglementation financière basé sur MiFID II, Bâle III, et guides AMF.

### Outils Utilisés
- **LLM :** Groq (llama-3.3-70b-versatile)
- **Embeddings :** sentence-transformers (distiluse-base-multilingual-cased-v2)
- **Index Vectoriel :** FAISS (IndexFlatL2)
- **Data Source :** EUR-Lex, Banque de France, AMF (scraping)

---

## 🏗️ Architecture

```
documents/                          (Données brutes)
├─ mifid_ii.txt
├─ basel_iii.txt
└─ amf_guide.txt
          ↓
    indexation.py (Une fois)
          ↓
    index.faiss + metadata.json (Persisté)
          ↓
    rag.py (N fois - interactive)
          ↓
    Réponse avec [Source: Article X]
```

---

## 📦 Installation

```bash
# 1. Installer dépendances
pip install -r requirements.txt

# 2. Créer .env avec clé Groq
echo "GROQ_API_KEY=gsk_..." > .env
```

**Obtenir clé Groq :** https://console.groq.com/

---

## 🚀 Utilisation

### Mode Automatique (Recommended)
```bash
python main.py
```

### Mode Manuel
```bash
# 1. Télécharger sources (EUR-Lex, Banque de France, AMF)
python get_sources.py

# 2. Créer index FAISS
python indexation.py

# 3. Lancer assistant
python rag.py
```

### Usage Assistant
```
💬 Assistant Réglementation Financière

❓ Question: Qu'est-ce que MiFID II Article 24 ?
📝 Réponse: MiFID II Article 24 (EUR-Lex) exige que les professionnels...
           [Source: EUR-Lex - Directive MiFID II]
           ⚠️ Ceci ne constitue pas un conseil en investissement.

❓ Question: exit
(Quitter)
```

---

## 📂 Structure des Fichiers

```
my_first_vector_db/
├─ get_sources.py         (Scrape sources financières)
├─ indexation.py          (Chunking + Embeddings + FAISS)
├─ vector_db.py           (Utilitaires embeddings)
├─ rag.py                 (Main RAG logic)
├─ main.py                (Orchestrateur)
├─ context.txt            (System prompt)
├─ requirements.txt       (Dépendances)
├─ README.md              (Cette doc)
│
├─ documents/             (Créé auto)
│  ├─ mifid_ii.txt
│  ├─ basel_iii.txt
│  └─ amf_guide.txt
│
├─ index.faiss            (Créé par indexation.py)
└─ metadata.json          (Créé par indexation.py)
```

---

## ✅ Exigences du TP - Checklist

- [x] **From Scratch** → Pas LangChain/LlamaIndex
- [x] **FAISS** → Index vectoriel persisté
- [x] **sentence-transformers** → Embeddings
- [x] **Groq** → LLM llama-3.3-70b-versatile
- [x] **indexation.py** → Chunking + Embeddings + Sauvegarde index
- [x] **rag.py** → Retrieval + Groq + CLI interactive
- [x] **Citations sources** → [Article X du Règlement Y, Source: URL]
- [x] **Refus hors corpus** → "Je ne dispose pas d'informations..."
- [x] **Pas d'invention** → Basé uniquement sur documents
- [x] **Disclaimer** → "⚠️ Ceci ne constitue pas un conseil..."
- [x] **Métadonnées** → source, article, document, filename
- [x] **Index persisté** → Ne se recrée pas à chaque run

---

## 🎯 Optimisations Implémentées

| Optimisation | Impact |
|---|---|
| **Token Limiter** | 1500 tokens max context → 3-5x moins cher |
| **Chunk Ranking** | FAISS retourne top-k par similarité |
| **Idempotence** | Fetch/Index skippés si existant |
| **Metadata Tracking** | Traçabilité complète sources |

---

## 📈 Performance

```
Tokens par requête:
- Context template:    ~500 tokens
- Chunks (k=3, limité): ~1500 tokens
- Question:            ~50 tokens
- Réponse Groq:        ~200 tokens
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total par requête:     ~2250 tokens ≈ $0.015 USD
```

---

## 🧪 Test Rapide

```bash
python main.py

# Test 1: Citation sources
❓ Quelles sont les exigences de Bâle III ?
✅ Réponse doit contenir [Source: Banque de France] et chiffres (4.5%, 6%, 8%)

# Test 2: Disclaimer
✅ Réponse doit finir par ⚠️ Ceci ne constitue pas un conseil...

# Test 3: Refus hors corpus
❓ Quel est le meilleur restaurant à Paris ?
✅ Réponse: Je ne dispose pas d'informations à ce sujet...

❓ exit
```

---

## 🔄 Pipeline Détaillé

### Phase 1: Indexation (Une fois)
```python
# load_and_chunk_documents()
documents → chunks (500 tokens, overlap 50)

# get_embeddings() (de vector_db.py)
chunks → embeddings (150 vecteurs × 384 dimensions)

# build_index()
embeddings → faiss.IndexFlatL2()
→ sauvegarde index.faiss + metadata.json
```

### Phase 2: RAG Chat (N fois)
```python
# retrieve_chunks()
question → embedding → FAISS.search(k=3)
→ top 3 chunks + metadata

# build_context()
chunks + sources → limit tokens (1500 max)
→ remplace {{Chuncks}} dans context.txt

# answer_question()
context + question → Groq LLM
→ réponse avec citations
```

---

## 🐛 Troubleshooting

| Erreur | Solution |
|--------|----------|
| `index.faiss not found` | Lancer `python indexation.py` |
| `GROQ_API_KEY not found` | Créer `.env` avec clé |
| `Slow embeddings (1ère fois)` | Attendre (~3 min, télécharge modèle) |
| `Module not found` | `pip install -r requirements.txt` |

---

## 📝 Spécifications Techniques

**Chunking :** 500 mots, overlap 50
**Embeddings :** distiluse-base-multilingual-cased-v2 (384 dim)
**FAISS Index :** IndexFlatL2 (L2 distance)
**Token Limit :** 1500 max per query
**LLM :** Groq llama-3.3-70b-versatile
**Python :** 3.8+

---

**Status:** ✅ Production Ready (Mai 2026)




# my_first_vector_db
Pour ceux comme moi qui n'ont pas de GPU Nvidia voici comment installer les dépendances:

pip install -r requirements.txt --break-system-packages