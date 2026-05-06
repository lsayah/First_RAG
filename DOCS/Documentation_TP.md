
# ATTENTION 

- Ne pas Utiliser LangChain/LlamaIndex 
- Ne pas Commiter la clé API 
- Persister l'index FAISS (le recréer à chaque lancement) ❌
- modèle d'embedding à l'indexation et à la recherche doit être exactement le même
- deux fichiers : l'index FAISS + un JSON des métadonnées, dans le même ordre absolu

___________________________________________________________________________________________________________________

# ADAPTER LE TP (FINANCE)

Phrase qui explique pourquoi et comment j'ai adapter le TP (Création d'un 4éme sujet) (logique avec mon parcours, mon projet professionnel et mes projets perso)

**Consigne originale**                 **Adaptation finance**

Citer le numéro d'article             Citer l'article + le règlement (ex: Art. 25 MiFID II)Avertissement médical/juridique      "Ceci n'est pas un conseil en investissement" 
Refuser si hors corpus                Refuser si hors réglementation indexée
Chunking par section                  Chunking par article réglementaire

___________________________________________________________________________________________________________________

# ETAPE

## A Faire 

**indexation.py** (à créer)
├─ Charger documents Finance (MiFID II, Bâle III, AMF)
├─ Chunking (taille + overlap optimisés)
├─ Embeddings via sentence-transformers
└─ Sauvegarde index FAISS + métadonnées JSON

**rag.p**y** (à refactoriser)
├─ Charger index FAISS persisté
├─ Récupérer k documents + métadonnées
├─ Limiter tokens avant appel Groq
├─ Générer réponse avec sources
└─ CLI interactif


- Test Unitaires
- Documentation Complète
- Test de performance
    --> Écrire 5-10 tests d'intégration

- Ajouter timeout Groq ? Pourquoi ??
    --> éviter blocage si API lente ou indisponible
    --> améliorer robustesse système
    --> éviter freeze CLI
    --> ex : timeout = 10s
    --> FLEMME



## Ce que j'ai fait

- Modification du Context (system prompt)
    --> Destiné a groq 
    --> {{Chuncks}} = les 3 chuncks les plus pertinent en context
    --> Context = chunks pertinent + Instructions
    --> Chunks préalablement choisie via BDD vectorielle

- Modification de requirements.txt
    --> Ajout : dotenv / groq / faiss-cpu /
- Creation de __indexation.py__
    -->
    -->
    -->
- Création de get_sources.py --> va scrapper 3 sources financière (règlementaire) serieuse 
    -->
    --> Tentative scraping sites officiels (AMF / BIS / ESMA)
    --> Gestion erreurs HTTP / parsing HTML
    --> Fallback manuel (PDF téléchargés)
    --> Sauvegarde dans /documents

- Création de __main.py__ --> Ochestrateur de A à Z (attention certiaine tache que 1 seul fois (Idempotence)) 
    --> Vérifie si documents présents (sinon fetch)
    --> Vérifie si index.faiss existe (sinon lance indexation)
    --> Lance mode chat (rag.py)
    --> Gestion idempotence (ne relance pas index inutilement)

- Modification __vector_db__ 
    --> Renommer __Emnbedding.py__ et deplacement dans src/utils/
    --> suppression wrapper fonction retrieve()
    --> suppression lien avec chroma (remplacement par FAISS)
- Modification de rag.py
    --> 
    --> 
    -->
- Création de dossier src/ et src/utils/ 
- Comptage des tokens 
    --> tiktoken
    --> Ajouter tiktoken pour token counting réel
    --> token_count = len(text) // 4 => trop approximatif
    --> ajouter un compteur de token qui nous dit combien de token excatement on etait utilisé 

- Gestion des erreurs
    --> Ajouter vérification fichier index dans rag.py
    --> index = faiss.read_index("index.faiss") => PROBLEME => index.faiss not found
    --> modification de retrieve_chunks() dans __rag.py__
        --> transforme question en embedding
        --> recherche similarité FAISS
        --> récupère index des chunks
        --> map avec metadata.json
        --> retourne top_k chunks pertinents

- Problème de chemin 
    --> Remplacer chemins codés par Path()
    --> Chemin absolue

- Suppresion get_sources.py
    --> Scrapping ne fonctionne pas

- Sauvegarde les premiers mots chunks dans metadata 
    --> Identifier les chunks choisie


## Difficultés 

- Scrapping impossible (récuperation des sources)
    --> Solution => telecharger pdf manuellement

- Adaptation du projet 

- Choix modalités Chunks (pertinence et Performances)

- Indexation avec FAISS

- Gestion des path 
    --> Fichier Config.py

- Multi source
    - Exctraction PDF 
        --> attention a pas limité l'exctraction ( de 152 chunks à 1010 chunks)
    - Multilangue (anglais et francais)

- Limitation utilisation Tokens
    --> 

- Taille metadata.json --> 15 20mb

___________________________________________________________________________________________________________________

# LE CODE

## Architecture 

|- **main.py** => Ochestres → Execute tout
|- **indexation.py** => Prépare les données(lis/chunks/ embed avec FAISS) → FAISS (Une seul fois au départ)
|- **rag.py** => Repond Question (embed question / compare vecteur question avec chunks FAISS / n chunks ajouter au context / Envoie context a groq)
|- **context.txt** => charger par rag.py pour envoyé a groq
|- **index.faiss** => Index persisté (disque) --> Créé par indexation.py
|- **metadata.json** => Sources (article, URL, doc) --> Créé par indexation.py

--------------------------------------------------------------------------------------------------------------

## Articulation du Code 

**Etape 1**: Collecter les dources (get_sources.py)
├─ Télécharge 3 documents: MiFID II, Bâle III, AMF
└─ sauvegarde dans documents/

**Etape 2**: Prépare la bibliothèque (BDD vectorielle via chunks) (indexation.py)
├─ Découpe chaque sources en paragraphes (chunks)
├─ Transforme chaque paragraphe en code numérique/vecteur (embeddings)
├─ Range tous les codes dans un classeur (FAISS index)
└─ Sauvegarde le classeur pour le réutiliser (Idempotence)

**Etape 3**: Utilisateur pose question (rag.py)
├─ Transforme la question en code numérique/vecteur (embeddings)
├─ Cherche dans le classeur les paragraphes les plus similaires
├─ Envoie ces paragraphes + la question à un assistant IA (Groq)
└─ L'IA répond en citant les sources

--------------------------------------------------------------------------------------------------------------

## Details du Code

**main.py** 
    - fetch_sources 
        --> Récupère les sources dans documents / si elle existe pas lance les 3 fetch de __get_source.py__
        --> 

    - indexation
        --> 

    - chat
        --> 

    - if __name__ == "__main__":
        --> 

-------------------------
**rag.py**
    - retrieve_chunks() 
        --> 

    - count_tokens():
        --> Risque: Si un 4ème chunk très pertinent existe, on le perd.
        --> utilise tiktoken encoding
        --> compte réel tokens (prompt + context)
        --> coupe si dépassement
        --> limite coût + évite erreur API

    - buil_context() 
        --> concatène chunks sélectionnés
        --> ajoute instruction :
            - citer article + règlement
            - refuser hors corpus
            - disclaimer finance
        --> structure lisible pour LLM

    - answer_question() 
        --> 

    - if __name__ == "__main__"
        --> 

-------------------------
**indexation.py**
    - extract_metadata() 
        --> Parse [SOURCE:], [URL:], [DOCUMENT:] en début de fichier

    - chunk_text()
        --> Split texte en chunks (500 mots, 50 mots overlap) 

    - load_and_chunk_documents()	
        --> Lit documents/, chunke, extrait metadata

    - build_index()	
        --> Orchestre tout : load → embed → FAISS → save

-------------------------
**vector_db.py**

    - get_embeddings
        --> 

pas besoin du wrapper retrieve

--------------------------------------------------------------------------------------------------------------

## Test Unitaire 


**Question Test**

- Questions factuelles précises (test anti-hallucination)
    1. Quelle est la date d'entrée en vigueur de l'output floor à 72,5% selon le document d424 ?
    ✅ Réponse attendue : 1er janvier 2027 — toute autre date = hallucination

    2. Combien de principes de gouvernance le document d328 contient-il, et quel principe traite de l'audit interne ?
    ✅ Réponse attendue : 13 principes, Principe 10 — teste la précision numérique

    3. Quel est le numéro ISBN du document "Basel III: Finalising post-crisis reforms" ?    
    ✅ Réponse attendue : 978-92-9259-022-2 — détail très précis, facile à inventer


- Questions pièges (test de refus d'inventer)
    4. Quel est le ratio CET1 minimum imposé par le document d424 ?
    ✅ Réponse attendue : le RAG doit dire que ce ratio n'est pas dans ce document — c'était Bâle 3 initial (2010). Un mauvais RAG inventera "4,5%"

    5. Le document d328 impose-t-il des sanctions aux banques qui ne respectent pas les principes de gouvernance ?
    ✅ Réponse attendue : Non, ce sont des lignes directrices, pas des normes contraignantes — teste la compréhension du statut du document


- Questions de liens entre documents (test de raisonnement)
    6. Quelle est la différence entre le Pilier 1 et le Pilier 2 de Bâle, et dans lequel classe-t-on le document d328 ?
    ✅ Teste la capacité à relier deux documents et à raisonner sur leur nature

    7. Le document d424 mentionne-t-il la gouvernance interne des banques ? Si non, quel document du corpus couvre ce sujet ?
    ✅ Teste la complémentarité des deux documents


- Questions temporelles (test de cohérence chronologique)
    8. Lequel des deux documents a été publié en premier, et quel événement réglementaire justifie la publication du second ?
    ✅ Réponse : d328 (2015) avant d424 (2017) — la crise financière de 2008 justifie les deux

    9. À quelle date les révisions du cadre IRB (Internal Ratings-Based) devaient-elles être implémentées selon d424 ?
    ✅ Réponse attendue : 1er janvier 2022 — teste la précision sur les dates de transition


- Question hors corpus (test de périmètre)
    10. Quelles sont les exigences du ratio LCR (Liquidity Coverage Ratio) selon vos documents ?
    ✅ Réponse attendue : le RAG doit reconnaître que le LCR n'est pas couvert dans ces deux documents — un mauvais RAG inventera une réponse à partir de sa mémoire générale sur Bâle 3


--------------------------------------------------------------------------------------------------------------

## Optimisation

__Optimisation__	          __Gain__	          __Effort__
Cache index mémoire	        -1.5s/query	        ⭐ Low
tiktoken real counting	    +10% accuracy	    ⭐ Low
Re-ranking k=5→10	        +20% quality	    ⭐⭐ Medium
Better embedding model	    +30% quality	    ⭐⭐ Medium
Semantic chunking	        +25% relevance	    ⭐⭐⭐ High

	                            Priorité	            Effort	            Impact
Vraies sources (PDFs)	        🔴 HAUTE	            Bas         	Légit TP ✓
Streaming Groq	                🟠 Moyenne	            Moyen	        UX améliorée
Cache embeddings	            🟠 Moyenne          	Moyen	        +50% rapidité
Chunk size tuning	            🟡 Basse	            Bas	            Meilleure précision
Web UI	                        🟡 Basse	            Haut	        Nice-to-have

___________________________________________________________________________________________________________________

# GESTION DES PARAMETRES


## Modalité des chunks

- hyperparamtre de base => chunk_size=500, overlap=50


Test avec 300 --> limité utilisation tokens --> 1817 chunks créés => pas bon 
Test avec 400 --> limité utilisation tokens --> 1279 chunks créés => pas bon 


# Choix Embeddings

distiluse-base-multilingual-cased-v2
--> source multilangue (sources => anglais et francais)


--------------------------------------------------------------------------------------------------------------
## Choix du modèles

Modèle	           Speed	     Qualité	    Coût	    Use Case
llama-3.3-70b	⭐⭐⭐⭐	⭐⭐⭐⭐⭐	Gratuit	←   TON CAS (RAG)
llama-3.1-8b	⭐⭐⭐⭐⭐  ⭐⭐⭐	     Gratuit	  Tâches simples
mixtral-8x7b	⭐⭐⭐	      ⭐⭐⭐⭐	   Gratuit	    Équilibre
gemma-7b	    ⭐⭐⭐⭐⭐  ⭐⭐	       Gratuit	    Très rapide



**les impératif pour le modèles**
- Comprendre le francais
- Gratuit
- Accés Reglementaire/Finance




# AUTRES

explique moi cela 

problème que le prof nous a expliquer et montrer mais que j'ai pas compris
porter l'info de quelle matrice d'embdding utilisé 
--> meme matrice embedding pour chunks et questions. 



Attention 
- code doit absolument etre idempotent 

- sauvegarder nom matrice d'embedding dans bdd vectorielle 
--> pour que prof puisse l'exploiter sans nous.
--> sauvegarder le modèle dans metadata.json