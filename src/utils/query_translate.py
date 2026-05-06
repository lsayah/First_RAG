# ============================================================================
# query_expander.py — Bilingual Query Expansion (FR ↔ EN)
#
# Stratégie : on ne traduit PLUS les documents.
# On traduit uniquement la question au moment du retrieval.
# → Les termes techniques (CET1, BCBS, Basel III) restent intacts dans l'index.
# → Le recall couvre les deux corpus (FR + EN) en un seul passage.
#
# Usage interne : from utils.query_expander import expand_query
# ============================================================================

import sys
from pathlib import Path
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

sys.path.insert(0, str(Path(__file__).parent.parent))


def _translate(text: str, source: str, target: str) -> str:
    """Traduit un texte court (question) via Google Translate."""
    try:
        return GoogleTranslator(source=source, target=target).translate(text)
    except Exception as e:
        print(f"⚠️  Traduction {source}→{target} échouée : {e}")
        return text  # fallback : retourne l'original


def detect_language(text: str) -> str:
    """Détecte la langue d'un texte. Retourne 'fr' ou 'en' (défaut 'fr')."""
    try:
        lang = detect(text)
        return lang if lang in ("fr", "en") else "fr"
    except LangDetectException:
        return "fr"


def expand_query(question: str) -> dict:
    """
    Prend une question (FR ou EN) et retourne les deux versions.

    Retourne:
        {
            "original":  str,   # question telle que saisie
            "fr":        str,   # version française
            "en":        str,   # version anglaise
            "detected":  str,   # langue détectée ("fr" | "en")
        }

    Exemples:
        "Qu'est-ce que Bâle 3 ?"
        → fr: "Qu'est-ce que Bâle 3 ?"
        → en: "What is Basel 3?"

        "What is CET1?"
        → fr: "Qu'est-ce que CET1 ?"
        → en: "What is CET1?"
    """
    detected = detect_language(question)

    if detected == "fr":
        fr_query = question
        en_query = _translate(question, source="fr", target="en")
    else:
        en_query = question
        fr_query = _translate(question, source="en", target="fr")

    print(f"   🔍 Langue détectée : {detected.upper()}")
    print(f"   🇫🇷 Query FR : {fr_query}")
    print(f"   🇬🇧 Query EN : {en_query}")

    return {
        "original": question,
        "fr":       fr_query,
        "en":       en_query,
        "detected": detected,
    }


def deduplicate_chunks(chunks_fr: list, chunks_en: list, threshold: float = 0.85) -> list:
    """
    Fusionne et déduplique deux listes de chunks récupérés.

    Stratégie simple : si deux chunks partagent plus de `threshold`
    de mots communs (Jaccard), on garde celui avec le meilleur score.

    Args:
        chunks_fr: list of (score, text, metadata) depuis query FR
        chunks_en: list of (score, text, metadata) depuis query EN
        threshold: similarité Jaccard au-delà de laquelle on considère doublon

    Returns:
        Liste dédupliquée, triée par score décroissant.
    """
    def jaccard(a: str, b: str) -> float:
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    merged = list(chunks_fr) + list(chunks_en)
    # Tri par score décroissant (score = premier élément du tuple)
    merged.sort(key=lambda x: x[0], reverse=True)

    deduplicated = []
    for candidate in merged:
        is_duplicate = False
        for kept in deduplicated:
            if jaccard(candidate[1], kept[1]) >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            deduplicated.append(candidate)

    return deduplicated


# ============================================================================
# Test standalone
# ============================================================================

if __name__ == "__main__":
    test_questions = [
        "Qu'est-ce que Bâle 3 et le ratio CET1 ?",
        "What are the Basel III liquidity requirements?",
        "Quelles sont les exigences de fonds propres AMF ?",
    ]

    print("=" * 60)
    print("TEST — Bilingual Query Expansion")
    print("=" * 60)

    for q in test_questions:
        print(f"\n❓ Question : {q}")
        result = expand_query(q)
        print(f"   ✅ Expansion OK\n")