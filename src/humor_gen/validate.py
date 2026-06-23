from __future__ import annotations
import re

# Set espanso di stop-words per una solida euristica sulla lingua inglese
ENGLISH_HINT_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "with", 
    "for", "is", "are", "was", "were", "why", "when", "because", 
    "it", "its", "you", "that", "this", "they", "their", "them", "but", "not"
}

def clean_joke(text: str) -> str:
    text = (text or "").strip()
    # Rimuove le premesse comuni degli LLM (case insensitive)
    text = re.sub(r"^(sure[,.!]?\s*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(here('s| is)\s+a\s+joke[:\s-]*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(joke|answer|response)\s*:\s*", "", text, flags=re.IGNORECASE)
    return " ".join(text.strip().strip('"').split())

def validate_joke(joke: str, item: dict[str, str], max_words: int = 45) -> tuple[bool, list[str]]:
    errors: list[str] = []
    
    # 1. Pulizia preventiva del testo prima del controllo dei vincoli
    cleaned_text = clean_joke(joke)
    lowered = cleaned_text.casefold()
    
    if not cleaned_text:
        errors.append("empty_output")
        return False, errors
        
    if len(cleaned_text.split()) > max_words:
        errors.append("too_long")
        
    # Controllo linee multiple sul testo pulito
    if "\n" in cleaned_text:
        errors.append("multiple_lines")
        
    if _looks_like_explanation(lowered):
        errors.append("contains_explanation")
        
    if not _looks_english(cleaned_text):
        errors.append("not_english_like")
        
    # 2. Controllo Vincolo: Word Inclusion (Allineato al dataset reale)
    if item["input_type"] == "word_pair":
        if not _contains_word(lowered, item["word1"]):
            errors.append("missing_word1")
        if not _contains_word(lowered, item["word2"]):
            errors.append("missing_word2")
            
    # 3. Controllo Vincolo: News Headline (Allineato al dataset reale)
    if item["input_type"] == "headline":
        # Prende la chiave 'headline' passata nel dizionario
        headline_text = item.get("headline", "")
        if not _has_headline_overlap(lowered, headline_text):
            errors.append("low_headline_relevance")
            
    return not errors, errors

def _looks_like_explanation(lowered: str) -> bool:
    return any(marker in lowered for marker in ("this joke", "the humor", "explanation:", "the punchline is", "funny because"))

def _contains_word(lowered: str, word: str) -> bool:
    word = str(word).strip().casefold()
    if not word:
        return False
    # Ottimizzazione: Supporta variazioni morfologiche comuni (plurali, passati, gerundi)
    # Es: se la parola è 'weigh', accetta anche 'weighed', 'weighing', 'weighs'
    pattern = rf"(?<![a-z0-9]){re.escape(word)}(s|es|ed|ing)?(?![a-z0-9])"
    return re.search(pattern, lowered) is not None or word in lowered

def _looks_english(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z']+", text.casefold())
    if not tokens:
        return False
    ascii_ratio = sum(ch.isascii() for ch in text) / max(len(text), 1)
    hint_ratio = sum(tok in ENGLISH_HINT_WORDS for tok in tokens) / max(len(tokens), 1)
    
    # Ottimizzato: allentata la soglia hint_ratio al 4% e salvate le battute brevi (<= 12 token)
    return ascii_ratio > 0.9 and (hint_ratio > 0.04 or len(tokens) <= 12)

def _has_headline_overlap(lowered: str, headline: str) -> bool:
    # Estrae parole chiave significative (lunghe almeno 3 caratteri) escludendo le stop-words
    headline_terms = {
        token
        for token in re.findall(r"[A-Za-z]{3,}", headline.casefold())
        if token not in ENGLISH_HINT_WORDS
    }
    if not headline_terms:
        return True
    return any(term in lowered for term in headline_terms)