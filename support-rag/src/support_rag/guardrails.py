"""Guardrails around the LLM: redact personal data, block prompt injection, verify grounding.

Input side (before retrieval and the model):
- PII redaction: Swiss phone numbers, e-mail addresses, IBANs, payment cards (Luhn-checked) and
  AHV/AVS numbers are replaced by placeholders, so they never reach the model provider or logs.
- Prompt-injection detection in DE/FR/IT/EN: blocked before any model call.

Output side (after the model):
- Prompt-leak detection: a per-process canary token sits in the system prompt; an answer that
  contains it is blocked.
- Grounding: an answer must cite at least one source, and every number in it (prices, days,
  limits) must appear in the sources it cites. Otherwise the fallback is returned.
"""

import re
import secrets
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

# Nordalp's own service numbers are not personal data (hotline, SMS shortcode).
ALLOWED_NUMBERS = {"0800700700", "444"}

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,3})?\b"),
    "AHV": re.compile(r"\b756[.\s]?\d{4}[.\s]?\d{4}[.\s]?\d{2}\b"),
    "CARD": re.compile(r"\b\d(?:[ -]?\d){12,18}\b"),
    "PHONE": re.compile(r"(?<![\w+])(?:\+41|0041|0)\s?\(?\d{2}\)?(?:[\s./-]?\d){7}\b"),
}


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


@dataclass(frozen=True)
class Redaction:
    text: str
    found: list[str] = field(default_factory=list)  # PII types that were replaced


def redact_pii(text: str) -> Redaction:
    found: list[str] = []

    def replace(kind: str) -> Callable[[re.Match[str]], str]:
        def _sub(m: re.Match[str]) -> str:
            digits = re.sub(r"\D", "", m.group())
            if kind in {"PHONE", "CARD"} and digits in ALLOWED_NUMBERS:
                return m.group()
            if kind == "CARD" and not _luhn_ok(digits):
                return m.group()
            found.append(kind)
            return f"[{kind}]"

        return _sub

    # Order matters: structured identifiers first, so their digits aren't taken for a phone.
    for kind, pattern in _PII_PATTERNS.items():
        text = pattern.sub(replace(kind), text)
    return Redaction(text, found)


def _normalize(text: str) -> str:
    """Lower-case, strip accents and collapse whitespace, so 'Règles' matches 'regles'."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text)


# Phrasings that try to override the assistant's instructions or extract them, per language.
_INJECTION_PATTERNS = [
    # English
    r"\b(ignore|disregard|forget|override)\b.{0,30}\b(instructions?|rules|prompt|guidelines)\b",
    r"\byou are (now|no longer)\b",
    r"\b(pretend|act) (to be|as if|like you are)\b",
    r"\b(system|initial|hidden) (prompt|instructions?|message)\b",
    r"\b(reveal|show|print|repeat|tell me)\b.{0,25}\b(your|the) (instructions|prompt|rules)\b",
    r"\b(developer|debug|god|dan) mode\b",
    r"\bjailbreak\b",
    r"\bfrom now on\b.{0,40}\b(answer|reply|respond|say|only)\b",
    r"\b(repeat|print|output|copy)\b.{0,30}\b(text|message|words|everything)\b.{0,25}"
    r"\b(above|before|previous)\b",
    # German
    r"\b(ignorier\w*|vergiss|missachte\w*)\b.{0,30}\b(anweisung\w*|regeln|instruktion\w*|vorgaben)\b",
    r"\bdu bist (jetzt|ab sofort|nun)\b",
    r"\bsystem-?prompt\b",
    r"\b(zeig|nenn|verrat)\w*\b.{0,25}\b(deine|die) (anweisungen|regeln|instruktionen)\b",
    r"\btu so,? als (ob|warst|waerst)\b",
    r"\bab sofort\b.{0,40}\b(antwort\w*|sag\w*|sprich\w*)\b",
    r"\bwiederhol\w*\b.{0,30}\b(text|nachricht|alles)\b.{0,30}\b(uber|oben|vorher|davor)\b",
    # French
    r"\b(ignore|oublie|oubliez|ne tiens pas compte)\w*\b.{0,30}"
    r"\b(instructions?|regles|consignes)\b",
    r"\b(tu es|vous etes) (maintenant|desormais)\b",
    r"\bprompt (systeme|initial)\b",
    r"\b(revele|montre|affiche)\w*\b.{0,25}\b(tes|vos|les) (instructions|regles|consignes)\b",
    r"\b(fais|faites) comme si\b",
    r"\b(a partir de maintenant|desormais)\b.{0,40}\b(reponds|repondez|dis|dites)\b",
    r"\b(repete|repetez|affiche|recopie)\w*\b.{0,30}\b(texte|message)\b.{0,40}"
    r"\b(au-dessus|dessus|precedent)\b",
    # Italian
    r"\b(ignora|dimentica|trascura)\w*\b.{0,30}\b(istruzion\w*|regole|indicazioni)\b",
    r"\b(ora|adesso) sei\b",
    r"\bprompt di sistema\b",
    r"\b(rivela|mostra|mostrami)\w*\b.{0,25}\b(le tue|le) (istruzioni|regole)\b",
    r"\bfingi (di essere|che)\b",
    r"\b(d'ora in poi|da adesso|da ora)\b.{0,40}\b(rispondi|di'|dimmi)\b",
    r"\b(ripeti|stampa|copia)\b.{0,30}\b(testo|messaggio)\b.{0,40}\b(sopra|precedente)\b",
]
_INJECTION = [re.compile(p) for p in _INJECTION_PATTERNS]


def detect_injection(text: str) -> str | None:
    """The matched pattern if ``text`` looks like a prompt-injection attempt, else None."""
    normalized = _normalize(text)
    for pattern in _INJECTION:
        if pattern.search(normalized):
            return pattern.pattern
    return None


# A fresh canary per process: if it ever shows up in an answer, the system prompt leaked.
CANARY = f"NA-{secrets.token_hex(6)}"

_NUMBER = re.compile(r"(?<![\w\[])\d+(?:[.,']\d+)*")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", ".").replace("'", "") for n in _NUMBER.findall(text)}


def unsupported_numbers(answer: str, cited_texts: list[str], question: str = "") -> list[str]:
    """Numbers in the answer that appear neither in the cited sources nor in the question.

    Citation markers like [2] are not numbers. A number the model made up, e.g. a wrong price,
    shows up here.
    """
    allowed = _numbers(" ".join(cited_texts)) | _numbers(question)
    answer_numbers = _numbers(re.sub(r"\[\d+\]", " ", answer))
    return sorted(n for n in answer_numbers if n not in allowed)
