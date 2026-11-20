"""Nettoyage de texte, partage entre le pipeline d'entrainement et l'API."""
import html
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
RT_RE = re.compile(r"\brt\b")
NON_ALPHA_RE = re.compile(r"[^a-z\s']")
SPACES_RE = re.compile(r"\s+")


def clean_text(text) -> str:
    """Minuscules, sans URL/mentions/ponctuation/chiffres/emojis (texte anglais)."""
    if text is None:
        return ""
    t = html.unescape(str(text)).lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(r" \1 ", t)
    t = RT_RE.sub(" ", t)
    t = NON_ALPHA_RE.sub(" ", t)
    return SPACES_RE.sub(" ", t).strip()
