# src/phrase_tags.py

import re

# --- Spatial vocab ---

# Single-word spatial tokens - will still be tagged independently
SPATIAL_WORDS = {
    "left", "right",
    "top", "bottom",
    "upper", "lower",
    "middle", "center", "centre",
    "front", "back",
    "behind", "under", "below", "beneath",
    "over", "above",
    "inside", "outside",
    "near", "nearby", "close", "closer", "closest",
    "far", "farthest", "furthest",
    "corner", "side", "edge",
    "background", "foreground", "beside", "between",
}

# Multi-word spatial expressions – to catch common phrases
SPATIAL_PATTERNS = [
    "on the left",
    "on the right",
    "at the left",
    "at the right",
    "to the left of",
    "to the right of",
    "in front of",
    "in front",
    "in the front",
    "at the front",
    "at the back",
    "in the back",
    "next to",
    "close to",
    "on top of",
    "in the middle",
    "in middle",
    "in the center",
    "in centre",
    "in between",
    "at the top",
    "at the bottom",
    "top left",
    "top right",
    "bottom left",
    "bottom right",
    "upper left",
    "upper right",
    "lower left",
    "lower right",
    "on left",
    "on right",
    "on bottom",
    "on top",
    "in back",
    "in center",
    
]

# --- Attribute vocab ---

COLOR_WORDS = {
    "red", "blue", "green", "yellow",
    "black", "white", "brown", "orange",
    "pink", "purple", "gray", "grey",
}

SIZE_WORDS = {
    "small", "little", "tiny",
    "big", "large", "huge",
    "tall", "short",
}

PATTERN_WORDS = {
    "striped", "spotted", "dotted",
    "checkered", "chequered", "plaid",
}

ATTRIBUTE_WORDS = COLOR_WORDS | SIZE_WORDS | PATTERN_WORDS


def _has_spatial(phrase: str) -> bool:
    p = phrase.lower()
    tokens = re.findall(r"\w+", p)

    # 1) Any whole-word spatial token?
    if any(tok in SPATIAL_WORDS for tok in tokens):
        return True

    # 2) Any multi-word spatial pattern as substring?
    if any(pat in p for pat in SPATIAL_PATTERNS):
        return True

    return False


def _has_attribute(phrase: str) -> bool:
    p = phrase.lower()
    tokens = re.findall(r"\w+", p)
    return any(tok in ATTRIBUTE_WORDS for tok in tokens)


def classify_phrase_type(phrase: str) -> str:
    """
    Returns one of:
      - 'SPATIAL_ONLY'
      - 'ATTRIBUTE_ONLY'
      - 'MIXED'
      - 'OTHER'
    """
    has_spatial = _has_spatial(phrase)
    has_attr = _has_attribute(phrase)

    if has_spatial and has_attr:
        return "MIXED"
    elif has_spatial:
        return "SPATIAL_ONLY"
    elif has_attr:
        return "ATTRIBUTE_ONLY"
    else:
        return "OTHER"
