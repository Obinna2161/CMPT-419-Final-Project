from src.phrase_tags import classify_phrase_type

test_phrases = [
    "man on right",
    "guy on left",
    "person at the top left",
    "woman in the middle",
    "dog under the table",
    "small red ball",
    "big striped shirt",
    "tall man on the left",
    "the man",
    "bright light",
    "little left shoe",
]

for s in test_phrases:
    print(f"{s!r:30} -> {classify_phrase_type(s)}")
