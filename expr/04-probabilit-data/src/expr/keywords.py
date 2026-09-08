import random
from collections.abc import Collection
from functools import cache

from nltk.corpus import wordnet


@cache
def _word_pool(pos: str) -> tuple[str, ...]:
    try:
        synsets = wordnet.all_synsets(pos)
        words = {
            lemma.name().lower()
            for synset in synsets
            for lemma in synset.lemmas()
            if lemma.count() > 0
            and lemma.name().isalpha()
            and len(lemma.name()) >= 3
        }
    except LookupError as error:
        raise RuntimeError(
            "The English WordNet corpus is not installed. "
            "Run `uv run python -m nltk.downloader wordnet`."
        ) from error
    return tuple(sorted(words))


def _words_for_part_of_speech(pos: str, excluded: Collection[str]) -> list[str]:
    return [word for word in _word_pool(pos) if word not in excluded]


def random_keywords(
    noun_count: int = 3,
    verb_count: int = 2,
    *,
    seed: int | None = None,
) -> dict[str, list[str]]:
    """Select distinct, attested English nouns and verbs from WordNet."""
    if noun_count < 0 or verb_count < 0:
        raise ValueError("Keyword counts cannot be negative")

    rng = random.Random(seed)
    nouns = rng.sample(_words_for_part_of_speech("n", ()), noun_count)
    verbs = rng.sample(_words_for_part_of_speech("v", nouns), verb_count)
    return {"nouns": nouns, "verbs": verbs}
