import pytest
from nltk.corpus import wordnet

from expr.keywords import random_keywords


def test_selects_distinct_nouns_and_verbs():
    keywords = random_keywords(noun_count=3, verb_count=2, seed=42)
    words = keywords["nouns"] + keywords["verbs"]

    assert len(keywords["nouns"]) == 3
    assert len(keywords["verbs"]) == 2
    assert len(words) == len(set(words))
    assert all(wordnet.synsets(word, pos="n") for word in keywords["nouns"])
    assert all(wordnet.synsets(word, pos="v") for word in keywords["verbs"])


def test_seed_reproduces_keyword_selection():
    assert random_keywords(seed=7) == random_keywords(seed=7)


@pytest.mark.parametrize(("noun_count", "verb_count"), [(-1, 2), (3, -1)])
def test_rejects_negative_counts(noun_count, verb_count):
    with pytest.raises(ValueError, match="cannot be negative"):
        random_keywords(noun_count, verb_count)
