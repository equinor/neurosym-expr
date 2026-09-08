import os
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
from openai.types.responses import ResponseCustomToolCall

os.environ.setdefault("AZURE_OPENAI_URL", "https://example.invalid")

import expr.llm as llm
from expr.llm import (
    DIFFICULTIES,
    LEARNING_OBJECTIVES,
    STRUCTURAL_PROFILES,
    SYSTEM_PROMPT,
    USER_PROMPT,
    balanced_question_profiles,
    fapply,
    model_signature,
    random_question_profile,
    signature_similarity,
)


def test_fapply_samples_the_model():
    tool_call = SimpleNamespace(
        call_id="test-call",
        input="value ~ triangular(c=0.4)\nreturn value",
    )

    response = fapply(tool_call)

    assert response.output.startswith("error: ")
    assert "triangular" in response.output


def test_fapply_samples_correlated_model():
    tool_call = SimpleNamespace(
        call_id="test-call",
        input=(
            "left ~ norm()\n"
            "right ~ norm()\n"
            "correlate left with right at 0.5\n"
            "return left + right"
        ),
    )

    response = fapply(tool_call)

    assert response.output == "valid"


def test_fapply_rejects_near_duplicate_model():
    source = "left ~ norm()\nright ~ norm()\nreturn left > right"
    tool_call = SimpleNamespace(call_id="test-call", input=source)

    response = fapply(tool_call, [model_signature(source)])

    assert response.output.startswith("error: near-duplicate")


def test_fapply_default_allows_models_below_new_duplicate_threshold(monkeypatch):
    source = "left ~ norm()\nright ~ norm()\nreturn left > right"
    tool_call = SimpleNamespace(call_id="test-call", input=source)
    monkeypatch.setattr(llm, "signature_similarity", lambda _left, _right: 0.85)

    response = fapply(tool_call, [model_signature(source)])

    assert response.output == "valid"


def test_signature_similarity_distinguishes_model_structures():
    normal_threshold = model_signature(
        "demand ~ norm()\ncapacity ~ norm()\nreturn demand > capacity"
    )
    categorical_value = model_signature(
        'state ~ discrete(["low", "high"])\nreturn state'
    )

    assert signature_similarity(normal_threshold, normal_threshold) == 1
    assert signature_similarity(normal_threshold, categorical_value) < 0.8


def test_signature_similarity_accounts_for_dependency_topology():
    shared_dependency = model_signature(
        "base ~ norm()\n"
        "rate ~ uniform()\n"
        "adjusted = base + rate\n"
        "total = adjusted * 2\n"
        "return total"
    )
    independent_dependency = model_signature(
        "base ~ norm()\n"
        "rate ~ uniform()\n"
        "adjusted = base * 2\n"
        "total = adjusted + rate\n"
        "return total"
    )

    assert signature_similarity(shared_dependency, shared_dependency) == 1
    assert signature_similarity(shared_dependency, independent_dependency) < 1


def test_signature_similarity_accounts_for_repeated_operator_usage():
    one_addition = model_signature(
        "left ~ norm()\nright ~ norm()\ntotal = left + right\nreturn total"
    )
    two_additions = model_signature(
        "left ~ norm()\n"
        "right ~ norm()\n"
        "total = (left + right) + right\n"
        "return total"
    )

    assert signature_similarity(one_addition, two_additions) < 1


def _profile(structural_profile="test structure"):
    return llm.QuestionProfile("test difficulty", "test objective", structural_profile)


def _tool_call(call_id, source):
    return ResponseCustomToolCall(
        call_id=call_id,
        input=source,
        name="validate_prob_model",
        type="custom_tool_call",
    )


def _mock_generation(monkeypatch, outputs):
    responses = iter(SimpleNamespace(output=output) for output in outputs)
    monkeypatch.setattr(
        llm.client.responses, "create", lambda **_kwargs: next(responses)
    )
    monkeypatch.setattr(
        llm,
        "random_keywords",
        lambda **_kwargs: {"nouns": ["loss"], "verbs": ["exceed"]},
    )


def test_generation_accepts_third_valid_near_duplicate(monkeypatch):
    source = "left ~ norm()\nright ~ norm()\nreturn left > right"
    signature = model_signature(source)
    history = [signature]
    _mock_generation(
        monkeypatch,
        [
            [_tool_call("first", source)],
            [_tool_call("second", source)],
            [_tool_call("third", source)],
            [],
        ],
    )

    llm.generate_exam_question(
        "test-model",
        profile=_profile(),
        prior_signatures=history,
        noun_count=1,
        verb_count=1,
        seed=0,
    )

    assert history == [signature, signature]


@pytest.mark.parametrize(
    "invalid_source",
    [
        "this is not a model",
        "value ~ triangular(c=0.4)\nreturn value",
    ],
)
def test_duplicate_fallback_never_accepts_invalid_models(monkeypatch, invalid_source):
    duplicate = "left ~ norm()\nright ~ norm()\nreturn left > right"
    signature = model_signature(duplicate)
    history = [signature]
    _mock_generation(
        monkeypatch,
        [
            [_tool_call("first", duplicate)],
            [_tool_call("second", duplicate)],
            [_tool_call("invalid", invalid_source)],
            [],
        ],
    )

    llm.generate_exam_question(
        "test-model",
        profile=_profile(),
        prior_signatures=history,
        noun_count=1,
        verb_count=1,
        seed=0,
    )

    assert history == [signature]


def test_generation_history_keeps_latest_fifty_signatures(monkeypatch):
    base_signature = model_signature("value ~ norm()\nreturn value")
    history = [replace(base_signature, sample_count=index) for index in range(50)]
    new_source = "value ~ uniform()\nreturn value"
    monkeypatch.setattr(llm, "signature_similarity", lambda _left, _right: 0)
    _mock_generation(monkeypatch, [[_tool_call("new", new_source)], []])

    llm.generate_exam_question(
        "test-model",
        profile=_profile(),
        prior_signatures=history,
        noun_count=1,
        verb_count=1,
        seed=0,
    )

    assert len(history) == 50
    assert history[0] == replace(base_signature, sample_count=1)
    assert history[-1] == model_signature(new_source)


def test_main_scopes_generation_history_by_structural_profile(monkeypatch):
    first = _profile("first structure")
    second = _profile("second structure")
    seen_histories = []

    def fake_generate(_model, *, prior_signatures, **_kwargs):
        seen_histories.append(list(prior_signatures))
        prior_signatures.append(model_signature("value ~ norm()\nreturn value"))

    monkeypatch.setattr(
        llm, "balanced_question_profiles", lambda *_args, **_kwargs: [first, second, first]
    )
    monkeypatch.setattr(llm, "generate_exam_question", fake_generate)
    monkeypatch.setattr(
        "sys.argv", ["expr.llm", "--model", "test-model", "--runs", "3"]
    )

    llm.main()

    assert [len(history) for history in seen_histories] == [0, 0, 1]


def test_system_prompt_is_a_concise_language_reference():
    assert len(SYSTEM_PROMPT) < 3_000
    assert "There is no implicit binary-operator precedence" in SYSTEM_PROMPT
    assert "Use the validation tool for every model" in SYSTEM_PROMPT


def test_user_prompt_does_not_ask_the_task_to_introduce_the_language():
    assert "do not name, introduce, or\nexplain the language" in USER_PROMPT
    assert "The reference solution must use the\nlanguage." in USER_PROMPT


def test_user_prompt_contains_authoring_checklist():
    assert "the task asks one precise probabilistic question" in USER_PROMPT
    assert "the task and reference solution describe exactly the same model" in USER_PROMPT
    assert "no unstated assumptions are needed" in USER_PROMPT


def test_user_prompt_requires_model_first_workflow():
    assert "Construct the complete reference model first" in USER_PROMPT
    assert "Correct and revalidate it until valid" in USER_PROMPT
    assert "Write the task strictly from the validated model" in USER_PROMPT
    assert "Do not output drafts, validation details, or reasoning" in USER_PROMPT


def test_random_question_profile_is_reproducible():
    profile = random_question_profile(seed=42)

    assert profile == random_question_profile(seed=42)
    assert profile[0] in DIFFICULTIES
    assert profile[1] in LEARNING_OBJECTIVES


def test_user_prompt_accepts_question_profile():
    prompt = USER_PROMPT.format(
        keywords="Nouns: loss\nVerbs: exceed",
        difficulty=DIFFICULTIES[0],
        learning_objective=LEARNING_OBJECTIVES[0],
        structural_profile=STRUCTURAL_PROFILES[LEARNING_OBJECTIVES[0]][0],
    )

    assert DIFFICULTIES[0] in prompt
    assert LEARNING_OBJECTIVES[0] in prompt
    assert STRUCTURAL_PROFILES[LEARNING_OBJECTIVES[0]][0] in prompt


def test_balanced_profiles_are_reproducible():
    assert balanced_question_profiles(15, seed=42) == balanced_question_profiles(
        15, seed=42
    )


def test_balanced_profiles_cover_dimensions_evenly():
    profiles = balanced_question_profiles(15, seed=42)

    assert Counter(profile.difficulty for profile in profiles) == {
        difficulty: 5 for difficulty in DIFFICULTIES
    }
    assert Counter(profile.learning_objective for profile in profiles) == {
        objective: 3 for objective in LEARNING_OBJECTIVES
    }
    assert all(
        profile.structural_profile
        in STRUCTURAL_PROFILES[profile.learning_objective]
        for profile in profiles
    )
    for objective, variants in STRUCTURAL_PROFILES.items():
        counts = Counter(
            profile.structural_profile
            for profile in profiles
            if profile.learning_objective == objective
        )
        assert max(counts[variant] for variant in variants) - min(
            counts[variant] for variant in variants
        ) <= 1


def test_balanced_profiles_split_numeric_and_boolean_results_evenly():
    profiles = balanced_question_profiles(10, seed=42)
    instructions = [profile.structural_profile.casefold() for profile in profiles]

    assert sum("return a numeric" in profile for profile in instructions) == 5
    assert sum("return a boolean" in profile for profile in instructions) == 5


def test_balanced_profiles_reject_negative_count():
    with pytest.raises(ValueError, match="cannot be negative"):
        balanced_question_profiles(-1)


def test_structural_profiles_constrain_common_defaults():
    profiles = [
        profile
        for variants in STRUCTURAL_PROFILES.values()
        for profile in variants
    ]

    assert all("norm" in profile for profile in profiles)
    assert all("correlation" in profile for profile in profiles)


def test_structural_profiles_balance_numeric_and_boolean_results():
    for variants in STRUCTURAL_PROFILES.values():
        assert len(variants) == 2
        normalized = [profile.casefold() for profile in variants]
        assert sum("return a numeric" in profile for profile in normalized) == 1
        assert sum("return a boolean" in profile for profile in normalized) == 1


def test_user_prompt_does_not_force_quantitative_results_into_events():
    assert "expected value, standard deviation, or quantiles" in USER_PROMPT
    assert "do not turn it\ninto a boolean threshold" in USER_PROMPT.casefold()
    assert "returned risk event" not in USER_PROMPT


def test_user_prompt_contains_complete_output_template():
    assert "## Task: <concise title>" in USER_PROMPT
    assert "<self-contained scenario, assumptions, and one precise question>" in USER_PROMPT
    assert "```text\n<complete validated model>\n```" in USER_PROMPT
