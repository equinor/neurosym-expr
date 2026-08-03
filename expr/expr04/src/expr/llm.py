import contextlib
import json
import os
import random
import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import azure.identity as identity
from openai import OpenAI

from openai.types.responses import ResponseOutputMessage as Message
from openai.types.responses import ResponseCustomToolCall as ToolCall
from openai.types.responses import ResponseReasoningItem as ReasoningItem
from openai.types.responses.response_input_item import (
    ResponseCustomToolCallOutput as ToolResponse,
)

from .keywords import random_keywords
from .lang import grammar, parse_model


credential = identity.AzureCliCredential(process_timeout=60)

token_provider = identity.get_bearer_token_provider(
    credential,
    "https://cognitiveservices.azure.com/.default",
)


client = OpenAI(
    base_url=os.environ["AZURE_OPENAI_URL"],
    api_key=token_provider,
)


SYSTEM_PROMPT = """\
# Probabilistic expression language

Generate models using this line-oriented syntax:

- Sample: `x ~ norm(loc=0, scale=1)`
- Assign: `y = x * 2`
- Correlate: `correlate x with z at 0.5`
- Return: `return y > 3`

Every statement must be on its own line. Blank lines and `#` comments are
allowed. Every model must end with exactly one `return` statement, and no
statements may follow it.

Use unqualified `scipy.stats` distribution names and NumPy function names.
Distribution calls always require parentheses. Special distributions are:

- `empirical(data, method=...)` for a non-empty one-dimensional list;
- `discrete(values, probabilities=...)` for categorical values; and
- `cumulative(quantiles, values)` for strictly increasing quantile positions.

Assign multivariate distribution outputs to multiple variables:

```text
x, y ~ multivariate_normal(mean=[1, 2], cov=[[1, 0.5], [0.5, 1]])
```

Literals include numbers, escaped double-quoted strings, `true`, `false`,
`null`, and nested lists. Supported operators are `+`, `-`, `*`, `/`, `//`,
`%`, `**`, comparisons, `and`, `or`, `not`, and `in`.

There is no implicit binary-operator precedence. Group combined operations
explicitly with parentheses:

```text
total = base + (rate * duration)
inside = (lower <= value) and (value <= upper)
```

Correlated variables must be sampled before their correlation statement.
Correlation coefficients must be between -1 and 1. For more than two
variables, use `correlate [a, b, c] with <matrix>`. The matrix must be square,
symmetric, finite, contain values between -1 and 1, and have ones on its
diagonal.

Example:

```text
demand ~ norm(loc=100, scale=15)
capacity ~ norm(loc=110, scale=10)
correlate demand with capacity at 0.3
shortfall = demand > capacity
return shortfall
```

Use the validation tool for every model. If validation fails, correct the
model and validate it again before answering.
"""


USER_PROMPT = """\
You are tasked to create an exam question for an exam in
probabilistic risk modelling.

Difficulty:
{difficulty}

Learning objective:
{learning_objective}

Structural profile:
{structural_profile}

Create a scenario and question that require the student to build a
probabilistic model to answer. The exam context has already introduced
the probabilistic expression language, so do not name, introduce, or
explain the language in the task. The reference solution must use the
language. Match the model complexity and reasoning required to the
specified difficulty and learning objective.

Follow the structural profile exactly. Treat its required features and
prohibitions as validation criteria. In particular, do not fall back to
correlated normal variables followed by a threshold unless the profile
explicitly permits that structure.

The returned expression must directly represent the requested uncertain
quantity or event. For a numeric result, ask for meaningful simulated summaries
such as its expected value, standard deviation, or quantiles. Do not turn it
into a boolean threshold merely to ask for a probability.

Use every keyword naturally to influence the scenario. Do not merely
list or mention the keywords.

Work in this order:

1. Construct the complete reference model first.
2. Validate it with the validation tool.
3. Correct and revalidate it until valid.
4. Write the task strictly from the validated model.
5. Verify that every model value and relationship appears in the task.

Do not output drafts, validation details, or reasoning.

Before responding, ensure that:

- the task asks one precise probabilistic question requesting an estimate;
- every value, unit, dependency, correlation, and assumption required by
  the reference solution is stated unambiguously in the task;
- the task and reference solution describe exactly the same model;
- distributions and parameter values are plausible for the scenario;
- no unstated assumptions are needed to construct the model; and
- the response contains only the requested Markdown structure.

Model keywords:
{keywords}

Template:

## Task: <concise title>

<self-contained scenario, assumptions, and one precise question>

## Reference solution

```text
<complete validated model>
```
"""


DIFFICULTIES = (
    "Foundational: use a small model with direct relationships and one risk event.",
    "Intermediate: combine several uncertain quantities and at least one derived event.",
    "Advanced: require a multi-step model with dependencies or correlated risks.",
)


LEARNING_OBJECTIVES = (
    "Choose suitable probability distributions and parameterize them from the scenario.",
    "Translate dependencies between uncertain quantities into derived expressions.",
    "Model dependence between sampled variables using correlation.",
    "Represent observed or categorical uncertainty with empirical or discrete distributions.",
    "Combine multiple conditions into a clearly defined risk measure.",
)


@dataclass(frozen=True)
class QuestionProfile:
    difficulty: str
    learning_objective: str
    structural_profile: str


@dataclass(frozen=True)
class ModelSignature:
    distributions: tuple[tuple[str, int], ...]
    sample_count: int
    correlation_count: int
    derived_count: int
    operators: frozenset[str]
    return_shape: str
    operator_counts: tuple[tuple[str, int], ...] = ()
    dependency_topology: tuple[tuple[str, tuple[int, ...]], ...] = ()


STRUCTURAL_PROFILES = {
    LEARNING_OBJECTIVES[0]: (
        (
            "Use at least three distinct distribution families. Use `norm` at "
            "most once and do not use correlation. Return a numeric aggregate "
            "with physical or financial units derived from all sampled quantities."
        ),
        (
            "Use distributions suited to positive, count, bounded, or "
            "categorical quantities. Do not use `norm` or correlation. Include "
            "at least one derived quantity. Return a boolean event and ask for "
            "its probability."
        ),
    ),
    LEARNING_OBJECTIVES[1]: (
        (
            "Compute a distribution parameter from an earlier sampled value "
            "and use it in a later sample. Do not use correlation, use `norm` "
            "at most once, and return a numeric total, duration, cost, or loss."
        ),
        (
            "Build a multi-step arithmetic dependency chain using `minimum`, "
            "`maximum`, `clip`, or a nonlinear function. Use at least two "
            "distribution families, no correlation, and `norm` at most once. "
            "Return a boolean event and ask for its probability."
        ),
    ),
    LEARNING_OBJECTIVES[2]: (
        (
            "Use exactly one pairwise correlation. At least one member of the "
            "pair must be non-normal. Return a numeric derived quantity that "
            "depends on both members rather than directly comparing the pair."
        ),
        (
            "Use one correlation relationship plus at least one independent "
            "variable from a different distribution family. Use `norm` at "
            "most once, combine all sampled quantities in the result, and "
            "return a boolean event whose probability is requested."
        ),
    ),
    LEARNING_OBJECTIVES[3]: (
        (
            "Use `empirical` for observed data and at least one other "
            "non-normal distribution family. Do not use `norm` or correlation. "
            "Return a numeric total, cost, duration, count, or performance measure."
        ),
        (
            "Use both `discrete` and `cumulative` in meaningful roles. Do not "
            "use `norm` or correlation. Return a boolean event derived from both "
            "draws and ask for its probability."
        ),
    ),
    LEARNING_OBJECTIVES[4]: (
        (
            "Use a compound event containing at least three conditions and both "
            "`and` and `or` (or `not`). Do not use correlation and use `norm` "
            "at most once. Return a boolean event and ask for its probability."
        ),
        (
            "Define at least three boolean conditions, including a categorical "
            "condition from `discrete` and two numeric conditions. Convert the "
            "conditions into a count or weighted score. Return a numeric risk "
            "measure and do not use `norm` or correlation."
        ),
    ),
}


_DISTRIBUTION_PATTERN = re.compile(r"^\s*[\w,\s]+~\s*(\w+)\s*\(", re.MULTILINE)
_OPERATOR_PATTERN = re.compile(
    r"\band\b|\bor\b|\bnot\b|==|!=|<=|>=|//|\*\*|[+\-*/%<>]"
)
_IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_]\w*\b")
_STRING_PATTERN = re.compile(r'"(?:\\.|[^"\\])*"')
_SAMPLE_PATTERN = re.compile(
    r"^\s*(?P<targets>[\w,\s]+?)\s*~\s*(?P<distribution>\w+)\s*\((?P<body>.*)\)\s*$"
)
SIGNATURE_HISTORY_LIMIT = 50
MAX_NEAR_DUPLICATE_REJECTIONS = 2
MAX_GENERATION_TURNS = 8
_NEAR_DUPLICATE_ERROR = "error: near-duplicate model structure"


def _dependency_parents(
    expression: str, definitions: dict[str, int]
) -> tuple[int, ...]:
    expression = _STRING_PATTERN.sub("", expression)
    parents = []
    for match in _IDENTIFIER_PATTERN.finditer(expression):
        name = match.group()
        suffix = expression[match.end() :].lstrip()
        if name in definitions and not (
            suffix.startswith("=") and not suffix.startswith("==")
        ):
            parents.append(definitions[name])
    return tuple(parents)


def _dependency_topology(
    lines: list[str], return_shape: str
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    definitions: dict[str, int] = {}
    topology: list[tuple[str, tuple[int, ...]]] = []

    for line in lines:
        sample = _SAMPLE_PATTERN.match(line)
        if sample:
            expression = sample.group("body")
            node_kind = f"sample:{sample.group('distribution')}"
            targets = [
                target.strip() for target in sample.group("targets").split(",")
            ]
        elif line.startswith("correlate "):
            expression = line
            node_kind = "correlate"
            targets = []
        elif line.startswith("return "):
            expression = line.removeprefix("return").strip()
            node_kind = f"return:{return_shape}"
            targets = []
        elif "=" in line:
            target, expression = line.split("=", 1)
            node_kind = "derived"
            targets = [name.strip() for name in target.split(",")]
        else:
            continue

        parents = _dependency_parents(expression, definitions)
        node_index = len(topology)
        topology.append((node_kind, parents))
        for target in targets:
            definitions[target] = node_index

    return tuple(topology)


def model_signature(source: str) -> ModelSignature:
    distributions = Counter(_DISTRIBUTION_PATTERN.findall(source))
    lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return_expression = next(
        (line.removeprefix("return").strip() for line in lines if line.startswith("return ")),
        "",
    )
    if re.search(r"\band\b|\bor\b|\bnot\b", return_expression):
        return_shape = "compound_boolean"
    elif re.search(r"==|!=|<=|>=|<|>", return_expression):
        return_shape = "comparison"
    else:
        return_shape = "value"

    operator_counts = Counter(_OPERATOR_PATTERN.findall("\n".join(lines)))
    return ModelSignature(
        distributions=tuple(sorted(distributions.items())),
        sample_count=sum(distributions.values()),
        correlation_count=sum(line.startswith("correlate ") for line in lines),
        derived_count=sum(
            "=" in line and "~" not in line and not line.startswith("return ")
            for line in lines
        ),
        operators=frozenset(operator_counts),
        return_shape=return_shape,
        operator_counts=tuple(sorted(operator_counts.items())),
        dependency_topology=_dependency_topology(lines, return_shape),
    )


def _counter_similarity(
    left: Counter[str], right: Counter[str]
) -> float:
    total = sum((left | right).values())
    return sum((left & right).values()) / total if total else 1.0


def _sequence_similarity(
    left: tuple[tuple[str, tuple[int, ...]], ...],
    right: tuple[tuple[str, tuple[int, ...]], ...],
) -> float:
    total = max(len(left), len(right))
    return (
        sum(left_node == right_node for left_node, right_node in zip(left, right))
        / total
        if total
        else 1.0
    )


def signature_similarity(left: ModelSignature, right: ModelSignature) -> float:
    if left == right:
        return 1.0

    left_distributions = Counter(dict(left.distributions))
    right_distributions = Counter(dict(right.distributions))
    distribution_overlap = _counter_similarity(
        left_distributions, right_distributions
    )
    left_operators = Counter(dict(left.operator_counts)) or Counter(left.operators)
    right_operators = Counter(dict(right.operator_counts)) or Counter(right.operators)
    operator_overlap = _counter_similarity(left_operators, right_operators)
    topology_overlap = _sequence_similarity(
        left.dependency_topology, right.dependency_topology
    )
    size_similarity = (
        1
        - abs(left.sample_count - right.sample_count)
        / max(left.sample_count, right.sample_count, 1)
    )
    derived_similarity = (
        1
        - abs(left.derived_count - right.derived_count)
        / max(left.derived_count, right.derived_count, 1)
    )
    return (
        0.15 * distribution_overlap
        + 0.15 * operator_overlap
        + 0.45 * topology_overlap
        + 0.08 * size_similarity
        + 0.07 * derived_similarity
        + 0.04 * (left.correlation_count == right.correlation_count)
        + 0.06 * (left.return_shape == right.return_shape)
    )


def random_question_profile(seed: int | None) -> tuple[str, str]:
    rng = random.Random(seed)
    return rng.choice(DIFFICULTIES), rng.choice(LEARNING_OBJECTIVES)


def balanced_question_profiles(
    count: int, seed: int | None = None
) -> list[QuestionProfile]:
    """Build a reproducible batch balanced across difficulty and objective."""
    if count < 0:
        raise ValueError("Profile count cannot be negative")

    rng = random.Random(seed)
    difficulties: list[str] = []
    objectives: list[str] = []
    while len(difficulties) < count:
        block = list(DIFFICULTIES)
        rng.shuffle(block)
        difficulties.extend(block)
    while len(objectives) < count:
        block = list(LEARNING_OBJECTIVES)
        rng.shuffle(block)
        objectives.extend(block)

    variant_offsets = {
        objective: rng.randrange(len(STRUCTURAL_PROFILES[objective]))
        for objective in LEARNING_OBJECTIVES
    }
    objective_occurrences = dict.fromkeys(LEARNING_OBJECTIVES, 0)
    profiles = []
    for difficulty, objective in zip(difficulties[:count], objectives[:count]):
        variants = STRUCTURAL_PROFILES[objective]
        occurrence = objective_occurrences[objective]
        variant = variants[(variant_offsets[objective] + occurrence) % len(variants)]
        objective_occurrences[objective] += 1
        profiles.append(QuestionProfile(difficulty, objective, variant))
    return profiles


def fapply(
    fc: ToolCall,
    prior_signatures: list[ModelSignature] | None = None,
    duplicate_threshold: float = 0.9,
    allow_near_duplicate: bool = False,
):

    try:
        model = parse_model(fc.input)
        model.sample(size=2, random_state=0)

        signature = model_signature(fc.input)
        similarities = [
            signature_similarity(signature, prior)
            for prior in (prior_signatures or ())[-SIGNATURE_HISTORY_LIMIT:]
        ]
        if (
            not allow_near_duplicate
            and similarities
            and max(similarities) >= duplicate_threshold
        ):
            output = (
                f"{_NEAR_DUPLICATE_ERROR}; regenerate with different "
                "distribution families, dependency shape, operators, or return form"
            )
        else:
            output = "valid"
    except Exception as error:
        output = f"error: {type(error).__name__}: {error}"

    return ToolResponse(
        type="custom_tool_call_output",
        call_id=fc.call_id,
        output=output,
    )


def generate_exam_question(
    model: str,
    *,
    profile: QuestionProfile,
    prior_signatures: list[ModelSignature],
    noun_count: int,
    verb_count: int,
    seed: int | None,
) -> None:
    validated_signature = None
    near_duplicate_rejections = 0
    keywords = random_keywords(
        noun_count=noun_count,
        verb_count=verb_count,
        seed=seed,
    )

    keywords = (
        f"Nouns: {', '.join(keywords['nouns'])}\nVerbs: {', '.join(keywords['verbs'])}"
    )

    print(keywords)

    print(profile.difficulty)
    print(profile.learning_objective)
    print(profile.structural_profile)

    user_prompt = USER_PROMPT.format(
        keywords=keywords,
        difficulty=profile.difficulty,
        learning_objective=profile.learning_objective,
        structural_profile=profile.structural_profile,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for _ in range(MAX_GENERATION_TURNS):
        response = client.responses.create(
            model=model,
            store=False,
            include=["reasoning.encrypted_content"],
            reasoning={"effort": "low", "summary": "concise"},
            input=messages,
            tools=[
                {
                    "type": "custom",
                    "name": "validate_prob_model",
                    "description": (
                        "Validates a probabilistic expression model's syntax, "
                        "resolves its distributions and functions, and executes "
                        "a sample to detect runtime errors."
                    ),
                    "format": {
                        "type": "grammar",
                        "syntax": "lark",
                        "definition": grammar,
                    },
                },
            ],
            parallel_tool_calls=False,
        )
        messages.extend(response.output)

        for message in response.output:
            match message:
                case ToolCall(call_id=call_id, input=input):
                    output = f"{call_id=}\n{input}"
                    output = textwrap.indent(input, "! ")
                    print(output)
                case ReasoningItem(summary=summary):
                    output = "\n".join(s.text for s in summary)
                    output = textwrap.indent(output, "* ")
                    print(output)
                case Message(content=content):
                    output = "\n".join(c.text for c in content)
                    output = textwrap.indent(output, "  ")
                    print(output)

        tool_calls = [fc for fc in response.output if hasattr(fc, "call_id")]
        results = [
            fapply(
                fc,
                prior_signatures,
                allow_near_duplicate=(
                    near_duplicate_rejections
                    >= MAX_NEAR_DUPLICATE_REJECTIONS
                ),
            )
            for fc in tool_calls
        ]
        for tool_call, result in zip(tool_calls, results):
            if result.output == "valid":
                validated_signature = model_signature(tool_call.input)
            elif result.output.startswith(_NEAR_DUPLICATE_ERROR):
                near_duplicate_rejections += 1

        for res in results:
            call_id = res.call_id
            output = f"{call_id=}\n{res.output}"
            output = textwrap.indent(output, ": ")
            print(output)

        if not results:
            if validated_signature is not None:
                prior_signatures.append(validated_signature)
                del prior_signatures[:-SIGNATURE_HISTORY_LIMIT]
            break

        messages.extend(results)


def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--noun-count", type=int, default=3)
    parser.add_argument("--verb-count", type=int, default=2)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write generated questions to this file instead of standard output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed used to reproduce keyword selections",
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    model = args.model or os.environ["AZURE_OPENAI_MODEL"]
    profiles = balanced_question_profiles(args.runs, args.seed)
    prior_signatures_by_profile: dict[str, list[ModelSignature]] = {}
    with contextlib.ExitStack() as stack:
        if args.output:
            output = stack.enter_context(args.output.open("w", encoding="utf-8"))
            stack.enter_context(contextlib.redirect_stdout(output))

        for run_index in range(args.runs):
            if run_index:
                print("\n---\n")
            generate_exam_question(
                model,
                profile=profiles[run_index],
                prior_signatures=prior_signatures_by_profile.setdefault(
                    profiles[run_index].structural_profile, []
                ),
                noun_count=args.noun_count,
                verb_count=args.verb_count,
                seed=None if args.seed is None else args.seed + run_index,
            )


if __name__ == "__main__":
    main()
