import json

from expr.output_to_jsonl import convert, parse_records


SOURCE = """\
Nouns: alpha, beta
Verbs: compare
Foundational: direct model.
Choose suitable distributions.
Return a boolean event.
* private reasoning
! bad ~ unknown()
! return bad
: call_id='bad'
: error: ValueError: unknown distribution
* more private reasoning
! value ~ norm(loc=2, scale=1)
! return value > 3
: call_id='good'
: valid
* final private reasoning
  ## Task: Example probability

  Estimate the probability that the value exceeds 3.

  ## Reference solution

  ```text
  value ~ norm(loc=2, scale=1)
  return value > 3
  ```

---

Nouns: unfinished
Verbs: omit
Foundational: direct model.
Choose suitable distributions.
Return a boolean event.
"""


def test_parse_records_extracts_only_completed_generation():
    assert parse_records(SOURCE) == [
        {
            "nouns": ["alpha", "beta"],
            "verbs": ["compare"],
            "difficulty": "Foundational: direct model.",
            "learning_objective": "Choose suitable distributions.",
            "structural_profile": "Return a boolean event.",
            "model": "value ~ norm(loc=2, scale=1)\nreturn value > 3",
            "task": (
                "Example probability\n\n"
                "Estimate the probability that the value exceeds 3."
            ),
            "reference_solution": (
                "value ~ norm(loc=2, scale=1)\nreturn value > 3"
            ),
        }
    ]


def test_convert_writes_one_json_object_per_line(tmp_path):
    input_path = tmp_path / "output.md"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(SOURCE, encoding="utf-8")

    assert convert(input_path, output_path) == 1
    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1
    assert json.loads(lines[0])["model"].startswith("value ~ norm")
