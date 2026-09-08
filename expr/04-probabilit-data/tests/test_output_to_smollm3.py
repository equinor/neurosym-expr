import json

import pytest

from expr.output_to_smollm3 import convert


def test_convert_writes_task_and_model_as_chat_messages(tmp_path):
    input_path = tmp_path / "output.jsonl"
    output_path = tmp_path / "smollm3_dataset.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "task": "Estimate the probability.",
                "model": "value ~ norm(loc=0, scale=1)\nreturn value > 1",
                "reference_solution": "not included",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert convert(input_path, output_path) == 1
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "messages": [
            {"role": "user", "content": "Estimate the probability."},
            {
                "role": "assistant",
                "content": "value ~ norm(loc=0, scale=1)\nreturn value > 1",
            },
        ]
    }


def test_convert_reports_missing_fields_with_line_number(tmp_path):
    input_path = tmp_path / "output.jsonl"
    output_path = tmp_path / "smollm3_dataset.jsonl"
    input_path.write_text('{"task": "Valid"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 1: 'model'"):
        convert(input_path, output_path)
