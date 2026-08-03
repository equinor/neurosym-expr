import argparse
import json
from pathlib import Path


def _message(record: object, line_number: int) -> dict[str, object]:
    if not isinstance(record, dict):
        raise ValueError(f"line {line_number}: expected a JSON object")

    task = record.get("task")
    model = record.get("model")
    if not isinstance(task, str) or not task.strip():
        raise ValueError(f"line {line_number}: 'task' must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"line {line_number}: 'model' must be a non-empty string")

    return {
        "messages": [
            {"role": "user", "content": task},
            {"role": "assistant", "content": model},
        ]
    }


def convert(input_path: Path, output_path: Path) -> int:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must be different")

    count = 0
    with (
        input_path.open(encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as output,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"line {line_number}: invalid JSON: {error.msg}"
                ) from error

            json.dump(
                _message(record, line_number),
                output,
                ensure_ascii=False,
            )
            output.write("\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert output.jsonl to a SmolLM3 chat dataset."
    )
    parser.add_argument("input", type=Path, nargs="?", default=Path("output.jsonl"))
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("smollm3_dataset.jsonl"),
    )
    args = parser.parse_args()

    count = convert(args.input, args.output)
    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
