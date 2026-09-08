import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path


_RECORD_SEPARATOR = re.compile(r"^---\s*$")
_TASK_HEADING = re.compile(r"^\s*## Task:\s*(.+?)\s*$")
_REFERENCE_HEADING = re.compile(r"^\s*## Reference solution\s*$")


def _split_records(source: str) -> Iterable[list[str]]:
    record: list[str] = []
    for line in source.splitlines():
        if _RECORD_SEPARATOR.fullmatch(line):
            if record:
                yield record
            record = []
        else:
            record.append(line)
    if record:
        yield record


def _profile(lines: list[str]) -> dict[str, object] | None:
    values = [line.strip() for line in lines if line.strip()]
    if len(values) < 5 or not values[0].startswith("Nouns: "):
        return None
    if not values[1].startswith("Verbs: "):
        return None

    return {
        "nouns": [item.strip() for item in values[0][len("Nouns: ") :].split(",")],
        "verbs": [item.strip() for item in values[1][len("Verbs: ") :].split(",")],
        "difficulty": values[2],
        "learning_objective": values[3],
        "structural_profile": values[4],
    }


def _last_valid_model(lines: list[str]) -> str | None:
    pending_model: str | None = None
    valid_model: str | None = None
    index = 0

    while index < len(lines):
        if lines[index].startswith("! "):
            model_lines = []
            while index < len(lines):
                if lines[index].startswith("! "):
                    model_lines.append(lines[index][2:])
                    index += 1
                elif (
                    not lines[index].strip()
                    and index + 1 < len(lines)
                    and lines[index + 1].startswith("! ")
                ):
                    model_lines.append("")
                    index += 1
                else:
                    break
            pending_model = "\n".join(model_lines).strip()
            continue

        if lines[index].strip() == ": valid" and pending_model:
            valid_model = pending_model
        index += 1

    return valid_model


def _generated_answer(lines: list[str]) -> tuple[str, str] | None:
    task_index = next(
        (index for index, line in enumerate(lines) if _TASK_HEADING.match(line)),
        None,
    )
    if task_index is None:
        return None

    reference_index = next(
        (
            index
            for index in range(task_index + 1, len(lines))
            if _REFERENCE_HEADING.match(lines[index])
        ),
        None,
    )
    if reference_index is None:
        return None

    title_match = _TASK_HEADING.match(lines[task_index])
    assert title_match is not None
    task_lines = lines[task_index + 1 : reference_index]
    body = "\n".join(
        line[2:] if line.startswith("  ") else line for line in task_lines
    ).strip()
    task = title_match.group(1)
    if body:
        task = f"{task}\n\n{body}"

    reference_lines = lines[reference_index + 1 :]
    fence_start = next(
        (
            index
            for index, line in enumerate(reference_lines)
            if line.strip().startswith("```")
        ),
        None,
    )
    if fence_start is None:
        return None
    fence_end = next(
        (
            index
            for index in range(fence_start + 1, len(reference_lines))
            if reference_lines[index].strip() == "```"
        ),
        None,
    )
    if fence_end is None:
        return None

    reference_solution = "\n".join(
        line[2:] if line.startswith("  ") else line
        for line in reference_lines[fence_start + 1 : fence_end]
    ).strip()
    return task, reference_solution


def parse_records(source: str) -> list[dict[str, object]]:
    records = []
    for lines in _split_records(source):
        profile = _profile(lines)
        model = _last_valid_model(lines)
        answer = _generated_answer(lines)
        if profile is None or model is None or answer is None:
            continue

        task, reference_solution = answer
        records.append(
            profile
            | {
                "model": model,
                "task": task,
                "reference_solution": reference_solution,
            }
        )
    return records


def convert(input_path: Path, output_path: Path) -> int:
    records = parse_records(input_path.read_text(encoding="utf-8"))
    with output_path.open("w", encoding="utf-8") as output:
        for record in records:
            json.dump(record, output, ensure_ascii=False)
            output.write("\n")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert completed expr.llm Markdown records to JSON Lines."
    )
    parser.add_argument("input", type=Path, nargs="?", default=Path("output.md"))
    parser.add_argument("output", type=Path, nargs="?", default=Path("output.jsonl"))
    args = parser.parse_args()

    count = convert(args.input, args.output)
    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
