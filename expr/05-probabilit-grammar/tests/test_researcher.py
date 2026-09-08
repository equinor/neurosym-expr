from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import unittest
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

from expr05.researcher import ModelEvaluation, PythonSession, Researcher


def evaluation(accepted: bool, modeled_rate: float) -> ModelEvaluation:
    return ModelEvaluation(
        accepted=accepted,
        observed_failure_rate=0.1,
        modeled_failure_rate=modeled_rate,
        absolute_error=abs(modeled_rate - 0.1),
        tolerance=0.01,
        sample_size=100,
    )


def tool_call(description: str) -> str:
    call = {"name": "model_builder", "arguments": {"description": description}}
    return f"<tool_call>{json.dumps(call)}</tool_call>"


class ResearcherLoopTests(unittest.TestCase):
    def make_researcher(
        self,
        responses: Iterator[str],
        model_evaluations: Iterator[ModelEvaluation],
    ) -> Researcher:
        researcher = Researcher.__new__(Researcher)
        researcher.evaluations = []

        def evaluate_model(_: str) -> ModelEvaluation:
            result = next(model_evaluations)
            researcher.evaluations.append(result)
            return result

        researcher.python = PythonSession(pd.DataFrame(), evaluate_model)
        researcher.time_limit_seconds = 300
        researcher._generate = lambda _messages, _max_time: next(responses)
        return researcher

    def test_iterates_until_model_passes_then_returns_answer(self) -> None:
        researcher = self.make_researcher(
            iter(
                [
                    "I am done too early.",
                    tool_call("first model"),
                    tool_call("revised model"),
                    "The revised model reproduces the observed failure rate.",
                ]
            ),
            iter([evaluation(False, 0.25), evaluation(True, 0.1)]),
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            answer = researcher("Model equipment failure")

        self.assertEqual(
            answer, "The revised model reproduces the observed failure rate."
        )
        self.assertEqual(len(researcher.evaluations), 2)
        transcript = output.getvalue()
        self.assertIn("[system]\n", transcript)
        self.assertIn("[user]\nModel equipment failure", transcript)
        self.assertIn("[assistant]\nI am done too early.", transcript)
        self.assertIn("[user]\nYou have not yet produced a model", transcript)
        self.assertIn("[assistant]\n" + tool_call("first model"), transcript)
        self.assertIn(
            "[tool]\n<tool_response>\n"
            + str(evaluation(False, 0.25))
            + "\n</tool_response>",
            transcript,
        )
        self.assertIn(
            "[assistant]\n"
            "The revised model reproduces the observed failure rate.",
            transcript,
        )

    def test_keeps_only_three_most_recent_turns(self) -> None:
        responses = iter(
            [
                tool_call("first model"),
                tool_call("second model"),
                tool_call("third model"),
                tool_call("fourth model"),
                "The fourth model reproduces the observed failure rate.",
            ]
        )
        researcher = self.make_researcher(
            responses,
            iter(
                [
                    evaluation(False, 0.4),
                    evaluation(False, 0.3),
                    evaluation(False, 0.2),
                    evaluation(True, 0.1),
                ]
            ),
        )
        generated_with: list[list[dict[str, str]]] = []

        def generate(messages: list[dict[str, str]], _: float) -> str:
            generated_with.append(messages)
            return next(responses)

        researcher._generate = generate

        researcher("Model equipment failure")

        final_context = generated_with[-1]
        contents = [message["content"] for message in final_context]
        self.assertNotIn(tool_call("first model"), contents)
        self.assertIn(tool_call("second model"), contents)
        self.assertIn(tool_call("third model"), contents)
        self.assertIn(tool_call("fourth model"), contents)
        self.assertEqual(len(final_context), 8)

    def test_stops_when_time_limit_is_exceeded(self) -> None:
        researcher = self.make_researcher(
            iter(["Still researching"]),
            iter([]),
        )
        researcher.time_limit_seconds = 1

        with (
            patch(
                "expr05.researcher.time.monotonic",
                side_effect=[100.0, 100.0, 101.1],
            ),
            self.assertRaisesRegex(
                TimeoutError,
                "Researcher exceeded its 1-second time limit",
            ),
        ):
            researcher("Model equipment failure")


class ResearcherGenerationTests(unittest.TestCase):
    def test_requests_input_ids_tensor_from_chat_template(self) -> None:
        class Tokenizer:
            eos_token_id = 0

            def apply_chat_template(self, *_args, **kwargs):
                self.template_kwargs = kwargs
                return torch.tensor([[1, 2, 3]])

            def decode(self, tokens, **_kwargs):
                return tokens.tolist()

        class Model:
            device = torch.device("cpu")
            config = SimpleNamespace(max_position_embeddings=10)

            def generate(self, input_ids, **_kwargs):
                return torch.cat((input_ids, torch.tensor([[4, 5]])), dim=-1)

        researcher = Researcher.__new__(Researcher)
        researcher.tokenizer = Tokenizer()
        researcher.model = Model()
        researcher.max_new_tokens = 5

        result = researcher._generate([{"role": "user", "content": "Hi"}], 1.0)

        self.assertFalse(researcher.tokenizer.template_kwargs["return_dict"])
        self.assertEqual(result, [4, 5])


class ResearcherScriptTests(unittest.TestCase):
    def test_direct_invocation_runs_main(self) -> None:
        script = Path(__file__).parents[1] / "src" / "expr05" / "researcher.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Analyze a CSV with researcher", result.stdout)


if __name__ == "__main__":
    unittest.main()
