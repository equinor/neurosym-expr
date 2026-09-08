from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import math
import operator
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from lark import Lark, Transformer, v_args
from lark.exceptions import VisitError
from probabilit import (
    All,
    Any as ProbabilitAny,
    Constant,
    CumulativeDistribution,
    DiscreteDistribution,
    Distribution,
    EmpiricalDistribution,
    Equal,
    MultivariateDistribution,
    scalar_transform,
)
from probabilit.modeling import Node, NotEqual
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from xgrammar import Grammar, GrammarCompiler, TokenizerInfo
from xgrammar.contrib.hf import LogitsProcessor


MODEL_GRAMMAR = """\
start: _NL* (instruction _NL+)* return_stmt _NL*

?instruction: VAR "~" distribution -> random_assign
            | variable_tuple "~" distribution -> multivariate_random_assign
            | VAR "=" expr -> assign
            | correlate
variable_tuple: VAR ("," VAR)+
return_stmt: "return" expr
?expr: unary | unary OPERATOR unary -> binary
?multiline_expr: unary | unary OPERATOR _NL* unary -> binary
?unary: "-" unary -> neg | _NOT unary -> not_ | primary
?primary: NUMBER -> number
        | ESCAPED_STRING -> string
        | "true" -> true
        | "false" -> false
        | "null" -> null
        | list
        | function
        | VAR -> var
        | "(" _NL* multiline_expr _NL* ")"
list: "[" _NL* "]"
    | "[" _NL* multiline_expr (_comma multiline_expr)* _comma? _NL* "]"
OPERATOR: "+" | "*" | "and" | "or" | "%" | "/" | "**" | "-" | "=="
        | "<" | "<=" | ">" | ">=" | "!=" | "//" | "in"
function: VAR "(" _NL* ")" -> func
        | VAR "(" _NL* _arguments _comma? _NL* ")" -> func
distribution: VAR "(" _NL* ")"
            | VAR "(" _NL* _arguments _comma? _NL* ")"
_arguments: positional_argument _comma _arguments
          | positional_argument
          | _keyword_arguments
positional_argument: multiline_expr
_keyword_arguments: keyword_argument (_comma keyword_argument)*
keyword_argument: VAR "=" _NL* multiline_expr
_comma: "," _NL*
_NOT: "not"
?correlate: "correlate" VAR "with" VAR "at" SIGNED_NUMBER -> correlate_pair
          | "correlate" variable_list "with" list -> correlate_matrix
variable_list: "[" VAR ("," VAR)+ "]"

%import common.CNAME -> VAR
%import common.ESCAPED_STRING
%import common.NEWLINE -> _NL
%import common.NUMBER
%import common.SH_COMMENT
%import common.SIGNED_NUMBER
%import common.WS_INLINE
%ignore SH_COMMENT
%ignore WS_INLINE
"""

MODEL_PARSER = Lark(MODEL_GRAMMAR, parser="lalr")


class ModelConversionError(ValueError):
    pass


@dataclass(frozen=True)
class _Keyword:
    name: str
    value: Any


@dataclass(frozen=True)
class _DistributionCall:
    name: str
    arguments: tuple[Any, ...]


@dataclass(frozen=True)
class _Correlation:
    variables: tuple[str, ...]
    matrix: np.ndarray


@dataclass(frozen=True)
class _Return:
    value: Any


_BINARY_OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def _empirical_distribution(*args: Any, **kwargs: Any) -> EmpiricalDistribution:
    if len(args) > 1 or (args and "data" in kwargs):
        raise ModelConversionError(
            "empirical() accepts its data once, either positionally or by keyword"
        )
    if args:
        data = args[0]
    elif "data" in kwargs:
        data = kwargs.pop("data")
    else:
        raise ModelConversionError("empirical() requires sample data")
    if not isinstance(data, list) or not data:
        raise ModelConversionError("empirical() sample data must be a non-empty list")
    if any(isinstance(item, list) for item in data):
        raise ModelConversionError("empirical() sample data must be one-dimensional")
    return EmpiricalDistribution(data, **kwargs)


def _discrete_distribution(*args: Any, **kwargs: Any) -> DiscreteDistribution:
    try:
        return DiscreteDistribution(*args, **kwargs)
    except (TypeError, ValueError) as error:
        raise ModelConversionError(
            f"Invalid discrete() distribution: {error}"
        ) from error


def _cumulative_distribution(*args: Any, **kwargs: Any) -> CumulativeDistribution:
    try:
        return CumulativeDistribution(*args, **kwargs)
    except (TypeError, ValueError) as error:
        raise ModelConversionError(
            f"Invalid cumulative() distribution: {error}"
        ) from error


_DISTRIBUTION_FACTORIES = {
    "cumulative": _cumulative_distribution,
    "discrete": _discrete_distribution,
    "empirical": _empirical_distribution,
}


def _has_node(value: Any) -> bool:
    if isinstance(value, Node):
        return True
    if isinstance(value, (list, tuple)):
        return any(_has_node(item) for item in value)
    return False


def _probabilit_operand(value: Any) -> Any:
    if isinstance(value, Node) or isinstance(value, (int, float, complex)):
        return value
    return Constant(value)


@v_args(inline=True)
class _ModelTransformer(Transformer):
    def __init__(self) -> None:
        super().__init__()
        self.variables: dict[str, Any] = {}
        self.random_variables: set[str] = set()

    def start(self, *statements: Any) -> Node:
        returned = next(
            (item for item in statements if isinstance(item, _Return)), None
        )
        if returned is None:
            raise ModelConversionError("The model does not return a value")
        result = (
            returned.value
            if isinstance(returned.value, Node)
            else Constant(returned.value)
        )
        for statement in statements:
            if isinstance(statement, _Correlation):
                variables = [self.variables[name] for name in statement.variables]
                try:
                    result.correlate(*variables, corr_mat=statement.matrix)
                except ValueError as error:
                    raise ModelConversionError(
                        "Correlated variables must be ancestors of the returned value"
                    ) from error
        return result

    def random_assign(self, name: Any, distribution: _DistributionCall) -> None:
        variable_name = str(name)
        self._define(variable_name, self._make_distribution(distribution))
        self.random_variables.add(variable_name)

    def multivariate_random_assign(
        self, names: tuple[str, ...], distribution: _DistributionCall
    ) -> None:
        args, kwargs = self._arguments(distribution.arguments)
        try:
            marginals = tuple(
                MultivariateDistribution(distribution.name, *args, **kwargs)
            )
        except (TypeError, ValueError) as error:
            raise ModelConversionError(
                f"Invalid multivariate distribution {distribution.name!r}: {error}"
            ) from error
        if len(marginals) != len(names):
            raise ModelConversionError(
                f"Multivariate distribution {distribution.name!r} produces "
                f"{len(marginals)} values, but {len(names)} variables were provided"
            )
        for name, marginal in zip(names, marginals, strict=True):
            self._define(name, marginal)
            self.random_variables.add(name)

    def variable_tuple(self, *names: Any) -> tuple[str, ...]:
        result = tuple(str(name) for name in names)
        if len(result) != len(set(result)):
            raise ModelConversionError(
                "Multivariate assignment variables must be unique"
            )
        return result

    def assign(self, name: Any, value: Any) -> None:
        self._define(str(name), value)

    def _define(self, name: str, value: Any) -> None:
        if name in self.variables:
            raise ModelConversionError(f"Variable {name!r} is already defined")
        self.variables[name] = value

    def correlate_pair(self, left: Any, right: Any, coefficient: Any) -> _Correlation:
        names = (str(left), str(right))
        if names[0] == names[1]:
            raise ModelConversionError("A variable cannot be correlated with itself")
        value = float(coefficient)
        return self._correlation(
            names, np.array([[1.0, value], [value, 1.0]])
        )

    def correlate_matrix(
        self, names: tuple[str, ...], matrix: list[Any]
    ) -> _Correlation:
        try:
            correlation_matrix = np.asarray(matrix, dtype=float)
        except (TypeError, ValueError) as error:
            raise ModelConversionError(
                "Correlation matrix must contain only numbers"
            ) from error
        return self._correlation(names, correlation_matrix)

    def variable_list(self, *names: Any) -> tuple[str, ...]:
        return tuple(str(name) for name in names)

    def _correlation(self, names: tuple[str, ...], matrix: np.ndarray) -> _Correlation:
        for name in names:
            if name not in self.variables:
                raise ModelConversionError(f"Unknown variable {name!r}")
            if name not in self.random_variables:
                raise ModelConversionError(
                    f"Correlation requires sampled variable {name!r}"
                )
        if len(names) != len(set(names)):
            raise ModelConversionError("Correlated variables must be unique")
        expected_shape = (len(names), len(names))
        if matrix.shape != expected_shape:
            raise ModelConversionError(
                f"Correlation matrix must have shape {expected_shape}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ModelConversionError("Correlation matrix values must be finite")
        if not np.allclose(matrix, matrix.T):
            raise ModelConversionError("Correlation matrix must be symmetric")
        if not np.allclose(np.diag(matrix), 1):
            raise ModelConversionError("Correlation matrix diagonal must contain 1")
        if np.any((matrix < -1) | (matrix > 1)):
            raise ModelConversionError(
                "Correlation matrix values must be between -1 and 1"
            )
        return _Correlation(names, matrix.copy())

    def return_stmt(self, value: Any) -> _Return:
        return _Return(value)

    def distribution(self, name: Any, *arguments: Any) -> _DistributionCall:
        return _DistributionCall(str(name), arguments)

    def _make_distribution(self, distribution: _DistributionCall) -> Node:
        args, kwargs = self._arguments(distribution.arguments)
        factory = _DISTRIBUTION_FACTORIES.get(distribution.name)
        if factory is not None:
            return factory(*args, **kwargs)
        return Distribution(distribution.name, *args, **kwargs)

    def func(self, name: Any, *arguments: Any) -> Any:
        function_name = str(name)
        function = getattr(np, function_name, None)
        if function is None or not callable(function):
            raise ModelConversionError(f"Unknown NumPy function {function_name!r}")
        args, kwargs = self._arguments(arguments)
        if any(_has_node(value) for value in (*args, *kwargs.values())):
            return scalar_transform(function)(*args, **kwargs)
        try:
            return function(*args, **kwargs)
        except (TypeError, ValueError) as error:
            raise ModelConversionError(
                f"Invalid arguments for NumPy function {function_name!r}: {error}"
            ) from error

    @staticmethod
    def _arguments(arguments: tuple[Any, ...]) -> tuple[list[Any], dict[str, Any]]:
        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        for argument in arguments:
            if isinstance(argument, _Keyword):
                kwargs[argument.name] = argument.value
            else:
                args.append(argument)
        return args, kwargs

    def positional_argument(self, value: Any) -> Any:
        return value

    def keyword_argument(self, name: Any, value: Any) -> _Keyword:
        return _Keyword(str(name), value)

    def var(self, name: Any) -> Any:
        variable_name = str(name)
        try:
            return self.variables[variable_name]
        except KeyError as error:
            raise ModelConversionError(f"Unknown variable {variable_name!r}") from error

    def binary(self, left: Any, token: Any, right: Any) -> Any:
        symbol = str(token)
        has_node = _has_node((left, right))
        if symbol == "==":
            return (
                Equal(_probabilit_operand(left), _probabilit_operand(right))
                if has_node
                else left == right
            )
        if symbol == "!=":
            return (
                NotEqual(_probabilit_operand(left), _probabilit_operand(right))
                if has_node
                else left != right
            )
        if symbol == "and":
            return (
                All(_probabilit_operand(left), _probabilit_operand(right))
                if has_node
                else np.logical_and(left, right)
            )
        if symbol == "or":
            return (
                ProbabilitAny(_probabilit_operand(left), _probabilit_operand(right))
                if has_node
                else np.logical_or(left, right)
            )
        if symbol == "in":
            if _has_node(left):
                return scalar_transform(operator.contains)(right, left)
            return left in right
        if has_node:
            left = _probabilit_operand(left)
            right = _probabilit_operand(right)
        return _BINARY_OPERATORS[symbol](left, right)

    def neg(self, value: Any) -> Any:
        return -value

    def not_(self, value: Any) -> Any:
        if isinstance(value, Node):
            return scalar_transform(np.logical_not)(value)
        return not value

    def list(self, *values: Any) -> list[Any]:
        if any(_has_node(value) for value in values):
            raise ModelConversionError(
                "Lists containing probabilistic values are not supported"
            )
        return list(values)

    def number(self, value: Any) -> int | float:
        text = str(value)
        return float(text) if "." in text or "e" in text.lower() else int(text)

    def string(self, value: Any) -> str:
        return json.loads(str(value))

    def true(self) -> bool:
        return True

    def false(self) -> bool:
        return False

    def null(self) -> None:
        return None


def model_from_tree(tree: Any) -> Node:
    try:
        return _ModelTransformer().transform(tree)
    except VisitError as error:
        if isinstance(error.orig_exc, ModelConversionError):
            raise error.orig_exc from None
        raise


def parse_model(source: str) -> Node:
    return model_from_tree(MODEL_PARSER.parse(source))


PROBABILITIER_SYSTEM_PROMPT = """\
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
/system_override
"""

PROBABILITIER_GRAMMAR = Grammar.from_ebnf(r"""
root ::= line* ws return_stmt ws comment? newline*
line ::= ws instruction? ws comment? newline
instruction ::= variable ws "~" ws distribution
               | variable_tuple ws "~" ws distribution
               | variable ws "=" ws expr
               | correlate
variable_tuple ::= variable (ws "," ws variable)+
return_stmt ::= "return" ws expr
expr ::= unary (ws operator ws unary)?
unary ::= "-" ws unary | "not" ws unary | primary
primary ::= number
           | string
           | "true"
           | "false"
           | "null"
           | list
           | function
           | variable
           | "(" space expr space ")"
list ::= "[" space (expr (space "," space expr)* (space ",")?)? space "]"
operator ::= "**" | "//" | "==" | "<=" | ">=" | "!="
            | "+" | "*" | "and" | "or" | "%" | "/" | "-" | "<" | ">" | "in"
function ::= variable "(" space (arguments (space ",")?)? space ")"
distribution ::= function
arguments ::= positional_arguments (space "," space keyword_arguments)?
            | keyword_arguments
positional_arguments ::= expr (space "," space expr)*
keyword_arguments ::= keyword_argument (space "," space keyword_argument)*
keyword_argument ::= variable ws "=" space expr
correlate ::= "correlate" ws variable ws "with" ws variable ws "at" ws signed_number
             | "correlate" ws variable_list ws "with" ws list
variable_list ::= "[" ws variable (ws "," ws variable)+ ws "]"
variable ::= [a-zA-Z_] [a-zA-Z0-9_]*
number ::= [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
signed_number ::= [+-]? number
string ::= "\"" ([^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* "\""
comment ::= "#" [^\r\n]*
space ::= [ \t\r\n]*
ws ::= [ \t]*
newline ::= "\r"? "\n"
""")


class Probabiliter:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_new_tokens: int = 4_096,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        tokenizer_info = TokenizerInfo.from_huggingface(
            self.tokenizer, vocab_size=config.vocab_size
        )
        compiler = GrammarCompiler(tokenizer_info)
        compiled_grammar = compiler.compile_grammar(PROBABILITIER_GRAMMAR)
        self.logits_processors = [LogitsProcessor(compiled_grammar)]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map=device,
        )
        self.max_new_tokens = max_new_tokens

    def __call__(self, description: str) -> str:
        messages = [
            {"role": "system", "content": PROBABILITIER_SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            logits_processor=self.logits_processors,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


RANDOM_STATE = 15_205
DEFAULT_FAILURE_RATE_TOLERANCE = 0.01


class SampleableModel(Protocol):
    def sample(self, size: int, random_state: int) -> object: ...


class ModelEvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelEvaluation:
    accepted: bool
    observed_failure_rate: float
    modeled_failure_rate: float
    absolute_error: float
    tolerance: float
    sample_size: int

    def __str__(self) -> str:
        return "\n".join(
            [
                f"models_failure_rate: {'yes' if self.accepted else 'no'}",
                f"observed_failure_rate: {self.observed_failure_rate:.6f}",
                f"modeled_failure_rate: {self.modeled_failure_rate:.6f}",
                f"absolute_error: {self.absolute_error:.6f}",
                f"required_tolerance: {self.tolerance:.6f}",
                f"sample_size: {self.sample_size}",
            ]
        )


class EvaluatedModelBuilder:
    def __init__(
        self,
        generator: Callable[[str], str],
        data: pd.DataFrame,
        target: str,
        parser: Callable[[str], SampleableModel] = parse_model,
        failure_rate_tolerance: float = DEFAULT_FAILURE_RATE_TOLERANCE,
    ) -> None:
        if target not in data:
            raise ValueError(f"Target column {target!r} is not present in the CSV")
        if not 0 <= failure_rate_tolerance <= 1:
            raise ValueError("Failure-rate tolerance must be between 0 and 1")

        self.generator = generator
        self.target = np.asarray(data[target])
        self.parser = parser
        self.failure_rate_tolerance = failure_rate_tolerance
        self._validate_binary_values(self.target, "Target")

    @staticmethod
    def _validate_binary_values(values: np.ndarray, name: str) -> None:
        try:
            valid = np.isin(values, [0, 1]).all()
        except TypeError:
            valid = False
        if not valid:
            raise ValueError(f"{name} values must all be binary (0 or 1)")

    def __call__(self, description: str) -> ModelEvaluation:
        try:
            source = self.generator(description)
            model = self.parser(source)
            predictions = np.asarray(
                model.sample(size=len(self.target), random_state=RANDOM_STATE)
            )
            if predictions.shape != self.target.shape:
                raise ValueError(
                    "The generated model must return one prediction per data row"
                )
            self._validate_binary_values(predictions, "Generated model")

            observed_failure_rate = float(np.mean(self.target))
            modeled_failure_rate = float(np.mean(predictions))
            absolute_error = abs(modeled_failure_rate - observed_failure_rate)
            return ModelEvaluation(
                accepted=absolute_error <= self.failure_rate_tolerance,
                observed_failure_rate=observed_failure_rate,
                modeled_failure_rate=modeled_failure_rate,
                absolute_error=absolute_error,
                tolerance=self.failure_rate_tolerance,
                sample_size=len(self.target),
            )
        except Exception as error:
            raise ModelEvaluationError(
                "Generated model failed evaluation: "
                f"{type(error).__name__}: {error}"
            ) from None


DEFAULT_DATASET = Path("res/ai4i2020.csv")
DEFAULT_RESEARCHER_MODEL = "HuggingFaceTB/SmolLM3-3B"
DEFAULT_RESEARCH_TIME_LIMIT_SECONDS = 300.0
MAX_TOOL_TURNS = 20
MAX_CONTEXT_TURNS = 3

SYSTEM_PROMPT = """\
You are a data researcher building a probabilistic model of equipment failure.
Investigate the user's question using the pandas
DataFrame `df`, which contains the input CSV, and the pandas module `pd`.

Use Python in <code>...</code> blocks to inspect or analyze the data. Variables
persist between calls. You must call `model_builder(description)` to propose a
probabilistic model. It measures the generated model's failure rate against the
failure rate in the data and reports whether the model is within tolerance.
The generated model itself is never returned to you.
Inspect the data before assuming column names, types, ranges, or distributions.
If a model does not pass, use the measured rates and your data analysis to
revise its description and try again. Do not give a final answer until a model
has passed. Once one passes, give a concise, evidence-based answer without
calling another tool.
"""

MODEL_BUILDER_TOOL = {
    "name": "model_builder",
    "description": (
        "Build a probabilistic model and measure its failure rate against the data"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "The problem and all relevant evidence from the CSV",
            }
        },
        "required": ["description"],
    },
}

_TOOL_PATTERN = re.compile(
    r"<code>(?P<code>.*?)</code>|<tool_call>(?P<tool>.*?)</tool_call>",
    re.DOTALL,
)


class PythonSession:
    def __init__(
        self,
        data: pd.DataFrame,
        model_builder: Callable[[str], ModelEvaluation],
    ) -> None:
        self.namespace: dict[str, Any] = {
            "df": data,
            "pd": pd,
            "model_builder": model_builder,
        }

    def execute(self, source: str) -> str:
        tree = ast.parse(source, mode="exec")
        body = tree.body
        final_expression = None
        if body and isinstance(body[-1], ast.Expr):
            final_expression = ast.Expression(body.pop().value)

        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            if body:
                exec(
                    compile(
                        ast.Module(body=body, type_ignores=[]),
                        "<researcher>",
                        "exec",
                    ),
                    self.namespace,
                )
            result = (
                eval(compile(final_expression, "<researcher>", "eval"), self.namespace)
                if final_expression
                else None
            )

        text = output.getvalue()
        if result is not None:
            text += repr(result)
        return text or "Code executed successfully with no output."


class Researcher:
    def __init__(
        self,
        model_name: str,
        model_builder: Callable[[str], ModelEvaluation],
        data: pd.DataFrame,
        device: str = "auto",
        max_new_tokens: int = 8_192,
        time_limit_seconds: float = DEFAULT_RESEARCH_TIME_LIMIT_SECONDS,
    ) -> None:
        if not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
            raise ValueError("Research time limit must be a positive finite number")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map=device,
        )
        self.max_new_tokens = max_new_tokens
        self.time_limit_seconds = time_limit_seconds
        self.evaluations: list[ModelEvaluation] = []

        def evaluate_model(description: str) -> ModelEvaluation:
            evaluation = model_builder(description)
            self.evaluations.append(evaluation)
            return evaluation

        self.python = PythonSession(data, evaluate_model)

    def _generate(
        self,
        messages: list[dict[str, str]],
        max_time: float,
    ) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=True,
            xml_tools=[MODEL_BUILDER_TOOL],
            python_tools=[MODEL_BUILDER_TOOL],
            return_tensors="pt",
            return_dict=False,
        ).to(self.model.device)
        context_limit = getattr(
            self.model.config, "max_position_embeddings", 65_536
        )
        available_tokens = context_limit - input_ids.shape[-1]
        if available_tokens <= 0:
            raise ValueError("The conversation exceeds the model context window")

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=min(self.max_new_tokens, available_tokens),
            max_time=max_time,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(
            outputs[0][input_ids.shape[-1] :],
            skip_special_tokens=True,
        )

    def _execute_tool(self, match: re.Match[str]) -> str:
        if code := match.group("code"):
            return self.python.execute(code.strip())

        call = json.loads(match.group("tool"))
        if call.get("name") != "model_builder":
            raise ValueError(f"Unknown tool: {call.get('name')!r}")
        arguments = call.get("arguments", {})
        description = arguments.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("model_builder requires a non-empty description")
        return str(self.python.namespace["model_builder"](description))

    def _remaining_time(self, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "Researcher exceeded its "
                f"{self.time_limit_seconds:g}-second time limit"
            )
        return remaining

    @staticmethod
    def _output_message(message: dict[str, str]) -> None:
        print(f"\n[{message['role']}]\n{message['content']}", flush=True)

    def __call__(self, prompt: str) -> str:
        deadline = time.monotonic() + self.time_limit_seconds
        first_evaluation = len(self.evaluations)
        initial_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        for message in initial_messages:
            self._output_message(message)

        turns: deque[list[dict[str, str]]] = deque(maxlen=MAX_CONTEXT_TURNS)

        for _ in range(MAX_TOOL_TURNS):
            messages = initial_messages + [
                message for turn in turns for message in turn
            ]
            answer = self._generate(messages, self._remaining_time(deadline))
            self._remaining_time(deadline)
            assistant_message = {"role": "assistant", "content": answer}
            self._output_message(assistant_message)
            tool_calls = list(_TOOL_PATTERN.finditer(answer))
            if not tool_calls:
                accepted = any(
                    evaluation.accepted
                    for evaluation in self.evaluations[first_evaluation:]
                )
                if accepted:
                    return answer
                continuation_message = {
                    "role": "user",
                    "content": (
                        "You have not yet produced a model that passes the "
                        "failure-rate evaluation. Continue the iterative "
                        "analysis and call model_builder."
                    ),
                }
                self._output_message(continuation_message)
                turns.append(
                    [
                        assistant_message,
                        continuation_message,
                    ]
                )
                continue

            results = []
            for tool_call in tool_calls:
                self._remaining_time(deadline)
                try:
                    result = self._execute_tool(tool_call)
                except Exception as error:
                    result = f"{type(error).__name__}: {error}"
                self._remaining_time(deadline)
                results.append(result)
            tool_message = {
                "role": "tool",
                "content": "\n\n".join(
                    f"<tool_response>\n{result}\n</tool_response>"
                    for result in results
                ),
            }
            self._output_message(tool_message)
            turns.append(
                [
                    assistant_message,
                    tool_message,
                ]
            )

        raise RuntimeError(
            "Researcher did not produce an acceptable failure-rate model "
            f"within {MAX_TOOL_TURNS} turns"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a CSV with researcher and probabilitier models."
    )
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--res-model", default=DEFAULT_RESEARCHER_MODEL)
    parser.add_argument("--prob-model", required=True)
    parser.add_argument("--target", default="Machine failure")
    parser.add_argument(
        "--failure-rate-tolerance",
        type=float,
        default=DEFAULT_FAILURE_RATE_TOLERANCE,
        help="Maximum absolute error in modeled failure rate (default: 0.01)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--research-time-limit",
        type=float,
        default=DEFAULT_RESEARCH_TIME_LIMIT_SECONDS,
        metavar="SECONDS",
        help="Maximum wall-clock time per research request (default: 300)",
    )
    parser.add_argument("--prompt", help="Run one prompt instead of interactive mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.dataset)
    probabilitier = Probabiliter(args.prob_model, device=args.device)
    model_builder = EvaluatedModelBuilder(
        probabilitier,
        data,
        args.target,
        failure_rate_tolerance=args.failure_rate_tolerance,
    )
    researcher = Researcher(
        args.res_model,
        model_builder=model_builder,
        data=data,
        device=args.device,
        time_limit_seconds=args.research_time_limit,
    )

    if args.prompt:
        researcher(args.prompt)
        return

    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt:
            researcher(prompt)


if __name__ == "__main__":
    main()
