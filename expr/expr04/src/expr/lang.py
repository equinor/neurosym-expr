import json
import logging
import operator
from dataclasses import dataclass
from typing import Any

import numpy as np
from lark import Lark, Transformer, logger, v_args
from lark.exceptions import VisitError
from probabilit import (
    All,
    Any,
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


grammar = """\
start: _NL* (instruction _NL+)* return_stmt _NL*

?instruction: VAR "~" distribution -> random_assign
            | variable_tuple "~" distribution -> multivariate_random_assign
            | VAR "=" expr -> assign
            | correlate

variable_tuple: VAR ("," VAR)+

return_stmt: "return" expr

?expr: unary
     | unary OPERATOR unary -> binary

?multiline_expr: unary
               | unary OPERATOR _NL* unary -> binary

?unary: "-" unary -> neg
      | _NOT unary -> not_
      | primary

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

OPERATOR: "+"
        | "*"
        | "and"
        | "or"
        | "%"
        | "/"
        | "**"
        | "-"
        | "=="
        | "<"
        | "<="
        | ">"
        | ">="
        | "!="
        | "//"
        | "in"

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

parser = Lark(grammar, parser="lalr")


class ModelConversionError(ValueError):
    """Raised when valid language syntax cannot be converted to a model."""


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
            (statement for statement in statements if isinstance(statement, _Return)),
            None,
        )
        if returned is None:
            raise ModelConversionError("The model does not return a value")

        result = (
            returned.value
            if isinstance(returned.value, Node)
            else Constant(returned.value)
        )
        for statement in statements:
            if not isinstance(statement, _Correlation):
                continue
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
        matrix = np.array([[1.0, value], [value, 1.0]])
        return self._correlation(names, matrix)

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
            if has_node:
                return Equal(_probabilit_operand(left), _probabilit_operand(right))
            return left == right
        if symbol == "!=":
            if has_node:
                return NotEqual(_probabilit_operand(left), _probabilit_operand(right))
            return left != right
        if symbol == "and":
            return (
                All(_probabilit_operand(left), _probabilit_operand(right))
                if has_node
                else np.logical_and(left, right)
            )
        if symbol == "or":
            return (
                Any(_probabilit_operand(left), _probabilit_operand(right))
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
    """Convert a parse tree into a lazy probabilit computational graph."""
    try:
        return _ModelTransformer().transform(tree)
    except VisitError as error:
        if isinstance(error.orig_exc, ModelConversionError):
            raise error.orig_exc from None
        raise


def parse_model(source: str) -> Node:
    """Parse source code and convert it into a probabilit model."""
    return model_from_tree(parser.parse(source))


def main() -> None:
    logger.setLevel(logging.WARN)

    text = """
    roll1 ~ uniform(loc=1.0, scale=0.5)
    roll2 ~ uniform()
    die1 = floor(1 + (roll1 * 6))
    die2 = floor(1 + (roll2 * 6))
    return die1 == die2
    """

    model = parse_model(text)
    print(model)
    print(model.sample(10, random_state=42))


if __name__ == "__main__":
    main()
