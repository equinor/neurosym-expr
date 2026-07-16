import pytest
from lark.exceptions import UnexpectedInput

from expr.lang import parser


def parse(source):
    return parser.parse(source)


def test_model_distinguishes_random_and_computed_variables():
    tree = parse(
        """\
height ~ Normal(mean=176, sd=7.1)
threshold = height > 180
return threshold
"""
    )

    assert [child.data for child in tree.children] == [
        "random_assign",
        "assign",
        "return_stmt",
    ]


def test_multivariate_assignment_and_matrix_correlation_are_accepted():
    tree = parse(
        """\
x, y ~ multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]])
z ~ norm()
correlate [x, y, z] with [[1, 0.2, 0.3], [0.2, 1, 0.4], [0.3, 0.4, 1]]
return (x + y) + z
"""
    )

    assert [child.data for child in tree.children] == [
        "multivariate_random_assign",
        "random_assign",
        "correlate_matrix",
        "return_stmt",
    ]


def test_model_requires_a_return_value():
    with pytest.raises(UnexpectedInput):
        parse("height ~ Normal(mean=176, sd=7.1)")


@pytest.mark.parametrize(
    "expression",
    [
        "1 + (2 * 3)",
        "(1 + 2) * 3",
        "-x * y",
        "-(x * y)",
        "(a < b) and (b < c)",
    ],
)
def test_explicitly_grouped_expressions_are_accepted(expression):
    parse(f"return {expression}")


@pytest.mark.parametrize(
    "expression",
    [
        "1 + 2 * 3",
        "a < b < c",
    ],
)
def test_ungrouped_operator_chains_are_rejected(expression):
    with pytest.raises(UnexpectedInput):
        parse(f"return {expression}")


def test_unary_minus_binds_before_binary_operator():
    tree = parse("return -x ** 2")
    binary = tree.children[0].children[0]

    assert binary.data == "binary"
    assert binary.children[0].data == "neg"


def test_multiline_list_does_not_require_a_trailing_comma():
    parse(
        """\
return [
    1,
    2
]
"""
    )


def test_single_nested_list_preserves_both_list_nodes():
    tree = parse("return [[1, 2]]")

    assert len(list(tree.find_data("list"))) == 2


@pytest.mark.parametrize(
    ("categories", "expected_items"),
    [
        ('["red", "blue"]', [("string", '"red"'), ("string", '"blue"')]),
        ("[1, 2.5]", [("number", "1"), ("number", "2.5")]),
        ('["red", 2]', [("string", '"red"'), ("number", "2")]),
    ],
)
def test_categorical_lists_support_strings_and_numbers(categories, expected_items):
    tree = parse(f"choice ~ categorical({categories})\nreturn choice")
    category_list = next(tree.find_data("list"))

    assert [
        (item.data, str(item.children[0])) for item in category_list.children
    ] == expected_items


def test_positional_arguments_may_precede_keyword_arguments():
    parse("value ~ Normal(0, sd=1)\nreturn value")


def test_positional_arguments_cannot_follow_keyword_arguments():
    with pytest.raises(UnexpectedInput):
        parse("value ~ Normal(sd=1, 0)\nreturn value")


def test_function_has_only_one_parse_tree_node():
    tree = parse("return floor(1.5)")
    function = tree.children[0].children[0]

    assert function.data == "func"
    assert all(getattr(child, "data", None) != "function" for child in function.children)


def test_argument_container_rules_are_not_in_the_parse_tree():
    tree = parse("value ~ Normal(0, sd=1)\nreturn value")
    node_names = {node.data for node in tree.iter_subtrees()}

    assert "arguments" not in node_names
    assert "keyword_arguments" not in node_names


def test_literals_membership_and_logical_negation():
    tree = parse('return not (("red" in ["red", "blue"]) or false)')
    node_names = {node.data for node in tree.iter_subtrees()}

    assert "not_" in node_names
    assert "func" not in node_names
