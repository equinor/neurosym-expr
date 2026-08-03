from expr06.train import extract_program, program_similarity


def test_extract_program_from_wrapped_completion():
    content = (
        "<think>brief reasoning</think>\n"
        "<probabilit>\n"
        "x ~ norm(loc=0, scale=1)\n"
        "return x\n"
        "</probabilit>"
    )

    assert extract_program(content) == "x ~ norm(loc=0, scale=1)\nreturn x"


def test_extract_program_allows_unwrapped_cold_start_completion():
    content = "<think>brief reasoning</think>\nx ~ norm(loc=0, scale=1)\nreturn x"

    assert extract_program(content) == "x ~ norm(loc=0, scale=1)\nreturn x"


def test_program_similarity_provides_partial_reward():
    solution = "x ~ norm(loc=0, scale=1)\nreturn x > 0"
    completion = "<probabilit>x ~ norm(loc=0, scale=1)\nreturn x</probabilit>"

    similarity = program_similarity(completion, solution)

    assert 0.0 < similarity < 1.0
