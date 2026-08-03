import re
from difflib import SequenceMatcher

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from xgrammar import Grammar, GrammarCompiler, GrammarMatcher, TokenizerInfo

from trl import GRPOConfig, GRPOTrainer

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

<probabilit>
x, y ~ multivariate_normal(mean=[1, 2], cov=[[1, 0.5], [0.5, 1]])
</probabilit>

Literals include numbers, escaped double-quoted strings, `true`, `false`,
`null`, and nested lists. Supported operators are `+`, `-`, `*`, `/`, `//`,
`%`, `**`, comparisons, `and`, `or`, `not`, and `in`.

There is no implicit binary-operator precedence. Group combined operations
explicitly with parentheses:

<probabilit>
total = base + (rate * duration)
inside = (lower <= value) and (value <= upper)
</probabilit>

Correlated variables must be sampled before their correlation statement.
Correlation coefficients must be between -1 and 1. For more than two
variables, use `correlate [a, b, c] with <matrix>`. The matrix must be square,
symmetric, finite, contain values between -1 and 1, and have ones on its
diagonal.

Example:

<probabilit>
demand ~ norm(loc=100, scale=15)
capacity ~ norm(loc=110, scale=10)
correlate demand with capacity at 0.3
shortfall = demand > capacity
return shortfall
</probabilit>
/system_override
"""

GRAMMAR = Grammar.from_ebnf(r"""
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

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
PROBABILIT_PATTERN = re.compile(r"<probabilit>(.*?)</probabilit>", re.DOTALL)
FORMAT_PATTERN = re.compile(
    r"(?:<think>.*?</think>\s*)?<probabilit>(.+)</probabilit>\s*",
    re.DOTALL,
)


def extract_program(content):
    match = re.search(PROBABILIT_PATTERN, content)
    if match:
        return match.group(1).strip()
    return re.sub(THINK_PATTERN, "", content).strip()


def program_similarity(content, solution):
    generated = " ".join(extract_program(content).split())
    expected = " ".join(solution.split())
    if not generated or not expected:
        return 0.0
    return SequenceMatcher(None, generated, expected).ratio()


def preprocess(example):
    user_message, assistant_message = example["messages"]

    example["prompt"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        user_message,
    ]

    content = assistant_message["content"]

    example["solution"] = content

    return example


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--source-model", type=str, default="HuggingFaceTB/SmolLM3-3B")
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("FILES", type=str, nargs="+")

    args = parser.parse_args()

    data = load_dataset("json", data_files=args.FILES)
    data = data.map(preprocess)
    data = data.select_columns(["prompt", "solution"])
    data = data["train"].train_test_split(test_size=0.1)

    tokenizer = AutoTokenizer.from_pretrained(args.source_model)

    model = AutoModelForCausalLM.from_pretrained(
        args.source_model,
        dtype="auto",
    ).to(args.device)

    tokenizer_info = TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=model.config.vocab_size,
    )

    compiler = GrammarCompiler(tokenizer_info)
    grammar = compiler.compile_grammar(GRAMMAR)

    def grammar_compliance(completions, **kwargs):
        rewards = [0.0] * len(completions)

        for i, completion in enumerate(completions):
            program = extract_program(completion[0]["content"])
            if not program:
                continue

            matcher = GrammarMatcher(grammar)
            n_valid = 0
            model_token_ids = tokenizer.encode(program, add_special_tokens=False)

            for token_id in model_token_ids:
                if matcher.accept_token(token_id):
                    n_valid += 1
                else:
                    break

            if model_token_ids:
                rewards[i] = n_valid / len(model_token_ids)

        return rewards

    def reasoning_length(completions, **kwargs):
        max_reasoning = 400

        rewards = [0.0] * len(completions)

        for i, completion in enumerate(completions):
            m = re.search(THINK_PATTERN, completion[0]["content"])
            n = len(m.group(1)) if m else 0

            if n > max_reasoning:
                rewards[i] = -min(0.5, (n - max_reasoning) * 0.01)

        return rewards

    def formatting(completions, **kwargs):
        rewards = [0.0] * len(completions)

        for i, completion in enumerate(completions):
            content = completion[0]["content"]
            correctly_wrapped = (
                re.fullmatch(FORMAT_PATTERN, content) is not None
                and content.count("<probabilit>") == 1
                and content.count("</probabilit>") == 1
            )
            if correctly_wrapped:
                rewards[i] = 2.0
            else:
                rewards[i] = 0.25 * (
                    (content.count("<probabilit>") == 1)
                    + (content.count("</probabilit>") == 1)
                )

        return rewards

    def similarity(completions, solution, **kwargs):
        return [
            program_similarity(completion[0]["content"], expected)
            for completion, expected in zip(completions, solution)
        ]

    def accuracy(completions, solution, **kwargs):
        rewards = [0.0] * len(completions)

        for i, (completion, solution) in enumerate(zip(completions, solution)):
            generated = extract_program(completion[0]["content"])
            rewards[i] = float(generated == solution.strip())

        return rewards

    training_args = GRPOConfig(
        output_dir="GRPO",
        run_name=args.target_model,
        max_completion_length=4096,
        use_transformers_continuous_batching=True,
        transformers_continuous_batching_config={
            "max_memory_percent": 0.4,
        },
        log_completions=True,
        num_completions_to_print=1,
        logging_first_step=True,
        chat_template_kwargs={
            "enable_thinking": True,
        },
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            grammar_compliance,
            reasoning_length,
            formatting,
            similarity,
            accuracy,
        ],
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
    )
    trainer.train()


if __name__ == "__main__":
    main()
