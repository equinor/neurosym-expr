from __future__ import annotations

import argparse

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from xgrammar import TokenizerInfo, GrammarCompiler, Grammar
from xgrammar.contrib.hf import LogitsProcessor


system_prompt = """\
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

grammar = Grammar.from_ebnf(r"""
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


def get_input():
    lines = []
    while value := input("> "):
        lines.append(value)

    return "".join(lines)


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
        gcomp = GrammarCompiler(tokenizer_info)

        compiled_grammar = gcomp.compile_grammar(grammar)

        self.logits_processors = [LogitsProcessor(compiled_grammar)]

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map=device,
        )
        self.max_new_tokens = max_new_tokens

    def __call__(self, description: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
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
        outputs = outputs[0][inputs.input_ids.shape[-1] :]

        answer = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return answer


def main():

    args = argparse.ArgumentParser()
    args.add_argument("--model", required=True)
    args.add_argument("--device", default="auto")

    args = args.parse_args()

    probit = Probabiliter(args.model, args.device)

    while True:
        user_prompt = get_input()

        answer = probit(user_prompt)
        print(answer)


if __name__ == "__main__":
    main()
