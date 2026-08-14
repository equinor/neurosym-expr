#  Structured output and categorisation

> Author: Bjarte Johansen Date: 14. August 2026

Most resources on using function calling and structured output is
focused on getting results for an interactive user. The user wants to
ask what the temperature is in Paris or find the orders for a
particular customer in the database.

However, these neurosymbolic techniques are particularly useful in
categorisation as well. We are going to look at a few examples.

By neurosymbolic techniques we mean techniques where we use
algorithms, and especially classical or symbolic AI, to guide and
augment machine learning models like LLMs. Function calling,
structured output and grammars give structure, actions and
formalism to the probabilistic nature of LLMs.

To get an understanding of how this work, we can look at this example
using pydantic and the OpenAI `responses.parse` API to make a simple
and open query to categorize a message:

```python
from openai import OpenAI

from pydantic import BaseModel


client = OpenAI()

class Output(BaseModel):
    category: str


system_prompt = """\
Categorize the users message.
"""

user_prompt = """\
What is your purpose?
"""

response = client.responses.parse(
    model="gpt-5.6",
    input=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    text_format=Output,
)

category = response.output_parse.category
print(category)

```

This defines a pydantic model that we strictly ask the model to
follow. What happens is that the model is converted to a json schema
and the model (I expect) is not only trained to give json output in
these scenarios, but explicitly and mechanically forced to follow the
grammar defined by our json schema: The output has to be a json object
with a a key called "category" and a value that is a string and not
another type.

It is important to note that even though we might not be using the
model for a chat interface, the API we have access to is still
functionally built for that.

Open categorisation, like we do in the example above, is often not
very useful or effectful--but there are many things we can do to
restrict the model and give it direction. In open categorisation are
basically asking the model to come up with _any_ category that it sees
fit. It could put any text in that string. This means that if we would
run this over thousands of examples, it would be difficult to predict
how the output would look like.

Normally, we want to at least put are data into a particular domain.

```python
from pydantic import Field

class Sentiment(BaseModel):
    category: str = Field(description="The sentiment of the text")
```

We can update both the class name and give the category attribute a
description to signify the domain. In most categorisation or
classification tasks we also know what classes we want to put the
texts into:

```python
from typing import Literal

Tone = Literal["positive", "negative", "neutral"]

class Sentiment(BaseModel):
    category: Tone = Field(description="The sentiment of the text")
```

Here we use a Literal to say that the category attribute can only be
the labels 'positive', 'negative' or 'neutral'. It is possible to use an
`enum.Enum` as well if you prefer.

Even categories that seem naturally understandable, like in our
sentiment example above, can contain ambiguity. For example if a user wrote

> The interface looks great, but I still cannot complete checkout.


We could reasonably choose any of the 3 labels:
    
- `positive` :: as the user praised the interface,
- `negative` :: as the user because they are blocked to complete
  their task,
- `neutral` :: because the statement contains both postive and
   negative sentiment.

Depending on what we want to achieve with the classification, we could
(try) to resolve the ambiguity by adding explanations of the labels and
instructions for how to use them to the system prompt:

```text
Classify the overall sentiment expressed by the author toward the product or
experience.

Categories:
- positive: The author's overall evaluation is favorable. Minor criticism may
  be present, provided it does not outweigh the praise.
- negative: The author's overall evaluation is unfavorable, or the author
  reports a serious unresolved problem that prevents their main goal.
- neutral: The author expresses no clear evaluation, provides primarily factual
  information, or expresses positive and negative views of roughly equal
  importance.

Decision rules:
1. Classify the author's expressed attitude, not whether the described event is
   objectively good or bad.
2. Consider the entire message rather than individual sentiment words.
3. Functional blockers outweigh cosmetic praise.
4. Do not treat the absence of praise as negative.
5. Questions and factual statements are neutral unless they express a clear
   attitude.
6. When positive and negative evidence is balanced, choose neutral.
7. Do not infer sentiment that is not supported by the text.
```

The goal of the system prompt is not just to explain the task, but to
put the model into the right context and embedding space to be able to
complete the task. Every word and phrasing that you choose affects the
starting point of the model and by extension the results. It is
therefore important to be careful to use the right words and phrases
to get the best results. I find that it is often useful to ask the
base model you are working on to generate a system prompt.

If you are not working on this type of task daily, the language you or
I would choose is probably pretty far from the average language for
solving the task that you are trying to solve. It can therefore be
beneficial to use the model to move your language and ideas towards
that average as that is probably where you will get the best results.

You should think about what the purpose of the task is. Are you classifying
    - The user's overall emotional tone;
    - their opinion of the product;
    - or their satisfaction with a particular interaction?

Dependent on the reason for classifying the sentiment of the user's
texts, you should adjust the instructions to the model.

When the model misclassifies an example, it can be difficult to know
why the model did that. It can therefore be beneficial to ask for a
reason by adding another attribute to the pydantic model:

```pydantic
class Sentiment(BaseModel):
    category: Tone = Field(description="The sentiment of the text")
    reason: str = Field(description="The reason for the given sentiment category")
```

This can give a view into which embedding space the model is in and
what you should change in your messages to steer it in the right
direction.

You might even want to add metadata, change the instructions or
prompts dependent on the text you want to categorize. Instead of
asking the model to classify the user's message, I will instead ask in
the user message to categorize a particular text or document.

```text
<metadata>
user_often_uses_sarcasm: true
</metadata>

<document>
The interface looks great, but I still cannot complete checkout.
</document>

What is the category of the document?
```

I have found that the models generally get bad at categorising when
there more than 10 categories and completely fall apart when there are
100s or 1000s of categories. In these instances we need techniques to
reduce the number of categories any given document can be categorised
as. Some of the things I have had success with in these instances is
to embed my training set and find which 5-10 category centroids my new
document is closest to and let the model decide between those than all
1000 of them.

Pydantic doesn't support this type of dynamic binding of attribute
types, but if we use the underlaying `responses.create` api and define
our json schema manually, we can get around this constraint:

```python
class Sentiment(BaseModel):
    category: str
    reason: str

labels = ['positive', "negative', 'neutral']

response = client.responses.create(
    input=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    text={
        "format": {
            "name": "Sentiment",
            "type": "json_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "category": {
                        "description": "Sentiment category the document belongs to",
                        "enum": labels,
                        "type": string,
                    },
                    "reason": {
                        "description": "Reason for the choosing the sentiment category",
                        "type": "string",
                    },
                },
                "required": [
                    "category",
                    "reason",
                ],
            },
        }
    },
)
output = response.output[1].content[0].text
output = Sentiment.model_validate_json(output)
print(output.category)
```

Above, we can see how we can use `responses.create` to get the exact
same output, except that the labels are not defined as a type on the
Sentiment.category attribute.

This means that if we had more fine-grained labels and by some other
process we knew that the document could only be positive we set the
labels to something like this instead:

```python
labels = ['somewhat positive', 'positive', 'very positive']
```

Using the same type of technique, you can also affect what the string
can contain through a regex. I have used this to build a taxonomic
hierarchy pr document.

```json
"properties": {
    "hierarchy": {
        "description": "The class hierarchy of the part",
        "type": "array",
        "items": {
            "type": "string",
            "pattern": "^([A-Z]+[a-z]+)+$",
        }
    }
}
```

Here we can see a partial definition where we say that a class
hierarchy is an array of strings that are CamelCased that can be
empty.

In this case it is important that the hierarchy can be empty as the
content could potentially not contain any parts. In my experience, you
should always allow the model an escape hatch and allow an unknown or
empty label. In the sentiment case we could consider `neutral` as this
case or we could also put an `unknown` case. The reason for doing this
is that otherwise we can end up with a lot of noice in our output if
the model is forced to choose a category when it is not possible.
