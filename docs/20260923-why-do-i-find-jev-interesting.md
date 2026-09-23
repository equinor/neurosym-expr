# Why do I find Jev interesting?

> Author: Bjarte Johansen Date: 23. September 2026

Last week [TypeSafe.ai](https://typesafe.ai) released Jev. A
recontextualisation of an old idea: Natural Language Processing (NLP)
as classification. This how most NLP tasks were solved before GPT.
Information extraction was sequence classification/tagging and and
most other tasks were picking between a set of categories--there are a
few outliers like translation and summarisation. Chatbots and
conversational AI was seen as fringe research. The reason being that
we didn't know how to generalize language modelling and have models
that could be trained to solve more than one task at a time.

The ["Attention is all you
need"-paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
is the catalyst for changing that, but it didn't happen over night.
The first transformer models introduced in the attention paper were a
generalisation and parallelisation of the attention mechanism that was
already being used in other formalisations of language modelling. The
first models using the transformer architecture, like
[BERT](https://arxiv.org/abs/1810.04805), were pretrained models that
we then finetune for a specific task.

This turned out to be a pretty powerful shift. We could now take a
model that was first trained to predict next token probabilities
through unsupervised learning on general text and then continue to
train the model on a specific NLP task. Categorisation got reframed
as a classification over the embedding of a class token that you added
to the input sequence so that the class token could "attend" to the
full sequence and capture the true class through training.

The same model could also be finetuned to produce a sequence of
tokens, or words, through autoregressive generation. Autoregressive
means that you uses the previous tokens that the model has generated
as input to the next token generation.

OpenAI then showed us with ChatGPT that if we just trained on large
enough data, we mostly didn't even have to fine-tune. The model can be
trained through [Reinforcement Learning with Human Feedback
(RLHF)](https://rlhfbook.com) and learn to answer questions or do
tasks like categorisation--or any task--through [zero- or few-shot
learning](https://en.wikipedia.org/wiki/Few-shot_learning), i.e you
give a description of the task and zero or more examples to the model
and it generalises to solving the task for new data without any
explict finetuning.

Though the models can struggle to generalise over all tasks and data
in some situations, the latest generation of Large Language Models
(LLMs) has shown that this paradigm is powerful: Autoregressive token
generation has become the dominant method despite problems like
hallucinations.

Language model
[hallucinations](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence))
are responses from these models that contain false or misleading
information--especially claims that are not rooted in reality but are
rather an effect of token probabilities, statistical uncertainty, and
the difficult problem of making the model say "I don't know."

There have been many attempt to solve this particular problem and most
of the contemporary LLMs give you the ability to advantage of one or
more of these techniques. "Reasoning" being the most general attempt
to mitigate model mistakes.

A very nearby method to prevent these types of mistakes is by either
forcing or suggesting that the model should follow a grammar: F.ex,
you can say that through the token generation the model can only
output a next token if it continues to be a valid python program (up
to that token). The issue with this approach is that the next valid
python token might be a low probability token and this can take the
model out into uncertain areas of the model latent space and even
though the program is valid syntactically, it can still be
semantically invalid and hallucinated.
[Xgrammar](https://xgrammar.mlc.ai) provides a way to efficiently
filter which tokens are allowed given a context-free grammar and has
been adopted by Google, DeepSeek, NVIDIA and others.

OpenAI also provides similar functionality through their [custom tool
api](https://developers.openai.com/api/docs/guides/function-calling?api-mode=responses#context-free-grammars)
and [structured
output](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)
is a particular instance of this technique that produces JSON that
follows a JSON spec definition.


[Steering vectors](https://arxiv.org/html/2502.18862v2) are another
attempt at controlling model behaviour and preventing both
hallucinations and bad behaviour. The idea is that unwanted behaviour
could be thought of as a direction in the latent space of the model,
and if we prevent the model from moving in that direction--or if it
moves in the opposite direction--we should be able to prevent this
behaviour.

Another method that has been developed to reduce hallucination and let
the model explore is "reasoning". Reasoning is where you train a large
language model to follow a set of actions to search its own [latent
space](https://en.wikipedia.org/wiki/Latent_space). This is done by
training the model through reinforcement learning to find a way to
move the model parameters into a configuration to answer your
question.

The latent space of a language model is where the model represents our
input by long sequences of numbers (or vectors). Through the
manipulation of the latent representation of our input the model finds
an area in the latent space that has a high probability to answer your
question. It then decodes this new position into a token that we can
read. It does this until it runs out of space or it decodes a position
that tells the process to stop.

Normally when we train a model to learn a task we give it one and one
example and check if it can correctly complete the task. If it cannot
do that correctly we calculate the difference between what it did and
the correct conclusion. We use that difference to find the direction
we need to move the parameters of our model in to get the correct
answer. We then move the parameters a little in that direction. In
relation to LLMs, this is called [Supervised Fine-Tuning
(SFT)](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)) and
rely on
[backpropagation](https://en.wikipedia.org/wiki/Backpropagation) and
[gradient descent](https://en.wikipedia.org/wiki/Gradient_descent).
SFT needs examples of the given task that where we have annotated the
true result that we expect.

[Reinforcement learning
(RL)](https://en.wikipedia.org/wiki/Reinforcement_learning) does not
explicitly require an expected result. It instead reformulates
learning as a game. It strictly only requires a way to define actions
that can be taken as a transition between states and a way to measure
if the goal has been achieved. The model then learns a policy, or a
set of actions, it can take to achieve the goal. Normally, to avoid
situations where the learner starts very far away from the goal and RL
can collapse into a random search, we add a reward function that
optimally tells the learner if it is moving in the right direction,
but often is a heuristic that we believe will let the learner move
towards the goal.

When we train reasoning models we set the goal as the correct answer
to a question, the actions the model can take are generating a token
and the reward function measures closeness to the target output after
we stop reasoning and generate a text.

To actually calculate the update to the model you need a critic to
evaluate the actions and which parameters should update. The critic
model would be approximatly the same size as the model you were
training. It was DeepSeek who showed us [Group Relative Policy
Optimization (GRPO)](https://arxiv.org/pdf/2402.03300) and a way to
remove the critic from the learning process and we saw an explosion in
reasoning models and their performance as GRPO opened up the
possibility training the mdoels through Reinforcement Learning with
Verifiable Rewards (RLVR).

One of the key problems that all of these methods are pointing to is
that it is very difficult to, and actually
[impossible](https://www.nature.com/articles/d41586-025-00068-5), to
completely remove hallucinations from LLMs. Models are trained to be
confident, not just in the voice that they project in their output,
but in how they represent their token probabilities as well. The
training goals of the model in SFT and RLVR is to force the logit
representation of the training goal as high as possible. Logits are
the unnormalised values the model produces that are turned into
normalized probability values. If we want to be able to know and
represent the uncertainty of the model we need the model to keep the
information about the probability distribution of what it has been
trained on. The models are literally trained to be confident in all
predictions. If we want to investigate the true confidence the model
has in its output, we need to sample many times. This can get very
expensive when you are trying to automate the processing of millions
of document.

TypeSafe.ai seems to have been struggling with all of these issues and
thinking about how can we make AI more robust? How can we make the
uncertainty of a model cheap to calculate so that it can be used in
automated decision making? How can we make decisions fast?

Autoregression is inherently slow as every token is dependent on the
previous tokens. SFT, RLHF, and RLVF are problematic because we lose
the ability to easily reason about uncertainty.

Others have also been thinking about these problems as well and
TypeSafe.ai is not the first to talk about Reinforcement Learning for
Calibrated Decisions, or [Calibrated
Confidence](https://arxiv.org/html/2503.02623v6), and strictly proper
scoring rules.

Their innovation lies in the observation that we generalised NLP
through language models, but they are slow and costly. What if we used
the parts of the same technology, and the new things we have learned
about reinforcement learning and scaling to generalise categorisation?
Many (or maybe even most) NLP tasks can be thought of as
categorisation tasks and instead of finetuning a specific model to
solve a task, we can just zero- or few-shot solve categorisation.

Another reason why I like this approach is that it gives a framework
for how to think and solve tasks. It makes you ask the question: What
are the smallest and simplest questions I can ask to solve this task?
