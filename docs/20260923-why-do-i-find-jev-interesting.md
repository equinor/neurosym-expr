# Why do I find Jev interesting?

> Author: Bjarte Johansen Date: 23. September 2026

Last week [TypeSafe.ai](https://typesafe.ai) released Jev. A
recontextualisation of an old idea: Natural Language Processing (NLP)
as classification. This is how most NLP tasks were solved before GPT.
Information extraction was sequence classification/tagging and most
other tasks were picking between a set of categories--there are a few
outliers like translation and summarisation. Chatbots and
conversational AI was seen as fringe research. The reason being that
we didn't know how to generalise language modeling and have models
that could be trained to solve more than one task at a time.

The ["Attention is all you
need"-paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
was the catalyst for changing that, but it didn't happen over night.
The first transformer models introduced in the attention paper were a
generalisation and parallelisation of the attention mechanism that was
already being used in other formalisations of language modeling. The
most popular architecture at the time being
[LSTMs](https://en.wikipedia.org/wiki/Long_short-term_memory). The
first models using the transformer architecture, like
[BERT](https://arxiv.org/abs/1810.04805), were pretrained models that
we then finetune for a specific task.

This was a pretty powerful shift. We could now take a model that was
first trained to predict next token probabilities through unsupervised
learning on general text and then continue to train the model on a
specific NLP task. Categorisation got re-framed as classification over
the embedding of a class token/ The token was added to the input
sequence so that the class token could "attend" to the full sequence
and capture the true class through training.

The same model could also be finetuned to produce a sequence of
tokens, or words, through autoregressive generation. Autoregressive
means that you uses the previous tokens that the model has generated
as input to the next token generation.

OpenAI then showed us with ChatGPT that if we just trained on large
enough data, we mostly didn't even have to finetune. The model can be
trained through [Reinforcement Learning with Human Feedback
(RLHF)](https://rlhfbook.com) and learn to answer questions or do
tasks like categorisation--or any task--through [zero- or few-shot
learning](https://en.wikipedia.org/wiki/Few-shot_learning), i.e you
give a description of the task and zero or more examples to the model
and it generalises to solving the task for new data without any
explicit finetuning. This is basically what [prompt
engineering](https://en.wikipedia.org/wiki/Prompt_engineering) is.

Though the models can struggle to generalise in some situations, the
latest generation of Large Language Models (LLMs) has shown that this
paradigm works: Autoregressive token generation has become the
dominant method despite problems like hallucinations.

Language model
[hallucinations](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence))
are responses from these models that contain false or misleading
information--especially claims that are not rooted in reality but are
rather an effect of token probabilities, statistical uncertainty, and
the difficult problem of making the model say "I don't know."

There have been many attempt to solve this particular problem and most
of the contemporary LLMs give you the ability to advantage of one or
more of these techniques.

A very nearby method to prevent these types of mistakes is by either
forcing or suggesting that the model should follow a grammar: F.ex,
you can say that through the token generation the model can only
output a next token if it continues to be a valid python program (up
to that token). The issue with this approach is that the next valid
python token might be a low probability token. This can take the
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

Maybe the most general method that has been developed to reduce
hallucination and let the model explore is "reasoning". Reasoning is
where you train a large language model to follow a set of actions to
search its own [latent
space](https://en.wikipedia.org/wiki/Latent_space). This is done by
training the model through reinforcement learning to find a way to
move the model parameters into a configuration that will answer your
question.

The latent space of a language model is where the model represents our
input by long sequences of numbers (or vectors). Through the
manipulation of the latent representation of our input, the model finds
an area in the latent space that has a high probability to answer your
question. It then decodes this new position into a token that we can
read. It does this until it runs out of space or it decodes a position
that tells the process to stop.

Normally when we train a model to learn a task we give it one and one
example and check if it can confidently complete the task. If it
cannot do that correctly we calculate the difference between what it
did and the correct conclusion. We use that difference to find the
direction we need to move the parameters in to get the correct answer.
We then move the parameters a little in that direction. In relation to
LLMs, this is called [Supervised Fine-Tuning
(SFT)](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)) and
rely on
[backpropagation](https://en.wikipedia.org/wiki/Backpropagation) and
[gradient descent](https://en.wikipedia.org/wiki/Gradient_descent).
SFT needs examples of the given task where we have annotated our
expected result.

[Reinforcement learning
(RL)](https://en.wikipedia.org/wiki/Reinforcement_learning) does not
explicitly require an expected result. It instead reformulates
learning as a game. It strictly only requires a way to define actions
that can be taken as a transition between states and a way to measure
if the goal has been achieved. The model then learns a policy, or a
set of actions, it can take to achieve the goal. Normally, to avoid
situations where the learner starts very far away from the goal, and
RL can collapse into a random search, we add a reward function that
tells our model if it is moving in the right direction. Hopefully we
can find a reward function that is always correct, but most often the
reward function is a heuristic that we believe will let the model move
towards our goal.

When we train reasoning models we set the goal as the correct answer
to a question, the actions the model can take are generating a token
and the reward function measures closeness to the target output after
we stop reasoning and generate a text.

To actually calculate the update to the model you need a critic to
evaluate the actions and which parameters should update. The critic
model would be approximately the same size as the model you were
training. It was DeepSeek who showed us [Group Relative Policy
Optimization (GRPO)](https://arxiv.org/pdf/2402.03300) and a way to
remove the critic from the learning process and we saw an explosion in
reasoning models and their performance as GRPO opened up the
possibility training the models through Reinforcement Learning with
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
normalized probability values for the tokens.

 If we want to be able to know and represent the uncertainty of the
model we need the model to keep the information about the probability
distribution of what it has been trained on. However, the models are
literally trained to be as confident they can be in all predictions.
If we want to investigate the true confidence a model has in its
output, we need to sample form it many times. This can get very
expensive when you are trying to automate the processing of millions
of document.

TypeSafe.ai seems to have been struggling with all of these issues and
thinking about how can we make AI more robust? How can we make the
uncertainty of a model cheap to calculate so that it can be used in
automated decision making? How can we make decisions fast?

Autoregression is inherently slow as every token is dependent on the
previous tokens. SFT, RLHF, and RLVF are problematic because we lose
the ability to easily reason about uncertainty.

Others have also been thinking about these problems and
TypeSafe.ai is not the first to talk about Reinforcement Learning for
Calibrated Decisions, or [Calibrated
Confidence](https://arxiv.org/html/2503.02623v6), and [strictly proper
scoring rules](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).

Their innovation lies in the observation that we generalised NLP
through language models, but they are slow and costly. What if we used
the parts of the same technology, and the new things we have learned
about reinforcement learning and scaling to generalise categorisation?
Many (or maybe even most) NLP tasks can be thought of as
categorisation tasks and instead of finetuning a specific model to
solve a task, we can just zero- or few-shot learn any categorisation
task.

Another reason why I like this approach is that it gives a framework
for how to think about and solve problems. It makes you ask the
question: What are the smallest and simplest questions I can ask to
solve this task? What order do they need to be asked in? Before we
would have to make a classifier for each of the questions we wanted to
ask or use an autogregressive model that becomes slower and more
costly the more branches there are in the process. It can also be
difficult to say something about the performance or at what rate does
the different parts fail.

The Jev approach allows us to iterate faster and focus on the details
of the problem; and since we get a view of the uncertainty we can know
at what points we need to put more effort.


# How do I believe Jev is implemented?

Jev is generalized categorisation. The models key properties are that
it is fast, parallel, and calibrated for expressing uncertainty about
its output categories--it is also cheap. Fast means that a query takes
a short time to compute (they claim 70-500ms end-to-end), parallel
means that it can calculate the answer to many questions about the
same input at the same time, and calibrated means that you do not
just get the models categorisation; you also get to know the
uncertainty of the categorisation.

TypeSafe.ai's API introduces 2 concepts: state and questions.

The state is an object that you want to ask questions over. It can be
a single string or a JSON object (where all values are strings).

If you read the API it becomes a point that this can be a JSON object
and different questions can filter on that object, but this is not
important to the inner workings of the model. To simplify the
description we will just consider the state as the document we want to
ask questions about.

The API describes 3 types of questions: choice, score, and noul. They
claim these as primitives, but as we will see--I believe there
actually is just 1.

- Choice :: Select a single option from a defined set of options. E.g
  if we want to classify reviews into one of negative, neutral,
  and positive.
- Score :: A rating against ordered _descriptive_ levels. E.g from a
  score of "Hate it" to "Do not care" to "Love it", use a
  sliding scale from 1-3 to express the users feeling towards
  football.
- Noul :: A yes-no question, E.g: Does the writer like football? Choose
  Yes or No.

Choice returns one of the options, the score returns a rating value,
and the noul returns a boolean. All of them return the probabilities
for each of the options and the models confidence in the response.

You can ask as many questions as you like to a single state(, up to
probably standard http request limits).

The question can be followed by detailed instructions and each choice
option, score level, or boolean value should be followed by a
description of what it means to choose that option. There can be 255
choices and 10 levels on a score.

255 is only one off 2^8 and I don't think that is a coincidence. I
believe they have a hidden "I do not know" category that they add to
all choices. I think the reason is the same reason we do that when we
use autogregressive models to do categorisation: So that you do not
force the model to set a category when non of them fit. Especially
since the goal of Jev is to arrive at a generalised categorisation
model.

(I wonder if they need to explicitly train for the unknown category or
if that happens naturally through the reinforcement learning
process--later we will see that the unknown category might not be
necessary. Maybe they just leave out the correct answer?)

TypeSafe.ai says the model is a new architecture, but they also say
they are "primarily a data research lab." Though I think there might
be some new ways of thinking about how to process the questions in
parallel and how to get information out of a transformer model, I am
quite sure that their base model is, infact, some kind of
attention-based transformer model. Probably based on an open weight
model.

What a
[transformer](https://poloclub.github.io/transformer-explainer/) does
is to first embed the input tokens, the token embeddings are then
processed through multi-head attention. Each head has learned what
part of the embedding to attend to and how to relate that information
to the task it is solving. It is not always possible to say what an
attention head is trying to solve, but it could be something abstract
like which name the pronouns in the document are referring to--Without
going too deep in the transformer architecture, each attention layer
consists of a query, key, and value. The key and value can be
[cached](https://alechelbling.com/visualizations/kv-cache/) (KV cache)
and reused along the causal axis of the transformer processing path,
i.e. from the first token to the last. Why I say that it is causal is
because the next token cannot affect the value of any of the previous
tokens, it can only affect the value result of the next token.

The reason I mention this is that I believe this is how Jev allows for
the parallel processing of questions. They compute the KV cache and it
can be propagated along each question in parallel; without any question
affecting the result of any of the other questions. This also means
that we do not have to train the model to learn to answer (virtually)
more than 1 question at a time. This saves us having to teach the
model how to differentiate between different questions and answers
because the model only ever sees one at a time.

I believe there are two options for how the model makes a decision.
The first is that there could be a feed-forward neural network that
takes the last hidden state of the transformer and has an output of
256 categories. The second, which is the more likely one in my
opinion, is that they have added 256 new special tokens to the
transformer. This is very similar to how we used to use the CLS token
in BERT to finetune for specific classification tasks.

The reason why I believe there are 256 tokens is that you want to have
a way to relate the options and the answer. What I would do in the
serialisation of the question is to add one of these tokens to the
beginning of the each option. It will also allow us to filter the
token logit output to just the 256 classification tokens. I would also
allow the output token to attend to the full network instead of just
the last hidden state.

Since we don't have to process more than one question at a time, we
also only have to output one token as we are only giving one answer
and that token is a strict choice between the (current) classification
tokens. This is how Jev is fast. It doesn't have to go through the
autogregressive step(s) to output token by token and only processes
forward through the network once.

Since we also have the probability of each of the potential tokens, we
can calculate the confidence that the top probability is correct. This
is also where TypeSafe.ai's point about SFT, RLHF, and RLVR comes in.
These techniques will result in very "spiky" logits around the
training target and it will not give a good representation of what the
distribution is. The reason this happens is because the training is
explicitly trying to maximize the logits of correct answers and is not
trying to calibrate for the potential that the prediction is wrong. Many
have tried to use the logits as a measure of confidence (ex.
[1](https://arxiv.org/abs/2205.09310),
[2](https://arxiv.org/abs/2305.15508),
[3](https://proceedings.mlr.press/v216/ye23a.html)), but every choice
is always very confident or the model has totally collapsed and output
tends towards random.

I see this as _the_ key insights that TypeSafe.ai had while developing
Jev.

To make automatic and good decisions, we need to be able to evaluate a
reasonable estimate of the confidence that we have found the true
answer. We need to be able to say if we believe the input conforms to
the distribution of things we have seen or that we need to be careful
when it is outside of that distribution.

I am not aware of a loss function that could be used with SFT to solve
this issue; I think that is why they have used RL. Through RL the
model can test different outputs--also wrong ones--and by choosing an
appropriate strictly proper scoring rule we can strongly penalize the
model for confidently wrong answers and give a softer penalization if
the model is wrong but shows a low confidence. If the model cannot try
every option, this penalization and correction cannot happen. It would
also lead to a combinatorial explosion if we would train it for every
option.

I said earlier that I believe there is only one primitive
categorisation function. There might be slightt variances in the
reward function that considers the type, but all of them use the
categorisation tokens to make the decision.

The choice function is easy to understand: 1 token pr choice. The
score function probably uses multiple tokens pr level to get a sliding
score. The noul is just a special case of the choice function. (It
could be that there is further special case for the noul where they
use [BinaryPPO](https://arxiv.org/html/2602.02708v1) as the optimiser.)

To avoid overfitting and that the model learns spurious relationships
between the order of the categories etc, we also need to randomly
select the tokens for each task. We should also only optimise for the
tokens that are used (for that categorisation), and not all tokens
every time.

This is also why I believe the categorisation tokens are injected into
the context as markers for the actual questions. It gives the model a
target and an easy optimisation path. This could also be one of the
motivations for the strict schema: So they can put their special
tokens in the right place. There could be more special tokens that
mark the begining of the state or instructions etc. They would do this
to help the model reach an optimum faster as you are able to keep more
of the structural information and do not have to learn that for every
possible way a user of TypeSafe.ai could think of using their model.

The last problem that we have to solve is that large penalties for
being confidently wrong might lead to overfitting and eventual model
collapse. By forcing a confident model to calibrate to uncertainty we
might remove the
[anisotropic](https://en.wikipedia.org/wiki/Anisotropy), or spiky,
features of the weights of the model. In the same way that we want to
get a better distribution of the logits given the confidence of the
model, we want the weights to form a less spiky latent space
[manifold](https://en.wikipedia.org/wiki/Manifold). If we move the
model too fast we might do better on one question, but we destroy the
models ability to evaluate the next question.

We have to regularise the parameter update, but there is already a
known technique to deal with this problem: [Kullback–Leibler
divergence
penality](https://mbrenndoerfer.com/writing/kl-divergence-penalty-rlhf-training).
Kullback-Leibler divergence measure the distance between two
probability distributions and it can be used as a penalisation in RL.
It makes sure that the model's fundamental behaviour stays
intact--that the weights don't move away from their original positions
too fast and the model ends up forgetting how to evaluate language.

An interesting thing is that the JEV model is not constrained to the
exact pattern that we see from TypeSafe.ai, even if we want to keep
the core API. F.ex, if we do not care about speed, we could add
reasoning before we categorise.

TypeSafe.ai has an obvious focus on speed to first "token", but if we
wanted a more rigorous approach we could add a Rational-Quadratic
Spline to the logits to make the output an explicit distribution. This
would allow us to calculate the Baysian uncertainty and potentially
remove the need for the "Unknown" token (that I believe is there).

# A fast way to implement Jev

We could take an open weight model, add 256 tokens, freeze the weight
(except we should probably learn the embeddings for the 256 tokens),
use [Low-Rank Adaption (LoRA)](https://arxiv.org/abs/2106.09685) to
reduce the number of learnable parameters, and train it through GRPO
with a [Brier score](https://en.wikipedia.org/wiki/Brier_score) as the
reward function. We would also have to generate some data, but we
could base the data on categorisation, score giving, and binary
question answering NLP tasks. There are many such data sets available.
This would be a fast way to validate that we have found a right
direction.

There are some problems with the LoRA approach, and we might end up
with a multilayer perceptron or something similar on the hidden state
instead of the 256 tokens, but I don't believe this is very far away
from the architecture of how Jev is implemented. They probably train
the whole model and do not use LoRA. It might also be that it is a
different model like diffusion model, but I doubt it.
