# WordNet-reduced DiffusionGemma vocabulary

`main.py` builds a vocabulary from every English WordNet lemma and the
DiffusionGemma tokenizer. It also retains:

- tokens used at both the beginning and middle of text;
- text tokenizer special tokens, excluding image and video tokens;
- every numeric, punctuation, symbol, and byte-fallback token.

The model reduction remaps token IDs and slices the encoder embedding, decoder
embedding, and language-model head together. It also removes the vision tower
and multimodal projector, leaving a text-only model. This is required because
DiffusionGemma ties the text weights and samples directly from the configured
vocabulary size.

Build `reduced_vocabulary.json` without loading the 26B model:

```console
uv run python main.py --vocabulary-only
```

Reduce the model and save its text-only weights, processor, tokenizer, and
vocabulary mapping under `models/`, without running inference:

```console
uv run python save_reduced_model.py
```

The exporter loads only the text modules from the pretrained checkpoint, uses
Transformers' low-memory loading path, and offloads the state dictionary while
loading. It never calls the model's forward or generation methods. A Hugging
Face account with access to the gated model is required.

Reload the saved artifacts:

```console
uv run python load_reduced_model.py
```

Use `load_reduced_model.py` rather than loading the weights with the stock
DiffusionGemma class. Transformers currently initializes a vision tower
unconditionally; the loader uses the included text-only subclass and therefore
keeps those discarded modules out of memory.

Run generation from the saved model:

```console
uv run python load_reduced_model.py "Explain a normal distribution."
```

Tokens outside the reduced vocabulary map to the original tokenizer's unknown
token. WordNet does not contain proper names, recent terminology, or every
inflected form, so this reduction intentionally limits output to its retained
subword pieces.

## Expression grammar notes

Continous:
    alpha a loc scale
    anglit loc scale
    arcsine loc scale
    argus chi loc scale
    beta a b loc scale
    betaprime a b loc scale
    bradford c loc scale
    burr c d loc scale
    burr12 c d loc scale
    cauchy loc scale
    chi df loc scale
    chi2 df loc scale
    cosine loc scale
    crystalball beta m loc scale
    dgamma a loc scale
    dpareto_lognorm u s a b loc scale
    dweibull c loc scale
    erlang a loc scale
    expon loc scale
    exponnorm K loc scale
    exponweib a c loc scale
    exponpow b loc scale
    f dfn dfd loc scale
    fatiguelife c loc scale
    fisk c loc scale
    foldcauchy c loc scale
    foldnorm c loc scale
    genlogistic c loc scale
    gennorm beta loc scale
    genpareto c loc scale
    genexpon a b c loc scale
    genextreme c loc scale
    gausshyper a b c z loc scale
    gamma a loc scale
    gengamma a c loc scale
    genhalflogistic c loc scale
    genhyperholic p a b loc scale
    geninvgauss p b loc cale
    gibrat loc scale
    gompertz c loc scale
    gumbel_r loc scale
    gumbel_l loc scale
    halfcauchy loc scale
    halflogistic loc scale
    halfnorm loc scale
    halfgennorm beta loc scale
    hypsecant loc scale
    invgamma a loc scale
    invgauss mu loc scale
    invweibull c loc scale
    irwinhall n loc scale
    jf_skew_t a b loc scale
    johnsonsb a b loc scale
    johnsonsu a b loc scale
    kappa4 h k loc scale
    kappa3 a loc scale
    ksone n loc scale
    kstwo n loc scale
    kstwobign loc scale
    landau loc scale
    laplace loc scale
    laplace_asymmetric kappa loc scale
    levy loc scale
    levy_l loc scale
    levy_stable alpha beta loc scale
    logistic loc scale
    loggamma c loc scale
    loglaplace c loc scale
    lognorm s loc scale
    loguniform a b loc scale
    lomac loc scale
    maxwell loc scale
    mielke k s loc scale
    moyal loc scale
    nakagami nu loc scale
    ncx2 df nc loc scale
    ncf dfn dfd nc loc scale
    nct df nc loc scale
    norm loc scale
    norminvgauss a b loc scale
    pareto b loc scale
    pearson3 skew loc scale
    powerlaw a loc scale
    powerlognorm c s loc scale
    powernorm c loc scale
    rdist c loc scale
    rayleigh loc scale
    rel_breitwigner rho loc scale
    rice b loc scale
    recipinvgauss mu loc scale
    semicircular loc scale
    skewcauchy a loc scale
    skewnorm a loc scale
    studentized_range k df loc scale
    t df loc scale
    trapezoid c d loc scale
    triang c loc scale
    truncexpon b loc scale
    truncnorm a b loc scale
    truncpareto b c loc scale
    truncweibull_min c a b loc scale
    tukeylambda lam loc scale
    uniform loc scale
    vonmises kappa loc scale
    vonmises_line kappa loc scale
    wald loc scale
    weibull_min c loc scale
    weibull_mac loc scale
    wrapcauchy c loc scale

Multivariate:
    multivariate_normal mean cov allow_singular
    matrix_normal mean rowcov colcov
    dirichlet alpha
    dirichlet_multinomial alpha n
    wishart df scale
    invwishart df scale
    multinomial n p
    special_ortho_group dim
    ortho_group dim
    unitary_group dim
    random_correlation eigs
    multivariate_t loc df allow_singular
    multivariate_hypergeom m n
    normal_inverse_gamma mu lambda a b
    random_table row col
    uniform_direction dim
    vonmises_fisher mu kappa
    matrix_t mean row_spread col_spread df

Discrete:
    bernoulli p loc
    betabinom n a b loc
    betanbinom n a b loc
    binom n p loc
    botlzmann lambda_ N loc
    dlaplace a loc
    geom p loc
    hypergeom M n N loc
    logser p loc
    nbiom n p loc
    nchypergeom_fisher M n N odds loc
    nchypergeom_wallenus M n N odds loc
    nhypergeom M n r loc
    planck lambda_ loc
    poisson mu loc
    possion_binom p loc
    randint low high loc
    skellam mu1 mu2 loc
    yulesimon alpha loc
    zipf a loc
    zipfian a n loc


Operators:
   add +
   multiply *
   maximum max
   minimum min
   and and
   or or
   avg avg
   mod %
   div /
   pow **
   sub -
   eq ==
   lt <
   le <=
   gt >
   ge >=
   isclose ~
   neg neg
   abs abs
   log log
   exp exp
   floor floor
   ceil ceil
   sign sign
   sqrt sqrt
   log10 log10
   sin sin
   cos cos
   tan tan
   arcsin arcsin
   arccos arccos
   arctan arctan
   arctan2 arctan2
   sinh sinh
   cosh cosh
   tanh tanh
   arcsinh arcsinh
   arccosh arccos
   arctanh arctanh


Variable assignment:
   a = distr ...
   a = 1

Correlation:
   corr a b 0.4


method:
   lhs
   halton
   sobol
   pseudo-random

correlator:
    cholesky
    imanconover
    permutation
    composite
