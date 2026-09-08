library(dcg/basics).



constant(constant(C)) --> number(C).
variable(var(Var)) --> Var, { atom(Var) }.

operator(op(Op)) -->
    Op,
    {
        member(
           Op,
           [
               add, mult, max, min, and, or, avg, fdiv,
               div, mod, pow, sub, eq, lt, le, gt, ge,
               close, -, abs, log, exp, floor, ceil, sign,
               sqrt, log10, sin, cos , tan, arcsin, arccos,
               arctan, arctan2, sinh, cosh, tanh, arcsinh,
               arccosh, arctanh
           ]
        )
    }.


distr(Name, [a-F]) --> alpha, a, '=', float(F).

distribution(distr(Name, Args)) --> distr(Name, Args).

statement(assign(var(Var), Constant)) --> variable(Var) '=', constant(Constant).
statement(assign(var(Var), Distr)) --> variable(Var), '=', distribution(Distr).
statement(expr(Operator, Var, Var)) --> variable(Var), operator(Operator), variable(Var).

program([Line]) --> statement(Line), optional('\n').
program([Line | Lines]) --> statement(Line), '\n', program(Lines).
