"""Tokenizer, parser and evaluator for RooFormula / TFormula expressions.

Used for Combine ``rateParam`` formulas, ``RooFormulaVar`` and ``RooGenericPdf``.  The
grammar and the semantics follow what ROOT 6.32 does with the expression (checked against
``RooFormulaVar`` in tests/backend_zmodel_check.py):

* references: ``@i`` and ``x[i]`` (i-th dependent) and the dependents' names;
* numbers: C++ literals; integer literals are *integers*, so ``1/2`` is 0 and ``-3/2`` is -1
  (TFormula compiles the expression as C++).  Integer typing propagates through ``+ - * /``,
  unary minus, comparisons and logical operators (bool -> int), and the ternary operator;
  functions, ``^``/``**`` and references are double;
* operators, lowest to highest precedence: ``?:``, ``||``, ``&&``, ``== !=``,
  ``< <= > >=``, ``+ -``, ``* /``, unary ``- + !``, and ``^``/``**`` (right associative,
  binding tighter than a unary minus on its left: ``-a^2 = -(a^2)``, ``a^-1`` is allowed);
* constants ``pi``, ``e``, ``sqrt2`` have TFormula's (6 significant digit) values;
* functions: exp log log10 sqrt pow abs fabs sin cos tan asin acos atan atan2 sinh cosh tanh
  erf erfc min max, with ``TMath::`` and ``std::`` spellings, and TMath::Power, Sq, Sign,
  Gaus(x, mean=0, sigma=1, norm=false), Pi(), E().

Anything else raises ``FormulaError``: nothing is guessed.
"""

import math
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple


class FormulaError(ValueError):
    pass


# ----------------------------------------------------------------------------------------
# tokenizer
# ----------------------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<ref>@\d+)
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*)
  | (?P<op>\*\*|&&|\|\||==|!=|<=|>=|[-+*/^()<>!?:,\[\]])
""", re.VERBOSE)


@dataclass
class Token:
    kind: str   # num, ref, name, op, end
    text: str
    pos: int


def tokenize(text: str) -> List[Token]:
    out, pos = [], 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise FormulaError(f"unexpected character {text[pos]!r} at position {pos} in formula '{text}'")
        kind = m.lastgroup
        if kind != "ws":
            out.append(Token(kind, m.group(kind), pos))
        pos = m.end()
    out.append(Token("end", "", len(text)))
    return out


# ----------------------------------------------------------------------------------------
# AST
# ----------------------------------------------------------------------------------------

@dataclass
class Node:
    op: str                  # "num", "ref", "call", "neg", "not", "pow", binary op text, "?:"
    args: Tuple = ()
    value: object = None     # number (int or float) for "num", index for "ref", name for "call"
    is_int: bool = False     # C++ integer typing


_CONSTANTS = {"pi": 3.141593, "e": 2.718282, "sqrt2": 1.414214}  # TFormula's rounded values

# function name -> (canonical, min args, max args)
_FUNCS = {}
for _n, _c, _lo, _hi in [
    ("exp", "exp", 1, 1), ("log", "log", 1, 1), ("log10", "log10", 1, 1), ("sqrt", "sqrt", 1, 1),
    ("pow", "pow", 2, 2), ("abs", "abs", 1, 1), ("fabs", "abs", 1, 1), ("sin", "sin", 1, 1),
    ("cos", "cos", 1, 1), ("tan", "tan", 1, 1), ("asin", "asin", 1, 1), ("acos", "acos", 1, 1),
    ("atan", "atan", 1, 1), ("atan2", "atan2", 2, 2), ("sinh", "sinh", 1, 1), ("cosh", "cosh", 1, 1),
    ("tanh", "tanh", 1, 1), ("erf", "erf", 1, 1), ("erfc", "erfc", 1, 1), ("min", "min", 2, 2),
    ("max", "max", 2, 2),
]:
    _FUNCS[_n] = (_c, _lo, _hi)
    _FUNCS["std::" + _n] = (_c, _lo, _hi)
for _n, _c, _lo, _hi in [
    ("Exp", "exp", 1, 1), ("Log", "log", 1, 1), ("Log10", "log10", 1, 1), ("Sqrt", "sqrt", 1, 1),
    ("Power", "pow", 2, 2), ("Abs", "abs", 1, 1), ("Sin", "sin", 1, 1), ("Cos", "cos", 1, 1),
    ("Tan", "tan", 1, 1), ("ASin", "asin", 1, 1), ("ACos", "acos", 1, 1), ("ATan", "atan", 1, 1),
    ("ATan2", "atan2", 2, 2), ("SinH", "sinh", 1, 1), ("CosH", "cosh", 1, 1), ("TanH", "tanh", 1, 1),
    ("Erf", "erf", 1, 1), ("Erfc", "erfc", 1, 1), ("Min", "min", 2, 2), ("Max", "max", 2, 2),
    ("Sq", "sq", 1, 1), ("Sign", "sign2", 2, 2), ("Gaus", "gaus", 1, 4), ("Pi", "pi", 0, 0), ("E", "e", 0, 0),
]:
    _FUNCS["TMath::" + _n] = (_c, _lo, _hi)


class _Parser:
    def __init__(self, text: str, names: Sequence[str]):
        self.text = text
        self.toks = tokenize(text)
        self.i = 0
        self.names = {n: k for k, n in enumerate(names)}
        self.n = len(names)

    def peek(self) -> Token:
        return self.toks[self.i]

    def take(self, text: Optional[str] = None) -> Token:
        tok = self.toks[self.i]
        if text is not None and tok.text != text:
            raise FormulaError(f"expected '{text}' at position {tok.pos} in formula '{self.text}', found '{tok.text}'")
        self.i += 1
        return tok

    def parse(self) -> Node:
        node = self.ternary()
        if self.peek().kind != "end":
            tok = self.peek()
            raise FormulaError(f"unexpected '{tok.text}' at position {tok.pos} in formula '{self.text}'")
        return node

    def ternary(self) -> Node:
        cond = self.binary(0)
        if self.peek().text == "?":
            self.take("?")
            a = self.ternary()
            self.take(":")
            b = self.ternary()
            return Node("?:", (cond, a, b), is_int=a.is_int and b.is_int)
        return cond

    _LEVELS = [("||",), ("&&",), ("==", "!="), ("<", "<=", ">", ">="), ("+", "-"), ("*", "/")]

    def binary(self, level: int) -> Node:
        if level == len(self._LEVELS):
            return self.unary()
        node = self.binary(level + 1)
        while self.peek().kind == "op" and self.peek().text in self._LEVELS[level]:
            op = self.take().text
            rhs = self.binary(level + 1)
            is_int = (node.is_int and rhs.is_int) if op in "+-*/" else True
            node = Node(op, (node, rhs), is_int=is_int)
        return node

    def unary(self) -> Node:
        tok = self.peek()
        if tok.kind == "op" and tok.text in ("-", "+", "!"):
            self.take()
            arg = self.unary()
            if tok.text == "+":
                return arg
            if tok.text == "-":
                return Node("neg", (arg,), is_int=arg.is_int)
            return Node("not", (arg,), is_int=True)
        return self.power()

    def power(self) -> Node:
        base = self.primary()
        if self.peek().kind == "op" and self.peek().text in ("^", "**"):
            self.take()
            expo = self.unary()  # right associative; allows a^-1
            return Node("pow", (base, expo))
        return base

    def primary(self) -> Node:
        tok = self.take()
        if tok.kind == "num":
            is_int = re.fullmatch(r"\d+", tok.text) is not None
            return Node("num", value=int(tok.text) if is_int else float(tok.text), is_int=is_int)
        if tok.kind == "ref":
            return self._ref(int(tok.text[1:]), tok)
        if tok.kind == "name":
            name = tok.text
            if self.peek().text == "(":
                return self._call(name, tok)
            if name == "x" and self.peek().text == "[":  # always an index, even if a dependent is named x
                self.take("[")
                idx = self.take()
                if idx.kind != "num" or not idx.text.isdigit():
                    raise FormulaError(f"bad index in x[...] at position {idx.pos} in formula '{self.text}'")
                self.take("]")
                return self._ref(int(idx.text), tok)
            if name in self.names:
                return Node("ref", value=self.names[name])
            if name in _CONSTANTS:
                return Node("num", value=_CONSTANTS[name])
            if name in ("true", "false"):
                return Node("num", value=int(name == "true"), is_int=True)
            raise FormulaError(f"unknown name '{name}' in formula '{self.text}'")
        if tok.text == "(":
            node = self.ternary()
            self.take(")")
            return node
        raise FormulaError(f"unexpected '{tok.text or 'end of formula'}' at position {tok.pos} in formula '{self.text}'")

    def _ref(self, idx: int, tok: Token) -> Node:
        if idx >= self.n:
            raise FormulaError(f"reference {tok.text}{idx if tok.kind == 'name' else ''} beyond the {self.n} "
                               f"dependents of formula '{self.text}'")
        return Node("ref", value=idx)

    def _call(self, name: str, tok: Token) -> Node:
        if name not in _FUNCS:
            raise FormulaError(f"unsupported function '{name}' in formula '{self.text}'")
        canon, lo, hi = _FUNCS[name]
        self.take("(")
        args = []
        if self.peek().text != ")":
            args.append(self.ternary())
            while self.peek().text == ",":
                self.take(",")
                args.append(self.ternary())
        self.take(")")
        if not lo <= len(args) <= hi:
            raise FormulaError(f"{name} takes {lo}..{hi} arguments, got {len(args)} in formula '{self.text}'")
        if canon == "pi":
            return Node("num", value=math.pi)
        if canon == "e":
            return Node("num", value=math.e)
        return Node("call", tuple(args), value=canon)


def parse(text: str, names: Sequence[str]) -> Node:
    """Parse ``text`` whose dependents (``@i``/``x[i]``) have the given names."""
    if not text.strip():
        raise FormulaError("empty formula")
    return _Parser(text, names).parse()


def used_refs(node: Node) -> set:
    if node.op == "ref":
        return {node.value}
    out = set()
    for a in node.args:
        out |= used_refs(a)
    return out


# ----------------------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------------------

class NumpyOps:
    """Evaluation with numpy (reference / tests)."""

    def __init__(self):
        import numpy as np
        from scipy import special

        self.np = np
        self.f = {"exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt, "pow": np.power,
                  "abs": np.abs, "sin": np.sin, "cos": np.cos, "tan": np.tan, "asin": np.arcsin, "acos": np.arccos,
                  "atan": np.arctan, "atan2": np.arctan2, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
                  "erf": special.erf, "erfc": special.erfc, "min": np.minimum, "max": np.maximum}

    def const(self, v):
        return float(v)

    def where(self, c, a, b):
        return self.np.where(c, a, b)

    def to_bool(self, a):
        return self.np.asarray(a) != 0

    def from_bool(self, b):
        return self.np.asarray(b, dtype=float)

    def trunc(self, a):
        return self.np.trunc(a)

    def compare(self, op, a, b):
        np = self.np
        return {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
                "==": np.equal, "!=": np.not_equal}[op](a, b)

    def logical(self, op, a, b):
        return {"&&": self.np.logical_and, "||": self.np.logical_or}[op](a, b)

    def logical_not(self, a):
        return self.np.logical_not(a)

    def call(self, name, args):
        return self.f[name](*args)


class TFOps:
    """Evaluation with zfit.z.numpy / TensorFlow (float64)."""

    def __init__(self):
        import tensorflow as tf
        import zfit.z.numpy as znp

        self.tf, self.znp = tf, znp
        m = tf.math
        self.f = {"exp": znp.exp, "log": znp.log, "log10": lambda a: znp.log(a) / math.log(10.0), "sqrt": znp.sqrt,
                  "pow": znp.power, "abs": znp.abs, "sin": znp.sin, "cos": znp.cos, "tan": znp.tan,
                  "asin": m.asin, "acos": m.acos, "atan": m.atan, "atan2": m.atan2, "sinh": m.sinh,
                  "cosh": m.cosh, "tanh": m.tanh, "erf": m.erf, "erfc": m.erfc, "min": znp.minimum,
                  "max": znp.maximum}

    def const(self, v):
        return self.tf.constant(float(v), dtype=self.tf.float64)

    def _t(self, a):
        return self.tf.convert_to_tensor(a, dtype=self.tf.float64)

    def where(self, c, a, b):
        a, b = self._t(a), self._t(b)
        shape = self.tf.broadcast_dynamic_shape(self.tf.shape(a), self.tf.shape(b))
        shape = self.tf.broadcast_dynamic_shape(shape, self.tf.shape(c))
        return self.tf.where(self.tf.broadcast_to(c, shape), self.tf.broadcast_to(a, shape),
                             self.tf.broadcast_to(b, shape))

    def to_bool(self, a):
        return self.tf.not_equal(self._t(a), 0.0)

    def from_bool(self, b):
        return self.tf.cast(b, self.tf.float64)

    def trunc(self, a):
        a = self._t(a)
        return self.tf.sign(a) * self.tf.floor(self.tf.abs(a))

    def compare(self, op, a, b):
        tf = self.tf
        return {"<": tf.less, "<=": tf.less_equal, ">": tf.greater, ">=": tf.greater_equal,
                "==": tf.equal, "!=": tf.not_equal}[op](self._t(a), self._t(b))

    def logical(self, op, a, b):
        return {"&&": self.tf.logical_and, "||": self.tf.logical_or}[op](a, b)

    def logical_not(self, a):
        return self.tf.logical_not(a)

    def call(self, name, args):
        return self.f[name](*[self._t(a) for a in args])


def _cdiv(a: int, b: int) -> int:
    if b == 0:
        raise FormulaError("integer division by zero in formula")
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _fold(node: Node) -> Node:
    """Constant-fold integer literal sub-expressions with C++ semantics."""
    args = tuple(_fold(a) for a in node.args)
    node = Node(node.op, args, node.value, node.is_int)
    if node.is_int and args and all(a.op == "num" and a.is_int for a in args):
        vals = [a.value for a in args]
        if node.op in ("+", "-", "*", "/"):
            a, b = vals
            v = {"+": a + b, "-": a - b, "*": a * b}.get(node.op) if node.op != "/" else _cdiv(a, b)
            return Node("num", value=v, is_int=True)
        if node.op == "neg":
            return Node("num", value=-vals[0], is_int=True)
    return node


def compile_formula(node: Node, ops) -> Callable[[Sequence], object]:
    """Return ``f(refs)`` evaluating the AST with ``ops``; ``refs[i]`` is the value of the
    i-th dependent (tensor, array or float)."""
    node = _fold(node)

    def build(n: Node):
        op = n.op
        if op == "num":
            v = ops.const(n.value)
            return lambda r: v
        if op == "ref":
            k = n.value
            return lambda r: r[k]
        if op == "call":
            fs = [build(a) for a in n.args]
            name = n.value
            if name == "sq":
                f0 = fs[0]
                return lambda r: f0(r) * f0(r)
            if name == "sign2":  # TMath::Sign(a, b) = |a| if b >= 0 else -|a|
                fa, fb = fs
                return lambda r: ops.where(ops.compare(">=", fb(r), 0.0), ops.call("abs", [fa(r)]),
                                           -ops.call("abs", [fa(r)]))
            if name == "gaus":
                return _gaus(fs, ops)
            return lambda r: ops.call(name, [f(r) for f in fs])
        if op == "neg":
            f0 = build(n.args[0])
            return lambda r: -f0(r)
        if op == "not":
            f0 = build(n.args[0])
            return lambda r: ops.from_bool(ops.logical_not(ops.to_bool(f0(r))))
        if op == "pow":
            fa, fb = (build(a) for a in n.args)
            return lambda r: ops.call("pow", [fa(r), fb(r)])
        if op == "?:":
            fc, fa, fb = (build(a) for a in n.args)
            return lambda r: ops.where(ops.to_bool(fc(r)), fa(r), fb(r))
        fa, fb = (build(a) for a in n.args)
        if op == "+":
            return lambda r: fa(r) + fb(r)
        if op == "-":
            return lambda r: fa(r) - fb(r)
        if op == "*":
            return lambda r: fa(r) * fb(r)
        if op == "/":
            if n.is_int:
                return lambda r: ops.trunc(fa(r) / fb(r))
            return lambda r: fa(r) / fb(r)
        if op in ("<", "<=", ">", ">=", "==", "!="):
            return lambda r: ops.from_bool(ops.compare(op, fa(r), fb(r)))
        if op in ("&&", "||"):
            return lambda r: ops.from_bool(ops.logical(op, ops.to_bool(fa(r)), ops.to_bool(fb(r))))
        raise FormulaError(f"internal: unknown node {op}")

    return build(node)


def _gaus(fs, ops):
    """TMath::Gaus(x, mean=0, sigma=1, norm=false)."""
    fx = fs[0]
    fm = fs[1] if len(fs) > 1 else (lambda r: 0.0)
    fsig = fs[2] if len(fs) > 2 else (lambda r: 1.0)
    fnorm = fs[3] if len(fs) > 3 else None

    def f(r):
        sig = fsig(r)
        u = (fx(r) - fm(r)) / sig
        val = ops.call("exp", [-0.5 * u * u])
        if fnorm is not None:
            val = ops.where(ops.to_bool(fnorm(r)), val / (math.sqrt(2.0 * math.pi) * sig), val)
        return ops.where(ops.compare("==", sig, 0.0), 0.0 * val, val)  # TMath::Gaus: 0 for sigma == 0
    return f
