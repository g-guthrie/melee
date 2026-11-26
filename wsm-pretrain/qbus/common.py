"""Shared pieces for the pointer bus on a frozen Qwen host.

Run from the root of the weight-space-micromachines checkout (its data/ and micromachines/ are reused), e.g.
    PYTHONPATH=. python /workspace/qbus/build_data.py ...

The bus's reader is the only trained part.  At each position it reads the frozen host's hidden state at one layer
and produces: a gate (no call, or one of + - * /) and two pointers.  A pointer targets the token just AFTER an
operand's last digit, because that is the first position whose hidden state knows the number is complete.  The
operand itself is read exactly from the token text, the arithmetic is exact decimal code shared with the paper's
calculator (calc_ref), and the result is written by forcing the answer's tokens.
"""
import json, os, re, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from micromachines.calcfmt import calc_ref, strip_solution, _chain, Fail  # noqa: E402

OPS = ["+", "-", "*", "/"]                    # gate classes 1..4 (0 = no call)
OPCHAR = {"+": "+", "-": "-", "*": "*", "/": "/", "x": "*", "×": "*"}
# "A op B =" where A is not itself the tail of a longer expression
EXPR = re.compile(r"(?<![\d.,])(\d[\d,]*(?:\.\d+)?)(\s*)([+\-*/x×])(\s*\$?)(\d[\d,]*(?:\.\d+)?)\s*=")
NUMCH = set("0123456789.,")
# a whole expression: two or more numbers joined by operators, ending in '=' (no brackets)
_N = r"\d[\d,]*(?:\.\d+)?"
EXPRN = re.compile(r"(?<![\d.,])(" + _N + r")((?:\s*[+\-*/x×]\s*\$?" + _N + r")+)\s*=")
CLEAN = re.compile(r"^\d+(\.\d+)?([+\-*/]\d+(\.\d+)?)+$")


def clean_expr(text):
    """'3 * $4,748,457 - $20' -> '3*4748457-20' ('' if it is not a plain bracket-free expression)."""
    e = text.replace("$", "").replace(",", "").replace(" ", "").replace("x", "*").replace("×", "*")
    return e if CLEAN.match(e) else ""


def eval_expr(expr):
    """Exact value of a cleaned expression, with the paper's calculator semantics applied step by step
    (usual precedence; each division truncated to 6 decimals). '' if it cannot be evaluated."""
    try:
        return _chain(expr)[1]
    except Fail:
        return ""


def find_exprs(text):
    """Whole expressions in a text: (char index of '=', char index of the first operator, cleaned expression)."""
    out = []
    for m in EXPRN.finditer(text):
        before = text[: m.start(1)].rstrip()
        if before and before[-1] in "+-*/x×(":      # the tail of something longer or bracketed: leave it alone
            continue
        e = clean_expr(text[m.start(1): m.end() - 1])
        if not e or eval_expr(e) == "":
            continue
        op_c = m.start(2) + len(m.group(2)) - len(m.group(2).lstrip())
        out.append((m.end() - 1, op_c, e))
    return out


def number_span_before(strs, j):
    """Index of the first token of the number that ends in token j-1, or -1."""
    i = j - 1
    while i >= 0:
        t = strs[i].strip()
        if t and all(c in NUMCH for c in t):
            i -= 1
        elif t.startswith("$") and len(t) > 1 and all(c in NUMCH for c in t[1:]):
            i -= 1; break
        else:
            break
    return i + 1 if i + 1 <= j - 1 else -1


def build_prompt(tok, shots_path="data/gsm8k/shots_v1.json"):
    shots = json.load(open(shots_path))
    T = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    prompt = []
    for s in shots:
        body, fin = strip_solution(s["solution"] + f"\n#### {s['answer']}")
        prompt += T(f"Question: {s['question']}\nAnswer: {body}\n#### {fin}\n\n")
    return prompt, {s["question"] for s in shots}


def find_calls(text):
    """Binary calls in a text: (char index of '=', char index of the operator, op, A, B)."""
    out = []
    for m in EXPR.finditer(text):
        before = text[: m.start(1)].rstrip()
        if before and before[-1] in "+-*/x×(":      # part of a chain or a bracketed expression: not a clean binary call
            continue
        a, b = m.group(1).replace(",", ""), m.group(5).replace(",", "")
        op = OPCHAR[m.group(3)]
        if calc_ref(f"{a}{op}{b}") == "":
            continue
        out.append((m.end() - 1, m.start(3), op, a, b))
    return out


def number_before(strs, j):
    """The number whose last character sits in token j-1 (tokens are strings). Returns '' if there is none."""
    chars = []
    i = j - 1
    while i >= 0:
        s = strs[i].strip()
        if s and all(c in NUMCH for c in s):
            chars.append(s); i -= 1
        elif s.startswith("$") and len(s) > 1 and all(c in NUMCH for c in s[1:]):
            chars.append(s[1:]); break
        else:
            break
    num = "".join(reversed(chars)).replace(",", "").strip(".")
    return num if re.fullmatch(r"\d+(\.\d+)?", num) else ""


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, pad=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        if pad is not None:
            mask = mask & pad[:, None, None, :]
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(att.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class Reader(nn.Module):
    """Trainable wiring: gate + two pointers from the host's hidden states. Never writes into the host."""

    def __init__(self, d_host, d=256, layers=0, heads=4, max_len=512):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_host), nn.Linear(d_host, d))
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.gate = nn.Linear(d, 1 + len(OPS))
        self.q, self.k = nn.Linear(d, 2 * d), nn.Linear(d, 2 * d)
        self.d = d

    def forward(self, h, pad=None):
        """h: (B, T, d_host) host states for the answer so far; pad: (B, T) True where real. Returns gate logits
        (B, T, 5) and pointer logits (B, 2, T, T) over positions <= the current one."""
        B, T, _ = h.shape
        z = self.inp(h.float())
        if len(self.blocks):
            z = z + self.pos.weight[:T]
        for blk in self.blocks:
            z = blk(z, pad)
        z = self.ln(z)
        q, k = self.q(z).view(B, T, 2, self.d), self.k(z).view(B, T, 2, self.d)
        ptr = torch.einsum("bthd,bshd->bhts", q, k) / self.d ** 0.5
        mask = torch.ones(T, T, dtype=torch.bool, device=h.device).tril()
        if pad is not None:
            mask = mask & pad[:, None, :]
        ptr = ptr.masked_fill(~mask[:, None] if mask.dim() == 3 else ~mask, float("-inf"))
        return self.gate(z), ptr
