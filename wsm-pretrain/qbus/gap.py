"""Why does one run lose problems another run wins?  Classify each such problem by what went wrong in the loser's text.

    PYTHONPATH=. python gap.py out/hard_calc.json out/hard_bus_L12_r0.json      # winner, loser
"""
import collections, json, re, sys
from common import find_calls, calc_ref

RES = re.compile(r"(\s*\$?)(-?\d[\d,]*(?:\.\d+)?)")
CHAIN = re.compile(r"\d[\d,.]*\s*[+\-*/x×]\s*\$?\d[\d,.]*\s*[+\-*/x×]\s*\$?\d[\d,.]*[^=\n]*=")
PAREN = re.compile(r"\([^()\n]*\d[^()\n]*\)[^=\n]*=|=[^\n]*\(")


def wrong_calls(t):
    n = 0
    for eq, opc, op, a, b in find_calls(t):
        if re.search(r"[+\-*/x×]\s*\$?$", t[:opc - len(a)].rstrip() + " ") and False:
            continue
        want = calc_ref(f"{a}{op}{b}")
        m = RES.match(t[eq + 1:])
        try:
            ok = m is not None and abs(float(m.group(2).replace(",", "")) - float(want)) <= 1e-9 * max(1.0, abs(float(want)))
        except ValueError:
            ok = False
        n += not ok
    return n


win, lose = (json.load(open(p)) for p in sys.argv[1:3])
w = {r["i"]: r for r in win["rows"]}
c, shown = collections.Counter(), collections.defaultdict(list)
for r in lose["rows"]:
    if r["ok"] or not w[r["i"]]["ok"]:
        continue
    t = r["out"]
    if CHAIN.search(t):
        kind = "has a chained expression (a op b op c =)"
    elif PAREN.search(t):
        kind = "has a bracketed expression"
    elif wrong_calls(t):
        kind = "a plain two-number call came out wrong"
    elif not re.search(r"####\s*-?[\d,]*\.?\d+", t):
        kind = "no final answer line"
    else:
        kind = "arithmetic all right; different reasoning or final copy"
    c[kind] += 1
    if len(shown[kind]) < 2:
        shown[kind].append((t[:230].replace("\n", " / "), w[r["i"]]["out"][:230].replace("\n", " / "), r["gold"]))
print(f"{sys.argv[1]} right, {sys.argv[2]} wrong: {sum(c.values())}")
for k, v in c.most_common():
    print(f"  {v:4d}  {k}")
    for lo, wi, gold in shown[k]:
        print("        LOSER :", lo); print("        WINNER:", wi); print("        gold  :", gold)
