"""Audit the arithmetic in generated GSM answers: for every "A op B = C" the model wrote, is C exactly right?

    PYTHONPATH=. python audit.py out/hard_plain.json out/hard_calc.json out/hard_bus_L12_r0.json
"""
import collections, json, re, sys
from common import find_calls, calc_ref

RES = re.compile(r"(\s*\$?)(-?\d[\d,]*(?:\.\d+)?)")


def audit(path):
    rows = json.load(open(path))["rows"]
    c, miss = collections.Counter(), []
    for r in rows:
        t = r["out"]
        for eq, opc, op, a, b in find_calls(t):
            want = calc_ref(f"{a}{op}{b}")
            m = RES.match(t[eq + 1:])
            got = m.group(2).replace(",", "") if m else None
            try:
                ok = got is not None and abs(float(got) - float(want)) <= 1e-9 * max(1.0, abs(float(want)))
            except ValueError:
                ok = False
            big = max(len(a.split(".")[0]), len(b.split(".")[0])) >= 5
            dollar_res = bool(m and "$" in m.group(1))
            dollar_a = t[:opc].rstrip()[-len(a) - 1:-len(a)] == "$" if not t[:opc].rstrip().endswith(",") else False
            key = ("big" if big else "small") + (" $result" if dollar_res else " plain-result")
            c[(key, ok)] += 1
            c[("ALL", ok)] += 1
            if not ok and len(miss) < 8:
                miss.append((t[max(0, opc - 30): eq + 24].replace("\n", " / "), want))
    return c, miss


for p in sys.argv[1:]:
    c, miss = audit(p)
    print(p)
    for key in sorted({k for k, _ in c}):
        right, wrong = c[(key, True)], c[(key, False)]
        print(f"   {key:22s} right {right:5d}  wrong {wrong:5d}  -> {100 * right / max(right + wrong, 1):.1f}% correct")
    if "bus" in p:
        for text, want in miss:
            print("   WRONG:", repr(text), "should be", want)
