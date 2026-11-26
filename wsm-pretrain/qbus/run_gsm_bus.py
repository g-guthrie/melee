"""GSM8K / GSM-Hard with the frozen host plus the pointer bus (same prompt, decoding and scoring as the paper's
scripts/run_gsm.py).

At every generated token the reader looks at the host's hidden states for the answer so far.  When its gate fires,
pointer 1 gives the token after operand A and pointer 2 the token after operand B; both numbers are read exactly
from the token text, calc_ref computes the result, and the result's tokens are forced (one optional non-digit token
such as a space or ' $' is left to the host first).  The host's weights are never changed.

    PYTHONPATH=. python run_gsm_bus.py OUT.json --reader /workspace/qreaders/L12_r2.pt --split hard
"""
import argparse, json, os, re, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import Reader, build_prompt, number_span_before, clean_expr, eval_expr

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--reader", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
ap.add_argument("--split", default="hard", choices=["test", "hard"])
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--max_batches", type=int, default=0)
ap.add_argument("--max_new", type=int, default=320)
a = ap.parse_args()
dev = "cuda"
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
ck = torch.load(a.reader, map_location=dev)
reader = Reader(ck["d_host"], ck["d"], ck["reader_layers"], max_len=max(ck["max_len"], a.max_new + 8)).to(dev).eval()
state = ck["state"]
if state["pos.weight"].shape[0] < reader.pos.weight.shape[0]:      # longer generations than the training length
    w = reader.pos.weight.data.clone(); w[: state["pos.weight"].shape[0]] = state["pos.weight"]
    w[state["pos.weight"].shape[0]:] = state["pos.weight"][-1]; state["pos.weight"] = w
reader.load_state_dict(state)
LAYER = ck["layer"]
eos = tok.convert_tokens_to_ids("<|endoftext|>")
T = lambda s: tok(s, add_special_tokens=False)["input_ids"]
CH = {c: T(c)[0] for c in "0123456789.-"}
assert all(len(T(c)) == 1 for c in CH)
DIGITISH = torch.tensor([CH[c] for c in "0123456789."], device=dev)
prompt, shot_q = build_prompt(tok)
rows = [json.loads(l) for l in open("data/gsm8k/test.jsonl" if a.split == "test" else "data/gsmhard/test.jsonl")]
rows = [r for r in rows if r["question"] not in shot_q]
stats = {"fires": 0, "writes": 0, "parse_fail": 0, "range_fail": 0}


@torch.no_grad()
def generate(seqs):
    L = max(len(s) for s in seqs); B = len(seqs)
    x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device=dev)
    att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
    o = model.model(input_ids=x, attention_mask=att, use_cache=True, output_hidden_states=True); past = o.past_key_values
    lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    gen = [[] for _ in range(B)]; strs = [[] for _ in range(B)]; done = [False] * B
    queue = [[] for _ in range(B)]          # tokens still to be forced
    lead = [False] * B                      # a result is pending and the host may place one non-digit token first
    veto = [False] * B                      # the next token must not extend the number just written
    hist = []                               # host states (layer LAYER) of the generated tokens
    for _ in range(a.max_new):
        nxt = lg.argmax(-1); nl = nxt.tolist()
        for i in range(B):
            if done[i]:
                continue
            if veto[i]:
                lg[i, DIGITISH] = float("-inf"); nl[i] = int(lg[i].argmax()); veto[i] = False
            if lead[i]:
                s = tok.decode([nl[i]])
                lead[i] = False
                if not (s.strip() in ("", "$") and len(s) <= 2):      # the host went straight to a digit: force now
                    nl[i] = queue[i].pop(0)
                    if not queue[i]: veto[i] = True
            elif queue[i]:
                nl[i] = queue[i].pop(0)
                if not queue[i]: veto[i] = True
            gen[i].append(nl[i]); strs[i].append(tok.decode([nl[i]]))
            if nl[i] == eos or (len(gen[i]) % 4 == 0 and "\n\n" in tok.decode(gen[i][-8:])): done[i] = True
        if all(done): break
        nxt = torch.tensor([eos if done[i] else nl[i] for i in range(B)], device=dev)
        att = torch.cat([att, att.new_ones(B, 1)], 1)
        o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True, output_hidden_states=True)
        past = o.past_key_values
        lg = model.lm_head(o.last_hidden_state[:, -1]).float()
        hist.append(o.hidden_states[LAYER][:, -1])
        h = torch.stack(hist, 1)                                       # (B, t, d): states of generated tokens 0..t-1
        gl, ptr = reader(h)
        g = gl[:, -1].argmax(-1).tolist(); t = h.shape[1] - 1          # position of the token just fed
        tgt = ptr[:, :, -1].argmax(-1).tolist()
        for i in range(B):
            if done[i] or g[i] == 0 or queue[i] or lead[i]:
                continue
            stats["fires"] += 1
            # pointer 1 targets the token just after the expression's first number; the expression runs from that
            # number to the '=' just fed, and is read exactly from the token text
            start = number_span_before(strs[i], tgt[i][0])
            expr = clean_expr("".join(strs[i][start:t])) if start >= 0 else ""
            if not expr:
                stats["parse_fail"] += 1; continue
            res = eval_expr(expr)
            if res == "":
                stats["range_fail"] += 1; continue
            stats["chained"] = stats.get("chained", 0) + (sum(expr.count(c) for c in "+-*/") > 1)
            queue[i] = [CH[c] for c in res]; lead[i] = True; stats["writes"] += 1
    return gen


def correct(pred, gold):
    try: return pred is not None and abs(float(pred) - float(gold)) <= 1e-6 * max(1.0, abs(float(gold)))
    except ValueError: return False


t0 = time.time(); res = []
order = sorted(range(len(rows)), key=lambda i: len(rows[i]["question"]))
if a.max_batches: order = order[: a.max_batches * a.bs]
for b in range(0, len(order), a.bs):
    idx = order[b:b + a.bs]
    for i, g in zip(idx, generate([prompt + T(f"Question: {rows[i]['question']}\nAnswer:") for i in idx])):
        txt = tok.decode(g).replace("<|endoftext|>", "").split("\n\n")[0]
        gold = rows[i]["answer"].split("####")[1].strip().replace(",", "")
        m = re.search(r"####\s*(-?[\d,]*\.?\d+)", txt); pred = m.group(1).replace(",", "") if m else None
        res.append({"i": i, "q": rows[i]["question"], "gold": gold, "pred": pred, "ok": correct(pred, gold), "out": txt})
n = len(res); acc = 100 * sum(r["ok"] for r in res) / n
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump({"model": a.model, "mode": "bus", "reader": a.reader, "split": a.split, "n": n, "acc": acc, "stats": stats, "rows": res},
          open(a.out, "w"), indent=1)
print(f"RESULT bus reader={os.path.basename(a.reader)} split={a.split}: {sum(r['ok'] for r in res)}/{n} = {acc:.2f}% | "
      f"{json.dumps(stats)} | {time.time() - t0:.0f}s", flush=True)
