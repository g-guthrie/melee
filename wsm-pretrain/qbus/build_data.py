"""Build reader training data from GSM8K train solutions, teacher-forced through the frozen host.

For each problem: [8-shot prompt] Question ... Answer: <solution>.  Calls ("A op B =") are found in the solution
text; the '=' token gets the gate label, pointer 1 targets the operator token (the token right after A), pointer 2
targets the '=' token itself (the token right after B).  Half of the sequences have their call operands replaced by
large random numbers so the reader sees GSM-Hard-sized operands.  The host's hidden states over the solution tokens
are cached for a few layers.

    PYTHONPATH=. python build_data.py --n 2800 --out /workspace/qdata
"""
import argparse, json, os, random, re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import OPS, build_prompt, find_calls, calc_ref, strip_solution, EXPR, OPCHAR

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
ap.add_argument("--n", type=int, default=2800)
ap.add_argument("--max_len", type=int, default=200)
ap.add_argument("--layers", default="6,12,18,24")
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--out", default="/workspace/qdata")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)
dev = "cuda"
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
prompt, shot_q = build_prompt(tok)
layers = [int(x) for x in a.layers.split(",")]


def rand_num():
    n = str(rng.randint(1, 10 ** rng.randint(1, 7)))
    if rng.random() < 0.3:
        n += "." + str(rng.randint(1, 99)).rstrip("0") if rng.random() < 0.8 else ""
        n = n.rstrip(".")
    return n


def enlarge(text):
    """Replace each call's operands by random numbers and its written result by the true result."""
    out, last = [], 0
    for m in EXPR.finditer(text):
        a_, b_ = rand_num(), rand_num()
        op = OPCHAR[m.group(3)]
        res = calc_ref(f"{a_}{op}{b_}")
        if res == "":
            continue
        out.append(text[last:m.start(1)] + a_ + m.group(2) + m.group(3) + m.group(4) + b_ + text[m.end(5):m.end()])
        last = m.end()
        tail = re.match(r"(\s*\$?)(-?\d[\d,]*(?:\.\d+)?)", text[last:])
        if tail:
            out.append(tail.group(1) + res); last += tail.end()
    return "".join(out) + text[last:]


rows = [json.loads(l) for l in open("data/gsm8k/train.jsonl")]
rows = [r for r in rows if r["question"] not in shot_q]
rng.shuffle(rows)
rows = rows[: a.n]
os.makedirs(a.out, exist_ok=True)
N, L = len(rows), a.max_len
H = np.lib.format.open_memmap(f"{a.out}/h.npy", mode="w+", dtype=np.float16, shape=(N, len(layers), L, model.config.hidden_size))
gate, p1, p2 = np.zeros((N, L), np.int64), np.full((N, L), -1, np.int64), np.full((N, L), -1, np.int64)
length = np.zeros(N, np.int64)
meta, n_calls, n_lost = [], 0, 0
seqs = []
for i, r in enumerate(rows):
    body, fin = strip_solution(r["answer"])
    text = f" {body}\n#### {fin}"
    if i % 2 == 1:
        text = enlarge(text)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"][:L], enc["offset_mapping"][:L]
    char2tok = {}
    for t, (s, e) in enumerate(offs):
        for c in range(s, e):
            char2tok[c] = t
    for eq_c, op_c, op, a_, b_ in find_calls(text):
        if eq_c not in char2tok or op_c not in char2tok:
            n_lost += 1
            continue
        te, to = char2tok[eq_c], char2tok[op_c]
        gate[i, te], p1[i, te], p2[i, te] = 1 + OPS.index(op), to, te
        n_calls += 1
    length[i] = len(ids)
    q_ids = tok(f"Question: {r['question']}\nAnswer:", add_special_tokens=False)["input_ids"]
    seqs.append(prompt + q_ids + ids)
    meta.append({"text": text, "n_prefix": len(prompt) + len(q_ids)})

eos = tok.convert_tokens_to_ids("<|endoftext|>")
order = sorted(range(N), key=lambda i: len(seqs[i]))
with torch.no_grad():
    for b in range(0, N, a.bs):
        idx = order[b:b + a.bs]
        M = max(len(seqs[i]) for i in idx)
        x = torch.tensor([seqs[i] + [eos] * (M - len(seqs[i])) for i in idx], device=dev)
        att = torch.tensor([[1] * len(seqs[i]) + [0] * (M - len(seqs[i])) for i in idx], device=dev)
        hs = model.model(input_ids=x, attention_mask=att, output_hidden_states=True).hidden_states
        for j, i in enumerate(idx):
            s, n = meta[i]["n_prefix"], int(length[i])
            for li, layer in enumerate(layers):
                H[i, li, :n] = hs[layer][j, s:s + n].float().cpu().numpy().astype(np.float16)
        if (b // a.bs) % 50 == 0:
            print(f"{b}/{N}", flush=True)
H.flush()
np.savez(f"{a.out}/labels.npz", gate=gate, p1=p1, p2=p2, length=length, layers=np.array(layers))
json.dump(meta, open(f"{a.out}/meta.json", "w"))
print(f"DATA_DONE sequences {N} calls {n_calls} lost_to_truncation {n_lost} enlarged {N // 2}")
