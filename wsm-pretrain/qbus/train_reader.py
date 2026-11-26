"""Train the bus reader (gate + pointers) on cached hidden states of the frozen host.

    python train_reader.py --data /workspace/qdata --layer 12 --reader_layers 2 --out /workspace/qreaders/L12_r2.pt
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from common import Reader, OPS

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/workspace/qdata")
ap.add_argument("--layer", type=int, default=12)
ap.add_argument("--reader_layers", type=int, default=0)
ap.add_argument("--d", type=int, default=256)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--n_train", type=int, default=0, help="use only this many training sequences (0 = all but validation)")
ap.add_argument("--n_val", type=int, default=300)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="")
a = ap.parse_args()
torch.manual_seed(a.seed)
dev = "cuda"
lab = np.load(f"{a.data}/labels.npz")
layers = list(lab["layers"])
li = layers.index(a.layer)
H = np.load(f"{a.data}/h.npy", mmap_mode="r")
N, _, L, D = H.shape
Hl = torch.from_numpy(np.ascontiguousarray(H[:, li])).to(dev)          # (N, L, D) float16 on the GPU
gate, p1, p2 = (torch.from_numpy(lab[k]).to(dev) for k in ("gate", "p1", "p2"))
length = torch.from_numpy(lab["length"]).to(dev)
pad = torch.arange(L, device=dev)[None] < length[:, None]
val = torch.arange(N - a.n_val, N, device=dev)
n_train = a.n_train or (N - a.n_val)
train = torch.arange(n_train, device=dev)

reader = Reader(D, a.d, a.reader_layers, max_len=L).to(dev)
opt = torch.optim.AdamW(reader.parameters(), lr=a.lr, weight_decay=0.01)
n_params = sum(p.numel() for p in reader.parameters())


def losses(idx):
    gl, ptr = reader(Hl[idx], pad[idx])
    g, m = gate[idx], pad[idx]
    lg = F.cross_entropy(gl[m], g[m])
    fire = g > 0
    lp = sum(F.cross_entropy(ptr[:, h][fire], t[idx][fire]) for h, t in ((0, p1), (1, p2)))
    return lg + lp, gl, ptr


@torch.no_grad()
def evaluate(idx):
    reader.eval()
    _, gl, ptr = losses(idx)
    g, m = gate[idx], pad[idx]
    pred = gl.argmax(-1)
    fire, pfire = (g > 0) & m, (pred > 0) & m
    tp = (fire & (pred == g)).sum().item()
    ok1 = (ptr[:, 0].argmax(-1) == p1[idx]) & fire
    ok2 = (ptr[:, 1].argmax(-1) == p2[idx]) & fire
    whole = fire & (pred == g) & ok1 & ok2
    reader.train()
    n = max(fire.sum().item(), 1)
    return {"calls": int(fire.sum()), "gate_recall": tp / n, "gate_precision": tp / max(pfire.sum().item(), 1),
            "ptr1_acc": ok1.sum().item() / n, "ptr2_acc": ok2.sum().item() / n, "call_fully_right": whole.sum().item() / n,
            "false_fires_per_1k_tokens": 1000 * (pfire & ~fire).sum().item() / m.sum().item()}


t0 = time.time()
for step in range(a.steps + 1):
    for grp in opt.param_groups:
        grp["lr"] = a.lr * min(1.0, (step + 1) / 200) * (0.5 * (1 + np.cos(np.pi * step / a.steps)))
    idx = train[torch.randint(len(train), (a.bs,), device=dev)]
    loss, _, _ = losses(idx)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
res = evaluate(val)
res.update(layer=a.layer, reader_layers=a.reader_layers, params=n_params, n_train=n_train, steps=a.steps, seconds=round(time.time() - t0))
print("READER " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
if a.out:
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"state": reader.state_dict(), "layer": a.layer, "reader_layers": a.reader_layers, "d": a.d, "d_host": D,
                "max_len": L, "val": res}, a.out)
