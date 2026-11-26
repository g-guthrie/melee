"""Pilot: can a transformer learn, from scratch, to route to a frozen exact tool?

A small causal transformer is trained from random init on a mix of two tasks:

  tool task   v1 v2 ... vK Q si sj =           ->  f(v_i, v_j)
              (K value slots, then apply a binary function f to the two named slots)
  lookup task v1 v2 ... vK L1 si sj =          ->  v_i      (L2 -> v_j)
              (plain retrieval; teaches slot lookup, says nothing about f)
  filler task PAD ... F u w =                  ->  g(u, w)
              (a table of random facts that must be memorised; competes for capacity)

Arms (--arm):
  none      plain transformer (baseline)
  tool      frozen exact tool for f, latent interface, no supervision of the interface
  tool_aux  same, plus an auxiliary loss that tells the readers what the operands are
  mlp       same interface, but a *trainable* MLP sits where the tool would be
  oracle    tool is fed the true operands (upper bound for the tool pathway)

The tool is a multilinear lookup: given two categorical distributions p_a, p_b over
operand values it returns sum_ab p_a[a] p_b[b] onehot(f(a, b)).  It is exact when the
inputs are one-hot and differentiable everywhere, and it has no trainable parameters.
A fraction of (a, b) pairs is never queried in training; accuracy on those pairs is the
main readout, because a model can only get them right by actually using the tool.
"""
import argparse, json, math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="tool", choices=["none", "tool", "tool_aux", "mlp", "oracle"])
    p.add_argument("--table", default="random", choices=["random", "modmul"])
    p.add_argument("--n", type=int, default=61, help="operand / answer values (prime for modmul)")
    p.add_argument("--m", type=int, default=64, help="filler symbols; filler table has m*m facts")
    p.add_argument("--K", type=int, default=4, help="variables bound in the context")
    p.add_argument("--holdout", type=float, default=0.2, help="fraction of (a,b) pairs never trained")
    p.add_argument("--tool_frac", type=float, default=0.4, help="fraction of examples that are the tool task")
    p.add_argument("--lookup_frac", type=float, default=0.2, help="fraction that are plain retrieval")
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--tool_layer", type=int, default=2, help="tool runs after this many blocks")
    p.add_argument("--writer", default="tied", choices=["tied", "learned"])
    p.add_argument("--aux_weight", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task_seed", type=int, default=1234, help="fixes f, g and the held-out pairs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", default="")
    return p.parse_args(argv)


class Task:
    """Token layout: [0,n) values | n..n+K slot names | then m filler symbols | Q, EQ, F, PAD, L1, L2."""

    def __init__(self, a, device):
        g = torch.Generator().manual_seed(a.task_seed)
        n, m, K = a.n, a.m, a.K
        self.n, self.m, self.K, self.a = n, m, K, a
        if a.table == "random":
            f = torch.randint(n, (n, n), generator=g)
        else:
            r = torch.arange(n)
            f = (r[:, None] * r[None, :]) % n
        self.f = f.to(device)
        self.g = torch.randint(n, (m, m), generator=g).to(device)
        perm = torch.randperm(n * n, generator=g)
        n_test = int(a.holdout * n * n)
        self.test_pairs = perm[:n_test].to(device)
        self.train_pairs = perm[n_test:].to(device)
        self.all_pairs = perm.to(device)
        self.NAME0 = n
        self.SYM0 = n + K
        self.Q, self.EQ, self.F, self.PAD, self.L1, self.L2 = (n + K + m + i for i in range(6))
        self.vocab = n + K + m + 6
        self.seq_len = K + 4
        self.device = device

    def tool_batch(self, B, pairs, gen):
        dev, K, n = self.device, self.K, self.n
        idx = pairs[torch.randint(len(pairs), (B,), generator=gen, device=dev)]
        av, bv = idx // n, idx % n
        vals = torch.randint(n, (B, K), generator=gen, device=dev)
        i = torch.randint(K, (B,), generator=gen, device=dev)
        j = (i + torch.randint(1, K, (B,), generator=gen, device=dev)) % K
        rows = torch.arange(B, device=dev)
        vals[rows, i] = av
        vals[rows, j] = bv
        q = torch.full((B,), self.Q, device=dev)
        eq = torch.full((B,), self.EQ, device=dev)
        x = torch.cat([vals, q[:, None], (i + self.NAME0)[:, None],
                       (j + self.NAME0)[:, None], eq[:, None]], dim=1)
        return x, self.f[av, bv], av, bv

    def lookup_batch(self, B, gen):
        """Same context and query names as the tool task, but the answer is one of the two values."""
        x, _, av, bv = self.tool_batch(B, self.all_pairs, gen)
        first = torch.rand(B, generator=gen, device=self.device) < 0.5
        x[:, self.K] = torch.where(first, self.L1, self.L2)
        return x, torch.where(first, av, bv)

    def filler_batch(self, B, gen, all_pairs=False):
        dev, m = self.device, self.m
        if all_pairs:
            u = torch.arange(m, device=dev).repeat_interleave(m)
            w = torch.arange(m, device=dev).repeat(m)
            B = m * m
        else:
            u = torch.randint(m, (B,), generator=gen, device=dev)
            w = torch.randint(m, (B,), generator=gen, device=dev)
        x = torch.full((B, self.seq_len), self.PAD, device=dev)
        x[:, -4] = self.F
        x[:, -3] = u + self.SYM0
        x[:, -2] = w + self.SYM0
        x[:, -1] = self.EQ
        return x, self.g[u, w]

    def train_batch(self, B, gen):
        xt, yt, av, bv = self.tool_batch(B, self.train_pairs, gen)
        xf, yf = self.filler_batch(B, gen)
        xl, yl = self.lookup_batch(B, gen)
        r = torch.rand(B, generator=gen, device=self.device)
        is_tool = r < self.a.tool_frac
        is_look = (~is_tool) & (r < self.a.tool_frac + self.a.lookup_frac)
        x = torch.where(is_tool[:, None], xt, torch.where(is_look[:, None], xl, xf))
        y = torch.where(is_tool, yt, torch.where(is_look, yl, yf))
        return x, y, is_tool, av, bv


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class Model(nn.Module):
    def __init__(self, a, task):
        super().__init__()
        self.a, self.n = a, task.n
        d, n = a.d, task.n
        self.tok = nn.Embedding(task.vocab, d)
        self.pos = nn.Embedding(task.seq_len, d)
        self.blocks = nn.ModuleList([Block(d, a.heads) for _ in range(a.layers)])
        self.ln_f = nn.LayerNorm(d)
        self.has_slot = a.arm != "none"
        if self.has_slot:
            self.ln_t = nn.LayerNorm(d)
            self.read_a, self.read_b = nn.Linear(d, n), nn.Linear(d, n)
            self.gate = nn.Linear(d, 1)
            self.scale = nn.Parameter(torch.tensor(1.0))
            if a.writer == "learned":
                self.write = nn.Linear(n, d, bias=False)
            if a.arm == "mlp":
                self.core = nn.Sequential(nn.Linear(2 * n, 4 * n), nn.GELU(), nn.Linear(4 * n, n))
            else:
                self.register_buffer("f_flat", task.f.reshape(-1).clone())  # frozen tool
        self.apply(self._init)

    @staticmethod
    def _init(mod):
        if isinstance(mod, (nn.Linear, nn.Embedding)):
            nn.init.normal_(mod.weight, std=0.02)
            if getattr(mod, "bias", None) is not None:
                nn.init.zeros_(mod.bias)

    def tool(self, pa, pb):
        """Exact, parameter-free: multilinear extension of the lookup table f."""
        outer = (pa.unsqueeze(-1) * pb.unsqueeze(-2)).flatten(-2)
        q = torch.zeros(*pa.shape[:-1], self.n, device=pa.device, dtype=pa.dtype)
        return q.index_add(-1, self.f_flat, outer)

    def slot(self, h, hard, oracle):
        z = self.ln_t(h)
        la, lb = self.read_a(z), self.read_b(z)
        pa, pb = la.softmax(-1), lb.softmax(-1)
        if hard:
            pa = F.one_hot(la.argmax(-1), self.n).to(pa.dtype)
            pb = F.one_hot(lb.argmax(-1), self.n).to(pb.dtype)
        if oracle is not None:  # true operands at the answer position of tool-task rows
            is_tool, av, bv = oracle
            last_a = torch.where(is_tool[:, None], F.one_hot(av, self.n).to(pa.dtype), pa[:, -1])
            last_b = torch.where(is_tool[:, None], F.one_hot(bv, self.n).to(pb.dtype), pb[:, -1])
            pa = torch.cat([pa[:, :-1], last_a[:, None]], 1)
            pb = torch.cat([pb[:, :-1], last_b[:, None]], 1)
        if self.a.arm == "mlp":
            q = self.core(torch.cat([pa, pb], -1)).softmax(-1)
        else:
            q = self.tool(pa, pb)
        g = torch.sigmoid(self.gate(z))
        w = self.write(q) if self.a.writer == "learned" else q @ self.tok.weight[: self.n]
        return h + g * self.scale * w, la[:, -1], lb[:, -1], g[:, -1, 0]

    def forward(self, x, ablate=False, hard=False, oracle=None):
        h = self.tok(x) + self.pos.weight[: x.shape[1]]
        aux = {}
        for i, blk in enumerate(self.blocks):
            if self.has_slot and i == self.a.tool_layer and not ablate:
                h, aux["la"], aux["lb"], aux["gate"] = self.slot(h, hard, oracle)
            h = blk(h)
        if self.has_slot and self.a.tool_layer == len(self.blocks) and not ablate:
            h, aux["la"], aux["lb"], aux["gate"] = self.slot(h, hard, oracle)
        return self.ln_f(h[:, -1]) @ self.tok.weight.T, aux


@torch.no_grad()
def evaluate(model, task, a, gen):
    model.eval()
    out = {}
    N = 4096
    for split, pairs in (("train", task.train_pairs), ("heldout", task.test_pairs)):
        x, y, av, bv = task.tool_batch(N, pairs, gen)
        orc = (torch.ones(N, dtype=torch.bool, device=x.device), av, bv) if a.arm == "oracle" else None
        logits, aux = model(x, oracle=orc)
        out[f"tool_{split}"] = (logits.argmax(-1) == y).float().mean().item()
        if model.has_slot:
            out[f"read_a_{split}"] = (aux["la"].argmax(-1) == av).float().mean().item()
            out[f"read_b_{split}"] = (aux["lb"].argmax(-1) == bv).float().mean().item()
            out[f"gate_tool_{split}"] = aux["gate"].mean().item()
            out[f"tool_{split}_ablated"] = (model(x, ablate=True)[0].argmax(-1) == y).float().mean().item()
            out[f"tool_{split}_hard"] = (model(x, hard=True, oracle=orc)[0].argmax(-1) == y).float().mean().item()
    x, y = task.lookup_batch(N, gen)
    out["lookup"] = (model(x)[0].argmax(-1) == y).float().mean().item()
    x, y = task.filler_batch(0, gen, all_pairs=True)
    logits, aux = model(x)
    out["filler"] = (logits.argmax(-1) == y).float().mean().item()
    if model.has_slot:
        out["gate_filler"] = aux["gate"].mean().item()
        out["filler_ablated"] = (model(x, ablate=True)[0].argmax(-1) == y).float().mean().item()
    model.train()
    return out


def main(argv=None):
    a = get_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    dev = a.device
    task = Task(a, dev)
    model = Model(a, task).to(dev)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    egen = torch.Generator(device=dev).manual_seed(999)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": other, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.98))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log = open(a.out, "w") if a.out else None

    def emit(rec):
        line = json.dumps(rec)
        if log:
            log.write(line + "\n"); log.flush()
        else:
            print(line, flush=True)

    emit({"config": vars(a), "params": n_params})
    t0 = time.time()
    for step in range(a.steps + 1):
        if step % a.eval_every == 0:
            rec = {"step": step, "time": round(time.time() - t0, 1)}
            rec.update(evaluate(model, task, a, egen))
            if step:
                rec["loss"] = loss.item()
            emit(rec)
            if step == a.steps:
                break
        for grp in opt.param_groups:
            grp["lr"] = a.lr * min(1.0, (step + 1) / a.warmup)
        x, y, is_tool, av, bv = task.train_batch(a.batch, gen)
        logits, aux = model(x, oracle=(is_tool, av, bv) if a.arm == "oracle" else None)
        loss = F.cross_entropy(logits, y)
        if a.arm == "tool_aux":
            la, lb = aux["la"][is_tool], aux["lb"][is_tool]
            loss = loss + a.aux_weight * (F.cross_entropy(la, av[is_tool]) + F.cross_entropy(lb, bv[is_tool]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    if log:
        log.close()


if __name__ == "__main__":
    main()
