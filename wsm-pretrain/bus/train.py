"""Train a small transformer from scratch with frozen exact tools on a shared value bus.

Task: tiny straight-line programs over values 0..n-1.  Each statement is 8 tokens:

    NAME = a OP b -> result ;        OP in {id, +, *, max};  a, b are literals or earlier names

A statement's result is shown only some of the time; hidden results must be carried
internally to later statements.  The loss is next-token prediction of shown results.

The bus: every position has one value register (a distribution over 0..n-1).  A literal
token fills its own register.  Elsewhere, two pointer heads pick two earlier positions,
the frozen tools compute on the registers found there, and a gate picks which tool (if
any) writes this position's register.  A few parallel sweeps let values flow through
chains of pointers.  Registers are then added to the residual stream as token embeddings.

Trainable: where to point and which tool to fire.  Frozen: what the tools compute.
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools import Tools  # noqa: E402

N_OPS = 4  # copy, add, mul, max
SLOT = 8   # tokens per statement


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tools", type=int, default=1, help="1: bus with frozen tools, 0: plain transformer")
    p.add_argument("--guide", default="fade", choices=["none", "full", "fade"],
                   help="supervision of gates and pointers: never, always, or faded out early")
    p.add_argument("--fade_start", type=float, default=0.10)
    p.add_argument("--fade_end", type=float, default=0.30)
    p.add_argument("--n", type=int, default=97)
    p.add_argument("--S", type=int, default=6, help="statements per program")
    p.add_argument("--names", type=int, default=6)
    p.add_argument("--p_var", type=float, default=0.6, help="chance an operand is a name rather than a literal")
    p.add_argument("--p_print", type=float, default=0.5, help="chance a statement's result is shown")
    p.add_argument("--holdout", type=float, default=0.2, help="fraction of (op,a,b) never executed in training")
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--pre", type=int, default=3, help="layers before the bus in each stage")
    p.add_argument("--post", type=int, default=1, help="layers after the last stage")
    p.add_argument("--stages", type=int, default=1)
    p.add_argument("--loop", type=int, default=0, help="1: tie the stage layers' weights (looped transformer)")
    p.add_argument("--attnres", type=int, default=0, help="1: attention over earlier layer outputs replaces the residual sum")
    p.add_argument("--sweeps", type=int, default=0, help="bus sweeps per stage; 0 means 2*S")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task_seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="")
    p.add_argument("--wiring", type=int, default=0, help="1: after training, report where pointers point and which tools fire")
    return p.parse_args(argv)


class Task:
    """Tokens: [0,n) values | names | EQ ARROW SEMI HIDE NULLARG | one token per op."""

    def __init__(self, a, device):
        self.a, self.n, self.S, self.NV, self.dev = a, a.n, a.S, a.names, device
        n = a.n
        self.NAME0 = n
        self.EQ, self.ARROW, self.SEMI, self.HIDE, self.NULLARG = (n + a.names + i for i in range(5))
        self.OP0 = n + a.names + 5
        self.vocab = self.OP0 + N_OPS
        self.T = SLOT * a.S
        g = torch.Generator().manual_seed(a.task_seed)
        self.held = (torch.rand(N_OPS, n, n, generator=g) < a.holdout).to(device)
        self.held[0] = False
        self.arrow_pos = torch.arange(a.S, device=device) * SLOT + 5

    def batch(self, B, gen, train=True, p_print=None):
        dev, n, S, NV = self.dev, self.n, self.S, self.NV
        p_print = self.a.p_print if p_print is None else p_print
        rows = torch.arange(B, device=dev)
        rnd = lambda *s: torch.rand(*s, generator=gen, device=dev)
        rint = lambda hi, *s: torch.randint(hi, s, generator=gen, device=dev)
        x = torch.zeros(B, self.T, dtype=torch.long, device=dev)
        gate = torch.zeros(B, self.T, dtype=torch.long, device=dev)       # 0 none, 1..4 tool
        p1 = torch.full((B, self.T), -1, dtype=torch.long, device=dev)    # pointer targets, -1 = unlabeled
        p2 = torch.full((B, self.T), -1, dtype=torch.long, device=dev)
        val = torch.zeros(B, NV, dtype=torch.long, device=dev)
        defined = torch.zeros(B, NV, dtype=torch.bool, device=dev)
        defpos = torch.zeros(B, NV, dtype=torch.long, device=dev)
        ldepth = torch.zeros(B, NV, dtype=torch.long, device=dev)
        y = torch.zeros(B, S, dtype=torch.long, device=dev)
        shown = torch.zeros(B, S, dtype=torch.bool, device=dev)
        depth = torch.zeros(B, S, dtype=torch.long, device=dev)
        ops = torch.zeros(B, S, dtype=torch.long, device=dev)
        held = torch.zeros(B, S, dtype=torch.bool, device=dev)
        for s in range(S):
            base = SLOT * s
            name, op = rint(NV, B), rint(N_OPS, B)
            operands = []
            for off in (2, 4):
                is_var = (rnd(B) < self.a.p_var) & defined.any(1)
                var = rnd(B, NV).masked_fill(~defined, -1.0).argmax(1)
                lit = rint(n, B)
                operands.append(dict(v=torch.where(is_var, val[rows, var], lit),
                                     tok=torch.where(is_var, self.NAME0 + var, lit),
                                     dep=torch.where(is_var, ldepth[rows, var], torch.zeros_like(lit)),
                                     is_var=is_var, src=defpos[rows, var], pos=base + off))
            A, Bo = operands
            is_held = self.held[op, A["v"], Bo["v"]]
            if train:  # never execute a held-out (op, a, b) in training: fall back to a copy
                op = torch.where(is_held, torch.zeros_like(op), op)
            unary = op == 0
            res = Tools.exact(op, A["v"], Bo["v"], n)
            x[:, base] = self.NAME0 + name
            x[:, base + 1] = self.EQ
            x[:, base + 2] = A["tok"]
            x[:, base + 3] = self.OP0 + op
            x[:, base + 4] = torch.where(unary, torch.full_like(op, self.NULLARG), Bo["tok"])
            x[:, base + 5] = self.ARROW
            show = rnd(B) < p_print if s < S - 1 else torch.ones(B, dtype=torch.bool, device=dev)
            x[:, base + 6] = torch.where(show, res, torch.full_like(res, self.HIDE))
            x[:, base + 7] = self.SEMI
            # routing labels: a name used as an operand copies from where that name was last computed
            gate[:, A["pos"]] = A["is_var"].long()
            p1[:, A["pos"]] = torch.where(A["is_var"], A["src"], torch.full_like(op, -1))
            b_var = Bo["is_var"] & ~unary
            gate[:, Bo["pos"]] = b_var.long()
            p1[:, Bo["pos"]] = torch.where(b_var, Bo["src"], torch.full_like(op, -1))
            # the arrow position computes the statement from its two operand positions
            gate[:, base + 5] = op + 1
            p1[:, base + 5] = base + 2
            p2[:, base + 5] = torch.where(unary, torch.full_like(op, -1), torch.full_like(op, base + 4))
            d = 1 + torch.maximum(A["dep"], torch.where(unary, torch.zeros_like(op), Bo["dep"]))
            y[:, s], shown[:, s], depth[:, s], ops[:, s], held[:, s] = res, show, d, op, is_held & ~unary
            val[rows, name], defined[rows, name], defpos[rows, name] = res, True, base + 5
            ldepth[rows, name] = torch.where(show, torch.zeros_like(d), d)
        return dict(x=x, y=y, shown=shown, depth=depth, ops=ops, held=held, gate=gate, p1=p1, p2=p2)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        """Returns the block's update (what a residual connection would add to x)."""
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        att = self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D))
        return att + self.mlp(self.ln2(x + att))


class Bus(nn.Module):
    """Pointer heads and a gate (trainable) around the frozen tools."""

    def __init__(self, d, n, sweeps):
        super().__init__()
        self.n, self.sweeps = n, sweeps
        self.ln = nn.LayerNorm(d)
        self.q = nn.Linear(d, 2 * d)   # two pointer heads
        self.k = nn.Linear(d, 2 * d)
        self.gate = nn.Linear(d, 1 + N_OPS)
        self.tools = Tools(n)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h, lit, lit_mask, hard=False, knock=()):
        B, T, D = h.shape
        z = self.ln(h)
        q, k = self.q(z).view(B, T, 2, D), self.k(z).view(B, T, 2, D)
        ptr = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(D)         # (B, 2, T, T)
        causal = torch.ones(T, T, dtype=torch.bool, device=h.device).tril(-1)  # strictly earlier positions
        ptr = ptr.masked_fill(~causal, float("-inf"))
        ptr[:, :, 0, 0] = 0.0  # position 0 has nothing earlier; give it a harmless self-pointer
        gl = self.gate(z)
        att, g = ptr.softmax(-1), gl.softmax(-1)
        if hard:
            att = F.one_hot(ptr.argmax(-1), T).to(att.dtype)
            g = F.one_hot(gl.argmax(-1), 1 + N_OPS).to(g.dtype)
        if knock:
            keep = torch.ones(1 + N_OPS, device=h.device)
            keep[[1 + j for j in knock]] = 0.0
            g = g * keep
        g_tools = g[..., 1:].unsqueeze(-1)                                   # (B, T, 4, 1)
        reg = lit
        for _ in range(self.sweeps):
            a, b = att[:, 0] @ reg, att[:, 1] @ reg                          # operands found by the pointers
            out = (g_tools * self.tools.all(a, b)).sum(-2)
            reg = lit + (~lit_mask).unsqueeze(-1) * out
        return reg, dict(ptr=ptr, gate=gl)


class Model(nn.Module):
    def __init__(self, a, task):
        super().__init__()
        self.a, self.n = a, a.n
        d = a.d
        self.tok = nn.Embedding(task.vocab, d)
        self.pos = nn.Embedding(task.T, d)
        n_stage_layers = a.pre if a.loop else a.pre * a.stages
        self.stage_layers = nn.ModuleList([Block(d, a.heads) for _ in range(n_stage_layers)])
        self.post_layers = nn.ModuleList([Block(d, a.heads) for _ in range(a.post)])
        self.ln_f = nn.LayerNorm(d)
        if a.tools:
            n_bus = 1 if a.loop else a.stages
            self.bus = nn.ModuleList([Bus(d, a.n, a.sweeps or 2 * a.S) for _ in range(n_bus)])
        if a.attnres:  # one learned query per place a layer (or the head) reads its input
            self.res_q = nn.Parameter(torch.zeros(a.pre * a.stages + a.post + 1, d))
            self.res_ln = nn.LayerNorm(d)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def _mix(self, outs, i):
        """Attention residual: a per-token softmax over all earlier outputs, weighted by a learned query."""
        st = torch.stack(outs, 2)                                            # (B, T, L, D)
        w = (self.res_ln(st) @ self.res_q[i]).softmax(-1) * len(outs)        # scaled so equal weights give the plain sum
        return (st * w.unsqueeze(-1)).sum(2)

    def forward(self, x, hard=False, knock=(), no_bus=False):
        a = self.a
        lit_mask = x < self.n
        lit = F.one_hot(x.clamp(max=self.n - 1), self.n).float() * lit_mask.unsqueeze(-1)
        outs = [self.tok(x) + self.pos.weight[: x.shape[1]]]
        h = outs[0]
        aux, li = [], 0
        for st in range(a.stages):
            for j in range(a.pre):
                blk = self.stage_layers[j if a.loop else st * a.pre + j]
                h_in = self._mix(outs, li) if a.attnres else h
                outs.append(blk(h_in)); h = h_in + outs[-1] if not a.attnres else None
                li += 1
                if a.attnres:
                    h = self._mix(outs, li)
            if a.tools and not no_bus:
                bus = self.bus[0 if a.loop else st]
                reg, info = bus(h, lit, lit_mask, hard=hard, knock=knock)
                aux.append(info)
                outs.append(bus.scale * (reg @ self.tok.weight[: self.n]))
                h = self._mix(outs, li) if a.attnres else h + outs[-1]
        for blk in self.post_layers:
            h_in = self._mix(outs, li) if a.attnres else h
            outs.append(blk(h_in))
            li += 1
            h = self._mix(outs, li) if a.attnres else h_in + outs[-1]
        return self.ln_f(h) @ self.tok.weight.T, aux


def guide_loss(aux, b):
    """Cross-entropy of gates and pointers against the routing labels, averaged over stages."""
    tot = 0.0
    for info in aux:
        tot = tot + F.cross_entropy(info["gate"].flatten(0, 1), b["gate"].flatten())
        for h, tgt in ((0, b["p1"]), (1, b["p2"])):
            m = tgt >= 0
            if m.any():
                tot = tot + F.cross_entropy(info["ptr"][:, h][m], tgt[m])
    return tot / max(len(aux), 1)


@torch.no_grad()
def evaluate(model, task, a, gen):
    model.eval()
    out, N = {}, 2048
    ap = task.arrow_pos

    def acc(b, mask, **kw):
        pred = model(b["x"], **kw)[0][:, ap].argmax(-1)
        return ((pred == b["y"]) & mask).sum().item() / max(mask.sum().item(), 1)

    # mixed: results shown half the time, as in training but with held-out operations allowed
    b = task.batch(N, gen, train=False)
    binary = b["ops"] > 0
    out["shown_seen"] = acc(b, b["shown"] & ~b["held"])
    out["shown_heldout"] = acc(b, b["shown"] & b["held"])
    for i, name in enumerate(Tools.NAMES):
        out[f"op_{name}"] = acc(b, b["shown"] & (b["ops"] == i) & ~b["held"])
    # silent: no intermediate result is shown, only the last statement is asked
    s = task.batch(N, gen, train=False, p_print=0.0)
    last = torch.zeros_like(s["shown"]); last[:, -1] = True
    out["silent_final"] = acc(s, last)
    for d in (1, 2, 3, 4):
        m = last & ((s["depth"] == d) if d < 4 else (s["depth"] >= 4))
        out[f"silent_depth{d}{'plus' if d == 4 else ''}"] = acc(s, m)
        out[f"n_depth{d}"] = int(m.sum())
    if a.tools:
        logits, aux = model(b["x"])
        info = aux[-1]
        out["gate_acc"] = (info["gate"].argmax(-1) == b["gate"]).float().mean().item()
        fire = b["gate"] > 0
        out["gate_acc_fire"] = (info["gate"].argmax(-1) == b["gate"])[fire].float().mean().item()
        for h, tgt in ((0, b["p1"]), (1, b["p2"])):
            m = tgt >= 0
            out[f"ptr{h + 1}_acc"] = (info["ptr"][:, h].argmax(-1) == tgt)[m].float().mean().item()
        out["silent_final_hard"] = acc(s, last, hard=True)
        out["silent_final_nobus"] = acc(s, last, no_bus=True)
        out["shown_seen_nobus"] = acc(b, b["shown"] & ~b["held"], no_bus=True)
        for j, name in enumerate(Tools.NAMES):
            out[f"silent_final_no_{name}"] = acc(s, last, knock=(j,))
        out["silent_final_no_add_mul"] = acc(s, last, knock=(1, 2))
        out["silent_final_no_copy_add"] = acc(s, last, knock=(0, 1))
    model.train()
    return out


@torch.no_grad()
def wiring_report(model, task, gen):
    """Describe the routing the model actually uses, by position within a statement."""
    model.eval()
    N, S = 4096, task.S
    b = task.batch(N, gen, train=False)
    info = model(b["x"])[1][-1]
    gate = info["gate"].softmax(-1)                       # (N, T, 5)
    tgt = info["ptr"].argmax(-1)                          # (N, 2, T)
    rep = {"gate_by_slot": {}, "arrow": {}, "operand": {}}
    names = ["none"] + Tools.NAMES
    slot = torch.arange(task.T, device=gate.device) % SLOT
    for sl in range(SLOT):
        g = gate[:, slot == sl].mean((0, 1))
        rep["gate_by_slot"][str(sl)] = {n: round(float(v), 3) for n, v in zip(names, g)}
    base = torch.arange(1, S, device=gate.device) * SLOT   # skip the first statement (nothing earlier to use)
    for h, off in ((0, 2), (1, 4)):
        t = tgt[:, h][:, base + 5]                         # where the arrow's pointer h lands
        direct = b["p1"][:, base + off]                    # where that operand's value was computed, if it is a name
        is_var = direct >= 0
        binary = b["ops"][:, 1:] > 0
        m = is_var & (binary if h == 1 else torch.ones_like(binary))
        cats = {"operand_position_as_designed": t == (base + off), "straight_to_where_it_was_computed": t == direct,
                "the_printed_result_token": t == direct + 1}
        other = m.clone()
        for k, v in cats.items():
            rep["arrow"][f"ptr{h + 1}_{k}"] = round(float((v & m).sum() / m.sum().clamp(min=1)), 3)
            other &= ~v
        rep["arrow"][f"ptr{h + 1}_elsewhere"] = round(float(other.sum() / m.sum().clamp(min=1)), 3)
        lit = ~is_var & (binary if h == 1 else torch.ones_like(binary))
        rep["arrow"][f"ptr{h + 1}_literal_operand_hit"] = round(float(((t == (base + off)) & lit).sum() / lit.sum().clamp(min=1)), 3)
        # does the operand position itself fetch the value (the designed copy step)?
        g_op = gate[:, base + off][..., 1]
        rep["operand"][f"slot{off}_copy_gate_when_name"] = round(float(g_op[is_var].mean()), 3)
        t_op = tgt[:, 0][:, base + off]
        rep["operand"][f"slot{off}_points_to_source"] = round(float(((t_op == direct) & is_var).sum() / is_var.sum().clamp(min=1)), 3)
    model.train()
    return rep


def main(argv=None):
    a = get_args(argv)
    torch.manual_seed(a.seed)
    dev = a.device
    task = Task(a, dev)
    model = Model(a, task).to(dev)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    egen = torch.Generator(device=dev).manual_seed(999)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": other, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.98))
    log = open(a.out, "w") if a.out else None

    def emit(rec):
        line = json.dumps(rec)
        if log:
            log.write(line + "\n"); log.flush()
        else:
            print(line, flush=True)

    emit({"config": vars(a), "params": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    ap, t0 = task.arrow_pos, time.time()
    for step in range(a.steps + 1):
        if step % a.eval_every == 0:
            rec = {"step": step, "time": round(time.time() - t0, 1)}
            rec.update(evaluate(model, task, a, egen))
            if step:
                rec["loss"], rec["guide_w"] = main_loss.item(), gw
            emit(rec)
            if step == a.steps:
                break
        for grp in opt.param_groups:
            grp["lr"] = a.lr * min(1.0, (step + 1) / a.warmup)
        b = task.batch(a.batch, gen)
        logits, aux = model(b["x"])
        main_loss = F.cross_entropy(logits[:, ap][b["shown"]], b["y"][b["shown"]])
        frac = step / a.steps
        gw = 0.0
        if a.tools and a.guide == "full":
            gw = 1.0
        elif a.tools and a.guide == "fade":
            gw = min(1.0, max(0.0, (a.fade_end - frac) / (a.fade_end - a.fade_start)))
        loss = main_loss + (gw * guide_loss(aux, b) if gw > 0 else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    if a.wiring and a.tools:
        emit({"wiring": wiring_report(model, task, egen)})
    if log:
        log.close()


if __name__ == "__main__":
    main()
