"""Bus v2: eight frozen tools, a counter built from increments, and value-dependent branching.

Programs are a start token followed by 8-token statements:

    NAME = a OP b -> result ;

OP is one of: id + - * max < == (pure: computed by the same-named tool from a and b),
  cnt   how many times name `a` has been assigned so far (a chain of increments),
  sel   `a` if the previous statement's result is non-zero, else `b` (a branch).

Every position has one value register.  Two trainable pointer heads choose two earlier
positions and a trainable gate chooses which frozen tool (if any) writes the register.
`sel` cannot be routed in a single stage: the pointer has to depend on a value the bus
computed, so it needs a second stage (a second pass of the loop).
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools import Tools  # noqa: E402

TOOLS = Tools.NAMES                      # copy add sub mul max lt eq inc
NT = len(TOOLS)
OPS = ["copy", "add", "sub", "mul", "max", "lt", "eq", "cnt", "sel"]   # statement kinds
PURE = 7                                 # OPS[:PURE] map to the same-named tool
CNT, SEL = OPS.index("cnt"), OPS.index("sel")
LT = OPS.index("lt")
T_COPY, T_INC = TOOLS.index("copy"), TOOLS.index("inc")
SLOT = 8


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tools", type=int, default=1)
    p.add_argument("--guide", default="fade", choices=["none", "full", "fade"])
    p.add_argument("--fade_start", type=float, default=0.10)
    p.add_argument("--fade_end", type=float, default=0.30)
    p.add_argument("--n", type=int, default=97)
    p.add_argument("--S", type=int, default=6)
    p.add_argument("--names", type=int, default=5)
    p.add_argument("--p_var", type=float, default=0.6)
    p.add_argument("--p_print", type=float, default=0.5)
    p.add_argument("--p_branch", type=float, default=0.2, help="chance a statement starts a compare-then-branch pair")
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--pre", type=int, default=3)
    p.add_argument("--post", type=int, default=1)
    p.add_argument("--stages", type=int, default=1)
    p.add_argument("--loop", type=int, default=0)
    p.add_argument("--attnres", type=int, default=0)
    p.add_argument("--sweeps", type=int, default=0)
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
    return p.parse_args(argv)


class Task:
    """Tokens: [0,n) values | names | BOS EQ ARROW SEMI HIDE NULLARG | one token per statement kind."""

    def __init__(self, a, device):
        self.a, self.n, self.S, self.NV, self.dev = a, a.n, a.S, a.names, device
        n = a.n
        self.NAME0 = n
        self.BOS, self.EQ, self.ARROW, self.SEMI, self.HIDE, self.NULLARG = (n + a.names + i for i in range(6))
        self.OP0 = n + a.names + 6
        self.vocab = self.OP0 + len(OPS)
        self.T = 1 + SLOT * a.S
        g = torch.Generator().manual_seed(a.task_seed)
        self.held = (torch.rand(PURE, n, n, generator=g) < a.holdout).to(device)
        self.held[0] = False
        self.arrow_pos = 1 + torch.arange(a.S, device=device) * SLOT + 5

    def batch(self, B, gen, train=True, p_print=None):
        a, dev, n, S, NV = self.a, self.dev, self.n, self.S, self.NV
        p_print = a.p_print if p_print is None else p_print
        rows = torch.arange(B, device=dev)
        rnd = lambda *s: torch.rand(*s, generator=gen, device=dev)
        rint = lambda hi, *s: torch.randint(hi, s, generator=gen, device=dev)
        Z = lambda *s, dt=torch.long: torch.zeros(*s, dtype=dt, device=dev)
        full = lambda v: torch.full((B,), v, dtype=torch.long, device=dev)
        x = Z(B, self.T); x[:, 0] = self.BOS
        gate = Z(B, self.T)
        p1, p2 = Z(B, self.T) - 1, Z(B, self.T) - 1
        sel_arrow = Z(B, self.T, dt=torch.bool)
        val, defpos, namepos, count, ldepth = Z(B, NV), Z(B, NV), Z(B, NV), Z(B, NV), Z(B, NV)
        defined, tainted = Z(B, NV, dt=torch.bool), Z(B, NV, dt=torch.bool)
        y, depth, ops = Z(B, S), Z(B, S), Z(B, S)
        shown, held = Z(B, S, dt=torch.bool), Z(B, S, dt=torch.bool)
        sel_next = Z(B, dt=torch.bool)       # this statement must be the branch of a compare/branch pair
        prev_res, prev_dep = Z(B), Z(B)
        for s in range(S):
            base = 1 + SLOT * s
            name = rint(NV, B)
            op = rint(CNT + 1, B)                                  # copy .. cnt
            start_pair = (rnd(B) < a.p_branch) & ~sel_next & (s < S - 1)
            op = torch.where(start_pair, full(LT), op)
            op = torch.where(sel_next, full(SEL), op)
            # the name token of every statement counts assignments: one more than the previous one of that name
            gate[:, base] = 1 + T_INC
            p1[:, base] = namepos[rows, name]                       # 0 (the start token) if never assigned
            count[rows, name] += 1
            namepos[rows, name] = base
            any_def = defined.any(1)
            op = torch.where((op == CNT) & ~any_def, full(0), op)
            operands = []
            for k, off in enumerate((2, 4)):
                is_var = ((rnd(B) < a.p_var) | ((op == CNT) & (k == 0))) & any_def
                var = rnd(B, NV).masked_fill(~defined, -1.0).argmax(1)
                lit = rint(n, B)
                operands.append(dict(v=torch.where(is_var, val[rows, var], lit),
                                     tok=torch.where(is_var, self.NAME0 + var, lit), var=var, is_var=is_var,
                                     dep=torch.where(is_var, ldepth[rows, var], Z(B)),
                                     taint=is_var & tainted[rows, var], src=defpos[rows, var], pos=base + off))
            A, Bo = operands
            pure = op < PURE
            is_held = pure & self.held[op.clamp(max=PURE - 1), A["v"], Bo["v"]]
            if train:
                op = torch.where(is_held, full(0), op)
                start_pair = start_pair & ~is_held
            unary = (op == 0) | (op == CNT)
            # a compare can only feed a branch if it does not itself depend on an earlier branch
            start_pair = start_pair & (op == LT) & ~A["taint"] & ~Bo["taint"]
            flag = prev_res != 0
            res = Tools.exact(op.clamp(max=PURE - 1), A["v"], Bo["v"], n)
            res = torch.where(op == CNT, count[rows, A["var"]], res)
            res = torch.where(op == SEL, torch.where(flag, A["v"], Bo["v"]), res)
            x[:, base] = self.NAME0 + name
            x[:, base + 1] = self.EQ
            x[:, base + 2] = A["tok"]
            x[:, base + 3] = self.OP0 + op
            x[:, base + 4] = torch.where(unary, full(self.NULLARG), Bo["tok"])
            x[:, base + 5] = self.ARROW
            show = rnd(B) < p_print if s < S - 1 else torch.ones(B, dtype=torch.bool, device=dev)
            x[:, base + 6] = torch.where(show, res, full(self.HIDE))
            x[:, base + 7] = self.SEMI
            # operands that are names copy the value from where that name was last computed
            gate[:, A["pos"]] = A["is_var"].long() * (1 + T_COPY)
            p1[:, A["pos"]] = torch.where(A["is_var"], A["src"], full(-1))
            b_var = Bo["is_var"] & ~unary
            gate[:, Bo["pos"]] = b_var.long() * (1 + T_COPY)
            p1[:, Bo["pos"]] = torch.where(b_var, Bo["src"], full(-1))
            # the arrow position computes the statement
            arrow = base + 5
            g_arrow = torch.where(pure, 1 + op, full(1 + T_COPY))
            t1 = torch.where(op == CNT, namepos[rows, A["var"]], full(base + 2))
            t1 = torch.where(op == SEL, torch.where(flag, full(base + 2), full(base + 4)), t1)
            t2 = torch.where(pure & (op != 0), full(base + 4), full(-1))
            gate[:, arrow], p1[:, arrow], p2[:, arrow] = g_arrow, t1, t2
            sel_arrow[:, arrow] = op == SEL
            dep_b = torch.where(unary, Z(B), Bo["dep"])
            d = 1 + torch.maximum(A["dep"], dep_b)
            d = torch.where(op == SEL, torch.maximum(d, 1 + prev_dep), d)
            d = torch.where(op == CNT, full(1), d)
            y[:, s], shown[:, s], depth[:, s], ops[:, s], held[:, s] = res, show, d, op, is_held
            taint = torch.where(op == CNT, Z(B, dt=torch.bool),
                                (op == SEL) | A["taint"] | (Bo["taint"] & ~unary))
            val[rows, name], defined[rows, name], defpos[rows, name] = res, True, arrow
            tainted[rows, name] = taint
            ldepth[rows, name] = torch.where(show, Z(B), d)
            prev_res, prev_dep = res, torch.where(show, Z(B), d)
            sel_next = start_pair
        # routing labels per stage: a branch can only be routed in the last stage
        labels = []
        for st in range(max(a.stages, 1)):
            if st == a.stages - 1 or a.stages <= 1:
                labels.append((gate, p1, p2))
            else:
                labels.append((gate.masked_fill(sel_arrow, 0), p1.masked_fill(sel_arrow, -1), p2))
        return dict(x=x, y=y, shown=shown, depth=depth, ops=ops, held=held, labels=labels)


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
        self.q, self.k = nn.Linear(d, 2 * d), nn.Linear(d, 2 * d)
        self.gate = nn.Linear(d, 1 + NT)
        self.tools = Tools(n)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h, lit, lit_mask, hard=False, knock=()):
        B, T, D = h.shape
        z = self.ln(h)
        q, k = self.q(z).view(B, T, 2, D), self.k(z).view(B, T, 2, D)
        ptr = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(D)
        causal = torch.ones(T, T, dtype=torch.bool, device=h.device).tril(-1)
        causal[0, 0] = True                      # the start token has nothing earlier; it points at itself
        ptr = ptr.masked_fill(~causal, float("-inf"))
        gl = self.gate(z)
        att, g = ptr.softmax(-1), gl.softmax(-1)
        if hard:
            att = F.one_hot(ptr.argmax(-1), T).to(att.dtype)
            g = F.one_hot(gl.argmax(-1), 1 + NT).to(g.dtype)
        if knock:
            keep = torch.ones(1 + NT, device=h.device)
            keep[[1 + j for j in knock]] = 0.0
            g = g * keep
        g_tools = g[..., 1:].unsqueeze(-1)
        reg = lit
        for _ in range(self.sweeps):
            a, b = att[:, 0] @ reg, att[:, 1] @ reg
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
        self.stage_layers = nn.ModuleList([Block(d, a.heads) for _ in range(a.pre if a.loop else a.pre * a.stages)])
        self.post_layers = nn.ModuleList([Block(d, a.heads) for _ in range(a.post)])
        self.ln_f = nn.LayerNorm(d)
        if a.tools:
            self.bus = nn.ModuleList([Bus(d, a.n, a.sweeps or 2 * a.S) for _ in range(1 if a.loop else a.stages)])
        if a.attnres:
            self.res_q = nn.Parameter(torch.zeros(a.pre * a.stages + a.post + 1, d))
            self.res_ln = nn.LayerNorm(d)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def _state(self, outs, i):
        """What a layer reads. Plain: the sum of everything so far. Attention residual: a per-token
        softmax over earlier outputs under a learned query, scaled so equal weights give the plain sum."""
        if not self.a.attnres:
            return sum(outs)
        st = torch.stack(outs, 2)
        w = (self.res_ln(st) @ self.res_q[i]).softmax(-1) * len(outs)
        return (st * w.unsqueeze(-1)).sum(2)

    def forward(self, x, hard=False, knock=(), no_bus=False):
        a = self.a
        lit_mask = x < self.n
        lit = F.one_hot(x.clamp(max=self.n - 1), self.n).float() * lit_mask.unsqueeze(-1)
        outs = [self.tok(x) + self.pos.weight[: x.shape[1]]]
        aux, li = [], 0
        for st in range(a.stages):
            for j in range(a.pre):
                outs.append(self.stage_layers[j if a.loop else st * a.pre + j](self._state(outs, li)))
                li += 1
            if a.tools and not no_bus:
                bus = self.bus[0 if a.loop else st]
                reg, info = bus(self._state(outs, li), lit, lit_mask, hard=hard, knock=knock)
                aux.append(info)
                outs.append(bus.scale * (reg @ self.tok.weight[: self.n]))
        for blk in self.post_layers:
            outs.append(blk(self._state(outs, li)))
            li += 1
        return self.ln_f(self._state(outs, li)) @ self.tok.weight.T, aux


def guide_loss(aux, b):
    tot = 0.0
    for info, (gate, p1, p2) in zip(aux, b["labels"]):
        tot = tot + F.cross_entropy(info["gate"].flatten(0, 1), gate.flatten())
        for h, tgt in ((0, p1), (1, p2)):
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

    b = task.batch(N, gen, train=False)
    out["shown_seen"] = acc(b, b["shown"] & ~b["held"])
    out["shown_heldout"] = acc(b, b["shown"] & b["held"])
    for i, name in enumerate(OPS):
        out[f"op_{name}"] = acc(b, b["shown"] & (b["ops"] == i) & ~b["held"])
    s = task.batch(N, gen, train=False, p_print=0.0)
    last = torch.zeros_like(s["shown"]); last[:, -1] = True
    out["silent_final"] = acc(s, last)
    for d in (1, 2, 3, 4):
        m = last & ((s["depth"] == d) if d < 4 else (s["depth"] >= 4))
        out[f"silent_depth{d}{'plus' if d == 4 else ''}"] = acc(s, m)
    hid = ~s["shown"]  # every hidden statement in silent programs, by kind (asks the model what it holds internally)
    if a.tools:
        logits, aux = model(b["x"])
        info, (gate, p1, p2) = aux[-1], b["labels"][-1]
        out["gate_acc"] = (info["gate"].argmax(-1) == gate).float().mean().item()
        out["gate_acc_fire"] = (info["gate"].argmax(-1) == gate)[gate > 0].float().mean().item()
        for h, tgt in ((0, p1), (1, p2)):
            m = tgt >= 0
            out[f"ptr{h + 1}_acc"] = (info["ptr"][:, h].argmax(-1) == tgt)[m].float().mean().item()
        sel_m = (b["ops"] == SEL)
        if sel_m.any():
            apos = ap.unsqueeze(0).expand_as(b["ops"])[sel_m]
            rows = torch.arange(N, device=apos.device).unsqueeze(1).expand_as(b["ops"])[sel_m]
            out["ptr_acc_branch"] = (info["ptr"][rows, 0, apos].argmax(-1) == p1[rows, apos]).float().mean().item()
        out["silent_final_hard"] = acc(s, last, hard=True)
        out["silent_final_nobus"] = acc(s, last, no_bus=True)
        out["shown_seen_nobus"] = acc(b, b["shown"] & ~b["held"], no_bus=True)
        for j, name in enumerate(TOOLS):
            out[f"silent_final_no_{name}"] = acc(s, last, knock=(j,))
        for i, j in ((1, 3), (0, 1), (0, 7), (5, 0)):
            out[f"silent_final_no_{TOOLS[i]}_{TOOLS[j]}"] = acc(s, last, knock=(i, j))
    model.train()
    return out


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
        frac, gw = step / a.steps, 0.0
        if a.tools and a.guide == "full":
            gw = 1.0
        elif a.tools and a.guide == "fade":
            gw = min(1.0, max(0.0, (a.fade_end - frac) / (a.fade_end - a.fade_start)))
        loss = main_loss + (gw * guide_loss(aux, b) if gw > 0 else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    if log:
        log.close()


if __name__ == "__main__":
    main()
