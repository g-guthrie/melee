"""Frozen exact tools that operate on distributions over values 0..n-1.

Every tool is multilinear in its inputs: on one-hot inputs it returns the exact one-hot
answer, and on soft inputs it returns the expectation, so gradients flow through it.
None of them has a trainable parameter.  Inputs may carry less than unit mass (the
missing mass means "no value here"); output mass is the product of the input masses.
"""
import torch
import torch.nn as nn


def _is_prime(n):
    return n > 1 and all(n % k for k in range(2, int(n ** 0.5) + 1))


def _primitive_root(n):
    for g in range(2, n):
        seen, v = set(), 1
        for _ in range(n - 1):
            v = v * g % n
            seen.add(v)
        if len(seen) == n - 1:
            return g
    raise ValueError(n)


def _cconv(x, y):
    """Circular convolution along the last dimension."""
    n = x.shape[-1]
    return torch.fft.irfft(torch.fft.rfft(x) * torch.fft.rfft(y), n=n)


class Tools(nn.Module):
    NAMES = ["copy", "add", "sub", "mul", "max", "lt", "eq", "inc"]

    def __init__(self, n):
        super().__init__()
        assert _is_prime(n), "n must be prime (multiplication uses discrete logs)"
        self.n = n
        g = _primitive_root(n)
        exp = [pow(g, e, n) for e in range(n - 1)]  # exponent e -> value g^e
        self.register_buffer("exp", torch.tensor(exp))
        self.register_buffer("neg", (-torch.arange(n)) % n)  # index of -k mod n

    def copy(self, pa, pb=None):
        return pa

    def add(self, pa, pb):
        """(a + b) mod n: circular convolution of the two distributions."""
        return _cconv(pa, pb).clamp_min(0)

    def sub(self, pa, pb):
        """(a - b) mod n: add a to the negation of b."""
        return _cconv(pa, pb[..., self.neg]).clamp_min(0)

    def lt(self, pa, pb):
        """1 if a < b else 0."""
        below = torch.cat([torch.zeros_like(pa[..., :1]), pa.cumsum(-1)[..., :-1]], -1)  # P(a < k)
        p = (pb * below).sum(-1)
        return self._flag(p, pa.sum(-1) * pb.sum(-1), pa)

    def eq(self, pa, pb):
        """1 if a == b else 0."""
        return self._flag((pa * pb).sum(-1), pa.sum(-1) * pb.sum(-1), pa)

    @staticmethod
    def _flag(p_true, mass, like):
        out = torch.zeros_like(like)
        out[..., 1] = p_true
        out[..., 0] = (mass - p_true).clamp_min(0)
        return out

    def inc(self, pa, pb=None):
        """a + 1, where "no value" counts as zero (so the first increment gives 1)."""
        out = torch.roll(pa, 1, -1)
        missing = (1 - pa.sum(-1)).clamp(0, 1)
        out = out.clone()
        out[..., 1] = out[..., 1] + missing
        return out

    def mul(self, pa, pb):
        """(a * b) mod n: convolution over exponents for non-zero values, zero handled apart."""
        ea, eb = pa[..., self.exp], pb[..., self.exp]
        c = _cconv(ea, eb).clamp_min(0)
        out = torch.zeros_like(pa)
        out[..., self.exp] = c
        ma, mb = pa.sum(-1), pb.sum(-1)
        a0, b0 = pa[..., 0], pb[..., 0]
        out[..., 0] = a0 * mb + b0 * ma - a0 * b0
        return out

    def max(self, pa, pb):
        """max(a, b): product of the two cumulative distributions, differenced."""
        g = pa.cumsum(-1) * pb.cumsum(-1)
        return torch.cat([g[..., :1], g[..., 1:] - g[..., :-1]], -1).clamp_min(0)

    def all(self, pa, pb):
        """Outputs of every tool, stacked on a new dim -2, in NAMES order."""
        return torch.stack([self.copy(pa), self.add(pa, pb), self.sub(pa, pb), self.mul(pa, pb),
                            self.max(pa, pb), self.lt(pa, pb), self.eq(pa, pb), self.inc(pa)], -2)

    # integer reference implementations, used by the data generator and the tests
    @staticmethod
    def exact(op, a, b, n):
        """Integer result of tool `op` (index into NAMES)."""
        outs = [a, (a + b) % n, (a - b) % n, (a * b) % n, torch.maximum(a, b),
                (a < b).long(), (a == b).long(), (a + 1) % n]
        res = outs[0]
        for i in range(1, len(outs)):
            res = torch.where(op == i, outs[i], res)
        return res


def self_test(n=97, device="cpu"):
    """Check every tool against integer arithmetic on all pairs, and against brute force on soft inputs."""
    t = Tools(n).to(device)
    a = torch.arange(n, device=device).repeat_interleave(n)
    b = torch.arange(n, device=device).repeat(n)
    eye = torch.eye(n, device=device, dtype=torch.float64)
    pa, pb = eye[a], eye[b]
    out = t.double().all(pa, pb)
    report = {}
    for i, name in enumerate(Tools.NAMES):
        want = Tools.exact(torch.full_like(a, i), a, b, n)
        got = out[:, i]
        report[f"{name}_onehot_exact"] = bool((got.argmax(-1) == want).all() and (got.max(-1).values > 0.999).all())
    g = torch.Generator(device="cpu").manual_seed(0)
    sa = torch.rand(64, n, generator=g, dtype=torch.float64).to(device)
    sb = torch.rand(64, n, generator=g, dtype=torch.float64).to(device)
    sa, sb = sa / sa.sum(-1, keepdim=True) * 0.9, sb / sb.sum(-1, keepdim=True) * 0.7  # partial mass
    soft = t.all(sa, sb)
    outer = (sa[:, :, None] * sb[:, None, :]).reshape(64, n * n)
    for i, name in enumerate(Tools.NAMES):
        if name in ("copy", "inc"):
            continue
        want = Tools.exact(torch.full_like(a, i), a, b, n)
        ref = torch.zeros(64, n, dtype=torch.float64, device=device).index_add(1, want, outer)
        report[f"{name}_soft_max_err"] = float((soft[:, i] - ref).abs().max())
    null = torch.zeros(1, n, dtype=torch.float64, device=device)
    report["inc_of_nothing_is_one"] = bool(t.inc(null)[0].argmax() == 1 and abs(float(t.inc(null)[0, 1]) - 1) < 1e-12)
    return report


if __name__ == "__main__":
    import json, sys
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rep = self_test(int(sys.argv[1]) if len(sys.argv) > 1 else 97, dev)
    print(json.dumps(rep, indent=1))
    ok = all(v is True or (isinstance(v, float) and v < 1e-9) for v in rep.values())
    print("TOOLS OK" if ok else "TOOLS FAILED")
    sys.exit(0 if ok else 1)
