"""Run a named set of training configs, several at a time, skipping finished ones."""
import argparse, itertools, os, subprocess, sys, time

ARMS = ["none", "tool", "tool_aux", "mlp", "oracle"]


def configs(name, steps):
    runs = []
    if name in ("pilot", "smoke"):
        seeds = [0, 1, 2] if name == "pilot" else [0]
        # main grid: every arm on both tool functions
        for table, arm, seed in itertools.product(["random", "modmul"], ARMS, seeds):
            runs.append((f"{table}-d128-{arm}-s{seed}", ["--table", table, "--arm", arm, "--seed", seed]))
        # capacity-limited grid: a narrower model, where memorising both tables is hard
        if name == "pilot":
            for arm, seed in itertools.product(["none", "tool", "tool_aux", "oracle"], seeds):
                runs.append((f"random-d48-{arm}-s{seed}",
                             ["--table", "random", "--arm", arm, "--seed", seed, "--d", 48]))
    return [(n, [str(x) for x in a] + ["--steps", str(steps)]) for n, a in runs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="pilot")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    todo = [(n, c) for n, c in configs(a.name, a.steps) if not os.path.exists(f"{a.out}/{n}.done")]
    print(f"{len(todo)} runs to do", flush=True)
    live = {}
    while todo or live:
        for n in [n for n, pr in live.items() if pr.poll() is not None]:
            rc = live.pop(n).returncode
            if rc == 0:
                open(f"{a.out}/{n}.done", "w").close()
            print(f"{time.strftime('%H:%M:%S')} finished {n} rc={rc}  ({len(todo)} queued, {len(live)} running)", flush=True)
        while todo and len(live) < a.workers:
            n, c = todo.pop(0)
            err = open(f"{a.out}/{n}.err", "w")
            live[n] = subprocess.Popen([sys.executable, f"{here}/train.py", "--out", f"{a.out}/{n}.jsonl",
                                        "--threads", "1"] + c, stderr=err)
        time.sleep(2)
    open(f"{a.out}/SWEEP_DONE", "w").close()
    print("sweep done", flush=True)


if __name__ == "__main__":
    main()
