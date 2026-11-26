"""Run the bus comparison grid on one GPU, several runs at a time, skipping finished ones."""
import itertools, os, subprocess, sys, time

ARMS = {
    "base":            ["--tools", "0"],
    "tools-full":      ["--tools", "1", "--guide", "full"],
    "tools-fade":      ["--tools", "1", "--guide", "fade"],
    "tools-none":      ["--tools", "1", "--guide", "none"],
    "fade-loop2":      ["--tools", "1", "--guide", "fade", "--stages", "2", "--loop", "1"],
    "fade-attnres":    ["--tools", "1", "--guide", "fade", "--attnres", "1"],
    "fade-loop2-attn": ["--tools", "1", "--guide", "fade", "--stages", "2", "--loop", "1", "--attnres", "1"],
}

if __name__ == "__main__":
    steps, workers, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    os.makedirs(out, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    todo = [(f"{arm}-s{seed}", ARMS[arm] + ["--seed", str(seed)])
            for seed, arm in itertools.product([0, 1, 2], ARMS) if not os.path.exists(f"{out}/{arm}-s{seed}.done")]
    live = {}
    while todo or live:
        for n in [n for n, p in live.items() if p.poll() is not None]:
            if live.pop(n).returncode == 0:
                open(f"{out}/{n}.done", "w").close()
            print(time.strftime("%H:%M:%S"), "finished", n, f"({len(todo)} queued)", flush=True)
        while todo and len(live) < workers:
            n, c = todo.pop(0)
            live[n] = subprocess.Popen([sys.executable, f"{here}/train.py", "--steps", steps, "--out", f"{out}/{n}.jsonl"] + c,
                                       stderr=open(f"{out}/{n}.err", "w"))
        time.sleep(3)
    open(f"{out}/SWEEP_DONE", "w").close()
