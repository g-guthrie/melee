"""Drive a RunPod GPU over HTTPS only: create a pod, run commands, copy files, delete it.

Reads the API key from ~/.runpod_key and remembers the current pod id in ~/.runpod_pod.

    python rp.py up                  start a pod (cheapest suitable GPU with capacity)
    python rp.py sh 'nvidia-smi'     run a shell command, print its output
    python rp.py put DIR REMOTE      copy a local directory's .py files to REMOTE on the pod
    python rp.py get REMOTE LOCAL    copy a remote directory back (as a tar stream)
    python rp.py status | down
"""
import base64, io, json, os, subprocess, sys, tarfile, time

API = "https://api.runpod.io/v2"
KEYF, PODF = os.path.expanduser("~/.runpod_key"), os.path.expanduser("~/.runpod_pod")
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
PREFER = ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090", "NVIDIA RTX A6000",
          "NVIDIA A40", "NVIDIA RTX 4000 Ada Generation", "NVIDIA RTX A4500", "NVIDIA RTX A4000"]


def api(method, path, body=None, timeout=60):
    key = open(KEYF).read().strip()
    cmd = ["curl", "-sS", "-m", str(timeout), "-X", method, "-w", "\n%{http_code}", API + path,
           "-H", "Authorization: Bearer " + key]
    if body is not None:
        cmd += ["-H", "Content-Type: application/json", "--data-binary", "@-"]
    r = subprocess.run(cmd, input=json.dumps(body) if body is not None else None, capture_output=True, text=True)
    text, _, code = r.stdout.rpartition("\n")
    try:
        return int(code), (json.loads(text) if text.strip() else {})
    except ValueError:
        return int(code or 0), {"raw": (text or r.stderr)[:500]}


def pod_id():
    return open(PODF).read().strip()


def up(max_price=0.6, name="wsm"):
    code, cat = api("GET", "/catalog/gpus")
    price = {g["id"]: g.get("price") or {} for g in cat["gpus"]}
    tries = [(price[g].get(t.lower()), t, g) for g in PREFER if g in price for t in ("COMMUNITY", "SECURE")]
    for p, cloud, gpu in sorted(t for t in tries if t[0] and t[0] <= max_price):
        code, r = api("POST", "/pods", {"name": name, "image": IMAGE, "gpu": {"id": gpu, "count": 1},
                                        "cloud": cloud, "disk": 30})
        if code == 201:
            open(PODF, "w").write(r["id"])
            print(f"pod {r['id']} on {gpu} ({cloud}) at ${p}/h")
            for _ in range(60):  # wait until commands are accepted
                c, out = api("POST", f"/pods/{r['id']}/exec", {"command": ["true"], "timeoutSeconds": 10})
                if c == 200:
                    return r["id"]
                time.sleep(5)
            raise SystemExit("pod never accepted commands")
        if code == 202:
            api("DELETE", "/queued-pods/" + r["id"])
    raise SystemExit("no GPU with capacity under the price cap")


def sh(command, timeout=25, check=False):
    code, r = api("POST", f"/pods/{pod_id()}/exec", {"command": ["bash", "-lc", command], "timeoutSeconds": timeout},
                  timeout=timeout + 30)
    if code != 200:
        raise SystemExit(f"exec failed: HTTP {code} {r}")
    if check and r.get("exitCode") != 0:
        raise SystemExit(f"command failed ({r.get('exitCode')}): {r.get('stderr', '')[-800:]}")
    return r


def put(local_dir, remote_dir):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root, _, files in os.walk(local_dir):
            for f in files:
                if f.endswith(".py"):
                    full = os.path.join(root, f)
                    tar.add(full, arcname=os.path.relpath(full, local_dir))
    data = base64.b64encode(buf.getvalue()).decode()
    chunks = [data[i:i + 4000] for i in range(0, len(data), 4000)]
    script = ("import sys,base64,io,tarfile,os; d=sys.argv[1]; os.makedirs(d,exist_ok=True); "
              "tarfile.open(fileobj=io.BytesIO(base64.b64decode(''.join(sys.argv[2:]))),mode='r:gz').extractall(d)")
    assert len(chunks) < 250, "too large for one call"
    code, r = api("POST", f"/pods/{pod_id()}/exec",
                  {"command": ["python", "-c", script, remote_dir] + chunks, "timeoutSeconds": 25})
    if code != 200 or r.get("exitCode") != 0:
        raise SystemExit(f"put failed: {code} {r}")
    print(f"copied {local_dir} -> {remote_dir} ({len(data)} b64 bytes)")


def get(remote_dir, local_dir, pattern="*"):
    r = sh(f"cd {remote_dir} && tar czf - {pattern} | base64 -w0", check=True)
    if r.get("truncated"):
        raise SystemExit("output truncated at 1 MiB; fetch in smaller pieces")
    os.makedirs(local_dir, exist_ok=True)
    tarfile.open(fileobj=io.BytesIO(base64.b64decode(r["stdout"])), mode="r:gz").extractall(local_dir)
    print(f"fetched {remote_dir}/{pattern} -> {local_dir}")


def down():
    code, _ = api("DELETE", "/pods/" + pod_id())
    c2, r = api("GET", "/pods")
    print("delete ->", code, "| pods left:", [(p["id"], p.get("status")) for p in r.get("pods", [])])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "up":
        up(*(float(x) for x in sys.argv[2:3]))
    elif cmd == "sh":
        r = sh(sys.argv[2], *(int(x) for x in sys.argv[3:4]))
        sys.stdout.write(r.get("stdout", ""))
        if r.get("stderr"):
            sys.stderr.write(r["stderr"][-3000:])
        print(f"[exit {r.get('exitCode')} timedOut {r.get('timedOut')}]")
    elif cmd == "put":
        put(sys.argv[2], sys.argv[3])
    elif cmd == "get":
        get(*sys.argv[2:5])
    elif cmd == "status":
        print(api("GET", "/pods")[1])
    elif cmd == "down":
        down()
