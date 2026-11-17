"""(1) bit-exact check: C controller (host gcc) vs numpy fixed-point model;  (2) closed-loop sampling test of the integer-only
controller for 15 seeds;  (3) stored-size table (const data bytes, object sizes).   usage: python verify_c.py"""
import numpy as np, sys, os, subprocess, json, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bits2 as B
from export_c import build, fx_forward, fx_from_int, header, data_bytes
HERE = os.path.dirname(os.path.abspath(__file__))
def setup(make):
    pl = B.setup(make()); d = len(pl["Acl"]); rng0 = np.random.default_rng(123)
    pl["us"] = np.abs(np.clip(pl["expert"](B.samples(pl,20000,rng0,1.5)), -pl["umax"], pl["umax"])).max()
    X = np.vstack([B.samples(pl,30000,rng0), B.boundary(pl,5000,rng0)]); rc = np.random.default_rng(999)
    calib = np.vstack([B.samples(pl,20000,rc), B.boundary(pl,2000,rc)]); return pl, d, X, calib
def train(pl, d, seed): r0 = np.random.default_rng(seed); return B.train(B.init(r0, d), pl, r0)
def c_outputs(layers, b, nin, A, tmp):
    open(f"{tmp}/ctrl_params.h", "w").write(header(layers, b, nin))
    exe = f"{tmp}/t"; subprocess.run(["gcc", "-O2", "-Wall", "-Werror", "-I", tmp, "-o", exe, f"{HERE}/test_host.c", f"{HERE}/controller.c"], check=True)
    out = subprocess.run([exe], input=A.astype(np.int8).tobytes(), capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int32).astype(np.int64)
def safe(pl, X, layers):
    u = fx_forward(layers, X/pl["s"])*layers[3]["out_scale"]; return bool(B.pn(pl, pl["f"](X, u)).max() <= np.sqrt(pl["c"])*(1+1e-9))
if __name__ == "__main__":
    res = {}; tmp = tempfile.mkdtemp()
    for name, make in (("pendulum", B.make_pend), ("bicycle", B.make_4d)):
        pl, d, X, calib = setup(make); print(f"\n===== {name} =====")
        # (1) bit-exactness
        net = train(pl, d, 0)
        for b in (3, 4, 5, 8):
            layers = build(pl, net, b, calib); A = np.clip(np.round((X[:6000]/pl["s"])*127.0), -127, 127).astype(np.int64)
            ok = np.array_equal(c_outputs(layers, b, d, A, tmp), fx_from_int(layers, A))
            print(f"  C == numpy fixed-point model on 6000 points, seed 0, {b} bits: {ok}"); assert ok
        # (2) closed loop, 15 seeds
        R = []
        for seed in range(15):
            net = train(pl, d, seed)
            if B.pn(pl, pl["f"](X, B.ctrl(pl, net, X))).max()/np.sqrt(pl["c"]) >= 0.995: continue
            R.append({b: safe(pl, X, build(pl, net, b, calib)) for b in range(2, 13)})
        first = [next((b for b in range(2, 13) if r[b]), None) for r in R]
        true = [next((b for b in range(2, 13) if all(r[bb] for bb in range(b, 13))), None) for r in R]
        cnt = {b: sum(r[b] for r in R) for b in (3, 4, 5, 6, 8)}
        print(f"  integer-only controller, {len(R)} networks: first pass {np.mean(first):.2f}, true bits {np.mean(true):.2f}, safe at 3,4,5,6,8 bits: {cnt}")
        res[name] = dict(first=float(np.mean(first)), true=float(np.mean(true)), counts=cnt)
        # (3) sizes, seed 0
        net = train(pl, d, 0); print("  bits | weights  biases(int32)  requant  = const data (B) | .rodata  .text (host gcc -Os; x86, not Cortex-M)")
        for b in (3, 4, 5, 6, 8, 10):
            layers = build(pl, net, b, calib); w, bi, rq, tot = data_bytes(layers, b)
            open(f"{tmp}/ctrl_params.h", "w").write(header(layers, b, d))
            subprocess.run(["gcc", "-Os", "-c", "-I", tmp, "-o", f"{tmp}/c.o", f"{HERE}/controller.c"], check=True)
            sz = subprocess.run(["size", "-A", f"{tmp}/c.o"], capture_output=True, text=True).stdout
            g = lambda s: next((int(l.split()[1]) for l in sz.splitlines() if l.startswith(s)), 0)
            print(f"  {b:4d} | {w:6d}  {bi:6d}        {rq:6d}   = {tot:6d}            | {g('.rodata'):6d}  {g('.text'):6d}")
            res[name].setdefault("sizes", {})[b] = dict(weights=w, bias=bi, requant=rq, total=tot)
    json.dump(res, open(f"{HERE}/res_embedded.json", "w"), indent=1); shutil.rmtree(tmp)
