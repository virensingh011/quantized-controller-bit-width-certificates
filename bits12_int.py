"""Integer-arithmetic emulation of the quantized controllers (sampling test, NOT a proof).
Weights: signed b-bit integers with per-output-channel scales (as in the paper).  Input and hidden activations:
ba-bit integers (input signed symmetric, hidden unsigned after ReLU, scales calibrated on a separate sample set),
biases: integers with scale s_in*s_w, accumulation exact (float64 holds the integer sums), requantization
multipliers kept in float.  Output scale and control u = us*out stay in float.   Scheme follows Jacob et al. (2018)
with zero points = 0.   usage: python bits12_int.py [nseeds]"""
import numpy as np, sys, json
import bits2 as B
def int_forward(net, b, Z, ba, scales=None):
    qin = 2**(ba-1)-1; sa = 1.0/qin; a = np.clip(np.round(Z/sa), -qin, qin); rec = []
    for l,(W,bi) in enumerate(net):
        s = np.abs(W).max(0)/(2**(b-1)-1); s = np.where(s==0, 1.0, s); Wint = np.round(W/s)
        acc = a@Wint + np.round(bi/(sa*s)); real = acc*(sa*s)
        if l == 3: return real[:,0], rec
        r = np.maximum(real, 0); rec.append(r.max())
        sn = (scales[l] if scales is not None else max(r.max(),1e-12)/(2**ba-1))
        a = np.clip(np.round(r/sn), 0, 2**ba-1); sa = sn
def int_ctrl(pl, net, b, X, ba, calib):
    _, rec = int_forward(net, b, calib/pl["s"], ba); scales = [max(r,1e-12)/(2**ba-1) for r in rec]
    out, _ = int_forward(net, b, X/pl["s"], ba, scales); return pl["us"]*out
def study(make, nseeds):
    pl = B.setup(make()); d = len(pl["Acl"]); rng0 = np.random.default_rng(123)
    pl["us"] = np.abs(np.clip(pl["expert"](B.samples(pl,20000,rng0,1.5)), -pl["umax"], pl["umax"])).max()
    X = np.vstack([B.samples(pl,30000,rng0), B.boundary(pl,5000,rng0)]); rc = np.random.default_rng(999)
    calib = np.vstack([B.samples(pl,20000,rc), B.boundary(pl,2000,rc)]); sc = np.sqrt(pl["c"])*(1+1e-9)
    BR = range(2,13); rows = []
    for seed in range(nseeds):
        r0 = np.random.default_rng(seed); net = B.train(B.init(r0,d), pl, r0)
        u = B.ctrl(pl, net, X)
        if B.pn(pl, pl["f"](X,u)).max()/np.sqrt(pl["c"]) >= 0.995: continue
        res = {"seed": seed}
        for name, fn in (("float", lambda q,b: B.ctrl(pl,q,X)), ("int8", lambda q,b: int_ctrl(pl,net,b,X,8,calib)),
                         ("int12", lambda q,b: int_ctrl(pl,net,b,X,12,calib))):
            ok = {b: bool(B.pn(pl, pl["f"](X, fn(B.quant(net,b), b))).max() <= sc) for b in BR}
            res[name] = ok
        rows.append(res)
    return pl["name"], rows
def summarize(name, rows):
    def first(ok): return next((b for b in range(2,13) if ok[b]), None)
    def true(ok): return next((b for b in range(2,13) if all(ok[bb] for bb in range(b,13))), None)
    print(f"\n=== {name}: {len(rows)} networks")
    for k in ("float","int8","int12"):
        fp=[first(r[k]) for r in rows]; tb=[true(r[k]) for r in rows]
        cnt={b: sum(r[k][b] for r in rows) for b in (3,4,5,6,8)}
        print(f"{k:6s} first pass mean {np.mean(fp):.2f} | true bits mean {np.mean(tb):.2f} | safe networks at 3,4,5,6,8 bits: {cnt}")
    return rows
if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15; out = {}
    for make in (B.make_pend, B.make_4d):
        name, rows = study(make, n); out[name] = rows; summarize(name, rows)
    json.dump(out, open("res12_int.json","w"))
