"""Export a trained, rounded controller as integer-only C data (ctrl_params.h) and a bit-exact numpy model of it."""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bits2 as B
from bits12_int import int_forward

def build(pl, net, b, calib, ba=8):
    qin = 2**(ba-1)-1; sa = 1.0/qin; _, rec = int_forward(net, b, calib/pl["s"], ba)
    scales = [max(r, 1e-12)/(2**ba-1) for r in rec]; layers = []
    for l, (W, bi) in enumerate(net):
        s = np.abs(W).max(0)/(2**(b-1)-1); s = np.where(s == 0, 1.0, s)
        d = dict(Wint=np.round(W/s).astype(np.int64), bint=np.round(bi/(sa*s)).astype(np.int64))
        if l < 3:
            m0, ts = [], []
            for m in sa*s/scales[l]:
                f, e = np.frexp(m); q = int(round(f*2**31))
                if q == 2**31: q //= 2; e += 1
                m0.append(q); ts.append(31-e)
            assert min(ts) >= 1 and max(ts) <= 62
            d.update(m0=np.array(m0, dtype=np.int64), ts=np.array(ts, dtype=np.int64)); sa = scales[l]
        else: d["out_scale"] = sa*s[0]*pl["us"]
        layers.append(d)
    return layers

def fx_forward(layers, Z):
    """bit-exact numpy model of controller.c (input already scaled to z = x/s, quantized to int8 here)."""
    a = np.clip(np.round(Z*127.0), -127, 127).astype(np.int64); return fx_from_int(layers, a)
def fx_from_int(layers, a):
    for l, d in enumerate(layers):
        acc = a @ d["Wint"] + d["bint"][None, :]
        if l == 3: return acc[:, 0]
        acc = np.maximum(acc, 0); ts = d["ts"][None, :]
        a = np.clip((acc*d["m0"][None, :] + (np.int64(1) << (ts-1))) >> ts, 0, 255)

def pack(Wint, b):
    bits = []                                    # output-channel-major, LSB first
    for j in range(Wint.shape[1]):
        for i in range(Wint.shape[0]):
            v = int(Wint[i, j]) & ((1 << b)-1); bits += [(v >> k) & 1 for k in range(b)]
    bits += [0]*((-len(bits)) % 8); by = [sum(bits[8*k+t] << t for t in range(8)) for k in range(len(bits)//8)]
    return by + [0]                              # one padding byte for the 2-byte reads

def header(layers, b, nin):
    L = ["#ifndef CTRL_PARAMS_H", "#define CTRL_PARAMS_H", "#include <stdint.h>", f"#define CTRL_NIN {nin}", f"#define CTRL_BITS {b}",
         f"#define CTRL_OUT_SCALE ({float(layers[3]['out_scale']):.12e})   /* u = acc * CTRL_OUT_SCALE, then saturate */"]
    for l, d in enumerate(layers):
        L.append(f"static const uint8_t L{l}_W[] = {{" + ",".join(map(str, pack(d["Wint"], b))) + "};")
        L.append(f"static const int32_t L{l}_B[] = {{" + ",".join(map(str, d["bint"])) + "};")
        if l < 3:
            L.append(f"static const int32_t L{l}_M[] = {{" + ",".join(map(str, d["m0"])) + "};")
            L.append(f"static const uint8_t L{l}_S[] = {{" + ",".join(map(str, d["ts"])) + "};")
    L.append("#endif"); return "\n".join(L) + "\n"

def data_bytes(layers, b):
    w = sum(len(pack(d["Wint"], b)) for d in layers); bi = 4*sum(len(d["bint"]) for d in layers)
    rq = sum(5*len(d["m0"]) for d in layers[:3]); return w, bi, rq, w+bi+rq
