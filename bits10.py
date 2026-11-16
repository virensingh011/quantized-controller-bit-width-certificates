"""Four-state plant: full-precision certificate, direct proofs and local certificates (Section VII-C of the paper).
usage: python bits10.py <seed> [budget]      writes res10_seed<k>.json
(Re-implemented after the original file was lost; uses the verifier of bits9.py.)"""
import numpy as np, sys, json, time, os
import bits2 as B, bits9 as V
from test9 import setup4d
def run(seed, budget=40000, bmax=12):
    pl=setup4d(); c=pl["c"]; sc=np.sqrt(c); us,um=pl["us"],pl["umax"]; M,lbv,beta=V.plant_w(pl)
    rng0=np.random.default_rng(123); _=B.samples(pl,20000,rng0,1.5)
    X=np.vstack([B.samples(pl,30000,rng0),B.boundary(pl,5000,rng0)])
    # NOTE: identical sample set to bits2.run (same rng0 call order) -> same sampling test
    rng=np.random.default_rng(seed); net=B.train(B.init(rng,4),pl,rng); fn=V.fold(pl,net)
    inv={b:bool(B.pn(pl,pl["f"](X,B.ctrl(pl,B.quant(net,b),X))).max()<=sc*(1+1e-9)) for b in range(2,bmax+1)}
    true=next((b for b in range(2,bmax+1) if all(inv[bb] for bb in range(b,bmax+1))),None)
    first=next((b for b in range(2,bmax+1) if inv[b]),None)
    fname=f"res10_seed{seed}.json"
    out=json.load(open(fname)) if os.path.exists(fname) else {}
    out.update(seed=seed,true=true,first_pass=first,inv={str(k):v for k,v in inv.items()})
    save=lambda: json.dump(out,open(fname,"w"),default=float)
    t=time.time()
    if "fp_certified" not in out:
        okF,nF,_=V.prove(lambda lo,hi: V.U_bound(fn,M,lbv,beta,us,um,c,lo,hi)<=sc,c,4,budget)
        out.update(fp_certified=okF,fp_boxes=nF,fp_time=time.time()-t); save()
    okF=out["fp_certified"]
    print(f"seed {seed}: sampling true={true} first={first} | FP certified={okF} boxes={out['fp_boxes']} ({out['fp_time']:.0f}s)",flush=True)
    for k in ("direct_boxes","local_boxes"): out.setdefault(k,{})
    out.setdefault("direct",None); out.setdefault("local",None)
    if not okF: save(); return out
    for b in range(first,bmax+1):                       # direct proof: only where sampling passes
        if not inv[b]: continue
        if out["direct"] is not None: break
        if str(b) in out["direct_boxes"]: ok=out["direct_boxes"][str(b)][0]
        else:
            q=B.quant(net,b); fq=V.fold(pl,q)
            ok,nb,_=V.prove(lambda lo,hi: V.U_bound(fq,M,lbv,beta,us,um,c,lo,hi)<=sc,c,4,budget)
            out["direct_boxes"][str(b)]=(bool(ok),int(nb)); save()
        if ok: out["direct"]=b; save(); break
    print(f"   direct proof first certifying width: {out['direct']}",flush=True)
    for b in range(first,bmax+1):                       # local certificate: U(FP net) + beta*eps(diff-prop) <= sqrt(c)
        if not inv[b]: continue
        if out["local"] is not None: break
        if str(b) in out["local_boxes"]: ok=out["local_boxes"][str(b)][0]
        else:
            q=B.quant(net,b); fq=V.fold(pl,q)
            def chk(lo,hi):
                U=V.U_bound(fn,M,lbv,beta,us,um,c,lo,hi); e=us*B.eps_dibp(pl,fn,fq,lo,hi)
                return U+beta*e<=sc
            ok,nb,_=V.prove(chk,c,4,budget)
            out["local_boxes"][str(b)]=(bool(ok),int(nb)); save()
        if ok: out["local"]=b; save(); break
    print(f"   local certificate first certifying width: {out['local']}   (sampling true={true})",flush=True)
    save(); return out
if __name__=="__main__":
    run(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv)>2 else 40000)
