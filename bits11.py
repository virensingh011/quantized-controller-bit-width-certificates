"""Diagnostics for the four-state plant (Section VII-C): ablation of the verifier components, and why the local
certificate needs (almost) the same width for every seed.   usage: python bits11.py ablation <seeds...> | margin
(Re-implemented after the original file was lost.)"""
import numpy as np, sys, json, time, glob
import bits2 as B, bits9 as V
from test9 import setup4d
CONF={"full":{}, "interval intermediate bounds":dict(inter="ibp"), "no ball bound":dict(use_ball=False),
      "no vertex bound":dict(use_vertex=False), "no interval fallback":dict(use_interval=False),
      "no saturation relaxation":dict(use_sat=False)}
def ablation(seeds, budget=40000):
    pl=setup4d(); c=pl["c"]; sc=np.sqrt(c); M,lbv,beta=V.plant_w(pl); res={}
    for seed in seeds:
        r0=np.random.default_rng(seed); net=B.train(B.init(r0,4),pl,r0); fn=V.fold(pl,net)
        for name,kw in CONF.items():
            t=time.time(); ok,nb,_=V.prove(lambda lo,hi: V.U_bound(fn,M,lbv,beta,pl["us"],pl["umax"],c,lo,hi,**kw)<=sc,c,4,budget)
            res[(seed,name)]=(bool(ok),int(nb),time.time()-t)
            print(f"seed {seed:2d} | {name:30s} certified={ok!s:5s} boxes={nb:6d} ({time.time()-t:.0f}s)",flush=True)
    json.dump({f"{k[0]}|{k[1]}":v for k,v in res.items()},open("res11_ablation.json","w"))
def margin_table():
    rows=[json.load(open(f)) for f in sorted(glob.glob("res10_seed*.json"),key=lambda f:int(f.split("seed")[1].split(".")[0]))]
    pl=setup4d(); c=pl["c"]; rng0=np.random.default_rng(123); _=B.samples(pl,20000,rng0,1.5)
    X=np.vstack([B.samples(pl,30000,rng0),B.boundary(pl,5000,rng0)]); print("seed  rho    margin(ctrl units)  local  direct  FPboxes")
    for r in rows:
        s=r["seed"]; r0=np.random.default_rng(s); net=B.train(B.init(r0,4),pl,r0)
        rho=B.pn(pl,pl["f"](X,B.ctrl(pl,net,X))).max()/np.sqrt(c); m=(1-rho)*np.sqrt(c)/V.plant_w(pl)[2]
        print(f"{s:4d}  {rho:.3f}  {m:.4f}              {r['local']}      {r['direct']}       {r['fp_boxes']}")
if __name__=="__main__":
    if sys.argv[1]=="ablation": ablation([int(a) for a in sys.argv[2:]])
    else: margin_table()
