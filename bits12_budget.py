"""Local certificate with a larger box budget (150,000) and the 1e-6 safety margin.  usage: python bits12_budget.py <seed> ..."""
import numpy as np, sys, time, json
import bits2 as B, bits9 as V
from test9 import setup4d
pl=setup4d(); c=pl["c"]; sc=np.sqrt(c)*(1-1e-6); us,um=pl["us"],pl["umax"]; M,lbv,beta=V.plant_w(pl)
import os
out={int(k):v for k,v in json.load(open('res12_budget.json')).items()} if os.path.exists('res12_budget.json') else {}
for seed in [int(a) for a in sys.argv[1:]]:
    r0=np.random.default_rng(seed); net=B.train(B.init(r0,4),pl,r0); fn=V.fold(pl,net); out[seed]=None
    for b in range(6,13):
        fq=V.fold(pl,B.quant(net,b)); t=time.time()
        chk=lambda lo,hi: V.U_bound(fn,M,lbv,beta,us,um,c,lo,hi)+beta*us*B.eps_dibp(pl,fn,fq,lo,hi)<=sc
        ok,nb,_=V.prove(chk,c,4,150000)
        print(f"seed {seed} b={b}: proved={ok} boxes={nb} ({time.time()-t:.0f}s)",flush=True)
        if ok: out[seed]=b; break
    json.dump(out,open("res12_budget.json","w"))
print("local certificate first certifying width at 150,000 boxes (search from 6 bits):",out)
