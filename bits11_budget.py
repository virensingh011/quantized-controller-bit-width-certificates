"""Does the local certificate's width depend on the box budget?  usage: python bits11_budget.py <seed> <budget> <b1> <b2> ..."""
import numpy as np, sys, time
import bits2 as B, bits9 as V
from test9 import setup4d
seed,budget=int(sys.argv[1]),int(sys.argv[2]); bs=[int(a) for a in sys.argv[3:]]
pl=setup4d(); c=pl["c"]; sc=np.sqrt(c); us,um=pl["us"],pl["umax"]; M,lbv,beta=V.plant_w(pl)
r0=np.random.default_rng(seed); net=B.train(B.init(r0,4),pl,r0); fn=V.fold(pl,net)
for b in bs:
    fq=V.fold(pl,B.quant(net,b)); t=time.time()
    chk=lambda lo,hi: V.U_bound(fn,M,lbv,beta,us,um,c,lo,hi)+beta*us*B.eps_dibp(pl,fn,fq,lo,hi)<=sc
    ok,nb,_=V.prove(chk,c,4,budget)
    print(f"seed {seed} b={b} budget={budget}: local certificate proved={ok} boxes={nb} ({time.time()-t:.0f}s)",flush=True)
