"""Test of Corollary 3 (ultimate bound) on trained, rounded networks.
phi(x) <= rho*||x||_P + a0 on E,  eps = sup|pi_q-pi|,  condition rho*sqrt(c)+a0+beta*eps <= sqrt(c)
Then ||x_k||_P <= rho^k ||x_0||_P + (a0+beta*eps)(1-rho^k)/(1-rho)."""
import numpy as np, sys, json
import bits2 as B, bits6 as S
def run(pl,m,seeds,bits,K=150,n0=3000):
    pl,X,lo,hi=S.prep(pl,m); sc=np.sqrt(pl["c"]); beta=pl["beta"]; rows=[]
    r0=np.random.default_rng(5); X0=np.vstack([B.samples(pl,n0-500,r0),B.boundary(pl,500,r0)])
    zero=np.zeros((1,len(pl["Acl"])))
    for seed in seeds:
        rng=np.random.default_rng(seed); net=B.train(B.init(rng,len(pl["Acl"])),pl,rng)
        u=B.ctrl(pl,net,X); phi=B.pn(pl,pl["f"](X,u)); nx=B.pn(pl,X)
        a0=float(B.pn(pl,pl["f"](zero,B.ctrl(pl,net,zero)))[0]); ok=nx>1e-6
        rho=float(np.max((phi[ok]-a0)/nx[ok]))
        for b in bits:
            q=B.quant(net,b); eps=float(np.abs(B.ctrl(pl,q,X)-u).max())
            eps_s=float(pl["us"]*B.eps_dibp(pl,net,q,lo,hi).max())
            cond=(rho<1) and (rho*sc+a0+beta*eps<=sc)
            R=(a0+beta*eps)/(1-rho) if rho<1 else np.inf
            x=X0.copy(); n0_=B.pn(pl,x); viol=0; maxn=n0_.max()
            for k in range(1,K+1):
                x=pl["f"](x,B.ctrl(pl,q,x)); nk=B.pn(pl,x); maxn=max(maxn,nk.max())
                if rho<1:
                    bound=rho**k*n0_+(a0+beta*eps)*(1-rho**k)/(1-rho)
                    viol+=int((nk>bound*(1+1e-9)+1e-12).sum())
            Remp=float(B.pn(pl,x).max())
            rows.append(dict(plant=pl["name"],seed=seed,bits=b,rho=rho,a0=a0,eps=eps,eps_sound=eps_s,cond=bool(cond),
                             R_pred=float(R),R_emp=Remp,R_pred_over_sc=float(R/sc),R_emp_over_sc=Remp/sc,
                             inv=bool(maxn<=sc*(1+1e-9)),viol=viol))
            print(rows[-1],flush=True)
    return rows
if __name__=="__main__":
    ns=int(sys.argv[1]); bits=[3,4,5,6,7,8,10]
    out=run(B.make_pend(),8,range(ns),bits)+run(B.make_4d(),3,range(ns),bits)
    json.dump(out,open("res_cor3_full.json","w"))
