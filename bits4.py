import numpy as np, sys, json
import bits2 as B, bits3 as C
def fold(pl,net):                      # exact change of coordinates w=Lc x folded into first layer
    T=np.linalg.inv(pl["Lc"]).T@np.diag(1/pl["s"]); n=[[W.copy(),b.copy()] for W,b in net]; n[0][0]=T@n[0][0]; return n
def wcells(pl,m):
    d=4; sc=np.sqrt(pl["c"]); e=np.linspace(-sc,sc,m+1); idx=np.stack(np.meshgrid(*[np.arange(m)]*d,indexing="ij"),-1).reshape(-1,d)
    lo,hi=e[idx],e[idx+1]; keep=np.linalg.norm((lo+hi)/2,axis=1)<=sc+np.linalg.norm((hi-lo)/2,axis=1); return lo[keep],hi[keep]
def cert(pl,netw,qw,lo,hi,budget):
    sc=np.sqrt(pl["c"]); us=pl["us"]; Li=np.linalg.inv(pl["Lc"]); Mw=pl["Lc"]@pl["A"]@Li; LB=pl["Lc"]@pl["bcol"]
    while True:
        outs,(ol,oh)=B.ibp(netw,lo,hi); ul=np.clip(us*ol[:,0],-pl["umax"],pl["umax"]); uh=np.clip(us*oh[:,0],-pl["umax"],pl["umax"])
        a,b_=C.imat(Mw,lo,hi); lw=a+np.minimum(np.outer(ul,LB),np.outer(uh,LB)); hw=b_+np.maximum(np.outer(ul,LB),np.outer(uh,LB))
        ub=np.linalg.norm(np.maximum(np.abs(lw),np.abs(hw)),axis=1)
        bad=~(ub+pl["beta"]*us*B.eps_dibp(pl,netw,qw,lo,hi)<=sc)
        if not bad.any(): return True
        if len(lo)>budget: return False
        L,H=lo[bad],hi[bad]; ax=np.argmax(H-L,axis=1); mid=(L+H)/2; r=np.arange(len(L))
        H1=H.copy(); H1[r,ax]=mid[r,ax]; L2=L.copy(); L2[r,ax]=mid[r,ax]; lo=np.vstack([L,L2]); hi=np.vstack([H1,H])
        keep=np.linalg.norm((lo+hi)/2,axis=1)<=sc+np.linalg.norm((hi-lo)/2,axis=1); lo,hi=lo[keep],hi[keep]
if __name__=="__main__":
    n=int(sys.argv[1]); budget=int(sys.argv[2]); pl=B.setup(B.make_4d()); r0=np.random.default_rng(123)
    pl["us"]=np.abs(np.clip(pl["expert"](B.samples(pl,20000,r0,1.5)),-pl["umax"],pl["umax"])).max()
    X=np.vstack([B.samples(pl,30000,r0),B.boundary(pl,5000,r0)]); sc=np.sqrt(pl["c"]); lo0,hi0=wcells(pl,3); res=[]; viol=0
    for seed in range(n):
        rng=np.random.default_rng(seed); net=B.train(B.init(rng,4),pl,rng); nw=fold(pl,net)
        inv={b:B.pn(pl,pl["f"](X,B.ctrl(pl,B.quant(net,b),X))).max()<=sc*(1+1e-9) for b in range(2,13)}
        fp=cert(pl,nw,nw,lo0,hi0,budget); first=None
        for b in range(2,13):
            if cert(pl,nw,fold(pl,B.quant(net,b)),lo0,hi0,budget):
                first=first or b
                if not inv[b]: viol+=1
        tb=next((b for b in range(2,13) if all(inv[x] for x in range(b,13))),None); res.append((seed,fp,first,tb))
        print("seed %2d FP certified=%s | local certificate bits=%s | TRUE=%s"%(seed,fp,first,tb),flush=True)
    f=np.array([x[2] if x[2] else np.nan for x in res],float); t=np.array([x[3] for x in res],float)
    print("violations",viol,"| mean certified %.1f mean true %.1f gap %.1f (n certified %d/%d)"%(np.nanmean(f),t.mean(),np.nanmean(f-t),np.sum(~np.isnan(f)),n))
    json.dump(res,open("res4.json","w"),default=float)
