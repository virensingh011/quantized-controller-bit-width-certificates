import numpy as np, sys, json
import bits2 as B
def imat(M,l,h):                       # interval of x@M.T style: M (out,in), l,h (cells,in)
    c=(l+h)/2; r=(h-l)/2; return c@M.T-r@np.abs(M).T, c@M.T+r@np.abs(M).T
def wbox(pl,lo_n,hi_n,net,us):
    """interval of Lc*f(x,clip pi(x)) over cells given in normalised coords (sound)"""
    sb=pl["s"]; lo=lo_n*sb; hi=hi_n*sb; Lc=pl["Lc"]
    outs,(ol,oh)=B.ibp(net,lo_n,hi_n); ul=np.clip(us*ol[:,0],-pl["umax"],pl["umax"]); uh=np.clip(us*oh[:,0],-pl["umax"],pl["umax"])
    if pl["name"]=="4D bicycle":
        a,b_=imat(Lc@pl["A"],lo,hi); LB=Lc@pl["bcol"]
        lw=a+np.minimum(np.outer(ul,LB),np.outer(uh,LB)); hw=b_+np.maximum(np.outer(ul,LB),np.outer(uh,LB))
    else:
        dt=0.05; g,b=9.8,0.1; M=np.array([[1,dt],[0,1-dt*b]]); a,b_=imat(Lc@M,lo,hi)
        sl,sh=np.sin(lo[:,0]),np.sin(hi[:,0]); tl=dt*(g*sl+ul); th=dt*(g*sh+uh); L1=Lc[:,1]
        lw=a+np.minimum(np.outer(tl,L1),np.outer(th,L1)); hw=b_+np.maximum(np.outer(tl,L1),np.outer(th,L1))
    return np.linalg.norm(np.maximum(np.abs(lw),np.abs(hw)),axis=1)
def certify_local(pl,net,q,lo,hi,budget=6000):
    """Theorem 1' certificate: ||f(x,clip pi)||_P + beta*eps(x) <= sqrt(c) on every cell (sound)"""
    sb=pl["s"]; sc=np.sqrt(pl["c"]); us=pl["us"]
    while True:
        ok=wbox(pl,lo,hi,net,us)+pl["beta"]*us*B.eps_dibp(pl,net,q,lo,hi)<=sc; bad=~ok
        if not bad.any(): return True
        if len(lo)>budget: return False
        L,H=lo[bad],hi[bad]; ax=np.argmax(H-L,axis=1); mid=(L+H)/2; r=np.arange(len(L))
        H1=H.copy(); H1[r,ax]=mid[r,ax]; L2=L.copy(); L2[r,ax]=mid[r,ax]; lo=np.vstack([L,L2]); hi=np.vstack([H1,H])
        cen=(lo+hi)/2*sb; d=lo.shape[1]; cr_=np.stack(np.meshgrid(*[[0,1]]*d,indexing="ij"),-1).reshape(-1,d)
        rP=np.max([B.pn(pl,(np.where(c_,hi,lo)*sb)-cen) for c_ in cr_],axis=0)
        keep=B.pn(pl,cen)<=sc+rP+1e-12; lo,hi=lo[keep],hi[keep]
def run(pl,n,m,budget):
    pl=B.setup(pl); d=len(pl["Acl"]); r0=np.random.default_rng(123)
    pl["us"]=np.abs(np.clip(pl["expert"](B.samples(pl,20000,r0,1.5)),-pl["umax"],pl["umax"])).max()
    X=np.vstack([B.samples(pl,30000,r0),B.boundary(pl,5000,r0)]); lo,hi=B.cells(pl,m); sc=np.sqrt(pl["c"]); res=[]; viol=0
    print("\n===",pl["name"],flush=True)
    for seed in range(n):
        rng=np.random.default_rng(seed); net=B.train(B.init(rng,d),pl,rng)
        inv={}; first=None
        for b in range(2,13):
            q=B.quant(net,b); inv[b]=B.pn(pl,pl["f"](X,B.ctrl(pl,q,X))).max()<=sc*(1+1e-9)
        for b in range(2,13):
            q=B.quant(net,b)
            if certify_local(pl,net,q,lo,hi,budget):
                if first is None: first=b
                if not inv[b]: viol+=1
        tb=next((b for b in range(2,13) if all(inv[x] for x in range(b,13))),None)
        res.append((seed,first,tb)); print("seed %2d  local certificate (Thm 1'+splitting) bits=%s | TRUE=%s"%(seed,first,tb),flush=True)
    print("soundness violations:",viol)
    f=np.array([x[1] if x[1] else np.nan for x in res],float); t=np.array([x[2] for x in res],float)
    print("mean certified %.1f  mean true %.1f  gap %.1f"%(np.nanmean(f),t.mean(),np.nanmean(f-t)))
    return res
if __name__=="__main__":
    n=int(sys.argv[1]); out={"4d":run(B.make_4d(),n,3,int(sys.argv[2])),"pend":run(B.make_pend(),n,8,int(sys.argv[2]))}
    json.dump(out,open("res3.json","w"),default=float)
