"""Third plant: 2-state double integrator (linear, affine in control, saturated). Same pipeline as the paper."""
import numpy as np, scipy.linalg as sl, sys, json
import bits2 as B, bits3 as C, bits6 as S
def make_di():
    dt=0.1; A=np.array([[1,dt],[0,1]]); Bm=np.array([[dt*dt/2],[dt]])
    Pr=sl.solve_discrete_are(A,Bm,np.diag([1,.1]),np.array([[1.]])); K=np.linalg.solve(1+Bm.T@Pr@Bm,Bm.T@Pr@A)
    umax=1.0; lim=1.0
    def f(X,u): return X@A.T+np.clip(u,-umax,umax)[:,None]*Bm[:,0]
    return dict(name="double integrator",A=A,B=Bm,Acl=A-Bm@K,expert=lambda X:-(X@K.T)[:,0],f=f,umax=umax,lim=lim,bcol=Bm[:,0])
def wbox_lin(pl,lo_n,hi_n,net,us):   # same as bits3.wbox, linear-plant branch
    sb=pl["s"]; lo=lo_n*sb; hi=hi_n*sb; Lc=pl["Lc"]
    outs,(ol,oh)=B.ibp(net,lo_n,hi_n); ul=np.clip(us*ol[:,0],-pl["umax"],pl["umax"]); uh=np.clip(us*oh[:,0],-pl["umax"],pl["umax"])
    a,b_=C.imat(Lc@pl["A"],lo,hi); LB=Lc@pl["bcol"]
    lw=a+np.minimum(np.outer(ul,LB),np.outer(uh,LB)); hw=b_+np.maximum(np.outer(ul,LB),np.outer(uh,LB))
    return np.linalg.norm(np.maximum(np.abs(lw),np.abs(hw)),axis=1)
C.wbox=wbox_lin
if __name__=="__main__":
    n=int(sys.argv[1]); pl,X,lo,hi=S.prep(make_di(),8); sc=np.sqrt(pl["c"]); beta=pl["beta"]
    print("c=%.3g beta=%.3g us=%.3g cells=%d"%(pl["c"],beta,pl["us"],len(lo)),flush=True)
    rows=[]; viol=0
    for seed in range(n):
        rng=np.random.default_rng(seed); net=B.train(B.init(rng,2),pl,rng)
        u=B.ctrl(pl,net,X); inv={}; dp={}; lp={}
        for b in range(2,13):
            q=B.quant(net,b); inv[b]=bool(S.inv_ok(pl,net,b,X)); dp[b]=bool(S.cert_direct(pl,q,lo,hi)); lp[b]=bool(C.certify_local(pl,net,q,lo,hi))
            if (dp[b] or lp[b]) and not inv[b]: viol+=1
        first=lambda d:next((b for b in range(2,13) if d[b]),None)
        tb=next((b for b in range(2,13) if all(inv[x] for x in range(b,13))),None)
        r=dict(seed=seed,first=first(inv),true=tb,direct=first(dp),local=first(lp),inv3=inv[3],dp3=dp[3],inv4=inv[4],dp4=dp[4],inv5=inv[5],dp5=dp[5]); rows.append(r); print(r,flush=True)
    print("soundness violations:",viol)
    for k in("first","true","direct","local"):
        v=[x[k] for x in rows if x[k] is not None]; print(k,"mean %.2f over %d/%d seeds"%(np.mean(v),len(v),n))
    for b in (3,4,5): print(b,"bits: sampling-safe",sum(x["inv%d"%b] for x in rows),"direct-proved",sum(x["dp%d"%b] for x in rows))
    json.dump(rows,open("res_di.json","w"))
