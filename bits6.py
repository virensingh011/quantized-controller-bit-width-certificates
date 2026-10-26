import numpy as np, sys, json, time
import bits2 as B, bits3 as C
def train_qat(net,pl,rng,b,steps=2500,lr0=5e-4):
    net=[[W.copy(),bi.copy()] for W,bi in net]
    m=[[np.zeros_like(W),np.zeros_like(bi)] for W,bi in net]; v=[[np.zeros_like(W),np.zeros_like(bi)] for W,bi in net]
    for t in range(1,steps+1):
        x=B.samples(pl,512,rng,1.5); y=np.clip(pl["expert"](x),-pl["umax"],pl["umax"])/pl["us"]
        q=B.quant(net,b); a=B.fwd(q,x/pl["s"],True); d=2*(a[-1][:,0]-y)[:,None]/len(x)
        for i in range(3,-1,-1):
            gW=a[i].T@d; gb=d.sum(0)
            if i>0: d=(d@q[i][0].T)*(a[i]>0)
            for j,gg in enumerate((gW,gb)):
                m[i][j]=.9*m[i][j]+.1*gg; v[i][j]=.999*v[i][j]+.001*gg*gg
                net[i][j]-=lr0*(m[i][j]/(1-.9**t))/(np.sqrt(v[i][j]/(1-.999**t))+1e-8)
    return net
def cert_direct(pl,q,lo,hi,budget=6000):
    sc=np.sqrt(pl["c"]); sb=pl["s"]
    while True:
        bad=~(C.wbox(pl,lo,hi,q,pl["us"])<=sc)
        if not bad.any(): return True
        if len(lo)>budget: return False
        L,H=lo[bad],hi[bad]; ax=np.argmax(H-L,axis=1); mid=(L+H)/2; r=np.arange(len(L))
        H1=H.copy(); H1[r,ax]=mid[r,ax]; L2=L.copy(); L2[r,ax]=mid[r,ax]; lo=np.vstack([L,L2]); hi=np.vstack([H1,H])
        cen=(lo+hi)/2*sb; d=lo.shape[1]; cr=np.stack(np.meshgrid(*[[0,1]]*d,indexing="ij"),-1).reshape(-1,d)
        rP=np.max([B.pn(pl,(np.where(c_,hi,lo)*sb)-cen) for c_ in cr],axis=0); keep=B.pn(pl,cen)<=sc+rP+1e-12; lo,hi=lo[keep],hi[keep]
def prep(pl,m):
    pl=B.setup(pl); r0=np.random.default_rng(123)
    pl["us"]=np.abs(np.clip(pl["expert"](B.samples(pl,20000,r0,1.5)),-pl["umax"],pl["umax"])).max()
    X=np.vstack([B.samples(pl,30000,r0),B.boundary(pl,5000,r0)]); lo,hi=B.cells(pl,m); return pl,X,lo,hi
def inv_ok(pl,net,b,X): return B.pn(pl,pl["f"](X,B.ctrl(pl,B.quant(net,b),X))).max()<=np.sqrt(pl["c"])*(1+1e-9)
def bits_true(pl,net,X): 
    inv={b:inv_ok(pl,net,b,X) for b in range(2,13)}; return next((b for b in range(2,13) if all(inv[x] for x in range(b,13))),None)
if __name__=="__main__":
    n=int(sys.argv[1]); out={}
    # ---------- pendulum: direct vs error-margin certificate, PTQ vs QAT, width ----------
    pl,X,lo,hi=prep(B.make_pend(),8); P=[]; viol=0; t0=time.time()
    for seed in range(n):
        rng=np.random.default_rng(seed); net=B.train(B.init(rng,2),pl,rng)
        invd={b:bool(inv_ok(pl,net,b,X)) for b in range(2,13)}
        row=dict(seed=seed,true=bits_true(pl,net,X),true_first=next((b for b in range(2,13) if invd[b]),None),direct=None,local=None)
        for b in range(2,13):
            q=B.quant(net,b); dp=cert_direct(pl,q,lo,hi); lp=C.certify_local(pl,net,q,lo,hi)
            if dp and row["direct"] is None: row["direct"]=b
            if lp and row["local"] is None: row["local"]=b
            if (dp or lp) and not invd[b]: viol+=1
        for b in (3,4):
            row[f"ptq_inv{b}"]=bool(inv_ok(pl,net,b,X)); row[f"ptq_dir{b}"]=bool(cert_direct(pl,B.quant(net,b),lo,hi))
            qn=train_qat(net,pl,np.random.default_rng(1000+seed),b)
            row[f"qat_inv{b}"]=bool(inv_ok(pl,qn,b,X)); row[f"qat_dir{b}"]=bool(cert_direct(pl,B.quant(qn,b),lo,hi))
            if row[f"qat_dir{b}"] and not row[f"qat_inv{b}"]: viol+=1
            if row[f"ptq_dir{b}"] and not row[f"ptq_inv{b}"]: viol+=1
        P.append(row); print("pend",row,"%.0fs"%(time.time()-t0),flush=True)
    out["pend"]=P; out["pend_viol"]=viol; print("pendulum soundness violations",viol,flush=True)
    json.dump(out,open("res6.json","w"),default=float)
    W=[]
    for h in (16,64):
        for seed in range(min(n,8)):
            rng=np.random.default_rng(seed); net=B.train(B.init(rng,2,h),pl,rng); d=None; l=None
            for b in range(2,13):
                q=B.quant(net,b)
                if d is None and cert_direct(pl,q,lo,hi): d=b
            W.append(dict(h=h,seed=seed,true=bits_true(pl,net,X),direct=d)); print("width",W[-1],flush=True)
    out["width"]=W; json.dump(out,open("res6.json","w"),default=float)
    # ---------- 4D: sampled invariance, PTQ vs QAT ----------
    pl,X,lo,hi=prep(B.make_4d(),3); Q=[]
    for seed in range(n):
        rng=np.random.default_rng(seed); net=B.train(B.init(rng,4),pl,rng); row=dict(seed=seed,true=bits_true(pl,net,X))
        for b in (3,4,5):
            row[f"ptq_inv{b}"]=bool(inv_ok(pl,net,b,X)); qn=train_qat(net,pl,np.random.default_rng(1000+seed),b); row[f"qat_inv{b}"]=bool(inv_ok(pl,qn,b,X))
        Q.append(row); print("4d",row,flush=True)
    out["d4"]=Q; json.dump(out,open("res6.json","w"),default=float)
