import numpy as np, scipy.linalg as sl, sys, json
# ================= plants =================
def make_4d():
    m,Iz,lf,lr,Cf,Cr,v,dt=1500.,2500.,1.2,1.6,80000.,80000.,15.,0.05
    Ac=np.array([[0,1,0,0],[0,-(2*Cf+2*Cr)/(m*v),(2*Cf+2*Cr)/m,(-2*Cf*lf+2*Cr*lr)/(m*v)],
                 [0,0,0,1],[0,-(2*Cf*lf-2*Cr*lr)/(Iz*v),(2*Cf*lf-2*Cr*lr)/Iz,-(2*Cf*lf**2+2*Cr*lr**2)/(Iz*v)]])
    Bc=np.array([[0],[2*Cf/m],[0],[2*Cf*lf/Iz]])
    M=sl.expm(np.block([[Ac,Bc],[np.zeros((1,5))]])*dt); A=M[:4,:4]; B=M[:4,4:5]
    Pr=sl.solve_discrete_are(A,B,np.diag([1,.1,1,.1]),np.array([[5.]])); K=np.linalg.solve(5+B.T@Pr@B,B.T@Pr@A)
    umax=0.5; lim=1.5
    def f(X,u): return X@A.T+np.clip(u,-umax,umax)[:,None]*B[:,0]
    return dict(name="4D bicycle",A=A,B=B,Acl=A-B@K,expert=lambda X:-(X@K.T)[:,0],f=f,umax=umax,lim=lim,bcol=B[:,0])
def make_pend():
    g,b,dt=9.8,0.1,0.05; kp,kd=15.,5.9; umax=20.; lim=0.8
    Acl=np.array([[1,dt],[-dt*kp,1-dt*(b+kd)]]); B=np.array([[0],[dt]])
    def f(X,u):
        th,om=X[:,0],X[:,1]; uu=np.clip(u,-umax,umax)
        return np.stack([th+dt*om, om+dt*(g*np.sin(th)-b*om+uu)],1)
    return dict(name="pendulum",Acl=Acl,expert=lambda X:-g*np.sin(X[:,0])-kp*X[:,0]-kd*X[:,1],f=f,umax=umax,lim=lim,bcol=B[:,0])
def setup(pl):
    Acl=pl["Acl"]; rad=max(abs(np.linalg.eigvals(Acl))); rho0=max(0.93,rad+0.01)
    P=sl.solve_discrete_lyapunov((Acl/rho0).T,np.eye(len(Acl))); Pi=np.linalg.inv(P)
    pl.update(P=P,Pi=Pi,Lc=np.linalg.cholesky(P).T,c=pl["lim"]**2/Pi[0,0],rho0=rho0,
              beta=float(np.sqrt(pl["bcol"]@P@pl["bcol"])),s=np.sqrt(pl["c"]*np.diag(Pi)) if False else None)
    r=np.random.default_rng(7); cmax=pl["c"]; Xe=samples(pl,20000,r)
    lo_,hi_=1e-3,1.0
    for _ in range(30):
        g=(lo_+hi_)/2; Xg=Xe*np.sqrt(g); rr=pn(pl,pl["f"](Xg,pl["expert"](Xg))).max()/np.sqrt(cmax*g)
        if rr<=0.95: lo_=g
        else: hi_=g
    pl["c"]=cmax*lo_; pl["gamma"]=lo_
    pl["s"]=np.sqrt(pl["c"]*np.diag(Pi)); return pl
def pn(pl,X): return np.linalg.norm(X@pl["Lc"].T,axis=1)
def samples(pl,n,rng,rad=1.0):
    d=len(pl["Acl"]); U=rng.normal(size=(n,d)); U/=np.linalg.norm(U,axis=1,keepdims=True)
    W=U*np.sqrt(pl["c"])*rad*rng.uniform(0,1,(n,1))**(1/d); return np.linalg.solve(pl["Lc"],W.T).T
def boundary(pl,n,rng):
    d=len(pl["Acl"]); U=rng.normal(size=(n,d)); U/=np.linalg.norm(U,axis=1,keepdims=True)
    return np.linalg.solve(pl["Lc"],(U*np.sqrt(pl["c"])).T).T
# ================= numpy MLP: x/s -> 3x32 ReLU -> 1, output * us =================
def init(rng,d,h=32):
    sz=[d,h,h,h,1]; return [[rng.normal(0,np.sqrt(2/sz[i]),(sz[i],sz[i+1])),np.zeros(sz[i+1])] for i in range(4)]
def fwd(net,z,acts=False):
    a=[z]
    for i,(W,b) in enumerate(net):
        zz=a[-1]@W+b; a.append(np.maximum(zz,0) if i<3 else zz)
    return a if acts else a[-1][:,0]
def train(net,pl,rng,steps=5000):
    m=[[np.zeros_like(W),np.zeros_like(b)] for W,b in net]; v=[[np.zeros_like(W),np.zeros_like(b)] for W,b in net]
    for t in range(1,steps+1):
        x=samples(pl,512,rng,1.5); y=np.clip(pl["expert"](x),-pl["umax"],pl["umax"])/pl["us"]
        a=fwd(net,x/pl["s"],True); d=2*(a[-1][:,0]-y)[:,None]/len(x)
        for i in range(3,-1,-1):
            W,b=net[i]; gW=a[i].T@d; gb=d.sum(0)
            if i>0: d=(d@W.T)*(a[i]>0)
            lr=2e-3*(0.2 if t>steps*0.7 else 1)
            for j,gg in enumerate((gW,gb)):
                m[i][j]=.9*m[i][j]+.1*gg; v[i][j]=.999*v[i][j]+.001*gg*gg
                net[i][j]-=lr*(m[i][j]/(1-.9**t))/(np.sqrt(v[i][j]/(1-.999**t))+1e-8)
    return net
def quant(net,b):
    q=[]
    for W,bi in net:
        s=np.abs(W).max(0)/(2**(b-1)-1); s=np.where(s==0,1,s); q.append([np.round(W/s)*s,bi.copy()])
    return q
def ctrl(pl,net,X): return pl["us"]*fwd(net,X/pl["s"])
# ================= bounds on eps = sup_E |u_q-u| =================
def imv(M,l,h):                      # interval matmul  l,h: (cells,in);  M: (in,out)
    c=(l+h)/2; r=(h-l)/2; return c@M-0*1, r@np.abs(M)   # returns center, radius (bias added by caller)
def ibp(net,l,h):
    outs=[]
    for i,(W,b) in enumerate(net):
        c=(l+h)/2; r=(h-l)/2; zc=c@W+b; zr=r@np.abs(W); zl,zh=zc-zr,zc+zr
        outs.append((zl,zh))
        l,h=(np.maximum(zl,0),np.maximum(zh,0)) if i<3 else (zl,zh)
    return outs,(l,h)
def cells(pl,m):
    d=len(pl["Acl"]); sb=pl["s"]; edges=np.linspace(-1,1,m+1); idx=np.stack(np.meshgrid(*[np.arange(m)]*d,indexing="ij"),-1).reshape(-1,d)
    lo=edges[idx]; hi=edges[idx+1]; lo=lo*sb; hi=hi*sb               # real coordinates, bounding box of E
    cen=(lo+hi)/2; corners=np.stack(np.meshgrid(*[[0,1]]*d,indexing="ij"),-1).reshape(-1,d)
    rP=np.max([pn(pl,np.where(cr,hi,lo)-cen) for cr in corners],axis=0)
    keep=pn(pl,cen)<=np.sqrt(pl["c"])+rP+1e-12                        # sound: cell may meet E
    return lo[keep]/sb,hi[keep]/sb                                    # normalised coordinates
def eps_dibp(pl,net,q,lo,hi):
    """sound bound on sup |u_q-u| via difference interval propagation (cells already cover E)"""
    zs,_=ibp(net,lo,hi); zq,_=ibp(q,lo,hi)
    xl,xh=lo,hi; xql,xqh=lo,hi; dl=np.zeros_like(lo); dh=np.zeros_like(lo)
    for i in range(4):
        W=net[i][0]; Wq=q[i][0]; dW=Wq-W
        # x~ interval:   dz = dW x~ + W d
        c1=((xql+xqh)/2)@dW; r1=((xqh-xql)/2)@np.abs(dW)
        c2=((dl+dh)/2)@W;  r2=((dh-dl)/2)@np.abs(W)
        dzl=c1+c2-r1-r2; dzh=c1+c2+r1+r2
        if i==3: return np.maximum(np.abs(dzl),np.abs(dzh))[:,0]
        zl,zh=zs[i]; zql,zqh=zq[i]
        nl=np.minimum(0,dzl); nh=np.maximum(0,dzh)                    # relu is monotone & 1-Lipschitz
        act=(zl>0)&(zql>0); nl=np.where(act,dzl,nl); nh=np.where(act,dzh,nh)
        ina=(zh<=0)&(zqh<=0); nl=np.where(ina,0,nl); nh=np.where(ina,0,nh)
        xl,xh=np.maximum(zl,0),np.maximum(zh,0); xql,xqh=np.maximum(zql,0),np.maximum(zqh,0)
        dl=np.maximum(nl,xql-xh); dh=np.minimum(nh,xqh-xl)            # intersect with range difference
def eps_norm(pl,net,q,a_in):
    """norm recursion using the real rounding error  e_l=|dW|a~+|W|e"""
    e=0.; a=a_in
    for i in range(4):
        W=net[i][0]; dW=q[i][0]-W
        e=np.linalg.norm(dW,2)*(a+e)+np.linalg.norm(W,2)*e
        if i<3: a=np.linalg.norm(W,2)*a+np.linalg.norm(net[i][1])
    return e
def K_formula(net,a_in):
    L=4; Ws=[W.T for W,_ in net]; ab=[a_in]
    for i in range(L-1): ab.append(np.linalg.norm(Ws[i],2)*ab[-1]+np.linalg.norm(net[i][1]))
    K=0
    for l in range(L):
        w=np.abs(Ws[l]).max(); no,ni=Ws[l].shape; tail=np.prod([np.linalg.norm(Ws[k],2) for k in range(l+1,L)]) if l<L-1 else 1.
        K+=w*np.sqrt(no*ni)/2*ab[l]*tail
    return K
def certify(pl,net,q,tol,lo,hi,budget=6000):
    """adaptive splitting: True if eps_dibp<=tol on a cover of E (sound), else False"""
    sb=pl["s"]; cnt=0
    while True:
        e=eps_dibp(pl,net,q,lo,hi); bad=e>tol
        if not bad.any(): return True
        if len(lo)>budget: return False
        L,H=lo[bad],hi[bad]; w=(H-L); ax=np.argmax(w,axis=1); mid=(L+H)/2; r=np.arange(len(L))
        H1=H.copy(); H1[r,ax]=mid[r,ax]; L2=L.copy(); L2[r,ax]=mid[r,ax]
        lo=np.vstack([L,L2]); hi=np.vstack([H1,H])
        # prune cells that cannot meet E (sound)
        cen=(lo+hi)/2*sb; d=lo.shape[1]; corners=np.stack(np.meshgrid(*[[0,1]]*d,indexing="ij"),-1).reshape(-1,d)
        rP=np.max([pn(pl,(np.where(cr,hi,lo)*sb)-cen) for cr in corners],axis=0)
        keep=pn(pl,cen)<=np.sqrt(pl["c"])+rP+1e-12; lo,hi=lo[keep],hi[keep]
# ================= experiment =================
def run(pl,nseeds,m):
    pl=setup(pl); d=len(pl["Acl"]); rng0=np.random.default_rng(123)
    pl["us"]=np.abs(np.clip(pl["expert"](samples(pl,20000,rng0,1.5)),-pl["umax"],pl["umax"])).max()
    X=np.vstack([samples(pl,30000,rng0),boundary(pl,5000,rng0)]); lo,hi=cells(pl,m)
    a_in=np.linalg.norm(np.maximum(np.abs(lo),np.abs(hi)).max(0))     # sound radius of normalised input box
    print(f"\n=== {pl['name']}: gamma={pl['gamma']:.3g} c={pl['c']:.3g} beta={pl['beta']:.3g} us={pl['us']:.3g} rho0={pl['rho0']:.3f} cells={len(lo)}")
    res=[]; viol=0; BR=range(2,13)
    for seed in range(nseeds):
        rng=np.random.default_rng(seed); net=train(init(rng,d),pl,rng)
        u=ctrl(pl,net,X); xn=pl["f"](X,u); rho=pn(pl,xn).max()/np.sqrt(pl["c"])
        if rho>=0.995: print(f"seed {seed}: FP net not contracting (rho={rho:.3f}), skipped"); continue
        margin=(1-rho)*np.sqrt(c:=pl["c"])/pl["beta"]
        K=K_formula(net,a_in)*pl["us"]; b_form=1+np.log2(1+K/margin)
        first={}; inv={}
        for b in BR:
            q=quant(net,b); uq=ctrl(pl,q,X)
            inv[b]=pn(pl,pl["f"](X,uq)).max()<=np.sqrt(c)*(1+1e-9)
            e_emp=np.abs(uq-u).max(); e_db=pl["us"]*eps_dibp(pl,net,q,lo,hi).max(); e_nm=pl["us"]*eps_norm(pl,net,q,a_in)
            assert e_db>=e_emp-1e-9, "DIBP unsound!"
            cert=certify(pl,net,q,margin/pl["us"],lo,hi)
            if cert and "adapt" not in first: first["adapt"]=b
            if cert and not inv[b]: viol+=1
            for k,e in (("emp",e_emp),("dibp",e_db),("norm",e_nm)):
                if e<=margin and k not in first: first[k]=b
                if e<=margin and not inv[b]: viol+=1
        tb=next((b for b in BR if all(inv[bb] for bb in range(b,13))),None)
        res.append((seed,rho,margin,b_form,first.get("emp"),first.get("dibp"),first.get("norm"),first.get("adapt"),tb))
        print("seed %2d rho=%.3f margin=%.3g | bits: formula=%.1f norm-rec=%s DIBP=%s DIBP+split=%s measured-eps=%s | TRUE=%s"%(seed,rho,margin,b_form,first.get("norm"),first.get("dibp"),first.get("adapt"),first.get("emp"),tb))
    print("soundness violations (a bound said safe but sampling found a leak):",viol)
    return res
if __name__=="__main__":
    out={}
    out["4d"]=run(make_4d(),int(sys.argv[1]),3)
    out["pend"]=run(make_pend(),int(sys.argv[1]),8)
    json.dump(out,open("res.json","w"),default=float)
