import numpy as np, sys, json, itertools
import bits2 as B, bits3 as C, bits4 as D
def crown_out(net,lo,hi):
    """affine bounds of net output over boxes (IBP intermediate bounds, backward linear relaxation). returns Al,cl,Au,cu"""
    outs,_=B.ibp(net,lo,hi); W=[w for w,_ in net]; bb=[b for _,b in net]
    Au=np.repeat(W[3].T,len(lo),0); Al=Au.copy(); cu=np.full(len(lo),bb[3][0]); cl=cu.copy()
    for i in (2,1,0):
        l,u=outs[i]; unst=(l<0)&(u>0); act=(l>=0)
        s=np.where(unst,u/np.where(u-l==0,1,u-l),np.where(act,1.,0.)); t=np.where(unst,-s*l,0.)
        al=np.where(unst,(u>-l).astype(float),np.where(act,1.,0.))
        for which in ("u","l"):
            A=Au if which=="u" else Al; Aposi=A>=0
            pick=np.where(Aposi,s,al) if which=="u" else np.where(Aposi,al,s)
            const=(np.where(Aposi,t,0.)*A).sum(1) if which=="u" else (np.where(~Aposi,t,0.)*A).sum(1)
            Anew=(A*pick)
            c_add=Anew@bb[i]; Aprev=Anew@W[i].T
            if which=="u": Au=Aprev; cu=cu+c_add+const
            else: Al=Aprev; cl=cl+c_add+const
    return Al,cl,Au,cu
def conc(A,c,lo,hi,up):
    return c+(np.where(A>=0,A*hi,A*lo).sum(1) if up else np.where(A>=0,A*lo,A*hi).sum(1))
VERT=None
def phi_ub(pl,netw,lo,hi):
    us,um=pl["us"],pl["umax"]; Li=np.linalg.inv(pl["Lc"]); Mw=pl["Lc"]@pl["A"]@Li; LB=pl["Lc"]@pl["bcol"]
    # interval fallback
    outs,(ol,oh)=B.ibp(netw,lo,hi); ul=np.clip(us*ol[:,0],-um,um); uh=np.clip(us*oh[:,0],-um,um)
    a,b_=C.imat(Mw,lo,hi); lw=a+np.minimum(np.outer(ul,LB),np.outer(uh,LB)); hw=b_+np.maximum(np.outer(ul,LB),np.outer(uh,LB))
    ub=np.linalg.norm(np.maximum(np.abs(lw),np.abs(hw)),axis=1)
    Al,cl,Au,cu=crown_out(netw,lo,hi)
    L=us*conc(Al,cl,lo,hi,False); U=us*conc(Au,cu,lo,hi,True); ok=(L>=-um)&(U<=um)
    if ok.any():
        Am=us*(Al+Au)/2; cm=us*(cl+cu)/2; Dw=us*conc(Au-Al,cu-cl,lo,hi,True); rad=np.maximum(Dw,0)/2
        N=Mw[None]+LB[None,:,None]*Am[:,None,:]
        V=np.array(list(itertools.product([0,1],repeat=lo.shape[1])),float)               # 16 x d
        vert=lo[:,None,:]+(hi-lo)[:,None,:]*V[None]                                       # cells,16,d
        img=np.einsum("cij,cvj->cvi",N,vert)+LB[None,None,:]*cm[:,None,None]
        ua=np.linalg.norm(img,axis=2).max(1)+np.linalg.norm(LB)*rad
        ub=np.where(ok,np.minimum(ub,ua),ub)
    return ub
def cert(pl,netw,qw,lo,hi,budget,use_eps=True):
    sc=np.sqrt(pl["c"]); us=pl["us"]
    while True:
        ub=phi_ub(pl,qw if not use_eps else netw,lo,hi)
        e=pl["beta"]*us*B.eps_dibp(pl,netw,qw,lo,hi) if use_eps else 0.
        bad=~(ub+e<=sc)
        if not bad.any(): return True,len(lo)
        if len(lo)>budget: return False,len(lo)
        L,H=lo[bad],hi[bad]; ax=np.argmax(H-L,axis=1); mid=(L+H)/2; r=np.arange(len(L))
        H1=H.copy(); H1[r,ax]=mid[r,ax]; L2=L.copy(); L2[r,ax]=mid[r,ax]; lo=np.vstack([L,L2]); hi=np.vstack([H1,H])
        keep=np.linalg.norm((lo+hi)/2,axis=1)<=sc+np.linalg.norm((hi-lo)/2,axis=1); lo,hi=lo[keep],hi[keep]
if __name__=="__main__":
    import time
    pl=B.setup(B.make_4d()); r0=np.random.default_rng(123)
    pl["us"]=np.abs(np.clip(pl["expert"](B.samples(pl,20000,r0,1.5)),-pl["umax"],pl["umax"])).max()
    X=np.vstack([B.samples(pl,30000,r0),B.boundary(pl,5000,r0)]); sc=np.sqrt(pl["c"]); lo0,hi0=D.wcells(pl,3)
    # sanity: crown sound on random points
    rng=np.random.default_rng(0); net=B.train(B.init(rng,4),pl,rng); nw=D.fold(pl,net)
    Al,cl,Au,cu=crown_out(nw,lo0,hi0); k=np.random.default_rng(1); bad=0
    for _ in range(200):
        w=lo0+(hi0-lo0)*k.uniform(size=lo0.shape); y=B.fwd(nw,w)
        L=(Al*w).sum(1)+cl; U=(Au*w).sum(1)+cu; bad+=int(((y<L-1e-9)|(y>U+1e-9)).sum())
    print("CROWN soundness violations:",bad)
    for bud in (6000,20000,60000):
        t=time.time(); ok,n=cert(pl,nw,nw,lo0,hi0,bud,use_eps=False); print("FP certified at budget",bud,ok,n,"cells %.0fs"%(time.time()-t),flush=True)
        if ok: break
