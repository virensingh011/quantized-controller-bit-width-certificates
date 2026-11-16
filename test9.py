"""Soundness tests for the verifier of bits9.py (random boxes; every bound must contain what sampling finds)."""
import numpy as np, sys
import bits2 as B, bits9 as V
def setup4d():
    pl=B.setup(B.make_4d()); rng0=np.random.default_rng(123)
    pl["us"]=np.abs(np.clip(pl["expert"](B.samples(pl,20000,rng0,1.5)),-pl["umax"],pl["umax"])).max(); return pl
if __name__=="__main__":
    pl=setup4d(); c=pl["c"]; M,lbv,beta=V.plant_w(pl); rng=np.random.default_rng(5)
    seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
    r0=np.random.default_rng(seed); net=B.train(B.init(r0,4),pl,r0)
    for name,nn in (("full-precision",net),("3-bit",B.quant(net,3)),("5-bit",B.quant(net,5))):
        fn=V.fold(pl,nn); worst={"crown":0,"sat":0,"U":0}; nviol=0
        for inter in ("crown","ibp"):
            for trial in range(6):
                nb=200; cen=rng.normal(size=(nb,4)); cen*=np.sqrt(c)*rng.uniform(0,1,(nb,1))/np.linalg.norm(cen,axis=1,keepdims=True)
                rad=rng.uniform(0.02,1.2,(nb,4))*np.where(rng.uniform(size=(nb,1))<.5,1,0.15); lo,hi=cen-rad,cen+rad
                Al,cl,Au,cu,_,_=V.crown_out(fn,lo,hi,inter)
                a,c0,rr,ulo,uhi=V.sat_enclosure(Al,cl,Au,cu,(lo+hi)/2,(hi-lo)/2,pl["us"],pl["umax"])
                U=V.U_bound(fn,M,lbv,beta,pl["us"],pl["umax"],c,lo,hi,inter)
                for i in range(nb):
                    W=lo[i]+(hi[i]-lo[i])*rng.uniform(size=(400,4)); y=B.fwd(fn,W)
                    # fold check
                    X=np.linalg.solve(pl["Lc"],W.T).T; assert np.abs(B.ctrl(pl,nn,X)-pl["us"]*y).max()<1e-9
                    lo_c=W@Al[i]+cl[i]; hi_c=W@Au[i]+cu[i]
                    assert (y>=lo_c-1e-9).all() and (y<=hi_c+1e-9).all(),"CROWN bound violated"
                    u=np.clip(pl["us"]*y,-pl["umax"],pl["umax"]); e=u-(W@a[i]+c0[i])
                    assert (np.abs(e)<=rr[i]+1e-9).all(),"saturation enclosure violated"
                    assert (u>=ulo[i]-1e-9).all() and (u<=uhi[i]+1e-9).all(),"u interval violated"
                    inE=np.linalg.norm(W,axis=1)<=np.sqrt(c)
                    if inE.any():
                        nxt=pl["f"](X[inE],pl["us"]*y[inE]); s=B.pn(pl,nxt).max()
                        assert s<=U[i]+1e-9, f"U_bound violated {s} > {U[i]} ({inter})"
                        worst["U"]=max(worst["U"],s/U[i])
        print(f"{name:15s} all containment checks passed (max sampled/U ratio {worst['U']:.3f})",flush=True)
    print("verifier soundness tests passed")
