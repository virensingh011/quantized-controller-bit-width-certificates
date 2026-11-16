import numpy as np, json, sys, time
import bits2 as B, bits3 as C, bits6 as S
def finetune_fp(net,pl,rng,steps=2500,lr0=5e-4):
    """same data, optimizer and schedule as QAT but with NO rounding in the loop"""
    net=[[W.copy(),bi.copy()] for W,bi in net]
    m=[[np.zeros_like(W),np.zeros_like(bi)] for W,bi in net]; v=[[np.zeros_like(W),np.zeros_like(bi)] for W,bi in net]
    for t in range(1,steps+1):
        x=B.samples(pl,512,rng,1.5); y=np.clip(pl["expert"](x),-pl["umax"],pl["umax"])/pl["us"]
        a=B.fwd(net,x/pl["s"],True); d=2*(a[-1][:,0]-y)[:,None]/len(x)
        for i in range(3,-1,-1):
            gW=a[i].T@d; gb=d.sum(0)
            if i>0: d=(d@net[i][0].T)*(a[i]>0)
            for j,gg in enumerate((gW,gb)):
                m[i][j]=.9*m[i][j]+.1*gg; v[i][j]=.999*v[i][j]+.001*gg*gg
                net[i][j]-=lr0*(m[i][j]/(1-.9**t))/(np.sqrt(v[i][j]/(1-.999**t))+1e-8)
    return net
n=int(sys.argv[1]); out={"pend":[],"d4":[]}
pl,X,lo,hi=S.prep(B.make_pend(),8)
for seed in range(n):
    rng=np.random.default_rng(seed); net=B.train(B.init(rng,2),pl,rng); ft=finetune_fp(net,pl,np.random.default_rng(1000+seed))
    row=dict(seed=seed)
    for b in (3,4):
        row[f"ft_inv{b}"]=bool(S.inv_ok(pl,ft,b,X)); row[f"ft_dir{b}"]=bool(S.cert_direct(pl,B.quant(ft,b),lo,hi))
    out["pend"].append(row); print("pend",row,flush=True); json.dump(out,open("res7.json","w"))
pl,X,lo,hi=S.prep(B.make_4d(),3)
for seed in range(n):
    rng=np.random.default_rng(seed); net=B.train(B.init(rng,4),pl,rng); ft=finetune_fp(net,pl,np.random.default_rng(1000+seed))
    row=dict(seed=seed)
    for b in (3,4,5): row[f"ft_inv{b}"]=bool(S.inv_ok(pl,ft,b,X))
    out["d4"].append(row); print("4d",row,flush=True); json.dump(out,open("res7.json","w"))
print("done")
