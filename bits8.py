import numpy as np, json, sys
import bits2 as B, bits6 as S
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

n=int(sys.argv[1]); out=[]
pl,X,lo,hi=S.prep(B.make_pend(),8)
for seed in range(n):
    rng=np.random.default_rng(seed); net=B.train(B.init(rng,2),pl,rng)
    ft=finetune_fp(net,pl,np.random.default_rng(1000+seed)); qn=S.train_qat(net,pl,np.random.default_rng(1000+seed),5)
    row=dict(seed=seed)
    for name,m in (("ptq",net),("ft",ft),("qat",qn)):
        row[name+"_inv5"]=bool(S.inv_ok(pl,m,5,X)); row[name+"_dir5"]=bool(S.cert_direct(pl,B.quant(m,5),lo,hi))
    out.append(row); print("pend5",row,flush=True); json.dump(out,open("res8.json","w"))
print("done")
