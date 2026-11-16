"""Quick check (about 1 minute, CPU): trains one tiny pendulum network, then checks that
(1) the difference-propagation error bound is never below the sampled error,
(2) the CROWN-style bound contains the network output,
(3) a direct proof, when it succeeds, is not contradicted by sampling."""
import numpy as np
import bits2 as B, bits3 as C, bits5 as F, bits6 as S
pl,X,lo,hi=S.prep(B.make_pend(),8); rng=np.random.default_rng(0)
net=B.train(B.init(rng,2),pl,rng,steps=1500)
u=B.ctrl(pl,net,X)
for b in (3,4,6):
    q=B.quant(net,b); e_s=np.abs(B.ctrl(pl,q,X)-u).max(); e_b=pl["us"]*B.eps_dibp(pl,net,q,lo,hi).max()
    assert e_b>=e_s-1e-9, "difference bound is unsound"; print(f"bits {b}: sampled error {e_s:.4f} <= proved bound {e_b:.4f}  OK")
    proved=S.cert_direct(pl,q,lo,hi,budget=3000); safe=S.inv_ok(pl,net,b,X)
    assert (not proved) or safe, "a proof was contradicted by sampling"; print(f"   direct proof: {proved}, sampling test passes: {safe}")
Al,cl,Au,cu=F.crown_out(net,lo,hi); k=np.random.default_rng(1); w=lo+(hi-lo)*k.uniform(size=lo.shape); y=B.fwd(net,w)
assert ((y>=(Al*w).sum(1)+cl-1e-9)&(y<=(Au*w).sum(1)+cu+1e-9)).all(); print("CROWN-style bounds contain the output  OK")
print("smoke test passed")
