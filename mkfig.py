import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
r=json.load(open("res6.json")); P=r["pend"]; Q=r["d4"]
# Fig 1: bits by method
labs=["weight-norm\nformula","one-box\ndiff.","diff.+split\n(global margin)","local\ncertificate","direct\nproof","sampling\ntest"]
pen=[14.7,9.9,5.9,5.3,3.8,3.9]; d4=[15.7,np.nan,11.1,np.nan,np.nan,4.6]
x=np.arange(len(labs)); fig,ax=plt.subplots(figsize=(7.2,3.1))
ax.bar(x-0.2,pen,0.4,label="pendulum",color="#3b6ea5"); ax.bar(x+0.2,[0 if np.isnan(v) else v for v in d4],0.4,label="4D bicycle",color="#e0883a")
for i,v in enumerate(d4):
    if np.isnan(v): ax.text(i+0.2,0.3,"n/a",ha="center",fontsize=7,rotation=90,va="bottom")
ax.set_xticks(x); ax.set_xticklabels(labs,fontsize=7.5); ax.set_ylabel("bits needed (mean, 15 seeds)"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig("fig/fig_bits.pdf"); plt.close()
# Fig 2: per-seed pendulum
fig,ax=plt.subplots(figsize=(7.2,2.8)); s=[p["seed"] for p in P]
ax.plot(s,[p["local"] for p in P],"s",label="local certificate",color="#c0392b"); ax.plot(s,[p["direct"] for p in P],"o",label="direct proof",color="#3b6ea5"); ax.plot(s,[p["true"] for p in P],"x",label="sampling test",color="k")
ax.set_xlabel("training seed"); ax.set_ylabel("bits needed"); ax.set_yticks(range(2,8)); ax.legend(frameon=False,ncol=3,fontsize=8,loc="upper center"); ax.set_ylim(2,7.6); fig.tight_layout(); fig.savefig("fig/fig_seeds.pdf"); plt.close()
# Fig 3: PTQ vs fine-tune->PTQ vs QAT (needs res6.json, res7.json, res8.json)
r7=json.load(open("res7.json")); r8=json.load(open("res8.json")); P7=r7["pend"]; Q7=r7["d4"]
fig,axs=plt.subplots(1,2,figsize=(7.2,2.9),sharey=True); w=0.27
pb=[3,4,5]
a=[sum(p[f"ptq_dir{b}"] for p in P) for b in (3,4)]+[sum(p["ptq_dir5"] for p in r8)]
f=[sum(p[f"ft_dir{b}"] for p in P7) for b in (3,4)]+[sum(p["ft_dir5"] for p in r8)]
q=[sum(p[f"qat_dir{b}"] for p in P) for b in (3,4)]+[sum(p["qat_dir5"] for p in r8)]
x=np.arange(3); axs[0].bar(x-w,a,w,label="PTQ",color="#999"); axs[0].bar(x,f,w,label="fine-tune, then PTQ",color="#e0b050"); axs[0].bar(x+w,q,w,label="QAT",color="#3b6ea5")
axs[0].set_xticks(x); axs[0].set_xticklabels(["3 bits","4 bits","5 bits"]); axs[0].set_title("pendulum (proved safe)",fontsize=9); axs[0].set_ylabel("networks out of 15"); axs[0].legend(frameon=False,fontsize=7.5,loc="lower right")
a=[sum(p[f"ptq_inv{b}"] for p in Q) for b in pb]; f=[sum(p[f"ft_inv{b}"] for p in Q7) for b in pb]; q=[sum(p[f"qat_inv{b}"] for p in Q) for b in pb]
axs[1].bar(x-w,a,w,color="#999"); axs[1].bar(x,f,w,color="#e0b050"); axs[1].bar(x+w,q,w,color="#3b6ea5")
axs[1].set_xticks(x); axs[1].set_xticklabels(["3 bits","4 bits","5 bits"]); axs[1].set_title("4D bicycle (passes sampling test)",fontsize=9)
fig.tight_layout(); fig.savefig("fig/fig_qat.pdf"); plt.close(); print("figs ok")
