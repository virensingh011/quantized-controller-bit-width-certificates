"""Verifier for plants that are affine in the control (Appendix A of the paper).

Re-implemented from the description in Appendix A after the original file was lost.
Plant in coordinates w = Lc x:   L f = M w + (Lc B) sat(pi),   E = {||w|| <= sqrt(c)}.
Every step returns an UPPER bound on sup_{Q cap E} ||L f||  (so ||f||_P), hence a pass is a proof
(up to floating-point rounding: float64 without outward rounding).

  (i)   CROWN affine bounds with backward-bounded pre-activations (inter="crown")
        or interval pre-activations (inter="ibp", used for the ablation)
  (ii)  sound relaxation of the saturation (Lemma 1)
  (iii) bound of ||N w + d|| over the box vertices and over the ball (Lemma 2); min taken,
        and also the plain interval bound of the closed loop.
"""
import numpy as np

# ---------------------------------------------------------------- folding / plant in w coordinates
def fold(pl, net):
    """net takes z = x/s with x = Lc^{-1} w  ->  fold the affine input map into layer 0 (input is w)."""
    Tm = np.linalg.inv(pl["Lc"]).T / pl["s"][None, :]
    return [[Tm @ net[0][0], net[0][1]]] + [[W, b] for W, b in net[1:]]

def plant_w(pl):
    Lc = pl["Lc"]
    M = Lc @ pl["A"] @ np.linalg.inv(Lc)
    lb = Lc @ pl["B"][:, 0]
    return M, lb, float(np.linalg.norm(lb))      # beta = ||Lc B|| = ||B||_P

# ---------------------------------------------------------------- (i) CROWN bounds
def relax(l, u):
    act = l >= 0; inact = u <= 0; unst = ~(act | inact)
    d = np.where(unst, u - l, 1.0)
    su = np.where(act, 1.0, np.where(inact, 0.0, u / d))
    iu = np.where(unst, -u * l / d, 0.0)
    sl = np.where(act, 1.0, np.where(inact, 0.0, (u >= -l).astype(float)))
    return su, iu, sl, np.zeros_like(l)

def back(fnet, rl, k, c, r):
    """affine lower/upper bounds of the pre-activation z_k as functions of w, plus numeric bounds on the box."""
    N = c.shape[0]; W, b = fnet[k]
    Lu = np.broadcast_to(W.T, (N,) + W.T.shape).copy(); Ll = Lu.copy()
    cu = np.broadcast_to(b, (N, len(b))).copy(); cl = cu.copy()
    for j in range(k - 1, -1, -1):
        su, iu, sl, il = rl[j]
        Lp = np.maximum(Lu, 0); Ln = np.minimum(Lu, 0)
        cu = cu + (Lp * iu[:, None, :]).sum(2) + (Ln * il[:, None, :]).sum(2)
        Lu = Lp * su[:, None, :] + Ln * sl[:, None, :]
        Lp = np.maximum(Ll, 0); Ln = np.minimum(Ll, 0)
        cl = cl + (Lp * il[:, None, :]).sum(2) + (Ln * iu[:, None, :]).sum(2)
        Ll = Lp * sl[:, None, :] + Ln * su[:, None, :]
        Wj, bj = fnet[j]
        cu = cu + Lu @ bj; cl = cl + Ll @ bj
        Lu = Lu @ Wj.T; Ll = Ll @ Wj.T
    ub = (Lu * c[:, None, :]).sum(2) + (np.abs(Lu) * r[:, None, :]).sum(2) + cu
    lb = (Ll * c[:, None, :]).sum(2) - (np.abs(Ll) * r[:, None, :]).sum(2) + cl
    return Ll, cl, Lu, cu, lb, ub

def crown_out(fnet, lo, hi, inter="crown"):
    """affine enclosure Al w + cl <= net(w) <= Au w + cu on each box; also numeric bounds."""
    c = (lo + hi) / 2; r = (hi - lo) / 2
    W0, b0 = fnet[0]; zc = c @ W0 + b0; zr = r @ np.abs(W0)
    l, u = zc - zr, zc + zr
    rl = [relax(l, u)]
    for k in (1, 2):
        # interval pre-activation of layer k from the previous interval
        al, au = np.maximum(l, 0), np.maximum(u, 0)
        Wk, bk = fnet[k]; ac, ar = (al + au) / 2, (au - al) / 2
        li, ui = ac @ Wk + bk - ar @ np.abs(Wk), ac @ Wk + bk + ar @ np.abs(Wk)
        if inter == "crown":
            _, _, _, _, lc_, uc_ = back(fnet, rl, k, c, r)
            l, u = np.maximum(li, lc_), np.minimum(ui, uc_)     # both sound -> intersect
        else:
            l, u = li, ui
        rl.append(relax(l, u))
    Ll, cl, Lu, cu, lb, ub = back(fnet, rl, 3, c, r)
    return Ll[:, 0, :], cl[:, 0], Lu[:, 0, :], cu[:, 0], lb[:, 0], ub[:, 0]

# ---------------------------------------------------------------- (ii) saturation relaxation (Lemma 1)
def sat_relax(gmin, gmax, ub):
    cl = lambda x: np.clip(x, -ub, ub)
    w = gmax - gmin; ok = w > 1e-15
    alpha = np.where(ok, (cl(gmax) - cl(gmin)) / np.where(ok, w, 1.0), 0.0)
    h = lambda g: cl(g) - alpha * g
    gp = [h(gmin), h(gmax)]; gm = [h(gmin), h(gmax)]
    for bp in (-ub, ub):
        ins = (bp >= gmin) & (bp <= gmax); v = h(np.full_like(gmin, bp))
        gp.append(np.where(ins, v, -np.inf)); gm.append(np.where(ins, v, np.inf))
    return alpha, np.max(gp, axis=0), np.min(gm, axis=0)

def sat_enclosure(Al, cl_, Au, cu_, c, r, us, umax):
    """affine enclosure of sat(us*net(w)):  a w + c0 + [-rr, rr]  on each box."""
    gu_c = us * (Au * c).sum(1) + us * cu_; gu_r = us * (np.abs(Au) * r).sum(1)
    gl_c = us * (Al * c).sum(1) + us * cl_; gl_r = us * (np.abs(Al) * r).sum(1)
    al_u, gam_plus, _ = sat_relax(gu_c - gu_r, gu_c + gu_r, umax)     # clip(g_u) <= al_u*g_u + gam_plus
    al_l, _, gam_minus = sat_relax(gl_c - gl_r, gl_c + gl_r, umax)    # clip(g_l) >= al_l*g_l + gam_minus
    a_up = (al_u * us)[:, None] * Au; c_up = al_u * us * cu_ + gam_plus
    a_lo = (al_l * us)[:, None] * Al; c_lo = al_l * us * cl_ + gam_minus
    D_c = ((a_up - a_lo) * c).sum(1) + (c_up - c_lo); D_r = (np.abs(a_up - a_lo) * r).sum(1)
    rr = np.maximum(D_c + D_r, 0) / 2                                 # max over box of (upper-lower)/2
    a = (a_up + a_lo) / 2; c0 = (c_up + c_lo) / 2
    ulo = np.clip(gl_c - gl_r, -umax, umax); uhi = np.clip(gu_c + gu_r, -umax, umax)
    return a, c0, rr, ulo, uhi

# ---------------------------------------------------------------- (iii) bound over the ellipse
@np.errstate(all="ignore")
def ball_bound_sq(Nm, d, c):
    """max_{||w||^2<=c} ||N w + d||^2 <= lam*c + ||d||^2 + sum g_i^2/(lam - sigma_i), any lam > sigma_max."""
    NtN = np.swapaxes(Nm, 1, 2) @ Nm
    sig, V = np.linalg.eigh(NtN); g = np.einsum("nji,nj->ni", V, np.einsum("nji,nj->ni", Nm, d))
    smax = sig[:, -1]; gn = np.linalg.norm(g, axis=1)
    lo_ = smax.copy(); hi_ = smax + gn / np.sqrt(c) + 1e-9 * (smax + 1.0)
    for _ in range(60):
        mid = (lo_ + hi_) / 2
        f = c - (g ** 2 / np.maximum(mid[:, None] - sig, 1e-300) ** 2).sum(1)
        lo_ = np.where(f < 0, mid, lo_); hi_ = np.where(f < 0, hi_, mid)
    lam = hi_ + 1e-12 * (1 + smax)
    return lam * c + (d ** 2).sum(1) + (g ** 2 / (lam[:, None] - sig)).sum(1)

def U_bound(fnet, M, lbv, beta, us, umax, c, lo, hi, inter="crown", use_ball=True, use_vertex=True, use_interval=True, use_sat=True):
    """upper bound on sup over (box cap ball) of ||L f(w, sat pi(w))||  for every box."""
    cen = (lo + hi) / 2; rad = (hi - lo) / 2
    Al, cl_, Au, cu_, _, _ = crown_out(fnet, lo, hi, inter)
    a, c0, rr, ulo, uhi = sat_enclosure(Al, cl_, Au, cu_, cen, rad, us, umax)
    n = lo.shape[1]; Nm = M[None] + lbv[None, :, None] * a[:, None, :]; d = c0[:, None] * lbv[None, :]
    best = np.full(len(lo), np.inf)
    if use_vertex:
        cor = np.array(np.meshgrid(*[[0, 1]] * n, indexing="ij")).reshape(n, -1).T
        best = np.max([np.linalg.norm(np.einsum("nij,nj->ni", Nm, np.where(cr[None, :] == 1, hi, lo)) + d, axis=1) for cr in cor], axis=0)
    if use_ball:
        bb = np.sqrt(np.maximum(ball_bound_sq(Nm, d, c), 0)); best = np.minimum(best, bb)
    U = best + beta * rr
    if not use_sat: U = np.full(len(lo), np.inf)       # ablation: no affine saturation enclosure
    if use_interval:                                              # plain interval bound of ||M w + lbv u||
        mc = cen @ M.T; mr = rad @ np.abs(M).T
        t1 = lbv[None, :] * ulo[:, None]; t2 = lbv[None, :] * uhi[:, None]
        lo_i = mc - mr + np.minimum(t1, t2); hi_i = mc + mr + np.maximum(t1, t2)
        Ui = np.linalg.norm(np.maximum(np.abs(lo_i), np.abs(hi_i)), axis=1)
        U = np.minimum(U, Ui)
    return U

# ---------------------------------------------------------------- box splitting driver
def initial_cover(c, d, m=3):
    R = np.sqrt(c); e = np.linspace(-R, R, m + 1)
    idx = np.array(np.meshgrid(*[np.arange(m)] * d, indexing="ij")).reshape(d, -1).T
    lo, hi = e[idx], e[idx + 1]
    return prune(lo, hi, c)

def prune(lo, hi, c):
    dist = np.linalg.norm(np.clip(0, lo, hi), axis=1)             # distance from origin to box (exact)
    k = dist <= np.sqrt(c) + 1e-12
    return lo[k], hi[k]

def split(lo, hi):
    w = hi - lo; ax = np.argmax(w, axis=1); r = np.arange(len(lo)); mid = (lo + hi) / 2
    h1 = hi.copy(); h1[r, ax] = mid[r, ax]; l2 = lo.copy(); l2[r, ax] = mid[r, ax]
    return np.vstack([lo, l2]), np.vstack([h1, hi])

def prove(check, c, d, budget=40000, m=3, chunk=1500, start=None, verbose=False):
    """check(lo,hi)-> bool array (True = box certified).  Returns (proved, n_boxes_in_cover, n_evals)."""
    lo, hi = initial_cover(c, d, m) if start is None else start
    done = 0; evals = 0
    while len(lo):
        ok = np.zeros(len(lo), bool)
        for i in range(0, len(lo), chunk):
            ok[i:i + chunk] = check(lo[i:i + chunk], hi[i:i + chunk])
        evals += len(lo); done += int(ok.sum())
        lo, hi = lo[~ok], hi[~ok]
        if verbose: print(f"   passed so far {done}, failing {len(lo)}", flush=True)
        if not len(lo): return True, done, evals
        if done + 2 * len(lo) > budget: return False, done + len(lo), evals
        lo, hi = prune(*split(lo, hi), c)
    return True, done, evals
