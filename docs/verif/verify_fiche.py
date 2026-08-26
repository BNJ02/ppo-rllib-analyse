#!/usr/bin/env python3
"""Verification machine de chaque demonstration de rapport/fiche-policy-gradient.md.

Etage 1 : identites numeriques exactes sur un PDM fini enumere (voir mdp.py).
Etage 2 : etapes algebriques par sympy (voir verify_symbolic.py).

Chaque assertion porte le numero de section de la fiche. Sortie code 1 si un
test echoue. Les tests marques [NEG] sont des controles negatifs : ils doivent
DETECTER une erreur, sinon le banc d'essai ne prouve rien.

    python3 verify_fiche.py
"""
import sys

import numpy as np
import torch

import mdp
from mdp import (A, D, DISC, GAMMA, N_TRAJ, S, STATES, ACTIONS, T, THETA0, REW)

TOL = 1e-9
FAILURES = []
COUNT = 0


def check(name, ok, detail=""):
    global COUNT
    COUNT += 1
    tag = "OK  " if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def close(x, y, tol=TOL):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    denom = max(np.linalg.norm(y), 1.0)
    return float(np.linalg.norm(x - y) / denom)


def section(title):
    print(f"\n=== {title}")


theta = THETA0
G = mdp.grad_J(theta)
V, Q, Adv = mdp.value_functions(theta)
pi = mdp.policy_np(theta)
score = mdp.score_table(theta)
ptraj = mdp.traj_probs(theta)


# ==========================================================================
section("Reference : gradient de politique (deux voies independantes)")
# ==========================================================================
e = close(mdp.grad_J_findiff(theta), G)
check("grad autodiff == differences finies centrees", e < 1e-6, f"ecart={e:.2e}")


# ==========================================================================
section("2. REINFORCE")
# ==========================================================================
# 1 - identite brute somme x somme (2.1)
g1, var_brute = mdp.estimator(theta, mdp.psi_total_return())
e = close(g1, G)
check("#1  identite brute  E[(sum_t score)(sum_t gamma^t R_t)] = grad J (2.1)",
      e < TOL, f"ecart={e:.2e}")

# 2 - la dynamique disparait au gradient (2.1)
def grad_log_Ptraj(n):
    th = torch.tensor(np.asarray(theta, dtype=float), requires_grad=True)
    p = mdp.policy(th)
    lp = torch.log(p[STATES[n], ACTIONS[n]]).sum() + float(np.log(mdp.DYN[n]))
    g, = torch.autograd.grad(lp, th)
    return g.detach().numpy().ravel()

worst = max(close(grad_log_Ptraj(n), score[STATES[n], ACTIONS[n]].sum(axis=0))
            for n in (0, 17, 511, 2186))
check("#2  grad ln P_theta(tau) = sum_t score, dynamique absente (2.1)",
      worst < TOL, f"ecart={worst:.2e}")

# 3 - lemme du score nul, premiere partie (2.2)
worst = 0.0
for i in range(T + 1):
    for j in range(i, T + 1):
        for si in range(S):
            mask = STATES[:, i] == si
            w = ptraj[mask]
            if w.sum() < 1e-12:
                continue
            val = (w @ score[STATES[mask, j], ACTIONS[mask, j]]) / w.sum()
            worst = max(worst, float(np.abs(val).max()))
check("#3  lemme : E[score(A_j|S_j) | S_i=s_i] = 0 pour tout i<=j (2.2)",
      worst < TOL, f"max={worst:.2e}")

# 4 - lemme, seconde partie : Psi_i mesurable dans le passe (2.2)
worst = 0.0
for i in range(T + 1):
    for j in range(i, T + 1):
        for si in range(S):
            mask = STATES[:, i] == si
            w = ptraj[mask]
            if w.sum() < 1e-12:
                continue
            # STRICTEMENT anterieur a i : R_i depend de A_i, donc l'inclure
            # violerait l'hypothese du lemme (et le test echoue si on l'inclut).
            psi = (REW[mask, :i] * DISC[:i]).sum(axis=1) if i > 0 else np.ones(mask.sum())
            val = (w * psi) @ score[STATES[mask, j], ACTIONS[mask, j]] / w.sum()
            worst = max(worst, float(np.abs(val).max()))
check("#4  lemme : E[score(A_j|S_j) * Psi_i | S_i] = 0, Psi_i passe (2.2)",
      worst < TOL, f"max={worst:.2e}")

# 5 - astuce de causalite (2.3)
g5, var_caus = mdp.estimator(theta, mdp.psi_reinforce())
e = close(g5, G)
check("#5  astuce de causalite : sum_{tau>=t} = grad J (2.3)", e < TOL, f"ecart={e:.2e}")
check("#5b causalite reduit strictement la variance (2.3)",
      var_caus < var_brute, f"{var_caus:.4f} < {var_brute:.4f}")


# ==========================================================================
section("3. Baseline, critique, avantage")
# ==========================================================================
# 6 - baseline arbitraire (3.1)
rng = np.random.default_rng(1)
worst = 0.0
for _ in range(5):
    b = rng.normal(size=(T + 1, S)) * 3
    gb, _ = mdp.estimator(theta, mdp.psi_baseline(b))
    worst = max(worst, close(gb, G))
check("#6  baseline b(t,s) arbitraire : estimateur non biaise (3.1)",
      worst < TOL, f"max sur 5 tirages={worst:.2e}")

# 6bis - [NEG] baseline dependant de l'action
b_act = rng.normal(size=(T + 1, S, A)) * 3
gba, _ = mdp.estimator(theta, mdp.psi_baseline_action(b_act))
e = close(gba, G)
check("#6b [NEG] baseline dependant de l'ACTION : biaise (3.1)",
      e > 1e-3, f"ecart={e:.2e} (doit etre grand)")

# 7 - baseline optimal gamma^t V_t (3.2)
b_star = DISC[:, None] * V[:T + 1]
g7, var_star = mdp.estimator(theta, mdp.psi_baseline(b_star))
e = close(g7, G)
check("#7  baseline gamma^t V_t(S_t) : non biaise (3.2)", e < TOL, f"ecart={e:.2e}")
rand_vars = []
for _ in range(20):
    b = rng.normal(size=(T + 1, S)) * 3
    rand_vars.append(mdp.estimator(theta, mdp.psi_baseline(b))[1])
check("#7b baseline V : variance < sans baseline et < 20 baselines aleatoires (3.2)",
      var_star < var_caus and var_star < min(rand_vars),
      f"V={var_star:.4f}  sans={var_caus:.4f}  min alea={min(rand_vars):.4f}")

# 8 - forme Q (3.3)
g8, _ = mdp.estimator(theta, mdp.psi_Q(Q))
e = close(g8, G)
check("#8  forme Q : gamma^t Q_t(S_t,A_t) = grad J (3.3)", e < TOL, f"ecart={e:.2e}")

# 9 - forme avantage (3.4)
g9, var_adv = mdp.estimator(theta, mdp.psi_A(Adv))
e = close(g9, G)
check("#9  forme avantage : gamma^t A_t(S_t,A_t) = grad J (3.4)", e < TOL, f"ecart={e:.2e}")

# 10 - chaque ligne du tableau Psi_t (3.5)
rows = {
    "retour total": mdp.psi_total_return(),
    "REINFORCE": mdp.psi_reinforce(),
    "REINFORCE + baseline": mdp.psi_baseline(b_star),
    "TD 1 pas": mdp.psi_nstep(V, 1),
    "TD 2 pas": mdp.psi_nstep(V, 2),
    "Q": mdp.psi_Q(Q),
    "avantage": mdp.psi_A(Adv),
}
for n in range(1, T + 2):
    rows[f"TD {n} pas"] = mdp.psi_nstep(V, n)
worst, worst_name = 0.0, ""
for name, psi in rows.items():
    e = close(mdp.estimator(theta, psi)[0], G)
    if e > worst:
        worst, worst_name = e, name
check(f"#10 les {len(rows)} lignes du tableau Psi_t donnent grad J (3.5)",
      worst < TOL, f"pire={worst:.2e} ({worst_name})")

# 11 - combinaison lineaire quelconque (3.5)
keys = list(rows)
w = rng.normal(size=len(keys))
w = w / w.sum()                                  # poids sommant a 1
mix = sum(wi * rows[k] for wi, k in zip(w, keys))
e = close(mdp.estimator(theta, mix)[0], G)
check("#11 combinaison lineaire de poids 1 : = grad J (3.5)", e < TOL, f"ecart={e:.2e}")
w2 = rng.normal(size=len(keys))
mix2 = sum(wi * rows[k] for wi, k in zip(w2, keys))
e = close(mdp.estimator(theta, mix2)[0], w2.sum() * G)
check("#11b combinaison de poids W : = W * grad J (3.5)", e < TOL, f"ecart={e:.2e}")

# 12 - GAE : forme recursive == forme somme (3.6)
LAM = 0.7
rec = mdp.psi_gae_recursive(V, LAM)
som = mdp.psi_gae_sum(V, LAM)
e = close(rec.ravel(), som.ravel())
check("#12 GAE : A_t = delta_t + gamma*lam*A_{t+1}  ==  sum_l (gamma lam)^l delta (3.6)",
      e < TOL, f"ecart={e:.2e}")
e = close(mdp.estimator(theta, rec)[0], G)
check("#12b GAE (recursif) : estimateur non biaise (3.6)", e < TOL, f"ecart={e:.2e}")

# 13 - coefficient de ponderation : standard vs Wikipedia (3.5 / 3.6)
std = mdp.psi_gae_weighted(V, LAM, 'standard')
wik = mdp.psi_gae_weighted(V, LAM, 'wiki')
e = close(std.ravel(), rec.ravel())
check("#13 GAE : ponderation (1-lam)lam^{n-1} des n-step == forme recursive (3.6)",
      e < TOL, f"ecart={e:.2e}")
g_std = mdp.estimator(theta, std)[0]
g_wik = mdp.estimator(theta, wik)[0]
ratio = float(np.median(g_wik / g_std))
attendu = 1.0 / (1 - LAM) ** 2
check("#13b [ECART WIKIPEDIA] ponderation lam^{n-1}/(1-lam) : facteur 1/(1-lam)^2",
      abs(ratio - attendu) < 1e-6,
      f"mesure={ratio:.6f}  attendu={attendu:.6f}  (lam={LAM})")
check("#13c la version Wikipedia n'est donc PAS egale a grad J",
      close(g_wik, G) > 1e-3, f"ecart a grad J={close(g_wik, G):.3f}")

# 32 - omettre les gamma^t biaise l'estimateur (annexe B)
psi_nogamma = Adv[np.arange(T + 1)[None, :], STATES, ACTIONS]
e = close(mdp.estimator(theta, psi_nogamma)[0], G)
check("#32 [NEG] omettre gamma^t dans Psi_t biaise l'estimateur (annexe B)",
      e > 1e-3, f"ecart={e:.2e} (doit etre grand)")


# ==========================================================================
section("4. Natural Policy Gradient")
# ==========================================================================
# distribution d'etats visites sous pi_theta (la meme pour F et pour KL barre)
d_state = np.zeros(S)
for t in range(T + 1):
    for s in range(S):
        d_state[s] += ptraj[STATES[:, t] == s].sum()
d_state /= d_state.sum()

# matrice de Fisher (4.3)
F = np.zeros((D, D))
for s in range(S):
    for a in range(A):
        F += d_state[s] * pi[s, a] * np.outer(score[s, a], score[s, a])


def kl_bar(theta_new_flat, order="new||old"):
    """KL moyennee sur les etats, differentiable en theta_new."""
    tn = theta_new_flat.reshape(S, A - 1)
    p_new = mdp.policy(tn)
    p_old = mdp.policy(torch.as_tensor(theta))
    d = torch.as_tensor(d_state)
    if order == "new||old":
        kl = (p_new * (torch.log(p_new) - torch.log(p_old))).sum(dim=1)
    else:
        kl = (p_old * (torch.log(p_old) - torch.log(p_new))).sum(dim=1)
    return (d * kl).sum()


flat = torch.tensor(np.asarray(theta, dtype=float).ravel())
for order in ("new||old", "old||new"):
    H = torch.autograd.functional.hessian(lambda x: kl_bar(x, order), flat).numpy()
    e = close(H, F)
    check(f"#14 hessienne de D_KL({order}) en Delta=0 == F (4.3)", e < 1e-8, f"ecart={e:.2e}")

# 15 - identite de l'information : E[-grad^2 ln pi] = F
negH = np.zeros((D, D))
for s in range(S):
    for a in range(A):
        h = torch.autograd.functional.hessian(
            lambda x, s=s, a=a: torch.log(mdp.policy(x.reshape(S, A - 1))[s, a]), flat).numpy()
        negH += -d_state[s] * pi[s, a] * h
e = close(negH, F)
check("#15 identite de l'information : E[-grad^2 ln pi] = F (4.3)", e < 1e-8, f"ecart={e:.2e}")

# 16 - ordres 0 et 1 du developpement de la KL
val0 = float(kl_bar(flat).item())
x = flat.clone().requires_grad_(True)
g0, = torch.autograd.grad(kl_bar(x), x)
check("#16 KL : ordre 0 nul et ordre 1 nul en Delta=0 (4.3)",
      abs(val0) < 1e-12 and float(g0.abs().max()) < 1e-10,
      f"KL(0)={val0:.2e}  |grad|={float(g0.abs().max()):.2e}")

# 17/18 - programme quadratique et son pas
EPS = 0.01
Finv_g = np.linalg.solve(F, G)
alpha = np.sqrt(2 * EPS / (G @ Finv_g))
delta_closed = alpha * Finv_g

from scipy.optimize import minimize
res = minimize(lambda dl: -(G @ dl), np.zeros(D), method="SLSQP",
               constraints=[{"type": "ineq", "fun": lambda dl: EPS - 0.5 * dl @ F @ dl}],
               options={"maxiter": 500, "ftol": 1e-14})
e = close(res.x, delta_closed)
check("#17 solution fermee du QP == optimum numerique SLSQP (4.4)", e < 1e-5, f"ecart={e:.2e}")
sat = 0.5 * delta_closed @ F @ delta_closed
check("#18 le pas alpha = sqrt(2 eps / g'F^-1 g) sature la contrainte (4.4)",
      abs(sat - EPS) < 1e-12, f"0.5 Delta'F Delta = {sat:.12f}  eps={EPS}")

# 19 - argmax lineaire sur boule euclidienne -> Delta = alpha g (4.1)
alpha_e = 0.3
r = alpha_e * np.linalg.norm(G)
res2 = minimize(lambda dl: -(G @ dl), np.zeros(D), method="SLSQP",
                constraints=[{"type": "ineq", "fun": lambda dl: r - np.linalg.norm(dl)}],
                options={"maxiter": 500, "ftol": 1e-14})
e = close(res2.x, alpha_e * G)
check("#19 max g'Delta sur ||Delta||<=alpha||g||  =>  Delta = alpha g (4.1)",
      e < 1e-5, f"ecart={e:.2e}")


# ==========================================================================
section("5. TRPO")
# ==========================================================================
def surrogate(theta_new, theta_old_np, adv):
    """L(theta, theta_old) = E_{pi_old}[ sum_t gamma^t r_t(theta) A_t ]."""
    p_new = mdp.policy(theta_new)
    p_old_np = mdp.policy_np(theta_old_np)
    ratio = p_new[STATES, ACTIONS] / torch.as_tensor(p_old_np[STATES, ACTIONS])
    a_t = torch.as_tensor(adv[np.arange(T + 1)[None, :], STATES, ACTIONS])
    w = torch.as_tensor(mdp.traj_probs(theta_old_np))
    return (w * (torch.as_tensor(DISC) * ratio * a_t).sum(dim=1)).sum()


th_t = torch.tensor(np.asarray(theta, dtype=float), requires_grad=True)
L0 = surrogate(th_t, theta, Adv)
check("#20 L(theta_i, theta_i) = 0 (5.2)", abs(float(L0.item())) < 1e-12,
      f"L={float(L0.item()):.2e}")
gL, = torch.autograd.grad(L0, th_t)
e = close(gL.detach().numpy().ravel(), G)
check("#21 grad_theta L(theta,theta_i)|_{theta=theta_i} = grad J(theta_i) (5.2)",
      e < TOL, f"ecart={e:.2e}")

# 22 - [NEG] l'accord tombe des qu'on s'eloigne
ecarts = []
for step in (0.2, 0.6, 1.2):
    th2 = torch.tensor(np.asarray(theta + step, dtype=float), requires_grad=True)
    L2 = surrogate(th2, theta, Adv)
    g2, = torch.autograd.grad(L2, th2)
    ecarts.append(close(g2.detach().numpy().ravel(), mdp.grad_J(theta + step)))
check("#22 [NEG] grad L != grad J loin de theta_i, ecart croissant (5.2, 11)",
      ecarts[0] > 1e-3 and ecarts[0] < ecarts[1] < ecarts[2],
      "ecarts=" + ", ".join(f"{x:.3f}" for x in ecarts))

# 23 - identite d'echantillonnage preferentiel (5.1)
theta_b = theta + 0.4
pi_b = mdp.policy_np(theta_b)
X = np.arange(1, A + 1) * 1.7
worst = 0.0
for s in range(S):
    direct = pi[s] @ X
    isamp = pi_b[s] @ ((pi[s] / pi_b[s]) * X)
    worst = max(worst, abs(direct - isamp))
check("#23 echantillonnage preferentiel : E_pi[X] = E_pi'[(pi/pi')X] (5.1)",
      worst < 1e-12, f"max={worst:.2e}")

# 24 - TRPO avant line search == NPG
g_trpo = gL.detach().numpy().ravel()
delta_trpo = np.sqrt(2 * EPS / (g_trpo @ np.linalg.solve(F, g_trpo))) * np.linalg.solve(F, g_trpo)
e = close(delta_trpo, delta_closed)
check("#24 mise a jour TRPO (avant line search) == mise a jour NPG (5.3, 11)",
      e < TOL, f"ecart={e:.2e}")


# 25 - gradient conjugue
def conjugate_gradient(Amat, b, iters=50, tol=1e-14):
    x = np.zeros_like(b)
    r_ = b - Amat @ x
    p_ = r_.copy()
    rs = r_ @ r_
    for _ in range(iters):
        Ap = Amat @ p_
        a_ = rs / (p_ @ Ap)
        x += a_ * p_
        r_ -= a_ * Ap
        rs_new = r_ @ r_
        if np.sqrt(rs_new) < tol:
            break
        p_ = r_ + (rs_new / rs) * p_
        rs = rs_new
    return x


e = close(conjugate_gradient(F, G), Finv_g)
check("#25 gradient conjugue sur F x = g converge vers F^-1 g (5.4)", e < 1e-10, f"ecart={e:.2e}")


# ==========================================================================
section("6. PPO")
# ==========================================================================
def ppo_cases(r, adv, eps):
    """Forme par cas de la page Wikipedia (6.1)."""
    r = np.asarray(r, dtype=float)
    adv = np.asarray(adv, dtype=float)
    pos = np.minimum(r, 1 + eps) * adv
    neg = np.maximum(r, 1 - eps) * adv
    return np.where(adv > 0, pos, np.where(adv < 0, neg, 0.0))


def ppo_clip(r, adv, eps):
    """Forme min/clip du papier (6.2)."""
    r = np.asarray(r, dtype=float)
    adv = np.asarray(adv, dtype=float)
    return np.minimum(r * adv, np.clip(r, 1 - eps, 1 + eps) * adv)


# 26 - equivalence des deux ecritures, grille dense incluant les bords exacts
worst = 0.0
for eps in (0.05, 0.1, 0.2, 0.3):
    rr = np.concatenate([np.linspace(0.0, 3.0, 4001), [1 - eps, 1 + eps, 1.0]])
    aa = np.concatenate([np.linspace(-4, 4, 401), [0.0, 1e-12, -1e-12]])
    Rg, Ag = np.meshgrid(rr, aa)
    worst = max(worst, float(np.abs(ppo_cases(Rg, Ag, eps) - ppo_clip(Rg, Ag, eps)).max()))
check("#26 forme par cas == min(rA, clip(r)A), sur grille dense + bords (6.2)",
      worst < 1e-15, f"ecart max={worst:.2e} sur 4 valeurs de eps")

# 27 - clipping unilateral : quels regimes ont un gradient nul ?
eps = 0.2
h = 1e-7
regimes = {
    ("A>0", "r<1-eps"): (0.5, 2.0), ("A>0", "dans"): (1.0, 2.0), ("A>0", "r>1+eps"): (1.5, 2.0),
    ("A<0", "r<1-eps"): (0.5, -2.0), ("A<0", "dans"): (1.0, -2.0), ("A<0", "r>1+eps"): (1.5, -2.0),
}
nuls, actifs = [], []
for (sa, sr), (r0, a0) in regimes.items():
    d = (ppo_cases(r0 + h, a0, eps) - ppo_cases(r0 - h, a0, eps)) / (2 * h)
    (nuls if abs(d) < 1e-6 else actifs).append(f"{sa},{sr}")
check("#27 clipping unilateral : exactement 2 regimes a gradient nul, 4 actifs (6.3)",
      set(nuls) == {"A>0,r>1+eps", "A<0,r<1-eps"} and len(actifs) == 4,
      f"nuls={sorted(nuls)}")

# 28 - premier pas de la boucle interne : ratio = 1, aucune borne atteinte
r_init = mdp.policy_np(theta)[STATES, ACTIONS] / mdp.policy_np(theta)[STATES, ACTIONS]
check("#28 boucle interne, pas 0 : r = 1 partout, aucune borne active (6.4)",
      np.allclose(r_init, 1.0) and (1 - eps < 1.0 < 1 + eps), "r=1")
th_p = torch.tensor(np.asarray(theta, dtype=float), requires_grad=True)
p_new = mdp.policy(th_p)
ratio_t = p_new[STATES, ACTIONS] / torch.as_tensor(mdp.policy_np(theta)[STATES, ACTIONS])
a_t = torch.as_tensor(Adv[np.arange(T + 1)[None, :], STATES, ACTIONS])
w_t = torch.as_tensor(ptraj)
obj = (w_t * (torch.as_tensor(DISC) * torch.minimum(
    ratio_t * a_t, torch.clamp(ratio_t, 1 - eps, 1 + eps) * a_t)).sum(dim=1)).sum()
gclip, = torch.autograd.grad(obj, th_p)
e = close(gclip.detach().numpy().ravel(), G)
check("#28b au pas 0, grad de l'objectif clippe == grad de politique (6.4)",
      e < TOL, f"ecart={e:.2e}")

# 29 - estimateur k3 de la KL
rng2 = np.random.default_rng(3)
p_ref = rng2.random((S, A)) + 0.1
p_ref /= p_ref.sum(axis=1, keepdims=True)
worst = 0.0
for s in range(S):
    kl = float((pi[s] * np.log(pi[s] / p_ref[s])).sum())
    k3 = float((pi[s] * (np.log(pi[s] / p_ref[s]) + p_ref[s] / pi[s] - 1)).sum())
    worst = max(worst, abs(kl - k3))
check("#29 k3 : E_{a~pi_theta}[log(pi/pref) + pref/pi - 1] = KL(pi||pref) (6.5)",
      worst < 1e-13, f"max={worst:.2e}")

# 29b - [ECART] la page echantillonne sous pi_{theta_t}, pas sous pi_theta
pi_t = mdp.policy_np(theta + 0.5)          # politique de collecte, != pi_theta
worst_b = 0.0
for s in range(S):
    kl = float((pi[s] * np.log(pi[s] / p_ref[s])).sum())
    k3_t = float((pi_t[s] * (np.log(pi[s] / p_ref[s]) + p_ref[s] / pi[s] - 1)).sum())
    worst_b = max(worst_b, abs(kl - k3_t))
check("#29b [ECART] sous pi_{theta_t} != pi_theta, k3 n'est plus egal a la KL (6.5)",
      worst_b > 1e-3, f"ecart={worst_b:.3f} (exact seulement en theta = theta_t)")

# 30 - k3 : integrande positive et variance plus faible
xs = np.linspace(1e-6, 50, 200001)
check("#30 k3 : x - 1 - log x >= 0, nul en x = 1 (6.5)",
      (xs - 1 - np.log(xs) >= -1e-12).all() and abs(1 - 1 - np.log(1.0)) < 1e-15, "")
var_naif, var_k3 = [], []
for s in range(S):
    lr = np.log(pi[s] / p_ref[s])
    naif = lr
    k3v = lr + p_ref[s] / pi[s] - 1
    var_naif.append(float((pi[s] * (naif - naif @ pi[s]) ** 2).sum()))
    var_k3.append(float((pi[s] * (k3v - k3v @ pi[s]) ** 2).sum()))
check("#30b k3 : variance strictement plus faible que l'estimateur naif (6.5)",
      all(a < b for a, b in zip(var_k3, var_naif)),
      f"k3={np.round(var_k3,4).tolist()} vs naif={np.round(var_naif,4).tolist()}")


# ==========================================================================
section("7. GRPO")
# ==========================================================================
Gsz = 8
rews = rng2.normal(size=Gsz) * 2 + 1
mu, sig = rews.mean(), rews.std()
adv_g = (rews - mu) / sig
check("#31 GRPO : score standardise, somme nulle et ecart-type 1 (7)",
      abs(adv_g.sum()) < 1e-12 and abs(adv_g.std() - 1) < 1e-12,
      f"somme={adv_g.sum():.1e}  std={adv_g.std():.6f}")
# mu empirique comme baseline : non biaise seulement si mu est independant de
# l'action ponderee -- ici mu INCLUT a_j, donc le baseline n'est pas exactement
# celui du lemme. On mesure le biais introduit sur un bandit a un etat.
probs_b = np.array([0.5, 0.3, 0.2])
rew_b = np.array([1.0, -0.5, 2.0])
score_b = np.eye(3) - probs_b            # score d'une politique tabulaire
true_grad = (probs_b * rew_b) @ score_b
Ggrp, n_mc = 6, 400000
draws = rng2.choice(3, size=(n_mc, Ggrp), p=probs_b)
r_d = rew_b[draws]
mu_d = r_d.mean(axis=1, keepdims=True)
est = ((r_d - mu_d)[:, :, None] * score_b[draws]).mean(axis=1).mean(axis=0)
facteur = float(np.median(est / true_grad))
check("#31b GRPO : baseline mu empirique -> gradient proportionnel a (1 - 1/G)",
      abs(facteur - (1 - 1 / Ggrp)) < 0.02,
      f"mesure={facteur:.4f}  attendu={(1 - 1/Ggrp):.4f} (biais d'echelle, pas de direction)")


# ==========================================================================
section("3.5 bis - compromis biais-variance : il vient du critique APPROCHE")
# ==========================================================================
# Avec un V EXACT, toutes les lignes du tableau sont non biaisees (test #10).
# La phrase de la fiche "n petit => beaucoup de biais" ne vaut donc que pour un
# V approche. On le verifie en bruitant V.
V_hat = V.copy()
V_hat[:T + 1] += np.random.default_rng(5).normal(size=(T + 1, S)) * 1.5

biais, variances = [], []
for n in range(1, T + 2):
    g_n, v_n = mdp.estimator(theta, mdp.psi_nstep(V_hat, n))
    biais.append(close(g_n, G))
    variances.append(v_n)
check("#33 V approche : le biais du TD a n pas decroit avec n (3.5)",
      all(biais[i] > biais[i + 1] for i in range(len(biais) - 1)),
      "biais=" + ", ".join(f"{b:.4f}" for b in biais))
check("#33b V approche : n = T+1 (Monte-Carlo, sans bootstrap) est non biaise (3.5)",
      biais[-1] < TOL, f"biais={biais[-1]:.2e}")
# La croissance de la variance avec n se mesure avec un V EXACT : un V bruite
# est aussi un mauvais baseline, ce qui ajoute de la variance a tous les n et
# detruit la monotonie. Les deux moities du compromis ne se mesurent donc pas
# dans le meme regime -- le biais avec V approche, la variance avec V exact.
var_exact = [mdp.estimator(theta, mdp.psi_nstep(V, n))[1] for n in range(1, T + 2)]
check("#33c V exact : la variance croit avec n (3.5)",
      all(var_exact[i] < var_exact[i + 1] for i in range(len(var_exact) - 1)),
      "var=" + ", ".join(f"{v:.3f}" for v in var_exact))
var_lam = [mdp.estimator(theta, mdp.psi_gae_recursive(V, l))[1]
           for l in (0.0, 0.3, 0.6, 0.9, 1.0)]
check("#33c bis V exact : la variance de GAE croit avec lambda (3.6)",
      all(var_lam[i] < var_lam[i + 1] for i in range(len(var_lam) - 1)),
      "var=" + ", ".join(f"{v:.3f}" for v in var_lam))

b_lam = [close(mdp.estimator(theta, mdp.psi_gae_recursive(V_hat, l))[0], G)
         for l in (0.0, 0.5, 0.9, 1.0)]
check("#33d V approche : biais de GAE decroissant en lambda, nul en lambda=1 (3.6)",
      all(b_lam[i] > b_lam[i + 1] for i in range(3)) and b_lam[-1] < TOL,
      "lam=0,0.5,0.9,1 -> " + ", ".join(f"{b:.4f}" for b in b_lam))


# ==========================================================================
print(f"\n{'=' * 70}")
print(f"{COUNT} assertions - {len(FAILURES)} echec(s)")
for f in FAILURES:
    print(f"  ECHEC : {f}")
sys.exit(1 if FAILURES else 0)
