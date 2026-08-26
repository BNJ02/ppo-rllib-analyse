"""Banc d'essai : PDM fini, espérances exactes par énumération.

Aucun échantillonnage Monte-Carlo : toutes les trajectoires sont énumérées avec
leur probabilité exacte, donc une identité vraie tient à ~1e-14 et une identité
fausse saute immédiatement.

Paramétrisation de la politique : softmax sur des logits dont la dernière
composante est fixée à 0. Sans cette réduction, la matrice de Fisher est
singulière (ajouter une constante à theta[s, :] ne change pas pi), et F^-1
n'existe pas.

Convention d'horizon fini : V et Q dépendent du temps. V_t(s) est l'espérance du
retour actualisé DEPUIS t. La page Wikipédia écrit V^pi(S_t) sans indice de
temps, ce qui suppose implicitement l'horizon infini ou une politique
stationnaire ; en horizon fini il faut V_t, sinon les identités sont fausses.
"""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

S, A, T = 3, 3, 3           # etats, actions, dernier pas (t = 0..T)
GAMMA = 0.9
S0 = 0

_rng = np.random.default_rng(20260826)
P = _rng.random((S, A, S)) ** 2 + 0.05      # dynamique asymetrique
P /= P.sum(axis=-1, keepdims=True)
R = _rng.normal(size=(S, A))                # recompenses signees
THETA0 = _rng.normal(size=(S, A - 1)) * 1.1  # loin de l'uniforme

D = S * (A - 1)                             # dimension du parametre


# --------------------------------------------------------------------------- #
# politique
# --------------------------------------------------------------------------- #
def policy(theta):
    """theta : (S, A-1) -> pi : (S, A), torch."""
    zeros = torch.zeros(theta.shape[0], 1, dtype=theta.dtype)
    logits = torch.cat([theta, zeros], dim=1)
    return torch.softmax(logits, dim=1)


def policy_np(theta_np):
    return policy(torch.as_tensor(theta_np)).detach().numpy()


def score_table(theta_np):
    """score[s, a, :] = grad_theta ln pi(a|s), aplati en dimension D."""
    th = torch.tensor(np.asarray(theta_np, dtype=float), requires_grad=True)
    out = np.zeros((S, A, D))
    for s in range(S):
        for a in range(A):
            if th.grad is not None:
                th.grad = None
            lp = torch.log(policy(th)[s, a])
            g, = torch.autograd.grad(lp, th, retain_graph=False)
            out[s, a] = g.detach().numpy().ravel()
    return out


# --------------------------------------------------------------------------- #
# enumeration exhaustive des trajectoires
# --------------------------------------------------------------------------- #
def enumerate_trajectories():
    """Retourne (states, actions, dyn_prob) : arrays (N, T+1), (N, T+1), (N,).

    dyn_prob = produit des p(s'|s,a) : la part de P_theta(tau) qui NE depend PAS
    de theta. C'est ce qui rend le test 2 (la dynamique disparait au gradient)
    verifiable directement.
    """
    states, actions, probs = [], [], []

    def rec(t, s, st, ac, pr):
        if t > T:
            states.append(st.copy())
            actions.append(ac.copy())
            probs.append(pr)
            return
        for a in range(A):
            st.append(s)
            ac.append(a)
            if t == T:
                rec(t + 1, None, st, ac, pr)
            else:
                for s2 in range(S):
                    p = P[s, a, s2]
                    if p > 0:
                        rec(t + 1, s2, st, ac, pr * p)
            st.pop()
            ac.pop()

    rec(0, S0, [], [], 1.0)
    return np.array(states), np.array(actions), np.array(probs)


STATES, ACTIONS, DYN = enumerate_trajectories()
N_TRAJ = len(DYN)
REW = R[STATES, ACTIONS]                       # (N, T+1)
DISC = GAMMA ** np.arange(T + 1)               # gamma^t


def traj_probs(theta_np):
    """P_theta(tau) exact pour chaque trajectoire."""
    pi = policy_np(theta_np)
    pol = pi[STATES, ACTIONS].prod(axis=1)
    return DYN * pol


def J_torch(theta):
    """J(theta) exact, differentiable."""
    pi = policy(theta)
    pol = pi[STATES, ACTIONS].prod(dim=1)
    dyn = torch.as_tensor(DYN)
    ret = torch.as_tensor((REW * DISC).sum(axis=1))
    return (dyn * pol * ret).sum()


def grad_J(theta_np):
    th = torch.tensor(np.asarray(theta_np, dtype=float), requires_grad=True)
    g, = torch.autograd.grad(J_torch(th), th)
    return g.detach().numpy().ravel()


def grad_J_findiff(theta_np, h=1e-6):
    g = np.zeros(D)
    flat = theta_np.ravel().copy()
    for k in range(D):
        up, dn = flat.copy(), flat.copy()
        up[k] += h
        dn[k] -= h
        g[k] = (J_torch(torch.as_tensor(up.reshape(S, A - 1))).item()
                - J_torch(torch.as_tensor(dn.reshape(S, A - 1))).item()) / (2 * h)
    return g


# --------------------------------------------------------------------------- #
# fonctions de valeur, indexees par le temps
# --------------------------------------------------------------------------- #
def value_functions(theta_np):
    """V[t, s], Q[t, s, a], Adv[t, s, a] pour t = 0..T (retour depuis t)."""
    pi = policy_np(theta_np)
    V = np.zeros((T + 2, S))
    Q = np.zeros((T + 1, S, A))
    for t in range(T, -1, -1):
        for s in range(S):
            for a in range(A):
                Q[t, s, a] = R[s, a]
                if t < T:
                    Q[t, s, a] += GAMMA * P[s, a] @ V[t + 1]
            V[t, s] = pi[s] @ Q[t, s]
    Adv = Q - V[:T + 1, :, None]
    return V, Q, Adv


# --------------------------------------------------------------------------- #
# estimateur generique  E[ sum_t Psi_t * score(A_t|S_t) ]
# --------------------------------------------------------------------------- #
def estimator(theta_np, psi):
    """psi : (N, T+1). Retourne (esperance exacte (D,), variance totale)."""
    sc = score_table(theta_np)                  # (S, A, D)
    p = traj_probs(theta_np)                    # (N,)
    per_traj = np.einsum('nt,ntd->nd', psi, sc[STATES, ACTIONS])
    mean = p @ per_traj
    centred = per_traj - mean
    var = float(p @ (centred ** 2).sum(axis=1))
    return mean, var


# --------------------------------------------------------------------------- #
# les differents Psi_t du tableau de la fiche (section 3.5)
# --------------------------------------------------------------------------- #
def psi_total_return():
    """sum_{0<=tau<=T} gamma^tau R_tau : identique pour tous les t."""
    tot = (REW * DISC).sum(axis=1)
    return np.repeat(tot[:, None], T + 1, axis=1)


def psi_reinforce():
    """gamma^t sum_{tau>=t} gamma^{tau-t} R_tau = sum_{tau>=t} gamma^tau R_tau."""
    disc_rew = REW * DISC
    return np.cumsum(disc_rew[:, ::-1], axis=1)[:, ::-1]


def psi_baseline(b):
    """b : (T+1, S) baseline arbitraire dependant de (t, s)."""
    return psi_reinforce() - b[np.arange(T + 1)[None, :], STATES]


def psi_baseline_action(b):
    """Controle negatif : baseline dependant de l'ACTION -> doit biaiser."""
    return psi_reinforce() - b[np.arange(T + 1)[None, :], STATES, ACTIONS]


def psi_Q(Q):
    return DISC[None, :] * Q[np.arange(T + 1)[None, :], STATES, ACTIONS]


def psi_A(Adv):
    return DISC[None, :] * Adv[np.arange(T + 1)[None, :], STATES, ACTIONS]


def psi_nstep(V, n):
    """gamma^t ( sum_{k<n} gamma^k R_{t+k} + gamma^n V_{t+n}(S_{t+n}) - V_t(S_t) ).

    Tronque a l'horizon : V_{t'} = 0 pour t' > T.
    """
    out = np.zeros((N_TRAJ, T + 1))
    Vfull = np.zeros((T + 2, S))
    Vfull[:T + 1] = V[:T + 1]
    for t in range(T + 1):
        acc = np.zeros(N_TRAJ)
        for k in range(n):
            if t + k <= T:
                acc += GAMMA ** k * REW[:, t + k]
        tail = t + n
        if tail <= T:
            acc += GAMMA ** n * Vfull[tail][STATES[:, tail]]
        acc -= Vfull[t][STATES[:, t]]
        out[:, t] = DISC[t] * acc
    return out


def td_residuals(V):
    """delta_t = R_t + gamma V_{t+1}(S_{t+1}) - V_t(S_t), tronque."""
    d = np.zeros((N_TRAJ, T + 1))
    for t in range(T + 1):
        nxt = 0.0
        if t < T:
            nxt = GAMMA * V[t + 1][STATES[:, t + 1]]
        d[:, t] = REW[:, t] + nxt - V[t][STATES[:, t]]
    return d


def psi_gae_recursive(V, lam):
    """Forme recursive : A_t = delta_t + gamma*lam*A_{t+1}, puis facteur gamma^t."""
    d = td_residuals(V)
    adv = np.zeros((N_TRAJ, T + 1))
    nxt = np.zeros(N_TRAJ)
    for t in range(T, -1, -1):
        nxt = d[:, t] + GAMMA * lam * nxt
        adv[:, t] = nxt
    return DISC[None, :] * adv


def psi_gae_sum(V, lam):
    """Forme somme : gamma^t sum_l (gamma*lam)^l delta_{t+l}."""
    d = td_residuals(V)
    adv = np.zeros((N_TRAJ, T + 1))
    for t in range(T + 1):
        for l in range(T + 1 - t):
            adv[:, t] += (GAMMA * lam) ** l * d[:, t + l]
    return DISC[None, :] * adv


def psi_gae_weighted(V, lam, coeff):
    """Combinaison des n-step TD avec un poids coeff(n), n = 1..T+1.

    coeff='standard' -> (1-lam) lam^{n-1}   (definition usuelle de GAE)
    coeff='wiki'     -> lam^{n-1} / (1-lam) (coefficient de la page Wikipedia)

    Au-dela de n = T+1 tous les n-step TD sont identiques (Monte-Carlo) : leur
    poids cumule est ajoute au dernier terme.
    """
    nmax = T + 1
    out = np.zeros((N_TRAJ, T + 1))
    for n in range(1, nmax + 1):
        if n < nmax:
            w = (1 - lam) * lam ** (n - 1) if coeff == 'standard' else lam ** (n - 1) / (1 - lam)
        else:  # queue geometrique
            tail = lam ** (n - 1) / (1 - lam)
            w = (1 - lam) * tail if coeff == 'standard' else tail / (1 - lam)
        out += w * psi_nstep(V, n)
    return out


# --------------------------------------------------------------------------- #
# geometrie : distribution d'etats, Fisher, KL moyennee
#
# Ajouts purs (aucune fonction ci-dessus n'est modifiee). Ces trois fonctions
# vivaient dans verify_fiche.py ; elles sont remontees ici pour etre partagees
# avec rapport/figures/make_figures.py.
# --------------------------------------------------------------------------- #
def state_distribution(theta_np):
    """d(s) : frequence de visite de s, moyennee sur t = 0..T, normalisee."""
    p = traj_probs(theta_np)
    d = np.zeros(S)
    for t in range(T + 1):
        for s in range(S):
            d[s] += p[STATES[:, t] == s].sum()
    return d / d.sum()


def fisher(theta_np):
    """F = E_{s~d, a~pi}[ score score^T ], matrice (D, D) inversible."""
    d = state_distribution(theta_np)
    pi = policy_np(theta_np)
    sc = score_table(theta_np)
    F = np.zeros((D, D))
    for s in range(S):
        for a in range(A):
            F += d[s] * pi[s, a] * np.outer(sc[s, a], sc[s, a])
    return F


def kl_bar(theta_new_flat, theta_old_np, d_state, order="new||old"):
    """KL moyennee sur les etats, differentiable en theta_new (torch)."""
    tn = theta_new_flat.reshape(S, A - 1)
    p_new = policy(tn)
    p_old = policy(torch.as_tensor(np.asarray(theta_old_np, dtype=float)))
    d = torch.as_tensor(np.asarray(d_state, dtype=float))
    if order == "new||old":
        kl = (p_new * (torch.log(p_new) - torch.log(p_old))).sum(dim=1)
    else:
        kl = (p_old * (torch.log(p_old) - torch.log(p_new))).sum(dim=1)
    return (d * kl).sum()


def kl_bar_np(theta_new_np, theta_old_np, d_state, order="new||old"):
    """Version numpy, non differentiable."""
    t = torch.as_tensor(np.asarray(theta_new_np, dtype=float).ravel())
    return float(kl_bar(t, theta_old_np, d_state, order).item())
