#!/usr/bin/env python3
"""Test de mutation : on injecte des fautes et on verifie que le banc les DETECTE.

Un test qui passe toujours ne prouve rien. Chaque mutation ci-dessous est une
erreur qu'on pourrait realistement ecrire dans la fiche.

Deux mutations sont classees DOIT PASSER : elles semblent fausses mais sont
mathematiquement correctes, et le banc doit les accepter. Elles documentent la
portee exacte du theoreme du baseline.

    python3 mutation_test.py
"""
import sys

import numpy as np

import mdp
from mdp import A, ACTIONS, DISC, GAMMA, S, STATES, T, REW

theta = mdp.THETA0
G = mdp.grad_J(theta)
V, Q, Adv = mdp.value_functions(theta)
IDX = np.arange(T + 1)[None, :]
FAILS = []
rng = np.random.default_rng(11)


def ecart(psi):
    return float(np.linalg.norm(mdp.estimator(theta, psi)[0] - G) / np.linalg.norm(G))


def must_detect(name, psi, seuil=1e-8):
    e = ecart(psi)
    ok = e > seuil
    print(f"  [{'DETECTE    ' if ok else 'NON DETECTE'}] {name}   ecart={e:.2e}")
    if not ok:
        FAILS.append(name)


def must_pass(name, psi, seuil=1e-9):
    e = ecart(psi)
    ok = e < seuil
    print(f"  [{'CORRECT    ' if ok else 'FAUX POSITIF'}] {name}   ecart={e:.2e}")
    if not ok:
        FAILS.append(name)


# --------------------------------------------------------------------------- #
print("A. Poids Psi_t -- fautes qui DOIVENT etre detectees :")

must_detect("gamma au lieu de gamma^t devant l'avantage",
            GAMMA * Adv[IDX, STATES, ACTIONS])
must_detect("aucun gamma^t devant l'avantage",
            Adv[IDX, STATES, ACTIONS])
must_detect("causalite inversee (recompenses passees au lieu de futures)",
            np.cumsum(REW * DISC, axis=1))
must_detect("baseline dependant de l'ACTION",
            mdp.psi_baseline_action(rng.normal(size=(T + 1, S, A)) * 3))
must_detect("Q sans le facteur gamma^t",
            Q[IDX, STATES, ACTIONS])

# bootstrap au mauvais instant : V_t au lieu de V_{t+n} dans le TD a n pas
psi = np.zeros((mdp.N_TRAJ, T + 1))
Vf = np.zeros((T + 2, S))
Vf[:T + 1] = V[:T + 1]
n = 2
for t in range(T + 1):
    acc = np.zeros(mdp.N_TRAJ)
    for k in range(n):
        if t + k <= T:
            acc += GAMMA ** k * REW[:, t + k]
    if t + n <= T:
        acc += GAMMA ** n * Vf[t][STATES[:, t + n]]      # FAUTE : Vf[t] au lieu de Vf[t+n]
    acc -= Vf[t][STATES[:, t]]
    psi[:, t] = DISC[t] * acc
must_detect("TD 2 pas : bootstrap avec V_t au lieu de V_{t+n}", psi)

LAM = 0.7
must_detect("GAE : ponderation lam^{n-1}/(1-lam) de la page Wikipedia",
            mdp.psi_gae_weighted(V, LAM, 'wiki'))

# --------------------------------------------------------------------------- #
print("\nB. Variantes qui SEMBLENT fausses mais sont correctes (portee du theoreme du baseline) :")

b_star = DISC[:, None] * V[:T + 1]
must_pass("baseline ADDITIONNE au lieu d'etre soustrait : toujours non biaise",
          mdp.psi_reinforce() + b_star[IDX[0], STATES])
must_pass("V_0(s) au lieu de V_t(s) comme baseline : toujours non biaise",
          DISC[None, :] * (Q[IDX, STATES, ACTIONS] - V[0][STATES]))
must_pass("residu TD avec + V(S_t) au lieu de - V(S_t) : toujours non biaise",
          DISC[None, :] * (mdp.td_residuals(V) + 2 * V[IDX, STATES]))

_d = mdp.td_residuals(V)
_adv = np.zeros_like(_d)
_nxt = np.zeros(mdp.N_TRAJ)
for _t in range(T, -1, -1):
    _nxt = _d[:, _t] + LAM * _nxt                    # gamma manquant dans la recursion
    _adv[:, _t] = _nxt
must_pass("GAE : recursion en lambda seul au lieu de gamma*lambda : toujours non biaise",
          DISC[None, :] * _adv)
print()
print("     Ces quatre variantes restent NON BIAISEES, et c'est un resultat, pas une")
print("     faiblesse du banc :")
print("       - tout b(t,s) est un baseline admissible, quel que soit son signe et")
print("         quelle que soit sa qualite (theoreme du baseline, lemme 2.2) ;")
print("       - E[delta_{t+l} | S_{t+l}] = 0 pour l >= 1, donc N'IMPORTE QUEL")
print("         coefficient sur les residus TD futurs laisse l'esperance inchangee.")
print("       - seul le role de BOOTSTRAP exige la bonne valeur (mutation 'V_t au")
print("         lieu de V_{t+n}' en section A), et seulement parce qu'il change le")
print("         terme l = 0.")
print("     Corollaire : avec un V EXACT, GAE est non biaise pour tout lambda. Le")
print("     compromis biais-variance de la fiche porte sur un V APPROCHE -- c'est")
print("     l'objet du test #33 de verify_fiche.py.")

# --------------------------------------------------------------------------- #
print("\nC. Objectif PPO -- l'equivalence des deux ecritures :")


def ppo_cases(r, adv, eps):
    return np.where(adv > 0, np.minimum(r, 1 + eps) * adv, np.maximum(r, 1 - eps) * adv)


RR, AA = np.meshgrid(np.linspace(0.0, 3.0, 2001), np.linspace(-4, 4, 201))


def check_ppo(name, fn, eps=0.2, expect_detect=True):
    dd = float(np.abs(ppo_cases(RR, AA, eps) - fn(RR, AA, eps)).max())
    ok = (dd > 1e-6) if expect_detect else (dd < 1e-12)
    label = ('DETECTE    ' if ok else 'NON DETECTE') if expect_detect else \
            ('CORRECT    ' if ok else 'FAUX POSITIF')
    print(f"  [{label}] {name}   ecart max={dd:.2e}")
    if not ok:
        FAILS.append(name)


check_ppo("max au lieu de min dans la forme du papier",
          lambda r, a, e: np.maximum(r * a, np.clip(r, 1 - e, 1 + e) * a))
check_ppo("cas A<0 traite avec min au lieu de max",
          lambda r, a, e: np.where(a > 0, np.minimum(r, 1 + e) * a, np.minimum(r, 1 - e) * a))
check_ppo("bornes echangees : 1-eps pour A>0 et 1+eps pour A<0",
          lambda r, a, e: np.where(a > 0, np.minimum(r, 1 - e) * a, np.maximum(r, 1 + e) * a))
check_ppo("clip sans borne basse",
          lambda r, a, e: np.minimum(r * a, np.minimum(r, 1 + e) * a))
check_ppo("forme correcte du papier (controle positif)",
          lambda r, a, e: np.minimum(r * a, np.clip(r, 1 - e, 1 + e) * a),
          expect_detect=False)

# --------------------------------------------------------------------------- #
print("\nD. Matrice de Fisher :")
pi = mdp.policy_np(theta)
score = mdp.score_table(theta)
ptraj = mdp.traj_probs(theta)
d_state = np.zeros(S)
for t in range(T + 1):
    for s in range(S):
        d_state[s] += ptraj[STATES[:, t] == s].sum()
d_state /= d_state.sum()
F_ok = sum(d_state[s] * pi[s, a] * np.outer(score[s, a], score[s, a])
           for s in range(S) for a in range(A))
for nm, Fm in (
    ("distribution d'etats uniforme au lieu de d^pi",
     sum((1 / S) * pi[s, a] * np.outer(score[s, a], score[s, a])
         for s in range(S) for a in range(A))),
    ("oubli de la ponderation par pi(a|s)",
     sum(d_state[s] * np.outer(score[s, a], score[s, a])
         for s in range(S) for a in range(A))),
    ("score au carre terme a terme au lieu du produit exterieur",
     np.diag(np.diag(F_ok)) * 1.0 + np.eye(mdp.D) * 0.0),
):
    e = float(np.linalg.norm(Fm - F_ok) / np.linalg.norm(F_ok))
    ok = e > 1e-8
    print(f"  [{'DETECTE    ' if ok else 'NON DETECTE'}] {nm}   ecart={e:.2e}")
    if not ok:
        FAILS.append(nm)

print(f"\n{'=' * 70}")
print(f"{len(FAILS)} probleme(s)")
sys.exit(1 if FAILS else 0)
