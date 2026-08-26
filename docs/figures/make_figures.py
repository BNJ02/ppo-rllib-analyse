#!/usr/bin/env python3
"""Genere les figures de rapport/fiche-policy-gradient-intuition.md.

Toutes les valeurs tracees sont calculees EXACTEMENT sur le PDM jouet de
rapport/verif/mdp.py (enumeration des 2187 trajectoires, aucune simulation
Monte-Carlo). Le script ecrit aussi valeurs.txt : tout nombre cite dans le
markdown doit s'y retrouver, pour que la prose et les figures ne divergent
jamais.

Usage :  python3 rapport/figures/make_figures.py
"""
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "verif"))
import mdp  # noqa: E402

torch.set_default_dtype(torch.float64)

S, A, T, D = mdp.S, mdp.A, mdp.T, mdp.D
GAMMA, DISC = mdp.GAMMA, mdp.DISC
ST, AC, REW = mdp.STATES, mdp.ACTIONS, mdp.REW
TIDX = np.arange(T + 1)[None, :]

THETA = mdp.THETA0
V, Q, ADV = mdp.value_functions(THETA)
DSTATE = mdp.state_distribution(THETA)
FISH = mdp.fisher(THETA)
G = mdp.grad_J(THETA)
PI = mdp.policy_np(THETA)
SCORE = mdp.score_table(THETA)
PTRAJ = mdp.traj_probs(THETA)
J0 = float(mdp.J_torch(torch.as_tensor(THETA)).item())

# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #
BG = "#ffffff"
FG = "#1b1f23"
GRID = "#d8dde3"
C_BAD = "#c0392b"
C_GOOD = "#1f6f3f"
C_BLUE = "#1f4e9c"
C_TEAL = "#0f766e"
C_ORANGE = "#d97706"
C_GREY = "#7c8794"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": "#aab2bd",
    "xtick.color": FG, "ytick.color": FG,
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "grid.color": GRID, "grid.linewidth": 0.7,
    "legend.frameon": False, "figure.dpi": 160,
})

VALS = []


def note(key, value, fmt="{:.4f}"):
    """Enregistre une valeur citable dans valeurs.txt et la retourne."""
    txt = value if isinstance(value, str) else fmt.format(value)
    VALS.append(f"{key:<54s} {txt}")
    return value


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), bbox_inches="tight")
    plt.close(fig)
    print("  ecrit", name)


def tidy(ax, grid="y"):
    ax.grid(axis=grid, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# --------------------------------------------------------------------------- #
# outils partages
# --------------------------------------------------------------------------- #
def psi_reinforce_shift(c):
    """gamma^t G_t avec une constante c ajoutee a toutes les recompenses."""
    dr = (REW + c) * DISC
    return np.cumsum(dr[:, ::-1], axis=1)[:, ::-1]


def value_shift(c):
    """V_t(s) apres decalage : V + c * sum_{k=0}^{T-t} gamma^k."""
    horizon = np.array([sum(GAMMA ** k for k in range(T + 1 - t)) for t in range(T + 1)])
    return V[:T + 1] + c * horizon[:, None]


def surrogate_np(theta_new):
    pn = mdp.policy_np(theta_new)
    ratio = pn[ST, AC] / PI[ST, AC]
    return float((PTRAJ * (DISC * ratio * ADV[TIDX, ST, AC]).sum(axis=1)).sum())


def clipped_np(theta_new, eps):
    pn = mdp.policy_np(theta_new)
    ratio = pn[ST, AC] / PI[ST, AC]
    at = ADV[TIDX, ST, AC]
    obj = np.minimum(ratio * at, np.clip(ratio, 1 - eps, 1 + eps) * at)
    return float((PTRAJ * (DISC * obj).sum(axis=1)).sum())


def kl_np(theta_new):
    """KL moyennee sur les etats, version numpy (verifiee contre mdp.kl_bar)."""
    z = np.concatenate([np.asarray(theta_new, dtype=float).reshape(S, A - 1),
                        np.zeros((S, 1))], axis=1)
    z = z - z.max(axis=1, keepdims=True)
    pn = np.exp(z)
    pn /= pn.sum(axis=1, keepdims=True)
    return float((DSTATE * (pn * (np.log(pn) - np.log(PI))).sum(axis=1)).sum())


def J_np(theta_new):
    return float(mdp.J_torch(torch.as_tensor(np.asarray(theta_new, dtype=float))).item())


_rng = np.random.default_rng(7)
_worst = 0.0
for _ in range(20):
    _d = _rng.normal(size=(S, A - 1)) * 0.5
    _worst = max(_worst, abs(kl_np(THETA + _d) - mdp.kl_bar_np(THETA + _d, THETA, DSTATE)))
assert _worst < 1e-12, f"KL numpy != KL torch : {_worst:.2e}"
note("controle KL numpy vs torch (ecart max)", _worst, "{:.2e}")


# --------------------------------------------------------------------------- #
# F2 - l'echelle de variance
# --------------------------------------------------------------------------- #
def fig_variance():
    b_const = DISC[:, None] * np.full((T + 1, S), V[:T + 1].mean())
    b_value = DISC[:, None] * V[:T + 1]

    etapes = [
        ("retour total\n(REINFORCE brut)", mdp.psi_total_return(), C_BAD),
        ("+ causalité", mdp.psi_reinforce(), C_ORANGE),
        ("+ baseline\nconstant", mdp.psi_baseline(b_const), C_BLUE),
        ("+ baseline\n$V_t(S_t)$", mdp.psi_baseline(b_value), C_TEAL),
        ("avantage exact $A^\\pi$\n(critique parfait)", mdp.psi_A(ADV), C_GOOD),
    ]
    noms, vars_, cols, biais = [], [], [], 0.0
    for nom, psi, col in etapes:
        m, v = mdp.estimator(THETA, psi)
        biais = max(biais, float(np.abs(m - G).max()))
        noms.append(nom); vars_.append(v); cols.append(col)

    note("F2 biais max de toutes les variantes", biais, "{:.2e}")
    for nom, v in zip(noms, vars_):
        note("F2 variance " + nom.replace("\n", " "), v)
    note("F2 facteur retour total -> avantage", vars_[0] / vars_[-1], "{:.1f}")
    note("F2 facteur de la causalite seule", vars_[0] / vars_[1], "{:.2f}")
    note("F2 facteur du baseline V seul", vars_[1] / vars_[3], "{:.2f}")

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    bars = ax.bar(range(len(noms)), vars_, color=cols, width=0.60, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(noms)))
    ax.set_xticklabels(noms, fontsize=9.4)
    ax.set_ylabel("variance exacte de l'estimateur")
    ax.set_title("Le même gradient, soixante fois moins bruité\n"
                 "(variances exactes — PDM 3 états, $\\gamma$ = 0,9)", fontsize=11.2)
    for b, v in zip(bars, vars_):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.18, f"{v:.3g}",
                ha="center", fontsize=9.6, color=FG)
    ax.set_ylim(min(vars_) * 0.55, max(vars_) * 4.0)
    ax.annotate("", xy=(4, vars_[-1] * 1.55), xytext=(0.35, vars_[0] * 1.1),
                arrowprops=dict(arrowstyle="-|>", color=C_GOOD, lw=1.8,
                                connectionstyle="arc3,rad=-0.18"))
    ax.text(2.0, vars_[0] * 1.55, f"variance divisée par {vars_[0]/vars_[-1]:.0f}",
            color=C_GOOD, fontsize=11, ha="center", weight="bold")
    tidy(ax)
    fig.text(0.5, -0.035,
             f"les cinq colonnes estiment le MÊME gradient — écart max {biais:.1e}",
             ha="center", fontsize=9.2, color=C_GREY)
    save(fig, "02-variance.png")


# --------------------------------------------------------------------------- #
# F3 - le baseline
# --------------------------------------------------------------------------- #
def fig_baseline():
    s_star = int(np.argmax(DSTATE))
    q, adv = Q[0, s_star], ADV[0, s_star]
    note("F3 etat le plus visite", f"s = {s_star} (d = {DSTATE[s_star]:.3f})")
    note("F3 Q(s*,.)", "  ".join(f"{x:+.3f}" for x in q))
    note("F3 A(s*,.)", "  ".join(f"{x:+.3f}" for x in adv))
    note("F3 V(s*)", V[0, s_star])

    offsets = np.linspace(0.0, 6.0, 25)
    var_sans, var_avec, ecart_grad = [], [], 0.0
    for c in offsets:
        psi = psi_reinforce_shift(c)
        m1, v1 = mdp.estimator(THETA, psi)
        b = DISC[:, None] * value_shift(c)
        m2, v2 = mdp.estimator(THETA, psi - b[TIDX[0], ST])
        var_sans.append(v1); var_avec.append(v2)
        ecart_grad = max(ecart_grad, float(np.abs(m1 - G).max()), float(np.abs(m2 - G).max()))
    note("F3 variance sans baseline, c=0", var_sans[0])
    note("F3 variance sans baseline, c=6", var_sans[-1])
    note("F3 variance avec baseline, c=6", var_avec[-1])
    note("F3 facteur a c=6", var_sans[-1] / var_avec[-1], "{:.0f}")
    note("F3 ecart max sur le gradient (toutes valeurs de c)", ecart_grad, "{:.2e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.30})
    x = np.arange(A)
    ax1.bar(x - 0.19, q, width=0.36, color=C_BAD, label="$Q(s,a)$ brut", zorder=3)
    ax1.bar(x + 0.19, adv, width=0.36, color=C_GOOD, label="$A(s,a) = Q - V$", zorder=3)
    ax1.axhline(0, color=FG, lw=0.9)
    ax1.axhline(V[0, s_star], color=C_GREY, ls="--", lw=1.1)
    ax1.text(2.45, V[0, s_star] + 0.09, f"$V(s)$ = {V[0, s_star]:.2f}",
             color=C_GREY, fontsize=9, ha="right")
    ax1.set_xticks(x); ax1.set_xticklabels([f"action {a}" for a in range(A)])
    ax1.set_title(f"Au départ, tout monte\n(état le plus visité, s = {s_star})")
    ax1.legend(fontsize=9.2, loc="lower left")
    tidy(ax1)

    ax2.plot(offsets, var_sans, color=C_BAD, lw=2.4, label="sans baseline")
    ax2.plot(offsets, var_avec, color=C_GOOD, lw=2.4, label="baseline $V_t(S_t)$")
    ax2.set_yscale("log")
    ax2.set_xlabel("constante $c$ ajoutée à TOUTES les récompenses")
    ax2.set_ylabel("variance de l'estimateur")
    ax2.set_title("Un décalage sans effet sur le problème\nfait exploser l'estimateur naïf")
    ax2.legend(fontsize=9.5, loc="center right")
    ax2.text(0.02, 0.11, f"le gradient, lui, ne bouge pas\n(écart max {ecart_grad:.0e})",
             transform=ax2.transAxes, fontsize=9, color=C_GREY)
    tidy(ax2)
    save(fig, "03-baseline.png")


# --------------------------------------------------------------------------- #
# F4 - causalite
# --------------------------------------------------------------------------- #
def fig_causalite():
    sc = SCORE[ST, AC]
    dr = REW * DISC
    esp = np.zeros((T + 1, T + 1))
    var = np.zeros((T + 1, T + 1))
    for t in range(T + 1):
        for tau in range(T + 1):
            u = dr[:, tau][:, None] * sc[:, t, :]
            m = PTRAJ @ u
            esp[t, tau] = np.linalg.norm(m)
            var[t, tau] = float(PTRAJ @ ((u - m) ** 2).sum(axis=1))

    bas = np.tril_indices(T + 1, -1)
    haut = np.triu_indices(T + 1)
    note("F4 |esperance| max sous la diagonale (tau < t)", esp[bas].max(), "{:.2e}")
    note("F4 |esperance| max sur et au-dessus de la diagonale", esp[haut].max())
    note("F4 variance jetee par la causalite (somme tau < t)", var[bas].sum())
    note("F4 variance conservee (somme tau >= t)", var[haut].sum())
    note("F4 part de variance jetee (%)",
         100 * var[bas].sum() / (var[bas].sum() + var[haut].sum()), "{:.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6),
                             gridspec_kw={"wspace": 0.42})
    for ax, M, titre, cmap in (
            (axes[0], esp, "Ce que le terme APPORTE\n"
                           "$\\Vert\\,\\mathbb{E}[\\gamma^\\tau R_\\tau \\cdot \\mathrm{score}_t]\\,\\Vert$",
             "Blues"),
            (axes[1], var, "Ce que le terme COÛTE\nvariance du même terme", "Reds")):
        im = ax.imshow(M, cmap=cmap, vmin=0)
        for t in range(T + 1):
            for tau in range(T + 1):
                v = M[t, tau]
                ax.text(tau, t, "~0" if v < 1e-12 else f"{v:.2f}",
                        ha="center", va="center", fontsize=9.5,
                        color="white" if v > M.max() * 0.6 else FG)
        ax.set_xlabel("pas de la récompense  $\\tau$")
        ax.set_ylabel("pas de l'action  $t$")
        ax.set_xticks(range(T + 1)); ax.set_yticks(range(T + 1))
        ax.set_title(titre, fontsize=10.5)
        for t in range(T + 1):
            ax.plot([t - 0.5, t - 0.5], [t - 0.5, T + 0.5], color=C_GOOD, lw=2.4)
            ax.plot([t - 0.5, T + 0.5], [t - 0.5, t - 0.5], color=C_GOOD, lw=2.4)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].text(0.5, -0.34, "sous la diagonale : espérance nulle\n(lemme du score nul)",
                 transform=axes[0].transAxes, ha="center", fontsize=9.4, color=C_BLUE)
    axes[1].text(0.5, -0.34, "sous la diagonale : variance bien réelle\n→ à jeter",
                 transform=axes[1].transAxes, ha="center", fontsize=9.4, color=C_BAD)
    fig.suptitle("Le passé n'apporte rien et coûte cher\n"
                 "(le cadre vert est ce que la causalité garde)", fontsize=11.8, y=1.06)
    save(fig, "04-causalite.png")


# --------------------------------------------------------------------------- #
# F5 - biais / variance
# --------------------------------------------------------------------------- #
def fig_biais_variance():
    rng = np.random.default_rng(4242)
    V_noisy = V.copy()
    V_noisy[:T + 1] += rng.normal(size=(T + 1, S)) * 0.6
    note("F5 bruit ajoute au critique (ecart-type)", 0.6, "{:.1f}")

    ns = list(range(1, T + 2))
    biais_n, var_n = [], []
    for n in ns:
        m, _ = mdp.estimator(THETA, mdp.psi_nstep(V_noisy, n))
        biais_n.append(float(np.linalg.norm(m - G)))
        _, v = mdp.estimator(THETA, mdp.psi_nstep(V, n))
        var_n.append(v)

    lams = np.linspace(0.0, 0.98, 15)
    biais_l, var_l = [], []
    for lam in lams:
        m, _ = mdp.estimator(THETA, mdp.psi_gae_recursive(V_noisy, lam))
        biais_l.append(float(np.linalg.norm(m - G)))
        _, v = mdp.estimator(THETA, mdp.psi_gae_recursive(V, lam))
        var_l.append(v)

    for n, b, v in zip(ns, biais_n, var_n):
        note(f"F5 n={n} biais (V approche)", b)
        note(f"F5 n={n} variance (V exact)", v)
    note("F5 biais a lambda=0 (V approche)", biais_l[0])
    note("F5 biais a lambda=0,98 (V approche)", biais_l[-1], "{:.2e}")
    note("F5 variance a lambda=0 (V exact)", var_l[0])
    note("F5 variance a lambda=0,98 (V exact)", var_l[-1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                   gridspec_kw={"wspace": 0.46})
    for ax, xs, bi, va, xlabel, titre in (
            (ax1, ns, biais_n, var_n, "$n$ — nombre de vraies récompenses",
             "TD à $n$ pas"),
            (ax2, lams, biais_l, var_l, "$\\lambda$", "GAE")):
        ax.plot(xs, bi, "o-", color=C_BAD, lw=2.3, ms=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("biais  (critique approché)", color=C_BAD)
        ax.tick_params(axis="y", labelcolor=C_BAD)
        axb = ax.twinx()
        axb.plot(xs, va, "s--", color=C_BLUE, lw=2.3, ms=4.5)
        axb.set_ylabel("variance  (critique exact)", color=C_BLUE)
        axb.tick_params(axis="y", labelcolor=C_BLUE)
        axb.spines["top"].set_visible(False)
        ax.set_title(titre)
        ax.grid(alpha=0.5); ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
    ax1.set_xticks(ns)
    ax1.plot([], [], "o-", color=C_BAD, label="biais — critique approché")
    ax1.plot([], [], "s--", color=C_BLUE, label="variance — critique exact")
    ax1.legend(fontsize=8.8, loc="center right")
    fig.suptitle("Le curseur : plus on fait confiance au critique, moins on est bruité —\n"
                 "mais plus on hérite de ses erreurs", fontsize=11.5, y=1.08)
    save(fig, "05-biais-variance.png")


# --------------------------------------------------------------------------- #
# F6 - hyperplans, boule, ellipse   [la figure demandee]
# --------------------------------------------------------------------------- #
def plan_2d():
    """Base orthonormee (e1, e2) du plan engendre par g et F^-1 g."""
    e1 = G / np.linalg.norm(G)
    w = np.linalg.solve(FISH, G)
    e2 = w - (w @ e1) * e1
    e2 /= np.linalg.norm(e2)
    return e1, e2, w


def fig_hyperplan():
    e1, e2, w = plan_2d()
    gn = np.linalg.norm(G)
    ang = np.degrees(np.arccos(G @ w / (gn * np.linalg.norm(w))))
    note("F6 angle entre le gradient et le gradient naturel (degres)", ang, "{:.1f}")
    note("F6 conditionnement de F", np.linalg.cond(FISH), "{:.1f}")

    M = np.array([[e1 @ FISH @ e1, e1 @ FISH @ e2],
                  [e2 @ FISH @ e1, e2 @ FISH @ e2]])
    EPS = 0.02
    alpha = np.sqrt(2 * EPS / (G @ w))
    dnat = alpha * w
    p_nat = np.array([dnat @ e1, dnat @ e2])
    r_ball = np.linalg.norm(dnat)
    p_eucl = np.array([r_ball, 0.0])

    kl_e = kl_np(THETA + (r_ball * e1).reshape(S, A - 1))
    kl_n = kl_np(THETA + dnat.reshape(S, A - 1))
    gain_e, gain_n = gn * p_eucl[0], gn * p_nat[0]
    note("F6 rayon de la boule euclidienne (= longueur du pas naturel)", r_ball)
    note("F6 gain d'objectif linearise, pas euclidien", gain_e)
    note("F6 gain d'objectif linearise, pas naturel", gain_n)
    note("F6 vraie KL atteinte par le pas euclidien", kl_e)
    note("F6 vraie KL atteinte par le pas naturel", kl_n)
    note("F6 rayon KL demande (epsilon)", EPS, "{:.2f}")
    note("F6 gain linearise par unite de KL, pas euclidien", gain_e / kl_e, "{:.1f}")
    note("F6 gain linearise par unite de KL, pas naturel", gain_n / kl_n, "{:.1f}")
    note("F6 rapport des vraies KL, euclidien / naturel", kl_e / kl_n, "{:.2f}")
    note("F6 rapport d'efficacite naturel / euclidien",
         (gain_n / kl_n) / (gain_e / kl_e), "{:.2f}")

    lim = r_ball * 1.55
    grid = np.linspace(-lim, lim, 241)
    XX, YY = np.meshgrid(grid, grid)
    QUAD = 0.5 * (M[0, 0] * XX ** 2 + 2 * M[0, 1] * XX * YY + M[1, 1] * YY ** 2)
    KL = np.zeros_like(XX)
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            d = XX[i, j] * e1 + YY[i, j] * e2
            KL[i, j] = kl_np(THETA + d.reshape(S, A - 1))

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.0),
                             gridspec_kw={"wspace": 0.32})

    def frame(ax, titre):
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axhline(0, color="#c8ced6", lw=0.8); ax.axvline(0, color="#c8ced6", lw=0.8)
        ax.set_xlabel("direction du gradient  $g$")
        ax.set_ylabel("direction orthogonale")
        ax.set_title(titre, fontsize=10.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # --- (a) boule euclidienne --------------------------------------------- #
    ax = axes[0]
    frame(ax, "(a)  Contrainte EUCLIDIENNE\nune boule dans l'espace des paramètres")
    for c in np.linspace(-lim, lim, 13):
        ax.axvline(c, color=C_GREY, ls=(0, (4, 4)), lw=0.9, alpha=0.55)
    ax.add_patch(Circle((0, 0), r_ball, facecolor="#dbe7f7",
                        edgecolor=C_BLUE, lw=2.0, alpha=0.9, zorder=2))
    ax.axvline(p_eucl[0], color=C_BLUE, lw=2.2, zorder=3)
    ax.annotate("", xy=p_eucl, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=2.6), zorder=4)
    ax.plot(*p_eucl, "o", color=C_BLUE, ms=7, zorder=5)
    ax.text(0.42, -0.30, "$\\Delta \\propto g$", color=C_BLUE, fontsize=13)
    ax.text(-lim * 0.96, lim * 0.92,
            "traits verticaux = les HYPERPLANS de niveau\n"
            "de l'objectif linéarisé  $g^{T}\\Delta$",
            fontsize=8.4, color=FG, va="top")
    ax.text(-lim * 0.96, -lim * 0.80,
            f"vraie KL atteinte : {kl_e:.3f}", fontsize=9.4, color=C_BLUE)

    # --- (b) ellipse de Fisher --------------------------------------------- #
    ax = axes[1]
    frame(ax, "(b)  Contrainte de FISHER\nune ellipse : la distance entre politiques")
    for c in np.linspace(-lim, lim, 13):
        ax.axvline(c, color=C_GREY, ls=(0, (4, 4)), lw=0.9, alpha=0.55)
    ax.contourf(XX, YY, QUAD, levels=[0, EPS], colors=["#dff0e4"], zorder=2)
    ax.contour(XX, YY, QUAD, levels=[EPS], colors=[C_GOOD], linewidths=2.1, zorder=3)
    ax.axvline(p_nat[0], color=C_GOOD, lw=2.2, zorder=3)
    ax.annotate("", xy=p_eucl, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=1.5, alpha=0.5), zorder=4)
    ax.annotate("", xy=p_nat, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_GOOD, lw=2.6), zorder=5)
    ax.plot(*p_nat, "o", color=C_GOOD, ms=7, zorder=6)
    ax.text(p_nat[0] - 1.30, p_nat[1] + 0.26, "$\\Delta \\propto F^{-1}g$",
            color=C_GOOD, fontsize=13)
    ax.text(-lim * 0.96, -lim * 0.72,
            f"{ang:.0f}° entre les deux pas\nvraie KL atteinte : {kl_n:.3f}",
            fontsize=9.4, color=C_GOOD)

    # --- (c) vraie KL vs approximation ------------------------------------- #
    ax = axes[2]
    wlim = lim * 2.6
    wg = np.linspace(-wlim, wlim, 181)
    WX, WY = np.meshgrid(wg, wg)
    WQUAD = 0.5 * (M[0, 0] * WX ** 2 + 2 * M[0, 1] * WX * WY + M[1, 1] * WY ** 2)
    WKL = np.zeros_like(WX)
    for i in range(WX.shape[0]):
        for j in range(WX.shape[1]):
            d = WX[i, j] * e1 + WY[i, j] * e2
            WKL[i, j] = kl_np(THETA + d.reshape(S, A - 1))
    frame(ax, "(c)  L'ellipse n'est qu'une approximation\nexacte près de 0, fausse loin")
    ax.set_xlim(-wlim, wlim); ax.set_ylim(-wlim, wlim)
    for lv in (EPS, 8 * EPS):
        ax.contour(WX, WY, WQUAD, levels=[lv], colors=[C_GOOD], linewidths=2.0)
        ax.contour(WX, WY, WKL, levels=[lv], colors=[C_ORANGE], linewidths=2.0,
                   linestyles="--")
    ax.plot([], [], color=C_GOOD, lw=2.0, label="$\\frac{1}{2}\\Delta^{T}F\\Delta$  (Fisher)")
    ax.plot([], [], color=C_ORANGE, lw=2.0, ls="--", label="vraie  $\\bar{D}_{KL}$")
    ax.legend(fontsize=9.2, loc="lower left", frameon=True, facecolor=BG,
              edgecolor="none", framealpha=0.92)
    ax.text(-wlim * 0.96, wlim * 0.92,
            f"niveaux {EPS:g} (intérieur)\net {8*EPS:g} (extérieur)",
            fontsize=8.8, color=C_GREY, va="top")

    fig.suptitle("Même objectif, deux contraintes : c'est la FORME de la contrainte "
                 "qui choisit la direction du pas", fontsize=12.2, y=1.04)
    save(fig, "06-hyperplan-boule-ellipse.png")


# --------------------------------------------------------------------------- #
# F7 - dependance aux coordonnees, mesuree
# --------------------------------------------------------------------------- #
def fig_reparametrisation():
    Mdiag = np.array([1.0, 3.0, 0.35, 1.0, 4.0, 0.25])
    note("F7 rehelonnement M (diagonale)", "  ".join(f"{x:g}" for x in Mdiag))

    def montee(naturel, reparam, alpha, eps, n_iter=12):
        th = THETA.ravel().copy()
        if reparam:
            th = th / Mdiag
        hist = []
        for _ in range(n_iter):
            th_std = (th * Mdiag) if reparam else th
            hist.append(J_np(th_std.reshape(S, A - 1)))
            g = mdp.grad_J(th_std.reshape(S, A - 1))
            F = mdp.fisher(th_std.reshape(S, A - 1))
            if reparam:                       # chaine : g' = M g,  F' = M F M
                g = Mdiag * g
                F = (Mdiag[:, None] * F) * Mdiag[None, :]
            if naturel:
                d = np.linalg.solve(F, g)
                step = np.sqrt(2 * eps / (g @ d)) * d
            else:
                step = alpha * g
            th = th + step
        hist.append(J_np(((th * Mdiag) if reparam else th).reshape(S, A - 1)))
        return np.array(hist)

    ALPHA, EPS = 0.5, 0.01
    ord_a, ord_b = montee(False, False, ALPHA, EPS), montee(False, True, ALPHA, EPS)
    nat_a, nat_b = montee(True, False, ALPHA, EPS), montee(True, True, ALPHA, EPS)

    ecart_ord = float(np.abs(ord_a - ord_b).max())
    ecart_nat = float(np.abs(nat_a - nat_b).max())
    note("F7 pas de gradient ordinaire alpha", ALPHA, "{:.2f}")
    note("F7 rayon KL du gradient naturel", EPS, "{:.3f}")
    note("F7 ecart max entre les 2 courbes ORDINAIRES", ecart_ord)
    note("F7 ecart max entre les 2 courbes NATURELLES", ecart_nat, "{:.2e}")
    note("F7 J final, ordinaire, coordonnees d'origine", ord_a[-1])
    note("F7 J final, ordinaire, coordonnees reechelonnees", ord_b[-1])
    note("F7 J final, naturel (les deux)", nat_a[-1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True,
                                   gridspec_kw={"wspace": 0.12})
    it = np.arange(len(ord_a))
    ax1.plot(it, ord_a, color=C_BAD, lw=2.4, label="coordonnées d'origine")
    ax1.plot(it, ord_b, color=C_ORANGE, lw=2.4, ls="--", label="coordonnées rééchelonnées")
    ax1.set_title("Gradient ORDINAIRE\nle paramétrage change l'apprentissage")
    ax1.set_xlabel("itération"); ax1.set_ylabel("$J(\\theta)$")
    ax1.legend(fontsize=9.2, loc="lower right")
    ax1.text(0.04, 0.93, f"écart max : {ecart_ord:.3f}", transform=ax1.transAxes,
             fontsize=10, color=C_BAD, weight="bold")
    tidy(ax1, grid="both")

    ax2.plot(it, nat_a, color=C_GOOD, lw=4.5, alpha=0.40, label="coordonnées d'origine")
    ax2.plot(it, nat_b, color=C_GOOD, lw=1.6, ls="--", label="coordonnées rééchelonnées")
    ax2.set_title("Gradient NATUREL\nles deux courbes sont confondues")
    ax2.set_xlabel("itération")
    ax2.legend(fontsize=9.2, loc="lower right")
    ax2.text(0.04, 0.93, f"écart max : {ecart_nat:.0e}", transform=ax2.transAxes,
             fontsize=10, color=C_GOOD, weight="bold")
    tidy(ax2, grid="both")

    fig.suptitle("Rééchelonner les coordonnées ne change PAS le problème.\n"
                 "Le gradient ordinaire s'en aperçoit ; le gradient naturel non.",
                 fontsize=11.5, y=1.10)
    save(fig, "07-reparametrisation.png")


# --------------------------------------------------------------------------- #
# F8 - pourquoi TRPO recule
# --------------------------------------------------------------------------- #
def fig_trpo():
    w = np.linalg.solve(FISH, G)
    dirn = w / np.linalg.norm(w)
    ts = np.linspace(0, 9.0, 60)
    kl_vraie = np.array([kl_np((THETA.ravel() + t * dirn).reshape(S, A - 1)) for t in ts])
    kl_quad = np.array([0.5 * (t * dirn) @ FISH @ (t * dirn) for t in ts])
    L_vals = np.array([surrogate_np((THETA.ravel() + t * dirn).reshape(S, A - 1)) for t in ts])
    dJ_vals = np.array([J_np((THETA.ravel() + t * dirn).reshape(S, A - 1)) - J0 for t in ts])

    note("F8 KL vraie / KL quadratique, pas court", kl_vraie[3] / kl_quad[3], "{:.3f}")
    note("F8 KL vraie / KL quadratique, pas long", kl_vraie[-1] / kl_quad[-1], "{:.3f}")
    note("F8 surestimation du substitut, pas court (%)",
         100 * (L_vals[3] - dJ_vals[3]) / abs(dJ_vals[3]), "{:.2f}")
    note("F8 surestimation du substitut, pas long (%)",
         100 * (L_vals[-1] - dJ_vals[-1]) / abs(dJ_vals[-1]), "{:.2f}")

    EPS = 0.02
    d_faux = np.ones(S) / S
    F_faux = np.zeros((D, D))
    for s in range(S):
        for a in range(A):
            F_faux += d_faux[s] * PI[s, a] * np.outer(SCORE[s, a], SCORE[s, a])
    w_faux = np.linalg.solve(F_faux, G)
    delta_faux = np.sqrt(2 * EPS / (G @ w_faux)) * w_faux
    coefs, kls, Ls, accepte = [], [], [], None
    for j in range(8):
        c = 0.8 ** j
        thn = (THETA.ravel() + c * delta_faux).reshape(S, A - 1)
        k, l = kl_np(thn), surrogate_np(thn)
        coefs.append(c); kls.append(k); Ls.append(l)
        if accepte is None and k <= EPS and l > 0:
            accepte = j
    note("F8 epsilon de la region de confiance", EPS, "{:.2f}")
    note("F8 KL atteinte par le pas non recule (F mal estimee)", kls[0])
    note("F8 depassement (facteur)", kls[0] / EPS, "{:.2f}")
    note("F8 nombre de rebroussements avant acceptation", accepte, "{:d}")
    note("F8 KL apres rebroussement", kls[accepte])
    note("F8 KL du pas calcule avec la VRAIE F",
         kl_np((THETA.ravel() + np.sqrt(2 * EPS / (G @ w)) * w).reshape(S, A - 1)))

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3),
                             gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    ax.plot(ts, kl_quad, color=C_GOOD, lw=2.4,
            label="modèle quadratique  $\\frac{1}{2}\\Delta^{T}F\\Delta$")
    ax.plot(ts, kl_vraie, color=C_ORANGE, lw=2.4, ls="--", label="vraie  $\\bar{D}_{KL}$")
    ax.fill_between(ts, kl_vraie, kl_quad, color=C_ORANGE, alpha=0.13)
    ax.set_xlabel("longueur du pas"); ax.set_ylabel("divergence")
    ax.set_title("(a)  Le modèle quadratique dérive")
    ax.legend(fontsize=8.8, loc="upper left")
    tidy(ax, grid="both")

    ax = axes[1]
    ax.plot(ts, L_vals, color=C_BLUE, lw=2.4,
            label="substitut  $L(\\theta,\\theta_i)$  — ce qu'on optimise")
    ax.plot(ts, dJ_vals, color=C_BAD, lw=2.4, ls="--",
            label="vraie amélioration  $J - J_i$")
    ax.fill_between(ts, dJ_vals, L_vals, color=C_BAD, alpha=0.13)
    ax.set_xlabel("longueur du pas"); ax.set_ylabel("amélioration")
    ax.set_title("(b)  Le substitut promet plus qu'il ne tient")
    ax.legend(fontsize=8.6, loc="lower right")
    tidy(ax, grid="both")

    ax = axes[2]
    xs = np.arange(len(coefs))
    ax.bar(xs, kls, color=[C_BAD if k > EPS else C_GOOD for k in kls], width=0.6, zorder=3)
    ax.axhline(EPS, color=FG, lw=1.6, ls="--")
    ax.text(len(coefs) - 0.4, EPS * 1.06, f"$\\epsilon$ = {EPS:g}", ha="right", fontsize=9.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$0{{,}}8^{{{j}}}$" for j in range(len(coefs))], fontsize=8.6)
    ax.set_xlabel("coefficient de rebroussement")
    ax.set_ylabel("vraie  $\\bar{D}_{KL}$  atteinte")
    ax.set_title("(c)  Avec un $F$ mal estimé,\nle pas viole vraiment la contrainte")
    ax.annotate("accepté après\n1 recul", xy=(accepte + 0.1, kls[accepte]),
                xytext=(accepte + 1.4, kls[0] * 0.80),
                arrowprops=dict(arrowstyle="->", color=C_GOOD, lw=1.4),
                color=C_GOOD, fontsize=9.5)
    tidy(ax)

    fig.suptitle("TRPO : un pas calculé sur un modèle, accepté seulement après vérification",
                 fontsize=11.8, y=1.05)
    save(fig, "08-trpo-recul.png")


# --------------------------------------------------------------------------- #
# F9 - la fonction clippee
# --------------------------------------------------------------------------- #
def fig_clip():
    eps = 0.2
    r = np.linspace(0.4, 1.8, 1401)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6),
                             gridspec_kw={"wspace": 0.26})
    for ax, adv, titre, col in (
            (axes[0], 1.0, "$A > 0$  —  bonne action : on POUSSE", C_BLUE),
            (axes[1], -1.0, "$A < 0$  —  mauvaise action : on RETIRE", C_BAD)):
        obj = np.minimum(r * adv, np.clip(r, 1 - eps, 1 + eps) * adv)
        ax.set_ylim(-2.15, 2.15)
        ax.plot(r, r * adv, color=C_GREY, lw=1.7, ls=":",
                label="substitut non clippé  $rA$")
        ax.plot(r, obj, color=col, lw=3.0, label="objectif clippé")

        borne = 1 + eps if adv > 0 else 1 - eps
        plat = r >= borne if adv > 0 else r <= borne
        ax.fill_between(r[plat], -2.15, 2.15, color="#e6e9ee", zorder=0)
        ax.axvline(1.0, color=C_GREY, lw=1.0)
        ax.axvline(borne, color=C_GOOD, lw=1.8, ls="--")
        ax.text(borne + 0.02, -2.02, f"$1{'+' if adv > 0 else '-'}\\epsilon$",
                color=C_GOOD, fontsize=12, ha="left")
        ax.text(1.0 + (-0.03 if adv > 0 else 0.03), -2.02, "$r=1$", color=C_GREY,
                fontsize=10, ha="right" if adv > 0 else "left")

        x_mort = borne + 0.30 * np.sign(adv)
        x_vif = 1.0 - 0.42 * np.sign(adv)
        y_mort = 1.55 if adv > 0 else 0.55
        y_vif = -1.55 if adv > 0 else -0.25
        y = lambda x: obj[np.argmin(np.abs(r - x))]
        ax.annotate("pente MORTE\nle pas s'arrête",
                    xy=(x_mort, y(x_mort)), xytext=(x_mort, y_mort),
                    ha="center", fontsize=9.4, color="#4b5563",
                    arrowprops=dict(arrowstyle="->", color="#4b5563", lw=1.3))
        ax.annotate("pente VIVE\non peut revenir",
                    xy=(x_vif, y(x_vif)), xytext=(x_vif, y_vif),
                    ha="center", fontsize=9.4, color=C_GOOD,
                    arrowprops=dict(arrowstyle="->", color=C_GOOD, lw=1.3))

        ax.set_xlabel("ratio  $r = \\pi_\\theta(a \\mid s)\\,/\\,\\pi_{\\theta_t}(a \\mid s)$")
        ax.set_ylabel("objectif")
        ax.set_title(titre, fontsize=10.8)
        ax.legend(fontsize=9, loc="upper left" if adv > 0 else "upper right")
        tidy(ax, grid="both")

    note("F9 largeur de clipping epsilon", eps, "{:.1f}")
    note("F9 borne active si A>0", 1 + eps, "{:.1f}")
    note("F9 borne active si A<0", 1 - eps, "{:.1f}")
    fig.suptitle("Le clipping est UNILATÉRAL : une seule borne agit par cas,\n"
                 "l'autre côté reste libre pour rattraper un pas raté",
                 fontsize=11.5, y=1.07)
    save(fig, "09-clip.png")


# --------------------------------------------------------------------------- #
# F10 - TRPO vs PPO, les deux garde-fous
# --------------------------------------------------------------------------- #
def fig_trpo_vs_ppo():
    e1, e2, w = plan_2d()
    EPS_KL, EPS_CLIP = 0.02, 0.2
    lim, n = 3.4, 91
    grid = np.linspace(-lim, lim, n)
    XX, YY = np.meshgrid(grid, grid)
    Lsur = np.zeros_like(XX); Lclip = np.zeros_like(XX); KL = np.zeros_like(XX)
    for i in range(n):
        for j in range(n):
            d = (XX[i, j] * e1 + YY[i, j] * e2).reshape(S, A - 1)
            Lsur[i, j] = surrogate_np(THETA + d)
            Lclip[i, j] = clipped_np(THETA + d, EPS_CLIP)
            KL[i, j] = kl_np(THETA + d)

    note("F10 substitut non clippe, max sur la fenetre", Lsur.max())
    note("F10 objectif clippe, max sur la fenetre", Lclip.max())
    note("F10 objectif clippe, valeur au bord de la fenetre", Lclip[n // 2, -1])
    note("F10 epsilon KL (TRPO)", EPS_KL, "{:.2f}")
    note("F10 epsilon de clipping (PPO)", EPS_CLIP, "{:.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0),
                             gridspec_kw={"wspace": 0.34})
    for ax, Z, titre, sous in (
            (axes[0], Lsur, "TRPO : une barrière À CÔTÉ de l'objectif",
             "l'objectif monte sans fin ;\nc'est la contrainte qui retient"),
            (axes[1], Lclip, "PPO : le terrain APLATI au-delà de la borne",
             "plus rien ne tire vers l'extérieur ;\naucune contrainte n'est nécessaire")):
        cf = ax.contourf(XX, YY, Z, levels=18, cmap="viridis")
        ax.contour(XX, YY, Z, levels=18, colors="white", linewidths=0.35, alpha=0.5)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03, label="objectif")
        ax.set_aspect("equal")
        ax.set_xlabel("direction du gradient  $g$")
        ax.set_ylabel("direction orthogonale")
        ax.set_title(titre, fontsize=10.8)
        ax.text(0.5, -0.26, sous, transform=ax.transAxes, ha="center",
                fontsize=9.2, color=C_GREY)
        ax.plot(0, 0, "o", color="white", ms=6, mec=FG)
    axes[0].contour(XX, YY, KL, levels=[EPS_KL], colors=[C_BAD], linewidths=2.4)
    axes[0].text(0.95, 1.55, f"$\\bar{{D}}_{{KL}} \\leq {EPS_KL:g}$", color=C_BAD,
                 fontsize=11, weight="bold")
    fig.suptitle("Deux façons d'empêcher un pas trop grand", fontsize=12.2, y=1.02)
    save(fig, "10-trpo-vs-ppo.png")


# --------------------------------------------------------------------------- #
# F11 - la boucle interne
# --------------------------------------------------------------------------- #
def fig_boucle_interne():
    EPS, LR, N_EPOCH = 0.2, 0.02, 60
    at_np = ADV[TIDX, ST, AC]
    old_np = PI[ST, AC]

    def objectif(th, clip):
        pn = mdp.policy(th)
        ratio = pn[ST, AC] / torch.as_tensor(old_np)
        a_t = torch.as_tensor(at_np)
        obj = (torch.minimum(ratio * a_t, torch.clamp(ratio, 1 - EPS, 1 + EPS) * a_t)
               if clip else ratio * a_t)
        return (torch.as_tensor(PTRAJ) * (torch.as_tensor(DISC) * obj).sum(dim=1)).sum()

    hist = {"frac": [], "ratio": [], "gclip": [], "gsur": []}
    th = torch.tensor(np.asarray(THETA, dtype=float), requires_grad=True)
    masse = np.repeat(PTRAJ[:, None], T + 1, axis=1) / (T + 1)
    for _ in range(N_EPOCH):
        pn = mdp.policy_np(th.detach().numpy())
        ratio = pn[ST, AC] / old_np
        hist["frac"].append(float((masse * ((ratio > 1 + EPS) | (ratio < 1 - EPS))).sum()))
        hist["ratio"].append(float((masse * np.abs(ratio - 1)).sum()))

        if th.grad is not None:
            th.grad = None
        gc, = torch.autograd.grad(objectif(th, True), th)
        hist["gclip"].append(float(gc.norm().item()))

        th2 = torch.tensor(th.detach().numpy(), requires_grad=True)
        gs, = torch.autograd.grad(objectif(th2, False), th2)
        hist["gsur"].append(float(gs.norm().item()))

        with torch.no_grad():
            th += LR * gc

    note("F11 pas d'apprentissage de la boucle interne", LR, "{:.2f}")
    note("F11 nombre de pas dans la boucle interne", N_EPOCH, "{:d}")
    note("F11 fraction clippee, pas 0 (%)", 100 * hist["frac"][0], "{:.1f}")
    note("F11 fraction clippee, pas final (%)", 100 * hist["frac"][-1], "{:.1f}")
    note("F11 norme du gradient clippe, pas 0", hist["gclip"][0])
    note("F11 norme du gradient clippe, pas final", hist["gclip"][-1])
    note("F11 norme du gradient NON clippe, pas final", hist["gsur"][-1])
    note("F11 extinction du gradient clippe (facteur)",
         hist["gclip"][0] / hist["gclip"][-1], "{:.1f}")

    ep = np.arange(N_EPOCH)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3),
                                   gridspec_kw={"wspace": 0.46})

    ax1.plot(ep, np.array(hist["frac"]) * 100, color=C_BAD, lw=2.5)
    ax1.set_xlabel("pas dans la boucle interne (même lot)")
    ax1.set_ylabel("% d'échantillons clippés", color=C_BAD)
    ax1.tick_params(axis="y", labelcolor=C_BAD)
    axb = ax1.twinx()
    axb.plot(ep, hist["ratio"], color=C_BLUE, lw=2.5, ls="--")
    axb.set_ylabel("$\\mathbb{E}\\,|r - 1|$   éloignement", color=C_BLUE)
    axb.tick_params(axis="y", labelcolor=C_BLUE)
    axb.spines["top"].set_visible(False)
    ax1.set_title("Le lot s'éloigne de la politique\nqui l'a produit")
    ax1.grid(alpha=0.5); ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)

    ax2.plot(ep, hist["gsur"], color=C_GREY, lw=2.3, ls=":",
             label="substitut NON clippé")
    ax2.plot(ep, hist["gclip"], color=C_GOOD, lw=2.7, label="objectif clippé")
    ax2.set_xlabel("pas dans la boucle interne (même lot)")
    ax2.set_ylabel("norme du gradient")
    ax2.set_ylim(0, max(hist["gsur"]) * 1.35)
    ax2.set_title("Le lot cesse de lui-même de tirer\n(le substitut nu, lui, ne s'arrête jamais)")
    ax2.legend(fontsize=9.2, loc="lower left")
    tidy(ax2, grid="both")

    fig.suptitle("Pourquoi « proximal » : le même lot est réutilisé, "
                 "jusqu'à ce qu'il n'ait plus rien à dire", fontsize=11.5, y=1.06)
    save(fig, "11-boucle-interne.png")


# --------------------------------------------------------------------------- #
def main():
    print(f"PDM jouet : {S} etats, {A} actions, T={T}, gamma={GAMMA},",
          f"{mdp.N_TRAJ} trajectoires enumerees")
    note("PDM : nombre de trajectoires enumerees", mdp.N_TRAJ, "{:d}")
    note("PDM : J(theta_0)", J0)
    for f in (fig_variance, fig_baseline, fig_causalite, fig_biais_variance,
              fig_hyperplan, fig_reparametrisation, fig_trpo, fig_clip,
              fig_trpo_vs_ppo, fig_boucle_interne):
        f()
    with open(os.path.join(HERE, "valeurs.txt"), "w") as fh:
        fh.write("# Toutes les valeurs citees dans fiche-policy-gradient-intuition.md\n")
        fh.write("# Regenere par : python3 rapport/figures/make_figures.py\n\n")
        fh.write("\n".join(VALS) + "\n")
    print(f"  ecrit valeurs.txt  ({len(VALS)} valeurs)")


if __name__ == "__main__":
    main()
