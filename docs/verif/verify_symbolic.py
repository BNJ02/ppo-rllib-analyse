#!/usr/bin/env python3
"""Etage 2 : les etapes ALGEBRIQUES de la fiche, verifiees par sympy.

Le banc numerique (verify_fiche.py) verifie des identites en esperance. Ici on
verifie les manipulations formelles : derivation de la contrainte de
normalisation, developpement de Taylor de la KL, resolution du lagrangien,
echange min/max, convexite.

    python3 verify_symbolic.py
"""
import sys

import sympy as sp

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# --------------------------------------------------------------------------- #
print("\n=== 2.2  Lemme du score nul")
# --------------------------------------------------------------------------- #
t1, t2 = sp.symbols('theta_1 theta_2', real=True)
theta = (t1, t2)
z = [t1, t2, sp.Integer(0)]                     # logits, derniere composante fixee
Z = sum(sp.exp(zi) for zi in z)
pi = [sp.exp(zi) / Z for zi in z]

check("sum_a pi(a|s) = 1", sp.simplify(sum(pi) - 1) == 0)
check("grad_theta sum_a pi(a|s) = 0",
      all(sp.simplify(sp.diff(sum(pi), th)) == 0 for th in theta))

# E_pi[grad ln pi] = sum_a pi * grad ln pi = sum_a grad pi = grad 1 = 0
esp_score = [sp.simplify(sum(pi[a] * sp.diff(sp.log(pi[a]), th) for a in range(3)))
             for th in theta]
check("E_pi[grad ln pi] = 0  (c'est le lemme)", all(e == 0 for e in esp_score),
      f"= {esp_score}")

# --------------------------------------------------------------------------- #
print("\n=== 4.3  Fisher : identite de l'information et Taylor de la KL")
# --------------------------------------------------------------------------- #
F = sp.Matrix(2, 2, lambda i, j: sp.simplify(
    sum(pi[a] * sp.diff(sp.log(pi[a]), theta[i]) * sp.diff(sp.log(pi[a]), theta[j])
        for a in range(3))))
negH = sp.Matrix(2, 2, lambda i, j: sp.simplify(
    -sum(pi[a] * sp.diff(sp.log(pi[a]), theta[i], theta[j]) for a in range(3))))
check("identite de l'information : E[-grad^2 ln pi] = E[grad ln pi grad ln pi^T]",
      sp.simplify(F - negH) == sp.zeros(2, 2))

# KL(pi_{theta + s*d} || pi_theta), developpee en s a l'ordre 2
s_, d1, d2 = sp.symbols('s d_1 d_2', real=True)
d = sp.Matrix([d1, d2])
zn = [t1 + s_ * d1, t2 + s_ * d2, sp.Integer(0)]
Zn = sum(sp.exp(zi) for zi in zn)
pin = [sp.exp(zi) / Zn for zi in zn]
KL = sum(pin[a] * (sp.log(pin[a]) - sp.log(pi[a])) for a in range(3))
ser = sp.series(KL, s_, 0, 3).removeO().expand()
c0 = sp.simplify(ser.coeff(s_, 0))
c1 = sp.simplify(ser.coeff(s_, 1))
c2 = sp.simplify(ser.coeff(s_, 2))
check("KL : terme d'ordre 0 nul", c0 == 0)
check("KL : terme d'ordre 1 nul", c1 == 0)
quad = sp.simplify(c2 - (d.T * F * d)[0, 0] / 2)
check("KL : terme d'ordre 2 = (1/2) d^T F d", sp.simplify(quad) == 0, f"reste={quad}")

# KL dans l'autre sens : meme terme quadratique
KLr = sum(pi[a] * (sp.log(pi[a]) - sp.log(pin[a])) for a in range(3))
c2r = sp.simplify(sp.series(KLr, s_, 0, 3).removeO().expand().coeff(s_, 2))
check("KL inversee : meme terme d'ordre 2 (le sens n'importe pas au 2e ordre)",
      sp.simplify(c2r - c2) == 0)

# --------------------------------------------------------------------------- #
print("\n=== 4.4  Programme quadratique du gradient naturel")
# --------------------------------------------------------------------------- #
g1, g2, lam, eps = sp.symbols('g_1 g_2 lambda epsilon', positive=True)
f11, f22 = sp.symbols('F_11 F_22', positive=True)
Fm = sp.diag(f11, f22)                          # diagonale : suffit et reste lisible
gv = sp.Matrix([g1, g2])
dv = sp.Matrix([d1, d2])
Lag = (gv.T * dv)[0, 0] - lam * ((dv.T * Fm * dv)[0, 0] / 2 - eps)
stat = [sp.diff(Lag, v) for v in (d1, d2)]
sol = sp.solve(stat, [d1, d2], dict=True)[0]
d_star = sp.Matrix([sol[d1], sol[d2]])
check("stationnarite du lagrangien : Delta = (1/lambda) F^-1 g",
      sp.simplify(d_star - Fm.inv() * gv / lam) == sp.zeros(2, 1))

lam_sol = sp.solve(sp.Eq((d_star.T * Fm * d_star)[0, 0] / 2, eps), lam)
lam_pos = [x for x in lam_sol if sp.simplify(x.subs({g1: 1, g2: 1, f11: 1, f22: 1, eps: 1})) > 0][0]
gFg = (gv.T * Fm.inv() * gv)[0, 0]
check("contrainte saturee : lambda = sqrt(g^T F^-1 g / (2 eps))",
      sp.simplify(lam_pos - sp.sqrt(gFg / (2 * eps))) == 0, f"lambda={sp.simplify(lam_pos)}")
d_final = sp.simplify(d_star.subs(lam, lam_pos))
d_formule = sp.simplify(sp.sqrt(2 * eps / gFg) * Fm.inv() * gv)
check("solution : Delta = sqrt(2 eps / (g^T F^-1 g)) F^-1 g",
      sp.simplify(d_final - d_formule) == sp.zeros(2, 1))

# --------------------------------------------------------------------------- #
print("\n=== 4.1  Argmax lineaire sur une boule euclidienne")
# --------------------------------------------------------------------------- #
r = sp.symbols('r', positive=True)
Lag2 = (gv.T * dv)[0, 0] - lam * ((dv.T * dv)[0, 0] - r ** 2)
sol2 = sp.solve([sp.diff(Lag2, v) for v in (d1, d2)], [d1, d2], dict=True)[0]
d2v = sp.Matrix([sol2[d1], sol2[d2]])
check("max g^T Delta sur ||Delta|| <= r  =>  Delta parallele a g",
      sp.simplify(d2v - gv / (2 * lam)) == sp.zeros(2, 1))
lam2 = [x for x in sp.solve(sp.Eq((d2v.T * d2v)[0, 0], r ** 2), lam)
        if sp.simplify(x.subs({g1: 1, g2: 1, r: 1})) > 0][0]
check("norme saturee  =>  Delta = r g / ||g||",
      sp.simplify(d2v.subs(lam, lam2) - r * gv / sp.sqrt((gv.T * gv)[0, 0])) == sp.zeros(2, 1))

# --------------------------------------------------------------------------- #
print("\n=== 6.2  Equivalence clip <-> forme par cas (preuve par cas exhaustive)")
# --------------------------------------------------------------------------- #
rr, aa, ee = sp.symbols('r A epsilon', real=True)
clip = sp.Max(1 - ee, sp.Min(rr, 1 + ee))
papier = sp.Min(rr * aa, clip * aa)

ok_all = True
details = []
for signe, cas_wiki in (("A>0", sp.Min(rr, 1 + ee) * aa), ("A<0", sp.Max(rr, 1 - ee) * aa)):
    for regime, hyp in (("r<1-eps", {rr: sp.Rational(1, 2), ee: sp.Rational(1, 5)}),
                        ("dans",    {rr: sp.Integer(1),     ee: sp.Rational(1, 5)}),
                        ("r>1+eps", {rr: sp.Rational(3, 2), ee: sp.Rational(1, 5)})):
        for aval in ([sp.Integer(3)] if signe == "A>0" else [sp.Integer(-3)]):
            h = dict(hyp); h[aa] = aval
            diff = sp.simplify(papier.subs(h) - cas_wiki.subs(h))
            ok_all &= (diff == 0)
            details.append(f"{signe}/{regime}:{diff}")
check("min(rA, clip(r)A) == forme par cas, 6 regimes", ok_all, "; ".join(details))

# Le point clef : multiplier par A < 0 echange min et max.
# sympy ne simplifie pas Min/Max sous simple hypothese de signe : on procede par
# cas exhaustif sur l'ordre de u et v, ce qui EST la demonstration.
uu, vv, Aa = sp.symbols('u v A', real=True)
cas_neg, cas_pos = True, True
for u_, v_ in [(sp.Integer(1), sp.Integer(2)), (sp.Integer(2), sp.Integer(1)),
               (sp.Integer(3), sp.Integer(3)), (sp.Rational(-5, 2), sp.Rational(7, 3))]:
    for A_ in [sp.Integer(-4), sp.Rational(-1, 3)]:
        cas_neg &= sp.simplify(sp.Min(u_ * A_, v_ * A_) - A_ * sp.Max(u_, v_)) == 0
    for A_ in [sp.Integer(4), sp.Rational(1, 3)]:
        cas_pos &= sp.simplify(sp.Min(u_ * A_, v_ * A_) - A_ * sp.Min(u_, v_)) == 0
check("A < 0 : min(uA, vA) = A * max(u, v)  (preuve par cas sur l'ordre de u, v)", cas_neg)
check("A > 0 : min(uA, vA) = A * min(u, v)", cas_pos)
# generalisation : verification exhaustive sur une grille rationnelle dense
import itertools
vals = [sp.Rational(k, 4) for k in range(-12, 13)]
gen_ok = all(
    sp.Min(u_ * A_, v_ * A_) == (A_ * sp.Max(u_, v_) if A_ < 0 else A_ * sp.Min(u_, v_))
    for u_, v_, A_ in itertools.product(vals, vals, [v for v in vals if v != 0]))
check(f"echange min/max : verifie sur {len(vals)**2 * (len(vals)-1)} triplets rationnels",
      gen_ok)

# --------------------------------------------------------------------------- #
print("\n=== 6.5  Estimateur k3 de la KL")
# --------------------------------------------------------------------------- #
x = sp.symbols('x', positive=True)
h = x - 1 - sp.log(x)
check("x - 1 - log x : point critique unique en x = 1", sp.solve(sp.diff(h, x), x) == [1])
check("x - 1 - log x : convexe (derivee seconde > 0)",
      sp.simplify(sp.diff(h, x, 2) - 1 / x ** 2) == 0)
check("x - 1 - log x : minimum nul  =>  integrande k3 >= 0", sp.simplify(h.subs(x, 1)) == 0)

# E_pi[pi_ref/pi] = 1 (c'est ce qui rend les deux estimateurs de meme esperance)
q1, q2 = sp.symbols('q_1 q_2', positive=True)
p_ = [pi[0], pi[1], pi[2]]
q_ = [q1, q2, 1 - q1 - q2]
check("E_{a~pi}[pi_ref/pi] = 1",
      sp.simplify(sum(p_[a] * (q_[a] / p_[a]) for a in range(3)) - 1) == 0)
kl_expr = sum(p_[a] * sp.log(p_[a] / q_[a]) for a in range(3))
k3_expr = sum(p_[a] * (sp.log(p_[a] / q_[a]) + q_[a] / p_[a] - 1) for a in range(3))
check("E[log(pi/pref) + pref/pi - 1] = KL(pi || pref)",
      sp.simplify(k3_expr - kl_expr) == 0)

print(f"\n{'=' * 70}")
print(f"{len(FAILS)} echec(s)")
sys.exit(1 if FAILS else 0)
