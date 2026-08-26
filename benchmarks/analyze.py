"""Analyse des logs de sweep : tableaux prets a coller dans docs/05-mesures.md.

    .venv/bin/python analyze.py [--run-id PILOT] [--threshold 500]

Volontairement sans pandas ni matplotlib : ils ne sont pas installes sur le
Jetson (meme choix que marl-rllib-gpu-bench, ou l'analyse graphique se fait
apres rsync). Ce script produit les CHIFFRES du rapport ; les courbes viennent
apres, ailleurs.

Ne juge que ce qui est mesure. Un bras absent des logs n'est pas rapporte, il
n'est pas suppose.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

LOGS = Path(__file__).resolve().parent / "logs"

K_STEPS = "env_steps_sampled"
K_RETURN = "env_runners/episode_return_mean"
K_EVAR = "learners/default_policy/vf_explained_var"
K_VF = "learners/default_policy/vf_loss"
K_VF_UNCLIPPED = "learners/default_policy/vf_loss_unclipped"
K_KL = "learners/default_policy/mean_kl_loss"
K_SPS = "throughput_sps_measured"

# Retours d'une politique aleatoire, mesures sur cette machine (voir arms.py).
RANDOM_RETURN = {"HalfCheetah-v5": -270.7, "Hopper-v5": 11.1, "Walker2d-v5": -4.9}


def load(run_id: str | None) -> dict[tuple[str, str, int], list[dict]]:
    runs: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    pattern = f"{run_id}_*.jsonl" if run_id else "*.jsonl"
    for path in sorted(LOGS.glob(pattern)):
        rows = [json.loads(l) for l in open(path) if l.strip()]
        if not rows:
            continue
        r0 = rows[0]
        key = (r0["bench_config/env"], r0["bench_config/name"], r0["seed"])
        runs[key] = rows
    return runs


def tail_mean(rows: list[dict], key: str, frac: float = 0.2) -> float:
    """Moyenne sur la derniere fraction du run. Plus robuste que la derniere
    valeur (window=1 cote RLlib, donc tres bruitee)."""
    vals = [r[key] for r in rows if key in r and math.isfinite(r[key])]
    if not vals:
        return float("nan")
    n = max(1, int(len(vals) * frac))
    return st.fmean(vals[-n:])


def steps_to_threshold(rows: list[dict], threshold: float) -> float:
    """Pas d'environnement pour atteindre `threshold` en retour. inf si jamais
    atteint : le run est CENSURE, pas exclu -- l'exclure biaiserait la moyenne
    en faveur des bras qui echouent souvent."""
    for r in rows:
        if r.get(K_RETURN, -math.inf) >= threshold:
            return r[K_STEPS]
    return math.inf


def fmt(x: float, nd: int = 1) -> str:
    if x != x:
        return "n/a"
    if x == math.inf:
        return "non atteint"
    return f"{x:.{nd}f}"


def agg(vals: list[float]) -> str:
    """moyenne +/- ecart-type, ou la valeur seule si une seule graine."""
    clean = [v for v in vals if v == v and abs(v) != math.inf]
    if not clean:
        return "n/a"
    if len(clean) == 1:
        return f"{clean[0]:.1f}"
    return f"{st.fmean(clean):.1f} ± {st.pstdev(clean):.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="seuil de retour pour samples_to_threshold ; a fixer "
                         "depuis la courbe du pilote, pas a deviner")
    args = ap.parse_args()

    runs = load(args.run_id)
    if not runs:
        print(f"aucun log dans {LOGS} (run_id={args.run_id})")
        return

    envs = sorted({k[0] for k in runs})
    print(f"{len(runs)} run(s) charge(s) | envs : {', '.join(envs)}\n")

    for env in envs:
        arms = sorted({k[1] for k in runs if k[0] == env})
        print(f"=== {env} " + "=" * (58 - len(env)))
        header = (f"{'bras':10s} {'graines':7s} {'pas':>8s} {'retour final':>16s} "
                  f"{'vf_evar fin':>12s} {'vf_evar max':>12s} "
                  f"{'unclip/clip':>12s} {'pas/s':>7s}")
        print(header)
        print("-" * len(header))

        finals: dict[str, list[float]] = {}
        for arm in arms:
            seeds = sorted(s for (e, a, s) in runs if e == env and a == arm)
            rows_per_seed = [runs[(env, arm, s)] for s in seeds]

            fin = [tail_mean(r, K_RETURN) for r in rows_per_seed]
            evar_fin = [tail_mean(r, K_EVAR) for r in rows_per_seed]
            evar_max = [max((x[K_EVAR] for x in r
                             if math.isfinite(x.get(K_EVAR, float("nan")))),
                            default=float("nan"))
                        for r in rows_per_seed]
            ratio = []
            for r in rows_per_seed:
                pairs = [(x[K_VF_UNCLIPPED], x[K_VF]) for x in r
                         if K_VF in x and K_VF_UNCLIPPED in x and x[K_VF] > 0]
                ratio.append(st.median([u / c for u, c in pairs]) if pairs else float("nan"))
            sps = [tail_mean(r, K_SPS, frac=0.8) for r in rows_per_seed]

            finals[arm] = fin
            # Pas franchis par la graine la moins avancee : rend visible un run
            # encore en cours, dont les chiffres ne veulent rien dire.
            progress = min(r[-1][K_STEPS] for r in rows_per_seed)
            print(f"{arm:10s} {len(seeds):^7d} {progress:>8d} {agg(fin):>16s} "
                  f"{agg(evar_fin):>12s} {agg(evar_max):>12s} "
                  f"{agg(ratio):>12s} {fmt(st.fmean([s for s in sps if s == s]) if any(s == s for s in sps) else float('nan'), 0):>7s}")

        if args.threshold is not None:
            print(f"\n  pas pour atteindre un retour de {args.threshold:g} :")
            for arm in arms:
                seeds = sorted(s for (e, a, s) in runs if e == env and a == arm)
                sts = [steps_to_threshold(runs[(env, arm, s)], args.threshold) for s in seeds]
                reached = [s for s in sts if s != math.inf]
                label = (f"{st.fmean(reached):,.0f}".replace(",", " ")
                         if reached else "jamais atteint")
                censored = len(sts) - len(reached)
                suffix = f"  ({censored}/{len(sts)} graine(s) censuree(s))" if censored else ""
                print(f"    {arm:10s} {label:>12s}{suffix}")

        # Score normalise a la maniere de la Table 1 du papier : aleatoire = 0,
        # meilleur bras observe = 1. Ne compare RIEN aux chiffres du papier --
        # seulement les bras entre eux, sur le meme env et la meme version de
        # MuJoCo.
        rnd = RANDOM_RETURN.get(env)
        best = max((st.fmean([v for v in vs if v == v])
                    for vs in finals.values() if any(v == v for v in vs)),
                   default=None)
        if rnd is not None and best is not None and best > rnd:
            print("\n  score normalise (aleatoire = 0, meilleur bras = 1) :")
            for arm in arms:
                vals = [v for v in finals[arm] if v == v]
                if vals:
                    print(f"    {arm:10s} {(st.fmean(vals) - rnd) / (best - rnd):6.3f}")
        print()

    # --- verdicts sur les predictions, quand les donnees le permettent ---
    print("=== predictions ===")
    hc = "HalfCheetah-v5"

    def arm_metric(arm: str, key: str, frac: float = 0.2) -> list[float]:
        seeds = sorted(s for (e, a, s) in runs if e == hc and a == arm)
        return [tail_mean(runs[(hc, arm, s)], key, frac) for s in seeds]

    d_evar, v_evar = arm_metric("D", K_EVAR), arm_metric("D_vfclip", K_EVAR)
    if d_evar and v_evar:
        d_ok = all(v < 0.1 for v in d_evar if v == v)
        v_ok = any(v > 0.5 for v in v_evar if v == v)
        verdict = "CONFIRMEE" if (d_ok and v_ok) else "INFIRMEE (voir tableau)"
        print(f"P1 vf_clip gele le critique : {verdict}")
        print(f"   D vf_evar={agg(d_evar)} (predit < 0.1) | "
              f"D_vfclip vf_evar={agg(v_evar)} (predit > 0.5)")
    else:
        print("P1 : pas encore de donnees (bras D et D_vfclip requis)")

    for pred, arm in [("P2 lambda_=0.95 plus efficace", "D_lambda"),
                      ("P3 KL ajoutee inutile", "D_kl"),
                      ("P4 clip 0.2 >= 0.3", "D_eps"),
                      ("P5 config du papier > defauts", "P")]:
        vals, base = arm_metric(arm, K_RETURN), arm_metric("D", K_RETURN)
        if vals and base and any(v == v for v in vals):
            print(f"{pred} : {arm} retour={agg(vals)} vs D={agg(base)}")
        else:
            print(f"{pred} : pas encore de donnees")


if __name__ == "__main__":
    main()
