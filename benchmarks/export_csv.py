"""Exporte les logs JSONL en un CSV compact, versionnable.

    .venv/bin/python export_csv.py [--run-id PILOT]

Les JSONL bruts font 45 Mo pour 323 colonnes par ligne, dont 16 servent
reellement. Le CSV fait 600 Ko et permet de recalculer chaque chiffre du
rapport. C'est LUI qui est versionne ; les JSONL restent locaux (.gitignore).

Garder ce script versionne : sans lui, le CSV serait un artefact opaque.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
LOGS = BASE / "logs"
OUT = BASE / "results" / "runs.csv"

# Colonnes retenues : celles qu'analyze.py consomme, plus les diagnostics cites
# dans docs/05 (KL, entropie, perte de politique). Ajouter une colonne ici
# suffit a la rendre disponible ; il faut alors re-exporter.
COLS = [
    "env", "arm", "seed", "iter",
    "env_steps_sampled", "wall_time_s", "cumulative_time_s",
    "throughput_sps_measured",
    "env_runners/episode_return_mean",
    "learners/default_policy/vf_explained_var",
    "learners/default_policy/vf_loss",
    "learners/default_policy/vf_loss_unclipped",
    "learners/default_policy/mean_kl_loss",
    "learners/default_policy/curr_kl_coeff",
    "learners/default_policy/policy_loss",
    "learners/default_policy/entropy",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="PILOT")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    paths = sorted(LOGS.glob(f"{args.run_id}_*.jsonl"))
    if not paths:
        raise SystemExit(f"aucun log {args.run_id}_*.jsonl dans {LOGS}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for p in paths:
            stem = p.stem[len(args.run_id) + 1:]
            env, rest = stem.split("_", 1)
            arm, seed = rest.rsplit("_seed", 1)
            for line in open(p):
                if not line.strip():
                    continue
                r = json.loads(line)
                w.writerow([env, arm, int(seed)] + [r.get(c, "") for c in COLS[3:]])
                rows += 1

    size_kb = args.out.stat().st_size // 1024
    print(f"{len(paths)} run(s) -> {rows} lignes, {size_kb} Ko -> {args.out}")


if __name__ == "__main__":
    main()
