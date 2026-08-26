"""Sweep : bras x envs x graines, budget en PAS D'ENVIRONNEMENT.

    .venv/bin/python -m ppo_arms.run_sweep --envs HalfCheetah-v5 \
        --arms D,D_vfclip --seeds 0 --max-env-steps 300000

Budget en pas d'env et non en itérations : les bras n'ont pas le même
`train_batch_size` (D=4000, P=2048), donc à budget d'itérations égal ils ne
verraient pas la même quantité de données et la comparaison ne voudrait rien
dire.

Un fichier `logs/<run_id>_<env>_<arm>_seed<N>.jsonl` par run. `--resume` skippe
les runs déjà terminés : indispensable sur un sweep de plusieurs heures.

Adapté de `marl-rllib-gpu-bench/bench/run_sweep.py` (gestion de reprise, capture
d'OOM, détection des EnvRunners morts).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # AVANT tout import torch/ray
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

from ppo_arms.arms import ARMS, ENV_ARMS, ENVS, build_config  # noqa: E402
from ppo_arms.logging_utils import JsonlLogger, run_metadata  # noqa: E402

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
# Une itération sans le moindre échantillon est déjà anormale ; 3 d'affilée
# signent des EnvRunners morts (typiquement OOM sur les 8 Go partagés).
MAX_STALLED_ITERS = 3


def _already_done(path: Path, max_env_steps: int) -> bool:
    """Terminé = le log existe ET sa dernière ligne atteint le budget. Un run
    coupé à mi-parcours est donc relancé depuis zéro, pas repris en l'état."""
    if not path.exists():
        return False
    try:
        last = None
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    last = json.loads(line)
        return last is not None and last.get("env_steps_sampled", 0) >= max_env_steps
    except Exception:
        return False


def run_one(arm: str, env: str, seed: int, args, run_id: str, meta: dict) -> str:
    spec = ARMS[arm]
    log_path = LOGS_DIR / f"{run_id}_{env}_{arm}_seed{seed}.jsonl"
    if args.resume and _already_done(log_path, args.max_env_steps):
        print(f"  [skip] {env}/{arm}/seed{seed} déjà terminé", flush=True)
        return "skipped"

    cfg = build_config(spec, env, seed, args.num_env_runners,
                       args.num_envs_per_runner, args.gpu_learner)
    algo = cfg.build_algo()
    logger = JsonlLogger(log_path, {"run_id": run_id, "seed": seed, **meta})

    spec_row = {
        "name": arm, "lever": spec.lever, "prediction": spec.prediction,
        "env": env, "seed": seed,
        "batch": cfg.train_batch_size_per_learner,
        "minibatch": cfg.minibatch_size,
        "num_epochs": cfg.num_epochs,
        "lr": cfg.lr,
        "lambda_": cfg.lambda_,
        "clip_param": cfg.clip_param,
        "vf_clip_param": cfg.vf_clip_param,
        "use_kl_loss": cfg.use_kl_loss,
        "workers": args.num_env_runners,
        "envs_per_runner": args.num_envs_per_runner,
    }

    steps, iteration, t_cumulative, stalled = 0, 0, 0.0, 0
    try:
        while steps < args.max_env_steps:
            t0 = time.perf_counter()
            result = algo.train()
            dt = time.perf_counter() - t0
            t_cumulative += dt

            # EnvRunners morts (OOM) : Ray les redémarre et RLlib renvoie
            # num_env_steps_sampled_lifetime=0 avec des métriques nan. Sans le
            # max(), `steps` retomberait à 0 et la boucle tournerait sans fin.
            reported = int(result.get("env_runners", {})
                           .get("num_env_steps_sampled_lifetime", 0))
            if reported <= steps:
                stalled += 1
                if stalled >= MAX_STALLED_ITERS:
                    raise RuntimeError(
                        f"aucun échantillon depuis {stalled} itérations "
                        f"(EnvRunners morts, typiquement OOM) — run abandonné")
            else:
                stalled = 0
            delta = max(0, reported - steps)
            steps = max(steps, reported)

            learner = result.get("learners", {}).get("default_policy", {})
            logger.log_iteration(
                iteration=iteration, is_warmup=(iteration == 0), wall_time_s=dt,
                cumulative_time_s=t_cumulative, config_spec=spec_row, result=result,
                extra={"env_steps_sampled": steps, "env_steps_delta": delta,
                       "throughput_sps_measured": delta / dt if dt > 0 else 0.0},
            )
            ret = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
            evar = learner.get("vf_explained_var", float("nan"))
            print(f"  it {iteration:>3} | steps {steps:>7} | return {ret:9.1f} "
                  f"| vf_evar {evar:6.3f} | {dt:6.2f}s", flush=True)
            iteration += 1
    finally:
        logger.close()
        algo.stop()

    print(f"  -> TERMINÉ {env}/{arm}/seed{seed} : {steps} pas en "
          f"{t_cumulative/60:.1f} min ({steps/t_cumulative:.0f} pas/s) "
          f"-> {log_path.name}", flush=True)
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", default=",".join(ENVS))
    ap.add_argument("--arms", default=None,
                    help="par défaut : ENV_ARMS (ablation complète sur HalfCheetah, "
                         "D et P ailleurs)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max-env-steps", type=int, default=500_000)
    ap.add_argument("--num-env-runners", type=int, default=0,
                    help="0 = échantillonnage dans le driver, sans acteur Ray "
                         "(le plus économe en RAM, contrainte serrante ici)")
    ap.add_argument("--num-envs-per-runner", type=int, default=4,
                    help="vectorisation ; possible car les envs sont single-agent")
    ap.add_argument("--gpu-learner", action="store_true",
                    help="learner sur GPU ; ne change pas les maths, seulement le debit")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    if args.gpu_learner:
        # Wheel torch NVIDIA pre-release : segfault au premier forward GPU avec
        # cuDNN actif (cf. skill jetson-torch-setup). Un MLP n'utilise pas cuDNN,
        # mais le cout de la desactivation est nul.
        import torch
        torch.backends.cudnn.enabled = False

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = run_metadata(os.cpu_count())
    envs = args.envs.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    forced_arms = args.arms.split(",") if args.arms else None

    total = sum(len(forced_arms or ENV_ARMS[e]) for e in envs) * len(seeds)
    print(f"run_id={run_id}  {total} runs  budget={args.max_env_steps} pas/run", flush=True)
    print(f"{meta}\n", flush=True)

    outcomes: dict[str, str] = {}
    # Graines à l'EXTÉRIEUR : une interruption laisse tous les bras à 1 graine
    # (comparaison possible, juste bruitée) plutôt qu'un tiers des bras à 3
    # graines (rien à comparer).
    for seed in seeds:
        for env in envs:
            for arm in (forced_arms or ENV_ARMS[env]):
                tag = f"{env}/{arm}/seed{seed}"
                print(f"=== {tag} : {ARMS[arm].lever} ===", flush=True)
                try:
                    outcomes[tag] = run_one(arm, env, seed, args, run_id, meta)
                except Exception as exc:
                    # Un OOM ne doit pas tuer les 29 autres runs du sweep.
                    outcomes[tag] = f"FAILED: {type(exc).__name__}"
                    print(f"  !! {tag} a échoué : {exc}", flush=True)
                    traceback.print_exc()

    print("\n=== RÉCAPITULATIF ===", flush=True)
    for tag, outcome in outcomes.items():
        print(f"{tag:40s} {outcome}", flush=True)
    print(f"\nlogs : {LOGS_DIR}/{run_id}_*.jsonl", flush=True)


if __name__ == "__main__":
    main()
