"""Garde-fous à lancer AVANT tout run.

    .venv/bin/python test_arms.py

Trois vérifications, dans l'ordre d'importance :

1. Les défauts PPO de la version de Ray installée sont bien ceux décrits dans
   `docs/02-ppo-papier-vs-rllib.md`. Si une montée de version les change, ce
   test casse — et c'est le rapport qu'il faut corriger, pas le test.
2. Chaque bras d'ablation ne diffère de D que d'UN champ. Attrape la confusion
   accidentelle de deux facteurs, qui rendrait le résultat ininterprétable.
3. Les constantes partagées le sont vraiment sur les 6 bras.
"""

from __future__ import annotations

import sys

from ray.rllib.algorithms.ppo import PPOConfig

from ppo_arms.arms import ARMS, ENVS, _make_obs_filter, build_config, tracked

# Valeurs affirmées par docs/02 et docs/03, verifiees sur ray 2.57.0.
REPORTED_DEFAULTS = {
    "vf_clip_param": 10.0,
    "lambda_": 1.0,
    "use_kl_loss": True,
    "kl_coeff": 0.2,
    "kl_target": 0.01,
    "clip_param": 0.3,
    "lr": 5e-5,
    "num_epochs": 30,
    "minibatch_size": 128,
    "train_batch_size": 4000,
    "entropy_coeff": 0.0,
    "vf_loss_coeff": 1.0,
    "gamma": 0.99,
}

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'ECHEC'}  {label}{'  -- ' + detail if detail and not ok else ''}")
    if not ok:
        failures.append(f"{label}: {detail}")


print("1. Defauts PPO de la version de Ray installee vs docs/02")
base = PPOConfig()
for key, expected in REPORTED_DEFAULTS.items():
    actual = getattr(base, key)
    check(f"{key} == {expected!r}", actual == expected, f"vaut {actual!r}")

print("\n2. Purete des bras d'ablation (exactement 1 champ modifie vs D)")
ref_env = ENVS[0]
cfg_d = build_config(ARMS["D"], ref_env, seed=0, num_env_runners=0, num_envs_per_runner=1)
tracked_d = tracked(cfg_d)

for name, spec in ARMS.items():
    if not name.startswith("D_"):
        continue
    cfg = build_config(spec, ref_env, seed=0, num_env_runners=0, num_envs_per_runner=1)
    diff = {k: (tracked_d[k], v) for k, v in tracked(cfg).items() if v != tracked_d[k]}
    check(f"{name} ({spec.lever})", len(diff) == 1,
          f"{len(diff)} champs different: {diff}")

print("\n3. D correspond bien aux defauts nus (aucun override cache)")
check("D == PPOConfig() sur les champs suivis",
      tracked_d == tracked(PPOConfig()),
      f"{ {k: (tracked(PPOConfig())[k], v) for k, v in tracked_d.items() if v != tracked(PPOConfig())[k]} }")

print("\n4. Constantes partagees identiques sur les 6 bras")
for const in ("gamma", "use_gae", "use_critic"):
    vals = {}
    for name, spec in ARMS.items():
        cfg = build_config(spec, ref_env, seed=0, num_env_runners=0, num_envs_per_runner=1)
        vals[name] = getattr(cfg, const)
    check(f"{const} identique partout", len(set(vals.values())) == 1, f"{vals}")

# La normalisation de recompense masquerait l'effet de vf_clip_param : elle doit
# rester absente. RLlib ne l'active pas par defaut, mais un connecteur ajoute par
# megarde la reintroduirait -- d'ou le controle explicite.
print("\n5. Aucune normalisation de recompense (masquerait vf_clip_param)")
for name, spec in ARMS.items():
    cfg = build_config(spec, ref_env, seed=0, num_env_runners=0, num_envs_per_runner=1)
    conn = cfg._env_to_module_connector
    check(f"{name} : connecteur env->module == _make_obs_filter et rien d'autre",
          conn is _make_obs_filter, f"vaut {conn!r}")

print("\n6. Le bras P applique bien tous les reglages du papier")
cfg_p = build_config(ARMS["P"], ref_env, seed=0, num_env_runners=0, num_envs_per_runner=1)
for key, expected in [("lr", 3e-4), ("num_epochs", 10), ("minibatch_size", 64),
                      ("lambda_", 0.95), ("clip_param", 0.2), ("use_kl_loss", False),
                      ("entropy_coeff", 0.0)]:
    check(f"P.{key} == {expected!r}", getattr(cfg_p, key) == expected,
          f"vaut {getattr(cfg_p, key)!r}")
check("P.vf_clip_param == inf", cfg_p.vf_clip_param == float("inf"),
      f"vaut {cfg_p.vf_clip_param!r}")
check("P.train_batch_size_per_learner == 2048",
      cfg_p.train_batch_size_per_learner == 2048,
      f"vaut {cfg_p.train_batch_size_per_learner!r}")

print()
if failures:
    print(f"=== {len(failures)} ECHEC(S) ===")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("=== tous les garde-fous passent ===")
