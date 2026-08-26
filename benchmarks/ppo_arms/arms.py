"""Les 6 bras : config PPO par défaut de RLlib vs config fidèle au papier.

Chaque bras est un **dict d'overrides appliqué à `PPOConfig()`**, jamais une
config réécrite à la main. C'est ce qui garantit — et rend testable — qu'un bras
d'ablation ne diffère de la référence que d'un seul champ. `test_arms.py`
l'assert.

Les quatre bras `D_*` isolent les écarts documentés dans
`docs/02-ppo-papier-vs-rllib.md` et `docs/03-gae-papier-vs-rllib.md`.

Piège Ray hérité de `marl-rllib-gpu-bench/bench/sweep.py` : les builders passés
à RLlib doivent être des fonctions top-level d'un module importable. Un lambda
défini dans un `if __name__` ou un import dynamique via `sys.path.insert()` fait
planter les `EnvRunner` distants (`ModuleNotFoundError` côté acteur, le sys.path
du driver n'est pas hérité). D'où `_make_obs_filter` ci-dessous.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import MeanStdFilter
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

# Les 3 envs du papier retenus. Retours d'une politique aléatoire mesurés sur
# cette machine (mujoco 3.12.0 / gymnasium 1.2.2), base du score normalisé :
#   HalfCheetah-v5  -270.7 sur 1000 pas (pas de terminaison)
#   Hopper-v5         11.1 sur   21 pas
#   Walker2d-v5       -4.9 sur    9 pas
ENVS = ("HalfCheetah-v5", "Hopper-v5", "Walker2d-v5")

RANDOM_RETURN = {"HalfCheetah-v5": -270.7, "Hopper-v5": 11.1, "Walker2d-v5": -4.9}


def _make_obs_filter(env=None, spaces=None, device=None):
    """Normalisation des observations, identique sur TOUS les bras.

    Constante partagée, pas un facteur : sans elle rien n'apprend sur MuJoCo, et
    la faire varier entre bras la rendrait confondante avec les facteurs testés.

    À ne pas confondre avec une normalisation de RÉCOMPENSE, volontairement
    absente : elle masquerait l'effet de `vf_clip_param`, qui est précisément
    l'objet du bras `D_vfclip`.
    """
    return MeanStdFilter()


# Config du papier, Table 3 (MuJoCo, 1 M pas). Réseau à part, voir PAPER_MODEL.
PAPER_TRAINING = {
    "lr": 3e-4,
    "num_epochs": 10,
    "minibatch_size": 64,
    "train_batch_size_per_learner": 2048,
    "lambda_": 0.95,
    "clip_param": 0.2,
    "use_kl_loss": False,          # §3 du papier : clipping SEUL
    "vf_clip_param": float("inf"),  # le papier n'écrête pas la perte de valeur
    "vf_loss_coeff": 1.0,
    "entropy_coeff": 0.0,          # « we don't use an entropy bonus »
}

# MLP 2x64 tanh, gaussienne à écart-type LIBRE (indépendant de l'état),
# réseaux politique/valeur séparés. RLlib par défaut : [256,256], free_log_std
# désactivé (vf_share_layers est déjà forcé à False par PPO).
PAPER_MODEL = DefaultModelConfig(
    fcnet_hiddens=[64, 64],
    fcnet_activation="tanh",
    vf_share_layers=False,
    free_log_std=True,
)


@dataclass
class ArmSpec:
    name: str
    lever: str          # ce qui change par rapport à D
    prediction: str     # enregistrée AVANT de mesurer
    training: dict = field(default_factory=dict)
    model: DefaultModelConfig | None = None


ARMS: dict[str, ArmSpec] = {
    # ---- référence ----
    "D": ArmSpec(
        "D", "aucun (défauts RLlib)", "référence",
    ),
    # ---- l'A/B principal ----
    "P": ArmSpec(
        "P", "config fidèle au papier (Table 3)",
        "bat D sur les trois envs",
        training=PAPER_TRAINING, model=PAPER_MODEL,
    ),
    # ---- ablation : un seul champ change par rapport à D ----
    # docs/02 §4.3 : la perte VF est écrêtée à 10, donc gradient du critique nul
    # dès |V - V_targ| > sqrt(10) ~ 3.16. Sur MuJoCo, V ~ 100+ des le depart.
    "D_vfclip": ArmSpec(
        "D_vfclip", "vf_clip_param=inf",
        "vf_explained_var depasse 0.5 alors que D reste sous 0.1",
        training={"vf_clip_param": float("inf")},
    ),
    # docs/03 §4.1 : lambda=1.0 degenere GAE en Monte-Carlo, soit l'estimateur
    # que le papier GAE a ete ecrit pour remplacer.
    "D_lambda": ArmSpec(
        "D_lambda", "lambda_=0.95",
        "atteint le seuil de retour avec MOINS d'echantillons que D",
        training={"lambda_": 0.95},
    ),
    # docs/02 §4.1 : RLlib cumule penalite KL et clipping, ce que le papier ne
    # fait jamais. use_kl_loss=False suffit a neutraliser le terme ET son
    # controleur (`after_gradient_based_update` teste ce meme drapeau), donc un
    # seul champ change -- kl_coeff devient inerte.
    "D_kl": ArmSpec(
        "D_kl", "use_kl_loss=False",
        "aucune difference significative avec D sur 3 graines",
        training={"use_kl_loss": False},
    ),
    # docs/02 §4.5 : le papier mesure eps=0.2 (0.82) meilleur que 0.3 (0.70).
    "D_eps": ArmSpec(
        "D_eps", "clip_param=0.2",
        "superieur ou egal a D",
        training={"clip_param": 0.2},
    ),
}

# Bras appliqués à chaque env. HalfCheetah porte l'ablation complète (retours
# les plus grands, pas de terminaison précoce donc un confondant en moins) ;
# Hopper et Walker2d ne servent qu'au contrôle de généralisation de l'A/B.
ENV_ARMS = {
    "HalfCheetah-v5": list(ARMS),
    "Hopper-v5": ["D", "P"],
    "Walker2d-v5": ["D", "P"],
}


def build_config(spec: ArmSpec, env: str, seed: int, num_env_runners: int,
                 num_envs_per_runner: int, gpu_learner: bool = False) -> PPOConfig:
    """`PPOConfig` pour un (bras, env, graine).

    Tout ce qui n'est pas dans `spec.training` reste au défaut de RLlib : c'est
    la définition même du bras D, et la condition pour que les bras d'ablation
    soient purs.
    """
    cfg = (
        PPOConfig()
        .environment(env)
        .debugging(seed=seed)
        .framework("torch")
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_runner,
            env_to_module_connector=_make_obs_filter,
        )
        # Choix mesure, pas suppose : le learner domine (83 % du temps
        # d'iteration a cause de num_epochs=30), donc c'est LUI qu'il faut
        # accelerer, pas l'echantillonnage. Le GPU ne change pas les maths --
        # c'est un levier de debit, identique sur tous les bras.
        .learners(num_learners=0, num_gpus_per_learner=1 if gpu_learner else 0)
    )
    if spec.training:
        cfg = cfg.training(**spec.training)
    if spec.model is not None:
        cfg = cfg.rl_module(model_config=spec.model)
    return cfg


# Champs de `.training()` compares par le test de purete. Une valeur absente ici
# et modifiee par un bras passerait le test sans etre vue -- garder la liste
# alignee sur les champs que les bras touchent.
TRACKED_FIELDS = (
    "lr", "num_epochs", "minibatch_size", "train_batch_size",
    "lambda_", "gamma", "clip_param", "vf_clip_param", "vf_loss_coeff",
    "entropy_coeff", "use_kl_loss", "kl_coeff", "kl_target",
    "use_critic", "use_gae", "shuffle_batch_per_epoch", "grad_clip",
)


def tracked(cfg: PPOConfig) -> dict:
    d = {f: getattr(cfg, f) for f in TRACKED_FIELDS}
    d["train_batch_size_per_learner"] = cfg.train_batch_size_per_learner
    return d
