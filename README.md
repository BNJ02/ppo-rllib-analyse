# PPO : le papier, et ce que Ray RLlib en a fait

Lecture croisée entre trois papiers de John Schulman et le code de RLlib qui prétend les implémenter. Ce que le code fait vraiment, où il s'écarte du papier, et ce que ces écarts changent en pratique.

Tous les extraits de code et numéros de ligne proviennent de `ray-project/ray` au commit [`7a5d7f1`](https://github.com/ray-project/ray/tree/7a5d7f1667f79a907c3106f9347d411285297219) (25 août 2026), **new API stack**.

---

## Si vous ne lisez qu'une chose

> Trois défauts de `PPOConfig()` s'écartent du papier au point de changer l'algorithme :
>
> | Défaut RLlib | Papier | Effet |
> |---|---|---|
> | `use_kl_loss=True`, `kl_coeff=0.2` | clipping **ou** KL, jamais les deux | objectif hybride évalué nulle part ; le papier mesure la KL seule comme **inférieure** au clipping (0,74 vs 0,82) |
> | `vf_clip_param=10.0` | pas de clipping de la perte de valeur | gradient du critique **exactement nul** dès que \|V − V_targ\| > 3,16 — sur un env à récompenses d'ordre 100, le critique n'apprend jamais |
> | `lambda_=1.0` | 0,95 partout | dégénère GAE en Monte-Carlo, c'est-à-dire **l'estimateur que le papier GAE a été écrit pour remplacer** |
>
> Le deuxième est le plus sévère : l'entraînement ne plante pas, il ne progresse simplement pas. Métrique de contrôle : `vf_explained_var`.

---

## Les documents

| # | Document | Contenu |
|---|---|---|
| 0 | [Résumé du papier PPO](docs/00-resume-papier-ppo.md) | une page — et pourquoi « Algorithm**s** » au pluriel |
| 1 | [Comprendre le papier](docs/01-comprendre-le-papier-ppo.md) | guide de lecture, équation (7) décortiquée cas par cas, six erreurs de lecture classiques, auto-test |
| 2 | [PPO : papier vs RLlib](docs/02-ppo-papier-vs-rllib.md) | correspondance ligne à ligne, 8 écarts classés par gravité, config de reproduction |
| 3 | [GAE : papier vs RLlib](docs/03-gae-papier-vs-rllib.md) | ce qu'est GAE, la récursion dans le code, `use_gae` mort et `use_critic` incohérent |
| 4 | [PPO en multi-agent](docs/04-ppo-multiagent-rllib.md) | ce que le code fait avec N modules ; lu à travers la taxonomie à trois axes de [`marl-rllib-sota`](https://github.com/BNJ02/marl-rllib-sota) |
| 5 | [**Mesures**](docs/05-mesures.md) | 30 runs, 3 graines : les écarts sont-ils réels ? Prédictions enregistrées **avant** les runs, trois d'entre elles infirmées |

Chaque document alterne formules LaTeX, extraits de code réels, et encarts **« En clair »** en langage courant. Le niveau technique n'est pas abaissé ; l'intuition est ajoutée à côté.

---

## Les trois papiers

| Papier | Référence | Ce qu'il apporte |
|---|---|---|
| **TRPO** | [arXiv:1502.05477](https://arxiv.org/abs/1502.05477) | contrainte KL dure, amélioration monotone, second ordre |
| **GAE** | [arXiv:1506.02438](https://arxiv.org/abs/1506.02438) | **comment** estimer Â_t : compromis biais/variance réglé par λ |
| **PPO** | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) | **quoi faire** de Â_t : clipping du ratio, premier ordre |

```
TRPO (2015)   →  contrainte KL, garantie monotone — mais second ordre, lourd
   │
GAE (2016)    →  Â_t = Σ (γλ)^l δ_{t+l}   ← RLlib l'implémente exactement,
   │                                          mais λ=1.0 par défaut le désactive
PPO (2017)    →  min(r·Â, clip(r,1±ε)·Â)  ← RLlib l'implémente exactement,
   │                                          puis ajoute une pénalité KL par-dessus
RLlib (2026)  →  les deux, plus 4 mécanismes absents des trois papiers
```

---

## Les écarts, en un tableau

La colonne **mesuré** vient de [05 §5.2](docs/05-mesures.md) : 30 runs, 3 graines, HalfCheetah/Hopper/Walker2d, 300k pas. Les écarts sans mesure n'ont pas été isolés dans un bras d'ablation.

| # | Écart | Gravité | Mesuré | Où |
|---|---|---|---|---|
| 1 | `lambda_=1.0` — désactive GAE | **élevée** | **×4,0 sur le retour** | [§4.1](docs/03-gae-papier-vs-rllib.md) |
| 2 | Clipping **et** pénalité KL cumulés par défaut | **élevée** | **×2,4 sur le retour** | [§4.1](docs/02-ppo-papier-vs-rllib.md) |
| 3 | Clipping de la **perte** de valeur à 10 | moyenne | 24× à 623× de gradient écrêté, **mais aucun effet isolé** | [§4.3](docs/02-ppo-papier-vs-rllib.md) |
| 4 | ε=0,3 alors que le papier mesure 0,2 meilleur | faible | aucun bénéfice à corriger seul | [§4.5](docs/02-ppo-papier-vs-rllib.md) |
| 5 | Règle d'adaptation de β aux constantes interverties | moyenne | non isolé | [§4.2](docs/02-ppo-papier-vs-rllib.md) |
| 6 | Standardisation des avantages, hors papier | moyenne | non isolé | [§4.4](docs/02-ppo-papier-vs-rllib.md) |
| 7 | `use_gae` sans effet, `use_critic=False` incohérent | moyenne | non isolé | [§4.2-4.3](docs/03-gae-papier-vs-rllib.md) |
| 8 | Pas de région de confiance sur le critique | moyenne | non isolé | [§4.5](docs/03-gae-papier-vs-rllib.md) |
| 9 | Architecture 256×256, σ dépendant de l'état | faible | non isolé | [§4.6](docs/02-ppo-papier-vs-rllib.md) |
| 10 | Minibatchs circulaires chevauchant deux époques | faible | non isolé | [§3.4](docs/02-ppo-papier-vs-rllib.md) |

**Tous les écarts corrigés ensemble** (bras `P`, config de la Table 3 du papier) : **×7,4 sur le retour**, 3,3× moins d'échantillons pour un même seuil, et 2× plus rapide en temps mur. Plus que le meilleur levier isolé — les leviers ne sont pas additifs.

> Le classement a été **réordonné après mesure**. Avant les runs, `vf_clip_param` était noté « critique » et `lambda_` en second : l'ablation montre l'inverse. Les deux se combinent — c'est `lambda_=1.0` qui rend l'écrêtage de la perte mordant, et lever l'écrêtage seul ne change rien ([05 §5.2](docs/05-mesures.md)).

Et, spécifiques au multi-agent ([§6](docs/04-ppo-multiagent-rllib.md)) : époques effectives inégales entre modules, avantages standardisés **par module**, `train_batch_size` compté en pas d'environnement, modules gelés payant GAE en entier.

---

## Reproduire le papier

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

config = (
    PPOConfig()
    .environment("HalfCheetah-v5")
    .env_runners(num_env_runners=1, rollout_fragment_length=2048)
    .training(
        use_kl_loss=False, kl_coeff=0.0, clip_param=0.2,   # clipping seul (§3 du papier)
        lambda_=0.95, gamma=0.99,                          # GAE, Table 3
        lr=3e-4, num_epochs=10, minibatch_size=64,
        train_batch_size_per_learner=2048,
        vf_clip_param=float("inf"),                        # neutralise le clipping de perte
        vf_loss_coeff=1.0, entropy_coeff=0.0,
    )
    .rl_module(model_config=DefaultModelConfig(
        fcnet_hiddens=[64, 64], fcnet_activation="tanh",
        vf_share_layers=False, free_log_std=True,
    ))
)
```

Restent non reproductibles sans code : la standardisation des avantages (câblée dans le connecteur `GeneralAdvantageEstimation`), le découpage exact des minibatchs, et l'annealing linéaire de ε du protocole Atari.

---

## Métriques de diagnostic

| Métrique | Ce qu'elle révèle |
|---|---|
| `vf_loss_unclipped` / `vf_loss` | **la métrique fiable** — ratio > 3 → le clipping de perte mord. Mesuré de 24 à 623 aux défauts |
| `vf_explained_var` | indicatif seulement — logué avec `window=1` (un seul minibatch), trop bruité pour un seuil |
| `curr_kl_coeff` | dérive vers le haut → la politique bouge trop vite |
| `num_module_steps_trained` par module | ratio > 2 entre modules → époques effectives inégales (multi-agent) |

---

## Portée et limites

- **Framework** : `torch` uniquement. TensorFlow est refusé sur le new API stack (`ValueError` explicite dans `PPOConfig.get_default_learner_class`).
- **Un seul commit** : les numéros de ligne dérivent. Les noms de symboles (`compute_loss_for_module`, `GeneralAdvantageEstimation`, `MiniBatchCyclicIterator`) restent le point d'entrée fiable.
- **Les mesures sont un A/B interne, pas une reproduction du papier.** MuJoCo v1 (mujoco-py) du papier et Gymnasium v5 (MuJoCo 3.x) diffèrent en récompenses, terminaisons et observations : les retours des Tables 1-6 ne sont pas comparables et ne sont jamais comparés. Ce qui est mesuré, ce sont les bras entre eux. Voir [05 §1](docs/05-mesures.md).
- **Budget de 300 000 pas**, non 1 M comme le papier, sur 3 graines et un seul environnement pour l'ablation complète. Aucun bras n'a convergé. Détail des limites en [05 §5.2](docs/05-mesures.md).
- **Pas de PDF redistribué.** Les papiers sont liés vers arXiv, pas copiés ici.

---

## Voir aussi

[`BNJ02/marl-rllib-sota`](https://github.com/BNJ02/marl-rllib-sota) — état de l'art du multi-agent dans RLlib : ce qu'il sait faire, ce qu'il ne sait pas faire, et comment câbler ce qui manque. Le document 4 de ce dépôt en est la contrepartie côté code PPO.

## Licence

[MIT](LICENSE).
