# Mesures : les dérives de RLlib aident-elles ou nuisent-elles ?

Les documents [02](02-ppo-papier-vs-rllib.md) et [03](03-gae-papier-vs-rllib.md) affirment, **par lecture de code**, que la config PPO par défaut de RLlib s'écarte du papier sur six points. Aucune de ces affirmations n'était mesurée. Ce document les met à l'épreuve.

> **Ce document est écrit en deux temps.** Le protocole et les prédictions ci-dessous ont été **commités avant le premier run** (commit de ce fichier). Les résultats sont ajoutés ensuite, sans retoucher les prédictions. C'est ce qui sépare une mesure d'une rationalisation.

---

## 1. Ce qui est mesurable, et ce qui ne l'est pas

**Non mesurable : « RLlib fait mieux que le papier ».** Le papier tourne sur MuJoCo v1 (mujoco-py, MuJoCo 1.31). Gymnasium ne fournit plus que v4/v5 (MuJoCo 3.x), avec des fonctions de récompense, des conditions de terminaison et des espaces d'observation différents. Les retours absolus des Tables 1 à 6 du papier **ne sont pas comparables** à ce qu'on mesure ici. Toute confrontation chiffrée à ces tables serait fausse.

**Mesurable : un A/B interne.** Même environnement, mêmes graines, même version de Ray — config par défaut contre config fidèle au papier, plus une ablation un-facteur-à-la-fois. C'est ce qui répond réellement à la question « les modifications accumulées depuis 2017 aident-elles ? ».

---

## 2. Protocole

### Les six bras

Chaque bras est un **dict d'overrides appliqué à `PPOConfig()`**, jamais une config réécrite à la main — c'est ce qui rend la pureté vérifiable par test.

| Bras | Différence vs `PPOConfig()` | Écart testé |
|---|---|---|
| `D` | aucune (défauts RLlib) | référence |
| `P` | config du papier, Table 3 | l'A/B principal |
| `D_vfclip` | `vf_clip_param=inf` | [02 §4.3](02-ppo-papier-vs-rllib.md) |
| `D_lambda` | `lambda_=0.95` | [03 §4.1](03-gae-papier-vs-rllib.md) |
| `D_kl` | `use_kl_loss=False` | [02 §4.1](02-ppo-papier-vs-rllib.md) |
| `D_eps` | `clip_param=0.2` | [02 §4.5](02-ppo-papier-vs-rllib.md) |

`P` en entier : `lr=3e-4`, `num_epochs=10`, `minibatch_size=64`, `train_batch_size_per_learner=2048`, `lambda_=0.95`, `clip_param=0.2`, `use_kl_loss=False`, `vf_clip_param=inf`, `entropy_coeff=0.0`, réseau `[64,64]` tanh, `free_log_std=True`.

**Constantes partagées sur les six bras** — sinon ce sont des variables confondantes :

- `gamma=0.99`
- `MeanStdFilter` sur les observations (sans lui rien n'apprend sur MuJoCo)
- **aucune normalisation de récompense** — elle masquerait exactement l'effet de `vf_clip_param`, qui est l'objet du bras `D_vfclip`
- CPU uniquement, `num_learners=0`

### Environnements et budget

| | |
|---|---|
| Envs | `HalfCheetah-v5` (6 bras) · `Hopper-v5`, `Walker2d-v5` (`D` et `P` seulement) |
| Graines | 0, 1, 2 |
| Budget | pas d'environnement, **pas itérations** — les bras n'ont pas le même `train_batch_size` |

HalfCheetah porte l'ablation complète : retours les plus grands, et **pas de terminaison précoce**, donc une variable confondante en moins. Hopper et Walker2d contrôlent la généralisation de l'A/B.

### Garde-fous, lancés avant tout run

`benchmarks/test_arms.py` vérifie, sur la version de Ray installée :

1. **Les six défauts affirmés par le rapport sont exacts.** Si une montée de version les change, le test casse — et c'est le rapport qu'il faut corriger.
2. **Chaque bras d'ablation ne diffère de `D` que d'un seul champ.** Attrape la confusion accidentelle de deux facteurs.
3. Les constantes partagées le sont réellement, et `P` applique bien tous les réglages du papier.

Sortie sur `ray==2.57.0` : **40 contrôles, tous verts**. Les défauts `vf_clip_param=10.0`, `lambda_=1.0`, `use_kl_loss=True`, `clip_param=0.3`, `lr=5e-5`, `num_epochs=30` sont donc confirmés programmatiquement, pas seulement par lecture.

---

## 3. Prédictions, enregistrées avant le premier run

Une prédiction fausse est un résultat : elle corrige le rapport correspondant.

**P1 — `vf_clip_param` gèle le critique.**
Sur HalfCheetah, bras `D` : `vf_explained_var` reste sous 0,1 pendant tout le run. Bras `D_vfclip` : dépasse 0,5. Le rapport `vf_loss_unclipped / vf_loss` reste > 3 en `D` et vaut ≈ 1 en `D_vfclip`.

Mécanisme affirmé en [02 §4.3](02-ppo-papier-vs-rllib.md) : la perte de valeur est écrêtée à 10, donc le gradient du critique est **exactement nul** dès que

```math
\left\lvert V_\theta(s_t) - V_t^{\text{targ}} \right\rvert > \sqrt{10} \approx 3{,}16
```

Sur MuJoCo avec `gamma=0.99`, $`V`$ est de l'ordre de 100 dès le départ.

**P2 — `lambda_=1.0` coûte en efficacité-échantillon.**
`D_lambda` (λ = 0,95) atteint un seuil de retour donné avec **moins** d'échantillons que `D`. Le seuil est fixé depuis la courbe du pilote, pas deviné.

**P3 — la KL cumulée au clipping ne sert à rien.**
`D_kl` ≈ `D`, pas de différence détectable sur trois graines. Le papier mesure la KL *seule* comme inférieure au clipping ; il ne dit rien de la KL *ajoutée* au clipping, qui est ce que fait RLlib.

**P4 — `clip_param=0.3` est un moins bon réglage que 0,2.**
`D_eps` ≥ `D`. Le papier mesure 0,2 (score 0,82) au-dessus de 0,3 (0,70).

**P5 — la config du papier bat les défauts.**
`P` > `D` sur les trois environnements.

**P6 — `lambda_=1.0` protège partiellement du critique gelé.** *(formulée après le pilote, avant que `D_lambda` n'ait tourné)*

Le pilote montre que `D` progresse malgré un critique qui ne converge pas. Mécanisme candidat : avec `lambda_=1.0`, GAE dégénère en Monte-Carlo ([03 §1.3](03-gae-papier-vs-rllib.md), éq. 18)

```math
\hat{A}_t = \sum_{l=0}^{\infty}\gamma^l r_{t+l} - V(s_t)
```

où $`V`$ n'intervient plus que comme **baseline**, jamais comme bootstrap. Un critique faux coûte alors de la variance, pas du biais. Les deux défauts de RLlib se compensent partiellement.

Si c'est vrai, `D_lambda` (λ = 0,95 **avec le clip toujours à 10**) devrait être **moins bon** que `D` : λ < 1 réintroduit une dépendance au critique, précisément celui que le clip empêche d'apprendre. Ce serait l'inverse de P2.

---

## 4. Conditions matérielles

Jetson Orin Nano, 6 cœurs, 8 Go de mémoire **unifiée** (~3,5 Go réellement libres), `ray==2.57.0`, `torch 2.5.0a0` (roue NVIDIA JetPack 6.1), `mujoco 3.12.0`, `gymnasium 1.2.2`.

Trois choix de débit, mesurés et non supposés. Aucun ne touche aux mathématiques de l'optimisation : ils sont identiques sur les six bras.

| Levier | Résultat | Décision |
|---|---|---|
| `OMP_NUM_THREADS` 1 → 3 | 155 → **185 pas/s** (+19 %), sorties numériques **identiques** | 3 |
| Learner sur GPU | `NvMapMemAllocInternalTagged: error 12` — OOM sur la mémoire unifiée | CPU |
| Ajouter des `EnvRunner` | inutile : voir ci-dessous | `num_env_runners=0`, 4 envs vectorisés |

### Le learner domine, et c'est déjà un résultat

Décomposition d'une itération du bras `D` sur HalfCheetah :

| | temps | part |
|---|---|---|
| `env_runner_sampling_timer` | 4,4 s | 17 % |
| `learner_update_timer` | 21,5 s | **83 %** |

Cause directe : les défauts `num_epochs=30` et `minibatch_size=128` sur un batch de 4000 font

```math
30 \times \frac{4000}{128} \approx 937 \text{ pas de gradient par itération}
```

là où la config du papier en fait 320 pour 2048 pas d'environnement, soit **1,5 fois moins par pas d'environnement**. Le défaut `num_epochs=30` de RLlib ne coûte donc pas seulement en fidélité au papier ([02 §4.5](02-ppo-papier-vs-rllib.md)) : il **triple le coût de calcul par échantillon collecté**. Et il rend l'ajout d'`EnvRunner` inutile — accélérer les 17 % ne sert à rien.

### Détails d'installation Jetson

Deux obstacles, tous deux couverts par les notes de la machine :

- la roue torch NVIDIA a un numéro de version malformé (`nv24.08` dans le nom de fichier, `nv24.8` dans les métadonnées) — `uv` ≥ 0.10 la refuse sans `UV_SKIP_WHEEL_FILENAME_CHECK=1` ;
- `ImportError: libcusparseLt.so.0` — corrigé par le pré-chargement en `RTLD_LOCAL` dans `.venv/.../_load_cusparselt.py`.

---

## 5. Résultats

### 5.1 Pilote — HalfCheetah, une seule graine

`D` contre `D_vfclip`, 300 000 pas, graine 0. **Une graine : aucune conclusion statistique, seulement un mécanisme et un ordre de grandeur.** Le sweep à trois graines tranche.

| | `D` (`vf_clip_param=10`) | `D_vfclip` (`inf`) |
|---|---|---|
| `vf_loss` sur tout le run | **7,1 – 9,9**, collée au plafond de 10 | 91 – 596, libre |
| `vf_loss_unclipped` | 758 → **967** (aucune tendance à la baisse) | 267 → 596 |
| ratio non-écrêtée / écrêtée | **34,3** (médiane), pointes à 110 | **1,0** exactement |
| `vf_explained_var` fin / max | 0,1 / 0,40 (passe plusieurs fois sous 0) | 0,3 / **0,71** |
| retour final | 116,6 | **178,0** |
| débit | 187 pas/s | 179 pas/s |

**Le mécanisme est confirmé, et de façon nette.** Dans `D`, la perte de valeur reste plaquée sous son plafond du début à la fin, et la perte non écrêtée **ne décroît jamais** — elle est plus élevée à 284 000 pas qu'à 4 000. Le critique ne converge pas. Le ratio de 34 dit combien de gradient est jeté : 97 % du signal d'erreur.

**Mais la prédiction P1, telle qu'écrite, est infirmée.** Les seuils étaient faux dans les deux sens : `D` monte à 0,40 (prédit : reste sous 0,1) et `D_vfclip` finit à 0,3 (prédit : dépasse 0,5). `vf_explained_var` est logué avec `window=1`, donc mesuré sur un seul minibatch : bien trop bruité pour un test à seuil sur une graine. **Le ratio non-écrêtée / écrêtée est la bonne métrique** — elle sépare 34 de 1,0 sans ambiguïté. Les prochaines prédictions porteront sur elle.

**Et une affirmation du rapport est trop forte.** [02 §4.3](02-ppo-papier-vs-rllib.md) dit que « le critique ne démarre jamais » et que « l'entraînement ne progresse simplement pas ». Faux : `D` passe de −282 à +117. Le critique, lui, ne converge effectivement pas — mais PPO progresse quand même. D'où P6.

Seuil retenu pour `samples_to_threshold` sur HalfCheetah : **retour de 100**. Un seuil de 0 ne discrimine pas (152 000 pas contre 160 000) : la phase initiale est portée par la politique, pas par le critique.

### 5.2 Sweep complet

> À compléter. 30 runs, trois graines.

---

## 6. Reproduire

```bash
cd benchmarks
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv sync

# garde-fous : doivent tous passer avant le moindre run
OMP_NUM_THREADS=3 .venv/bin/python test_arms.py

# pilote : calibre le débit et fixe le seuil depuis la courbe observée
OMP_NUM_THREADS=3 .venv/bin/python -m ppo_arms.run_sweep \
    --envs HalfCheetah-v5 --arms D,D_vfclip --seeds 0 --max-env-steps 300000

# sweep complet, reprise sans danger après interruption
OMP_NUM_THREADS=3 .venv/bin/python -m ppo_arms.run_sweep \
    --seeds 0,1,2 --max-env-steps 300000 --resume --run-id sweep1
```

Un fichier `benchmarks/logs/<run_id>_<env>_<arm>_seed<N>.jsonl` par run, une ligne par itération, toutes les métriques RLlib aplaties. Rien n'est écrasé.
