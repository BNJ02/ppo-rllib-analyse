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

### 5.2 Sweep complet — 30 runs, trois graines

30 runs sur 30 terminés, aucun échec. 300 000 pas d'environnement par run, graines 0/1/2, `PPOConfig()` de Ray 2.57.0 pour référence.

#### HalfCheetah-v5 — l'ablation complète

| bras | levier | retour final | pas → retour 100 | ratio non-écrêtée / écrêtée | score normalisé |
|---|---|---|---|---|---|
| `D` | défauts RLlib | 205 ± 106 | 234 700 | 24,0 | 0,265 |
| `D_eps` | `clip_param=0.2` | 113 ± 40 | 242 700 | 24,9 | 0,214 |
| `D_vfclip` | `vf_clip_param=inf` | 204 ± 44 | 225 300 | **1,0** | 0,264 |
| `D_kl` | `use_kl_loss=False` | 496 ± 82 | 133 300 | 35,8 | 0,427 |
| `D_lambda` | `lambda_=0.95` | 813 ± 262 | 144 000 | 5,0 | 0,603 |
| `P` | config du papier | **1526 ± 365** | **70 300** | 1,0 | **1,000** |

Score normalisé à la manière de la Table 1 du papier : politique aléatoire = 0, meilleur bras observé = 1. Il classe **les bras entre eux**, sur la même version de MuJoCo — il ne se compare à aucun chiffre du papier (voir §1).

#### Hopper-v5 et Walker2d-v5 — contrôle de généralisation

| env | bras | retour final | pas → retour 100 | ratio | `vf_explained_var` |
|---|---|---|---|---|---|
| Hopper | `D` | 370 ± 70 | 18 700 | **623** | 0,4 |
| Hopper | `P` | **1301 ± 105** | 8 900 | 1,0 | 0,9 |
| Walker2d | `D` | 313 ± 9 | 32 000 | **385** | 0,5 |
| Walker2d | `P` | **1324 ± 296** | 15 000 | 1,0 | 0,9 |

---

#### Verdicts sur les prédictions

**P5 — confirmée, sans ambiguïté.** `P` bat `D` sur les trois environnements et sur **les neuf graines**, sans le moindre chevauchement : le pire run de `P` (994) dépasse le meilleur run de `D` (452). Facteur 7,4× sur HalfCheetah, 3,5× sur Hopper, 4,2× sur Walker2d. Et sur les deux axes à la fois — 3,3× plus efficace en échantillons **et** 2× plus rapide en temps mur (380 contre 190 pas/s), parce que `num_epochs=30` avec `minibatch_size=128` fait 937 pas de gradient par itération là où le papier en fait 320.

**P2 — confirmée.** λ = 0,95 quadruple le retour (813 contre 205) et réduit de 39 % les échantillons nécessaires. Le mécanisme n'est pas celui que j'avais écrit : voir P6.

**P3 — infirmée, et c'est le résultat le plus inattendu.** J'avais prédit « aucune différence détectable ». La KL cumulée au clipping **coûte un facteur 2,4 sur le retour** (496 contre 205) et 43 % des échantillons. La séparation est totale sur les trois graines : le pire run de `D_kl` (391) dépasse le meilleur run de `D` (354). RLlib ajoute une pénalité que le papier présente explicitement comme une *alternative* au clipping, jamais comme un complément ([02 §4.1](02-ppo-papier-vs-rllib.md)) — et le coût est mesurable.

**P4 — infirmée, faiblement.** `clip_param=0.2` isolé n'apporte rien : 113 contre 205, donc plutôt **pire** que le défaut 0,3. Les intervalles se chevauchent (`D_eps` par graine : 62 / 117 / 160 ; `D` : 117 / 354 / 144) — à trois graines ce n'est pas concluant, seulement l'absence de tout bénéfice. Le papier mesure bien 0,2 (0,82) au-dessus de 0,3 (0,70), mais dans une config où λ = 0,95 et lr = 3e-4. Transplanté seul dans les défauts RLlib, l'effet ne survit pas. **Un réglage d'hyperparamètre issu d'un papier ne se transporte pas isolément.**

**P1 — la moitié confirmée, la moitié infirmée.**

La partie sur le ratio est exacte, et exactement aux valeurs annoncées : **24,0 en `D`** (prédit > 3), **1,000 en `D_vfclip`** (prédit ≈ 1). Sur Hopper et Walker2d le ratio monte à **623 et 385** : plus de 99,7 % du signal d'erreur du critique est jeté. L'écrêtage de la perte de valeur est donc bien réel, massif, et pire encore hors de HalfCheetah.

La partie sur `vf_explained_var` est fausse dans les deux sens : `D` ne reste pas sous 0,1 (pic à 0,7) et `D_vfclip` ne dépasse pas 0,5 (0,3 ± 0,1 en fin). La métrique est loguée avec `window=1`, donc mesurée sur un seul minibatch — trop bruitée pour un test à seuil.

**Et surtout : lever l'écrêtage ne change rien au résultat.** `D_vfclip` = 204 ± 44 contre `D` = 205 ± 106. Identiques. C'est le fait le plus important du sweep, et il contredit la hiérarchie des rapports.

**P6 — infirmée, et le mécanisme est l'inverse de celui que j'avais proposé.** Je prédisais que `D_lambda` serait *moins bon* que `D`, λ < 1 réintroduisant une dépendance à un critique empêché d'apprendre. `D_lambda` est **4× meilleur**.

Le chiffre qui l'explique est le ratio : **24,0 en `D`, 5,0 en `D_lambda`**. Changer λ, *sans toucher au clip*, divise par cinq la fraction de gradient jetée. La cible du critique dans RLlib est la cible TD(λ) et non le retour Monte-Carlo ([03 §3.2](03-gae-papier-vs-rllib.md)) :

```math
V_t^{\text{targ}} = V_{\text{old}}(s_t) + \hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)}
```

Avec λ = 1 elle dégénère en Monte-Carlo : loin de $`V`$, à forte variance, donc l'erreur quadratique dépasse presque toujours 10 et le clip mord. Avec λ = 0,95 la cible reste **proche de $`V`$**, l'erreur passe souvent sous le seuil, et le critique apprend — `vf_explained_var` monte de 0,1 à 0,7.

**Les deux défauts ne se compensent pas, ils s'aggravent mutuellement : `lambda_=1.0` est ce qui rend `vf_clip_param=10` mordant.** C'est pourquoi lever le clip seul ne sert à rien (`D_vfclip` ≈ `D`) alors que corriger λ seul suffit à faire tomber le ratio.

---

#### Ce que le sweep change dans les rapports

**1. `vf_clip_param` n'est pas l'écart le plus grave.** Il est spectaculaire à la mesure — 99 % du gradient du critique écrêté — et sans effet sur le retour quand il est corrigé seul. [02 §4.3](02-ppo-papier-vs-rllib.md) affirmait que « le critique ne démarre jamais » et que « l'entraînement ne progresse simplement pas » : les deux sont faux. `D` passe de −270 à +205, et son `vf_explained_var` atteint 0,7 en pic. Un fort ratio d'écrêtage **ne prédit pas** l'échec : `D_kl` a le pire ratio de HalfCheetah (35,8) et fait 2,4× mieux que `D` ; Hopper tourne à un ratio de 623 et apprend.

**2. Le classement mesuré des trois écarts testables est :**

| rang | écart | progrès au-dessus de l'aléatoire | échantillons pour un seuil |
|---|---|---|---|
| 1 | `lambda_=1.0` au lieu de 0,95 | ×2,3 | ×1,6 |
| 2 | pénalité KL cumulée au clipping | ×1,6 | ×1,8 |
| 3 | `clip_param=0.3` au lieu de 0,2 | ×0,8 (défavorable, non concluant) | ×1,0 |
| — | `vf_clip_param=10` | ×1,0 | ×1,0 |

**Sur quelle métrique.** « Progrès au-dessus de l'aléatoire » = $`(R_{\text{bras}} - R_{\text{aléatoire}}) / (R_{D} - R_{\text{aléatoire}})`$, la convention de la Table 1 du papier. Le **rapport brut** des retours finaux donne des chiffres plus flatteurs — ×4,0 et ×2,4 — mais il n'a pas de sens sur HalfCheetah, dont le retour aléatoire vaut −270,7 : le zéro de l'échelle y est arbitraire, et tout rapport brut est gonflé par ce décalage. Les deux nombres sont donnés partout dans ce document ; **c'est celui corrigé de l'aléatoire qui doit être cité.**

**3. La somme dépasse ses parties.** `P` (×3,8 au-dessus de l'aléatoire) fait mieux que le meilleur levier isolé (×2,3), et corrige en plus le réseau, le pas d'apprentissage et le nombre d'époques. Les leviers ne sont pas additifs et ne peuvent pas être classés indépendamment — ce que l'ablation un-facteur-à-la-fois montre justement en échouant à reconstituer `P`.

#### Limites

- **Trois graines.** Suffisant pour les séparations totales (P5, P3, P2), insuffisant pour P4.
- **300 000 pas**, non 1 M comme le papier. Aucun bras n'a convergé ; le classement pourrait bouger à budget plus long, en particulier pour `D` dont la courbe monte encore à l'arrêt.
- **Un seul environnement porte l'ablation.** Le ratio d'écrêtage varie d'un facteur 26 entre HalfCheetah (24) et Hopper (623) : le classement des leviers n'est pas garanti transférable.
- **Aucune comparaison aux chiffres du papier**, pour la raison donnée en §1.

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

Un fichier `benchmarks/logs/<run_id>_<env>_<arm>_seed<N>.jsonl` par run, une ligne par itération, toutes les métriques RLlib aplaties (323 clés). Rien n'est écrasé.

Puis, pour convertir les logs en CSV et régénérer tous les tableaux de la §5.2 :

```bash
.venv/bin/python export_csv.py --run-id sweep1
.venv/bin/python analyze.py --threshold 100
```

**Les résultats des 30 runs sont versionnés** sous [`benchmarks/results/runs.csv`](../benchmarks/results/runs.csv) — 2898 lignes, une par itération, 614 Ko. **Tous les chiffres de ce document sont recalculables à partir de ce seul fichier**, sans relancer les 13 heures de calcul :

```bash
.venv/bin/python analyze.py --threshold 100
```

Les JSONL bruts (45 Mo, 323 colonnes par ligne dont 16 utilisées) ne sont **pas** versionnés : ils resteraient dans l'historique git indéfiniment pour des colonnes que personne ne relit. `analyze.py --from-logs` les lit encore si vous relancez le sweep vous-même, et produit une sortie identique au octet près.

---

## 7. Conclusion — l'implémentation est-elle satisfaisante ?

### Le code est fidèle. La configuration par défaut ne l'est pas.

Il faut séparer deux questions que la formulation « est-ce que RLlib implémente bien PPO ? » mélange.

**L'objectif clippé est correct, ligne à ligne.** L'éq. (7) du papier se retrouve telle quelle dans `ppo_torch_learner.py`, y compris le `min` et le sens des deux branches ([02 §3](02-ppo-papier-vs-rllib.md)). La récursion GAE de l'éq. (16) est correcte, y compris la coupure aux frontières d'épisodes ([03 §3](03-gae-papier-vs-rllib.md)). Rien de ce qui a été mesuré ne remet cela en cause. **Ce n'est pas un bug d'implémentation.**

**Les valeurs par défaut, elles, coûtent cher.** Le plus simple est de regarder la seule courbe que tout entraînement RL affiche — `env_runners/episode_return_mean`, la somme des récompenses d'un épisode — aux mêmes budgets d'échantillons, moyennée sur les trois graines :

**HalfCheetah-v5**

| bras | 50k pas | 100k | 150k | 200k | 250k | 300k |
|---|---|---|---|---|---|---|
| `D` défauts | −297 | −218 | −89 | −36 | 131 | **303** |
| `D_kl` (KL désactivée) | −239 | −105 | 146 | 369 | 421 | **528** |
| `D_lambda` (λ=0,95) | −186 | −39 | 255 | 505 | 598 | **930** |
| `P` config du papier | −129 | 309 | 790 | 1051 | 1354 | **1757** |

**Hopper-v5** et **Walker2d-v5**

| env | bras | 50k | 100k | 150k | 200k | 250k | 300k |
|---|---|---|---|---|---|---|---|
| Hopper | `D` | 216 | 288 | 318 | 358 | 361 | **369** |
| Hopper | `P` | 319 | 465 | 636 | 835 | 1234 | **1689** |
| Walker2d | `D` | 233 | 266 | 302 | 310 | 312 | **318** |
| Walker2d | `P` | 299 | 381 | 603 | 899 | 937 | **1428** |

Trois choses se lisent directement sur ces lignes, et aucune n'a besoin d'un ratio :

1. **`P` est devant à tous les relevés, sur les trois environnements.** Pas de croisement, pas de rattrapage tardif, et aucun chevauchement entre graines sur les 9 runs.
2. **`D` plafonne sur Hopper et Walker2d** — 358 → 361 → 369 sur les 100 000 derniers pas — alors que `P` accélère encore à l'arrêt. Ce n'est pas un retard, c'est un plateau.
3. **`P` arrive plus tôt.** Le retour de 100 est franchi à 70 000 pas contre 235 000 pour `D` sur HalfCheetah. En temps réel sur la même machine : **3,1 minutes contre 20,4**, parce que `P` est aussi 2× plus rapide par pas (380 contre 190 pas/s, voir [§4](#4-conditions-matérielles)).

#### Et si l'on veut un seul chiffre

Les rapports de retours sont commodes mais fragiles, et il faut dire lequel on cite :

| | `lambda_=0.95` | KL désactivée | config `P` |
|---|---|---|---|
| retour final, rapport **brut** | ×4,0 | ×2,4 | ×7,4 |
| **progrès au-dessus de l'aléatoire** | ×2,3 | ×1,6 | ×3,8 |
| échantillons pour atteindre un retour de 100 | ×1,6 | ×1,8 | ×3,3 |

Le rapport **brut** divise deux retours finaux : 813 / 205 = 4,0. Il est trompeur ici, parce qu'une politique aléatoire sur HalfCheetah obtient **−270,7** : le zéro de l'échelle est arbitraire, et déplacer l'origine change le rapport à volonté.

Le **progrès au-dessus de l'aléatoire** corrige cela en mesurant le chemin parcouru depuis le point de départ réel — c'est la convention de la Table 1 du papier :

```math
\frac{R_{\text{bras}} - R_{\text{aléatoire}}}{R_{D} - R_{\text{aléatoire}}} = \frac{813 - (-270{,}7)}{205 - (-270{,}7)} = 2{,}3
```

Sur Hopper et Walker2d, dont le retour aléatoire est proche de 0, les deux rapports coïncident (×3,5 et ×3,6 ; ×4,2 et ×4,2) — c'est HalfCheetah seul qui gonflait le chiffre brut.

**Aucun de ces rapports n'est une « vitesse de convergence ».** Rien n'a convergé à 300 000 pas. La ligne « échantillons pour atteindre un retour de 100 » est ce qui s'en rapproche le plus, et c'est la seule à répondre à la question pratique : *combien de temps avant d'avoir quelque chose qui marche.*

> **En clair** : personne n'a écrit PPO de travers. Quelqu'un a laissé les réglages génériques d'`AlgorithmConfig` là où PPO avait besoin des siens, et le résultat par défaut est un algorithme qui apprend sept fois moins bien que ce que le même code sait faire.

### Oui, on peut faire nettement mieux, et le classement est mesuré

Par ordre d'effet décroissant sur HalfCheetah, chaque levier testé **isolément** :

| levier | correction | progrès au-dessus de l'aléatoire | pourquoi |
|---|---|---|---|
| `lambda_` | `1.0` → `0.95` | **×2,3** (brut ×4,0) | à λ=1 GAE dégénère en Monte-Carlo — l'estimateur que le papier GAE remplace ([03 §4.1](03-gae-papier-vs-rllib.md)) |
| `use_kl_loss` | `True` → `False` | **×1,6** (brut ×2,4) | RLlib cumule pénalité KL **et** clipping ; le papier les présente comme deux alternatives ([02 §4.1](02-ppo-papier-vs-rllib.md)) |
| `clip_param` | `0.3` → `0.2` | aucun bénéfice isolé | le réglage du papier ne se transporte pas hors de sa config d'origine ([02 §4.5](02-ppo-papier-vs-rllib.md)) |
| `vf_clip_param` | `10` → `inf` | aucun effet isolé | écrête pourtant 96 à 99,8 % du gradient du critique ([02 §4.3](02-ppo-papier-vs-rllib.md)) |

**Les deux premiers sont le vrai gisement.** Ce sont deux lignes.

### La config recommandée

```python
config = (
    PPOConfig()
    .training(
        lambda_=0.95,          # levier n°1 : x2,3 a lui seul
        use_kl_loss=False,     # levier n°2 : x2,4 a lui seul
        kl_coeff=0.0,
        clip_param=0.2,
        lr=3e-4,               # avec num_epochs=10 : voir ci-dessous
        num_epochs=10,
        minibatch_size=64,
        train_batch_size_per_learner=2048,
        vf_clip_param=float("inf"),
        entropy_coeff=0.0,
    )
    .rl_module(model_config=DefaultModelConfig(
        fcnet_hiddens=[64, 64], fcnet_activation="tanh",
        vf_share_layers=False, free_log_std=True,
    ))
    .env_runners(env_to_module_connector=lambda *a, **k: MeanStdFilter())
)
```

C'est le bras `P`, celui qui donne le ×3,8. Le `MeanStdFilter` sur les observations n'est pas dans le papier mais était **actif sur tous les bras**, référence comprise : ce n'est pas lui qui produit l'écart.

**Le couple `lr` / `num_epochs` mérite une note.** Les défauts RLlib (`lr=5e-5`, `num_epochs=30`) font **937 pas de gradient par itération** contre 320 pour le papier — trois fois le calcul pour chaque échantillon collecté, ce qui explique à soi seul l'écart de débit (190 contre 380 pas/s, [§4](#4-conditions-matérielles)). Ces deux réglages vont ensemble : baisser `num_epochs` sans monter `lr` sous-entraînerait.

### Ce qui reste non tranché

**La décomposition du gain est incomplète.** L'ablation n'a isolé que quatre champs. `P` change aussi `lr`, `num_epochs`, `minibatch_size`, la taille du batch et le réseau ([64,64] tanh + `free_log_std` contre [256,256] tanh + σ dépendant de l'état). **Impossible de dire quelle part du gain vient de l'optimisation ou de l'architecture** — il faudrait quatre bras de plus. Ce qui est certain : les leviers ne sont pas additifs, puisque `P` (×3,8) dépasse le meilleur levier isolé (×2,3).

**Le classement vaut à 300 000 pas.** Aucun bras n'a convergé et la courbe de `D` monte encore à l'arrêt. Un budget de 1 M pas — celui du papier — pourrait resserrer l'écart, sans qu'on sache de combien.

**Un seul environnement porte l'ablation.** Le ratio d'écrêtage varie d'un facteur 26 entre HalfCheetah (24) et Hopper (623) : le classement des leviers n'est pas garanti transférable, en particulier vers les environnements à récompenses de faible amplitude où `vf_clip_param=10` ne mord pas du tout.

### Ce que vous verrez concrètement dans vos propres logs

En appliquant les deux leviers principaux, voici les métriques qui bougent, dans l'ordre où vous les remarquerez.

**Dès les premières minutes — les itérations vont plus vite.** `num_epochs` passe de 30 à 10 : trois fois moins de pas de gradient par lot collecté. Mesuré ici : **190 → 380 pas/s**, une itération de 26 s tombe à 13 s. C'est visible immédiatement, avant même que la politique ait appris quoi que ce soit.

**Dans les premiers 10 % du budget — `env_runners/episode_return_mean` décolle plus tôt.** C'est le signal principal. Sur HalfCheetah, à 100 000 pas, les défauts sont encore à −218 pendant que la config du papier est déjà à +309. Vous ne verrez pas « la même courbe en plus rapide » : vous verrez une courbe qui quitte le plancher pendant que l'autre y reste.

**Sur toute la durée — la courbe ne plafonne plus.** C'est ce qui se voit le mieux sur Hopper et Walker2d : les défauts se figent autour de 310-370 et n'en bougent plus, alors que `P` monte encore quand on l'arrête. Si votre courbe actuelle a un plateau précoce que vous attribuiez à la tâche, il vaut la peine de vérifier que ce n'est pas `lambda_`.

**Dans les métriques du learner** — trois changements mécaniques, utiles comme vérification que les réglages ont bien pris :

| métrique | avant | après | ce que ça dit |
|---|---|---|---|
| `vf_loss_unclipped` / `vf_loss` | 24 à 623 | tombe vers 5, ou 1 avec `vf_clip_param=inf` | la part du gradient du critique qui n'est plus jetée |
| `vf_explained_var` | ~0,1 | ~0,7 | le critique explique enfin la variance des retours |
| `mean_kl_loss`, `curr_kl_coeff` | actives, β dérive | inertes | confirme que `use_kl_loss=False` est bien pris |

`vf_explained_var` est **bruitée** — RLlib la logue avec `window=1`, donc sur un seul minibatch. Elle sert d'indication de tendance, pas de test à seuil : c'est en partie ce qui a fait échouer la prédiction P1 (§5.2).

**Ce que vous ne verrez pas, et qui serait un signal d'alerte** : si `env_runners/episode_return_mean` s'effondre après le changement, le suspect est le couple `lr=3e-4` / `num_epochs=10`, pas `lambda_` ni la KL. Ces deux-là se règlent ensemble et dépendent de la taille du lot — voir le domaine de validité ci-dessous. Repartez alors des deux seuls leviers principaux, en gardant `lr` et `num_epochs` aux valeurs RLlib.

### Peut-on appliquer ça les yeux fermés sur un autre environnement ?

**Non.** Ce qui a été mesuré, ce sont trois tâches de locomotion continue MuJoCo, sur 300 000 pas. Voici ce que chaque levier suppose, et donc où il cesse de valoir.

| levier | transférable ? | ce dont l'effet dépend |
|---|---|---|
| `use_kl_loss=False` | **oui, largement** | ne dépend d'aucune propriété de l'environnement — c'est un terme d'objectif que le papier ne prévoit pas. Le seul risque à l'enlever est une politique qui bouge plus vite par pas, à surveiller via `mean_kl_loss` |
| `lambda_=0.95` | **oui, probablement** | recommandation quasi universelle de la littérature ($`\lambda \in [0{,}9\,;\,0{,}99]`$, papier GAE §5). L'effet devrait *diminuer* sur des épisodes courts, où Monte-Carlo n'est pas si bruité |
| `vf_clip_param=inf` | **dépend de l'échelle des récompenses** | ne mord que si $`(V - V^{targ})^2 > 10`$. Sur nos envs le ratio d'écrêtage va de 24 à 623 ; sur un environnement à récompenses de faible amplitude il vaudra 1 et le réglage sera sans objet |
| `clip_param=0.2` | **non** | mesuré comme ne se transportant pas, même entre deux configs du même environnement |
| `lr=3e-4` + `num_epochs=10` | **non, à retester** | c'est un couple. Le bon compromis dépend de la taille du batch et du bruit des gradients, donc de la tâche |
| réseau `[64,64]` tanh | **non** | dimensionné pour des observations MuJoCo de taille ~17. Sur des images ou des espaces bien plus grands, c'est trop petit |

**Trois angles morts complets** — rien de ce dépôt ne dit quoi que ce soit à leur sujet :

- **Actions discrètes.** Les trois envs sont continus. `free_log_std` n'a même pas de sens en discret, et l'écrêtage de la perte de valeur se comporte différemment sur des récompenses bornées.
- **Multi-agent.** Non testé. [04](04-ppo-multiagent-rllib.md) ajoute d'ailleurs ses propres écarts — époques effectives inégales entre modules, avantages standardisés par module — qui pourraient dominer ceux-ci.
- **Récompenses éparses ou de faible amplitude.** Tout le mécanisme de l'écrêtage repose sur l'échelle de $`V`$.

**La procédure sûre**, plutôt que de recopier la config : appliquer les deux premiers leviers, puis **mesurer sur sa propre tâche**. Deux runs suffisent à trancher, et le harnais de `benchmarks/` le fait déjà — un bras d'ablation, c'est une ligne dans `ARMS`.

Le diagnostic à regarder en premier, quel que soit l'environnement : le rapport `vf_loss_unclipped / vf_loss`. S'il vaut 1, `vf_clip_param` est hors sujet chez vous. S'il vaut 20 ou 600, il écrête, et `lambda_` est le premier réglage à corriger.

### Le résumé en trois phrases, pour transmettre

> RLlib implémente correctement les maths de PPO, mais `PPOConfig()` par défaut n'est pas la configuration du papier : il ajoute une pénalité KL au clipping, et il met `lambda_=1.0`, ce qui annule GAE. Sur trois tâches MuJoCo, trois graines, la configuration du papier progresse **3,8 fois plus** que les défauts pour un même budget d'échantillons, tout en tournant deux fois plus vite. Les deux lignes qui récupèrent l'essentiel sont `lambda_=0.95` et `use_kl_loss=False` — mais l'ampleur du gain dépend de la tâche, donc à vérifier sur la vôtre avant de généraliser.

Ce qu'il ne faut **pas** leur dire : « le critique de RLlib ne marche pas ». C'est ce que ce dépôt affirmait avant de mesurer, et c'est faux — voir [02 §4.3](02-ppo-papier-vs-rllib.md).

### La réponse courte

**L'implémentation est bonne, les défauts sont mauvais, et deux lignes récupèrent l'essentiel.** Le piège n'est pas dans le code de RLlib mais dans le fait que `PPOConfig()` a l'air d'être « PPO » alors que c'est PPO plus une pénalité KL, moins GAE, avec un critique en grande partie écrêté. Rien dans la documentation ni dans les logs ne le signale — sur le new API stack, même l'avertissement historique sur `vf_clip_param` a disparu ([02 §4.3](02-ppo-papier-vs-rllib.md)).
