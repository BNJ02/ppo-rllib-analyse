# PPO : du papier Schulman et al. (2017) à l'implémentation Ray RLlib

**Papier** : *Proximal Policy Optimization Algorithms*, John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (OpenAI), arXiv:1707.06347v2
**Code analysé** : `ray-project/ray` @ commit `7a5d7f1667f79a907c3106f9347d411285297219` (25/08/2026), répertoire `rllib/`
**Rapport lié** : [`03-gae-papier-vs-rllib.md`](03-gae-papier-vs-rllib.md) — le papier GAE, qui fournit le $`\hat{A}_t`$ que PPO consomme

---

## 0. PPO, c'est quoi ? (la version en français)

**PPO = Proximal Policy Optimization**, « optimisation de politique proximale ». *Proximale* = « qui reste dans le voisinage ».

Le problème à résoudre : en apprentissage par renforcement, on améliore une politique en poussant les probabilités des bonnes actions vers le haut. Mais les données dont on dispose ont été **collectées par l'ancienne politique**. Dès qu'on modifie trop la politique, ces données ne décrivent plus le comportement de l'agent, et la mise à jour devient une extrapolation hasardeuse — l'entraînement s'effondre.

> **En clair** : c'est comme corriger un tir d'artillerie à partir d'une seule observation. Une petite correction, c'est légitime. Une correction énorme fondée sur la même observation, c'est du hasard. Le gradient de politique naïf, lui, ne connaît pas cette limite : si l'avantage est grand, il pousse aussi loin qu'on le laisse aller.

**TRPO** (2015), l'algorithme précédent des mêmes auteurs, résolvait cela avec une **contrainte dure** : « améliore-toi, mais la divergence KL entre l'ancienne et la nouvelle politique doit rester sous $`\delta`$ ». Mathématiquement propre, mais coûteux (optimisation de second ordre, gradient conjugué, produit hessien-vecteur) et incompatible avec le dropout ou le partage de paramètres.

**L'idée de PPO** : obtenir le même effet avec de la simple descente de gradient du premier ordre. Au lieu d'interdire les grands pas par une contrainte, on **rend les grands pas inintéressants** en aplatissant l'objectif au-delà d'un seuil.

```math
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta)\hat{A}_t,\ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\big)\Big]
```

où $`r_t(\theta) = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}`$ mesure « de combien la nouvelle politique a changé d'avis sur cette action ».

> **En clair** : tant que la politique reste dans une bande de ±20 % autour de l'ancienne, l'objectif se comporte normalement. Au-delà, il devient **plat** : le gradient tombe à zéro, il n'y a plus rien à gagner à pousser plus loin. On ne l'interdit pas, on le rend sans intérêt. Le `min(...)` fait que ce plafonnement ne joue que dans le sens qui *avantagerait* la politique — on ne clippe jamais dans le sens qui la pénalise. D'où le nom du papier : c'est une **borne inférieure pessimiste** sur l'objectif réel.

Le reste de PPO est de la plomberie éprouvée : N acteurs collectent T pas en parallèle, on calcule les avantages par GAE, on fait K époques de SGD par minibatch sur ces données, puis on recopie les poids vers les acteurs et on recommence.

---

## 1. Le contenu formel du papier

### 1.1 L'objectif clippé (§3)

Avec $`r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{old}}(a_t \mid s_t)`$, donc $`r_t(\theta_{old}) = 1`$ :

```math
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta)\hat{A}_t,\ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\big)\Big] \tag{7}
```

Le comportement dépend du signe de l'avantage :

| Signe de $`\hat{A}_t`$ | Le clipping mord à | Interprétation |
|---|---|---|
| $`\hat{A}_t > 0`$ (bonne action) | $`r_t > 1+\epsilon`$ | on cesse de récompenser une hausse de probabilité déjà importante |
| $`\hat{A}_t < 0`$ (mauvaise action) | $`r_t < 1-\epsilon`$ | on cesse de récompenser une baisse de probabilité déjà importante |

> **En clair** : dans les deux cas, le gradient devient nul **du côté où la politique s'éloigne trop**. Mais s'il faut revenir en arrière (le ratio est déjà hors de la bande et il faut le ramener), le `min` laisse passer le gradient. C'est un frein asymétrique, pas un mur.

### 1.2 La pénalité KL adaptative (§4) — **variante alternative**

```math
L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat{A}_t - \beta\, \mathrm{KL}\big[\pi_{\theta_{old}}(\cdot|s_t),\, \pi_\theta(\cdot|s_t)\big]\right] \tag{8}
```

avec, après chaque mise à jour, $`d = \hat{\mathbb{E}}_t\big[\mathrm{KL}[\pi_{\theta_{old}}, \pi_\theta]\big]`$ et :

```math
\beta \leftarrow \begin{cases} \beta / 2 & \text{si } d < d_{targ} / 1{,}5 \\[4pt] \beta \times 2 & \text{si } d > d_{targ} \times 1{,}5 \end{cases}
```

> **En clair** : au lieu d'aplatir l'objectif, on taxe la distance à l'ancienne politique. Et comme personne ne sait choisir le taux de taxe $`\beta`$, on l'ajuste automatiquement : trop de mouvement → on double la taxe ; pas assez → on la divise par deux.

**Le papier est catégorique** : cette variante est présentée comme une **alternative**, testée, et **jugée inférieure** — *« we found that the KL penalty performed worse than the clipped surrogate objective, however, we've included it here because it's an important baseline »*.

| Réglage | Score normalisé moyen (Tab. 1) |
|---|---|
| Sans clipping ni pénalité | −0,39 |
| **Clipping, $`\epsilon = 0{,}2`$** | **0,82** |
| Clipping, $`\epsilon = 0{,}1`$ | 0,76 |
| Clipping, $`\epsilon = 0{,}3`$ | 0,70 |
| Meilleure KL adaptative ($`d_{targ}=0{,}01`$) | 0,74 |
| Meilleure KL fixe ($`\beta = 3`$) | 0,72 |

Retenir ces deux lignes : **$`\epsilon = 0{,}2`$ bat $`\epsilon = 0{,}3`$ (0,82 vs 0,70)**, et **le clipping bat la KL (0,82 vs 0,74)**. Les deux serviront au §4.

### 1.3 L'objectif combiné (§5)

```math
L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t\Big[L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t)\Big] \tag{9}
```

avec $`L_t^{VF} = \big(V_\theta(s_t) - V_t^{targ}\big)^2`$, erreur quadratique **sans aucun clipping**, et $`S`$ l'entropie de la politique.

> **En clair** : trois termes. (1) améliorer la politique, prudemment. (2) apprendre au critique à prédire les retours — c'est lui qui fournit les avantages. (3) une prime à l'indécision, pour que la politique ne se fige pas trop vite sur une seule action et continue d'explorer.

Précision du papier souvent négligée : $`c_1`$ n'a d'intérêt **que si politique et critique partagent des paramètres**. Dans les expériences MuJoCo ils ne les partagent pas, donc *« coefficient $`c_1`$ is irrelevant »* et *« we don't use an entropy bonus »* ($`c_2 = 0`$).

### 1.4 GAE tronqué (§5, éq. 11-12)

```math
\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1} \tag{11}
```

```math
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \tag{12}
```

C'est l'estimateur du papier GAE (arXiv:1506.02438), tronqué à l'horizon $`T`$ du rollout. Voir [`03-gae-papier-vs-rllib.md`](03-gae-papier-vs-rllib.md) pour son analyse détaillée.

### 1.5 Algorithme 1

```
pour itération = 1, 2, … :
    pour acteur = 1..N :
        exécuter π_{θ_old} dans l'environnement pendant T pas
        calculer les avantages Â_1, …, Â_T
    optimiser L par rapport à θ, K époques, minibatch de taille M ≤ NT
    θ_old ← θ
```

> **En clair** : la ligne qui compte est la dernière. $`\theta_{old}`$ n'est mis à jour **qu'une fois par itération**, à la fin. Pendant les K époques, le dénominateur du ratio reste figé sur la politique qui a réellement produit les données. C'est ce qui rend le ratio interprétable et le clipping légitime.

### 1.6 Hyperparamètres du papier

| | MuJoCo (Tab. 3) | Roboschool (Tab. 4) | Atari (Tab. 5) |
|---|---|---|---|
| Horizon $`T`$ | 2048 | 512 | 128 |
| Pas Adam | $`3\times10^{-4}`$ | KL-adaptatif | $`2{,}5\times10^{-4}\times\alpha`$ |
| Époques $`K`$ | 10 | 15 | 3 |
| Taille minibatch $`M`$ | 64 | 4096 | $`32\times8`$ |
| $`\gamma`$ | 0,99 | 0,99 | 0,99 |
| $`\lambda`$ (GAE) | **0,95** | **0,95** | **0,95** |
| Acteurs $`N`$ | 1 | 32 / 128 | 8 |
| $`\epsilon`$ (clip) | **0,2** | — | $`0{,}1\times\alpha`$ |
| $`c_1`$ (VF) | non pertinent | — | 1 |
| $`c_2`$ (entropie) | 0 | — | 0,01 |

Architecture MuJoCo : MLP $`2\times64`$, tanh, gaussienne diagonale à écart-type **libre** (indépendant de l'état), **pas de partage de paramètres** politique/critique. $`\alpha`$ décroît linéairement de 1 à 0 sur Atari.

---

## 2. Où PPO vit dans RLlib

Deux « API stacks » coexistent dans le dépôt. Seul le **new API stack** est maintenu ; c'est lui qu'il faut analyser.

| Fichier | Rôle | Statut |
|---|---|---|
| `rllib/algorithms/ppo/ppo.py` | `PPOConfig` + boucle `training_step()` | actif |
| `rllib/algorithms/ppo/ppo_learner.py` | montage du pipeline de connecteurs, schedulers, contrôleur KL | actif |
| `rllib/algorithms/ppo/torch/ppo_torch_learner.py` | **la perte PPO** (`compute_loss_for_module`) | actif |
| `rllib/algorithms/ppo/torch/default_ppo_torch_rl_module.py` | réseau par défaut (encodeur + tête pi + tête vf) | actif |
| `rllib/connectors/learner/general_advantage_estimation.py` | GAE + standardisation des avantages | actif |
| `rllib/utils/postprocessing/value_predictions.py` | récursion GAE (numpy) | actif |
| `rllib/connectors/learner/add_one_ts_to_episodes_and_truncate.py` | pas fantôme pour le bootstrap | actif |
| `rllib/algorithms/ppo/ppo_torch_policy.py`, `ppo_tf_policy.py` | ancienne API (Policy/RolloutWorker) | `@OldAPIStack`, TF refusé sur le new stack |

Notes sur l'état du dépôt à ce commit : `rllib/tuned_examples/` **n'existe plus**, et le `README.md` de PPO référence encore DDPPO, **supprimé** du dépôt. Seuls PPO et APPO subsistent comme variantes.

---

## 3. Correspondance avec le papier

### 3.1 Ratio et surrogate clippé — **conforme**

`ppo_torch_learner.py:74-95`

```python
logp_ratio = torch.exp(
    curr_action_dist.logp(batch[Columns.ACTIONS]) - batch[Columns.ACTION_LOGP]
)
...
surrogate_loss = torch.min(
    batch[Postprocessing.ADVANTAGES] * logp_ratio,
    batch[Postprocessing.ADVANTAGES]
    * torch.clamp(logp_ratio, 1 - config.clip_param, 1 + config.clip_param),
)
```

C'est l'équation (7) au caractère près, avec le ratio calculé en espace log pour la stabilité numérique :

```math
r_t(\theta) = \exp\big(\log\pi_\theta(a_t|s_t) - \log\pi_{\theta_{old}}(a_t|s_t)\big)
```

Point crucial : $`\log\pi_{\theta_{old}}`$ **n'est pas recalculé**. Il est enregistré au moment de l'échantillonnage (`Columns.ACTION_LOGP`, produit par l'EnvRunner) et transporté dans le batch. $`\pi_{\theta_{old}}`$ est donc bien la politique de collecte, figée pour toute l'itération — la sémantique exacte de l'Algorithme 1.

> **En clair** : RLlib ne garde pas une copie du réseau ancien. Il garde simplement, pour chaque action jouée, la probabilité qu'elle avait au moment où elle a été jouée. C'est suffisant, et bien moins coûteux.

Détail correct et subtil : la politique courante utilise `get_train_action_dist_cls()` tandis que la politique de collecte est reconstruite avec `get_exploration_action_dist_cls()` — la distribution réellement utilisée pour échantillonner est ainsi préservée.

### 3.2 Perte totale — **conforme à (9), plus un terme**

`ppo_torch_learner.py:108-121`

```python
total_loss = possibly_masked_mean(
    -surrogate_loss
    + config.vf_loss_coeff * vf_loss_clipped
    - (entropy_coeff * curr_entropy)
)
if config.use_kl_loss:
    total_loss += self.curr_kl_coeffs_per_module[module_id] * mean_kl_loss
```

soit, en formules :

```math
L^{RLlib} = \hat{\mathbb{E}}_t\Big[-L_t^{CLIP} + c_1 \tilde{L}_t^{VF} - c_2 S[\pi_\theta](s_t)\Big] + \beta\,\overline{\mathrm{KL}}
```

Les trois premiers termes sont l'opposé de l'éq. (9) — RLlib **minimise** là où le papier **maximise**. **Le quatrième terme n'est pas dans l'éq. (9)** : voir §4.1. Et $`\tilde{L}^{VF}`$ n'est pas $`L^{VF}`$ : voir §4.3.

### 3.3 GAE — **conforme, avec un raffinement absent du papier**

`utils/postprocessing/value_predictions.py` implémente la forme récursive de (11)-(12) :

```math
\hat{A}_t = \delta_t + \gamma\lambda \cdot \texttt{propagate}_t \cdot \hat{A}_{t+1}
```

```python
non_terminal = 1.0 - terminateds
propagate = non_terminal * (1.0 - truncateds)
td_residuals = rewards + gamma * non_terminal * next_state_values - values
for t in reversed(...):
    running_advantage = td_residuals[t] + gamma * lambda_ * propagate[t] * running_advantage
```

Le papier ne distingue pas **terminaison** (l'épisode est vraiment fini) et **troncature** (on a coupé la trajectoire, mais la suite existe). RLlib le fait, et c'est correct :

- `terminated[t]` → pas de $`s_{t+1}`$, le bootstrap $`\gamma V(s_{t+1})`$ est annulé dans $`\delta_t`$ ;
- `truncated[t]` → $`V(s_{t+1})`$ reste une prédiction **valide** dans $`\delta_t`$, mais la récursion GAE ne franchit pas la frontière.

> **En clair** : quand un épisode est coupé par une limite de temps, faire comme s'il n'y avait plus rien après apprend à l'agent que le temps qui passe est une punition — bug classique. RLlib l'évite.

Le mécanisme de bootstrap est astucieux : `AddOneTsToEpisodesAndTruncate` allonge chaque épisode d'un pas artificiel (dernière observation dupliquée, récompense nulle), ce qui permet de calculer **toutes** les $`V(s_t)`$ **et** les valeurs de bootstrap en **une seule passe avant** du critique. Un `LOSS_MASK` neutralise ensuite ce pas fantôme dans la perte.

### 3.4 Boucle K époques / minibatchs — **principe conforme, détail différent**

`ppo.py:391-449` échantillonne de façon synchrone jusqu'à `total_train_batch_size`, puis :

```python
learner_results = self.learner_group.update(
    episodes=episodes,
    num_epochs=self.config.num_epochs,
    minibatch_size=self.config.minibatch_size,
    shuffle_batch_per_epoch=self.config.shuffle_batch_per_epoch,
)
```

puis `sync_weights(..., inference_only=True)` vers tous les EnvRunners — c'est le `θ_old ← θ` de l'Algorithme 1.

La différence est dans `MiniBatchCyclicIterator` (`utils/minibatch_utils.py:56-175`) : il ne découpe **pas** le batch en $`\lceil NT/M \rceil`$ minibatchs disjoints par époque. Il fait avancer un curseur **circulaire**, remélange à chaque bouclage, et s'arrête quand chaque échantillon a été vu au moins `num_epochs` fois. Conséquence : **un minibatch peut chevaucher la frontière entre deux époques** (fin du batch mélangé + début du batch remélangé).

> **En clair** : équivalent en espérance, différent en pratique. Un échantillon peut apparaître deux fois dans le même minibatch au moment du bouclage, ce qui ne peut pas arriver avec un découpage classique.

### 3.5 Optimiseur

Adam (`core/learner/torch/torch_learner.py:128`), conforme au papier. Mais **pas de clipping de gradient** par défaut (`grad_clip=None`), et **pas d'annealing linéaire** de $`\alpha`$ — le schéma Atari (Tab. 5), où le pas d'apprentissage **et** $`\epsilon`$ décroissent de 1 à 0, doit être reconstruit à la main via un *schedule* sur `lr`, et n'est pas reproductible du tout pour $`\epsilon`$ (`clip_param` n'accepte pas de schedule).

---

## 4. Écarts par rapport au papier

### 4.1 ⚠️ Clipping **et** pénalité KL activés simultanément par défaut

`ppo.py:135-137` :

```python
self.use_kl_loss = True    # ← activé
self.kl_coeff = 0.2
self.kl_target = 0.01
```

L'objectif effectivement optimisé par défaut est donc :

```math
L^{CLIP} - c_1 L^{VF} + c_2 S - \beta\,\mathrm{KL}\big[\pi_{old},\pi_\theta\big]
```

c'est-à-dire **l'éq. (9) et l'éq. (8) additionnées**. Cette combinaison ne correspond à **aucune** des variantes évaluées dans le papier — et le papier mesure que la composante KL, seule, est **moins bonne** que le clipping (0,74 vs 0,82, Tab. 1).

> **En clair** : le papier propose deux freins et démontre que le premier est meilleur. RLlib installe les deux en même temps. Ce n'est pas absurde en soi — deux freins freinent —, mais ce n'est pas PPO tel que publié, et le comportement obtenu n'a été mesuré nulle part.

C'est un héritage historique : l'implémentation RLlib d'origine s'appuyait sur TRPO / $`L^{KLPEN}`$, et le clipping a été ajouté par-dessus sans retirer la KL. Le README l'assume à demi-mot (« *There are two formulations of PPO, which are both implemented in RLlib* ») mais ne dit pas qu'elles sont **actives ensemble**.

Incohérence interne au dépôt : APPO, dans le même répertoire, pose `use_kl_loss = False` par défaut.

#### Mesuré : le deuxième frein coûte un facteur 2,4

[05 §5.2](05-mesures.md), bras d'ablation ne changeant que `use_kl_loss`, HalfCheetah, 3 graines, 300k pas :

| | KL active (défaut) | `use_kl_loss=False` |
|---|---|---|
| retour final | 205 ± 106 | **496 ± 82** |
| pas pour atteindre un retour de 100 | 234 700 | **133 300** |

×1,6 de progrès au-dessus de l'aléatoire (×2,4 en rapport brut des retours, chiffre gonflé par le retour aléatoire de −270,7 — voir [05 §5.2](05-mesures.md)), −43 % d'échantillons, avec **séparation totale sur les trois graines** : le pire run sans KL (391) dépasse le meilleur run avec (354). C'est le deuxième levier le plus fort après `lambda_`.

La prédiction enregistrée avant les runs était « aucune différence détectable » — le papier ne mesure que la KL *seule*, jamais la KL *ajoutée*. Elle est infirmée : le deuxième frein n'est pas neutre, il coûte cher.

Pour reproduire le papier : `config.training(use_kl_loss=False, kl_coeff=0.0, clip_param=0.2)`.

### 4.2 ⚠️ Règle d'adaptation de $`\beta`$ aux constantes interverties

`ppo_torch_learner.py:162-166` :

```python
if kl_loss > 2.0 * config.kl_target:
    # TODO (Kourosh) why not 2?
    curr_var.data *= 1.5
elif kl_loss < 0.5 * config.kl_target:
    curr_var.data *= 0.5
```

```math
\text{RLlib :}\quad \beta \leftarrow \begin{cases} 1{,}5\,\beta & \text{si } d > 2\,d_{targ}\\ 0{,}5\,\beta & \text{si } d < 0{,}5\,d_{targ}\end{cases} \qquad\qquad \text{Papier :}\quad \beta \leftarrow \begin{cases} 2\,\beta & \text{si } d > 1{,}5\,d_{targ}\\ 0{,}5\,\beta & \text{si } d < d_{targ}/1{,}5\end{cases}
```

| | Seuil haut | Facteur haut | Seuil bas | Facteur bas |
|---|---|---|---|---|
| Papier §4 | $`1{,}5\,d_{targ}`$ | $`\times 2`$ | $`d_{targ}/1{,}5`$ | $`\div 2`$ |
| RLlib | $`2\,d_{targ}`$ | $`\times 1{,}5`$ | $`0{,}5\,d_{targ}`$ | $`\times 0{,}5`$ |

Les constantes 1,5 et 2 sont **interverties** entre seuil et facteur. De plus, le papier est **symétrique** ($`\times2`$ / $`\div2`$) alors que RLlib ne l'est pas ($`\times1{,}5`$ à la hausse, $`\times0{,}5`$ à la baisse) : le contrôleur relâche la contrainte plus vite qu'il ne la resserre.

> **En clair** : le régulateur réagit mollement quand la politique dérive trop, et énergiquement quand elle ne bouge pas assez. Ce n'est pas le comportement décrit dans le papier. Le `TODO (Kourosh) why not 2?` laissé dans le code montre que l'écart n'est pas assumé mais subi.

Deuxième différence, plus insidieuse : le papier met $`\beta`$ à jour **une fois par mise à jour de politique**, à partir de $`d`$ calculé sur **tout le batch**. RLlib le fait dans `after_gradient_based_update` (`ppo_learner.py:87-116`) à partir de la dernière valeur loggée de `mean_kl_loss` avec `window=1` — c'est-à-dire le KL du **dernier minibatch de la dernière époque**. Estimation nettement plus bruitée, sur 128 échantillons au lieu de 4000.

### 4.3 ⚠️ Clipping de la **perte** de valeur — absent du papier

`ppo_torch_learner.py:98-101` :

```python
vf_loss = torch.pow(value_fn_out - batch[Postprocessing.VALUE_TARGETS], 2.0)
vf_loss_clipped = torch.clamp(vf_loss, 0, config.vf_clip_param)   # défaut : 10.0
```

```math
\tilde{L}_t^{VF} = \min\Big(\big(V_\theta(s_t) - V_t^{targ}\big)^2,\ \texttt{vf\_clip\_param}\Big) \qquad \text{au lieu de} \qquad L_t^{VF} = \big(V_\theta(s_t) - V_t^{targ}\big)^2
```

Ce n'est **ni** le papier PPO (MSE nue), **ni** OpenAI baselines (qui clippe la **prédiction** autour de $`V_{old}`$, pas la perte), **ni** le papier GAE (qui utilise une région de confiance).

**Conséquence pratique**, et c'est le piège n°1 de PPO dans RLlib :

```math
\big|V_\theta(s_t) - V_t^{targ}\big| > \sqrt{10} \approx 3{,}16 \quad\Longrightarrow\quad \frac{\partial \tilde{L}^{VF}}{\partial \theta} = 0
```

> **En clair** : sur un environnement dont les retours sont de l'ordre de 100 ou 1000, le critique se trompe de beaucoup plus que 3,16 au démarrage. Sa perte est donc plafonnée et une grande part de son gradient est jetée.

RLlib le sait et émet un avertissement lorsque la récompense moyenne dépasse `vf_clip_param` — mais **uniquement sur l'ancienne API stack** (`ppo.py:549-561`). Sur le new API stack, il n'y a **aucun** avertissement.

#### Ce que la mesure a corrigé

Cette section affirmait auparavant que le critique « n'apprend jamais » et que « l'entraînement ne progresse simplement pas ». **Les deux sont faux**, et [05 §5.2](05-mesures.md) le mesure :

| affirmation | mesure (HalfCheetah, 3 graines, 300k pas) |
|---|---|
| gradient massivement écrêté | **vraie** — ratio perte non-écrêtée / écrêtée = **24,0** en `D`, exactement 1,0 sans le clip. 623 sur Hopper, 385 sur Walker2d |
| le critique n'apprend jamais | **fausse** — `vf_explained_var` atteint 0,7 en pic avec le clip actif |
| l'entraînement ne progresse pas | **fausse** — le retour passe de −270 à +205 |
| lever le clip corrige | **fausse** — `vf_clip_param=inf` seul donne 204 ± 44 contre 205 ± 106. Aucun effet |

L'écrêtage est donc bien réel et massif, mais il n'est **pas** l'écart le plus coûteux : `lambda_=1.0` (×2,3 de progrès) et la pénalité KL cumulée (×1,6) le dépassent tous les deux. Un fort ratio d'écrêtage ne prédit pas l'échec.

**Le clip et `lambda_=1.0` s'aggravent mutuellement.** La cible du critique est la cible TD(λ) ([03 §3.2](03-gae-papier-vs-rllib.md)) : avec λ = 1 elle dégénère en Monte-Carlo, donc loin de $`V`$ et à forte variance, donc l'erreur quadratique dépasse presque toujours 10. Avec λ = 0,95 le ratio tombe de 24,0 à 5,0 **sans toucher au clip**. C'est ce qui explique que lever le clip seul ne serve à rien.

Correctif : corriger `lambda_` d'abord ; `vf_clip_param=float("inf")` ensuite, ou normaliser les récompenses. `vf_explained_var` est un mauvais témoin — RLlib la logue avec `window=1`, donc sur un seul minibatch, bien trop bruitée. Surveiller le rapport `vf_loss_unclipped / vf_loss`.

### 4.4 Standardisation des avantages — absente du papier

`general_advantage_estimation.py:142-150` :

```python
module_advantages = module_value_targets - module_vf_preds
module_advantages = (module_advantages - module_advantages.mean()) / max(
    1e-4, module_advantages.std()
)
```

```math
\hat{A}_t \leftarrow \frac{\hat{A}_t - \mathrm{mean}(\hat{A})}{\max\big(10^{-4},\ \mathrm{std}(\hat{A})\big)}
```

Pratique standard héritée des baselines, mais absente des deux papiers. Deux conséquences :

1. **Elle interagit avec $`\epsilon`$.** Le clipping de PPO s'applique au ratio, mais le gradient est proportionnel à $`\hat{A}_t`$. En renormalisant $`\hat{A}`$, on change l'amplitude des pas, donc la vitesse à laquelle le ratio atteint la bande $`[1-\epsilon, 1+\epsilon]`$. Le même $`\epsilon=0{,}2`$ ne signifie plus la même chose selon que les avantages sont normalisés ou non.
2. **Portée et détails** : la normalisation est faite sur **tout le batch d'entraînement**, avant découpage en minibatchs, et **inclut les pas fantômes** de `AddOneTsToEpisodesAndTruncate` dans le calcul de moyenne/écart-type (le masque n'agit qu'au niveau de la perte). Biais faible mais réel.

Sur l'ancienne API stack, la même opération est faite ailleurs, dans la boucle d'algorithme : `standardize_fields(train_batch, ["advantages"])` (`ppo.py:495`).

### 4.5 Valeurs par défaut éloignées de celles du papier

| Hyperparamètre | Papier (MuJoCo) | RLlib défaut | Réf. |
|---|---|---|---|
| $`\epsilon`$ (`clip_param`) | **0,2** | **0,3** | `ppo.py:140` |
| $`\lambda`$ (`lambda_`) | **0,95** | **1,0** | `ppo.py:134` |
| pas Adam (`lr`) | **$`3\times10^{-4}`$** | **$`5\times10^{-5}`$** | `ppo.py:124` |
| époques $`K`$ (`num_epochs`) | 10 (15 Roboschool, 3 Atari) | **30** | `ppo.py:131` |
| taille minibatch $`M`$ | 64 | 128 | `ppo.py:132` |
| taille batch $`NT`$ | 2048 | 4000 | `ppo.py:126` |
| $`\gamma`$ | 0,99 | 0,99 ✓ | `algorithm_config.py:384` |
| $`c_2`$ (`entropy_coeff`) | 0 (MuJoCo) / 0,01 (Atari) | 0,0 ✓ | `ppo.py:139` |
| $`c_1`$ (`vf_loss_coeff`) | 1 | 1,0 ✓ | `ppo.py:138` |

Trois écarts méritent commentaire :

**$`\lambda = 1{,}0`$** — le plus lourd de conséquences. $`\lambda=1`$ réduit GAE au retour Monte-Carlo moins baseline (éq. 18 du papier GAE) : variance maximale, c'est-à-dire **exactement l'estimateur que le papier GAE a été écrit pour remplacer**. Le papier PPO utilise 0,95 dans toutes ses tables. Ce défaut vient de `AlgorithmConfig` et n'a jamais été spécialisé pour PPO.

**`num_epochs = 30` avec `lr = 5e-5`** — RLlib compense un pas d'apprentissage 6× plus petit par 3× plus de passes sur les mêmes données. Combiné à $`\epsilon = 0{,}3`$ (plus permissif que 0,2), le régime d'optimisation est nettement plus agressif en nombre de réutilisations du batch.

> **En clair** : le papier fait 10 passes prudentes avec des pas moyens. RLlib fait 30 passes avec des pas minuscules et une bande de sécurité plus large. Ça peut converger, mais c'est 3× plus de calcul par échantillon collecté, et ça s'éloigne du régime testé par les auteurs.

**$`\epsilon = 0{,}3`$** — alors que le papier mesure explicitement 0,2 comme meilleur que 0,3 (0,82 vs 0,70 sur le benchmark de la Tab. 1). C'est le seul écart où le papier fournit une mesure directe contredisant le défaut RLlib.

> **Mesuré, et le résultat ne se transporte pas.** [05 §5.2](05-mesures.md) a isolé ce seul champ : $`\epsilon = 0{,}2`$ donne 113 ± 40 contre 205 ± 106 pour le défaut — donc aucun bénéfice, plutôt une tendance défavorable, mais les intervalles se chevauchent et trois graines ne tranchent pas. Le papier mesure 0,2 > 0,3 *dans une config où $`\lambda = 0{,}95`$ et lr = 3e-4*. Transplanté seul dans les défauts RLlib, l'effet disparaît. Un réglage d'hyperparamètre tiré d'un papier ne se transporte pas isolément — il faut corriger $`\lambda`$ et la KL d'abord.

### 4.6 Architecture par défaut

| | Papier (MuJoCo) | RLlib défaut |
|---|---|---|
| couches cachées | $`2\times64`$ | **$`2\times256`$** (`fcnet_hiddens=[256, 256]`) |
| activation | tanh | tanh ✓ |
| têtes pi/vf | — | linéaires directes (`head_fcnet_hiddens=[]`) |
| partage politique/critique | **non** | **non** pour PPO ✓ |
| écart-type gaussien | **libre** (indépendant de l'état) | **dépendant de l'état** (`free_log_std=False`) |
| clipping de $`\log\sigma`$ | non | oui, $`\pm20`$ (`log_std_clip_param=20.0`) |

Le partage mérite une note. `DefaultModelConfig.vf_share_layers` vaut `True` globalement, mais PPO le force à `False` :

```python
# ppo.py:361-362
def _model_config_auto_includes(self):
    return super()._model_config_auto_includes | {"vf_share_layers": False}
```

**Conforme au papier** — et cela rend `vf_loss_coeff` sans effet réel, exactement comme le note le papier pour $`c_1`$.

En revanche `free_log_std=False` est un écart : le papier utilise des écarts-types **variables mais indépendants de l'état**, suivant [Sch+15b; Dua+16]. RLlib le propose (`free_log_std=True`) sans l'activer par défaut.

> **En clair** : chez RLlib, le réseau prédit à la fois où viser et à quel point hésiter, en fonction de l'état. Chez Schulman, il ne prédit que où viser ; le niveau d'hésitation est un paramètre global appris séparément. La seconde option est plus stable en contrôle continu, c'est pourquoi le papier la choisit.

### 4.7 Ce que RLlib ajoute, hors périmètre du papier

- **Multi-agent** : la perte est calculée par `module_id`, avec un `kl_coeff` et un scheduler d'entropie **par module** (`ppo_learner.py:31-52`).
- **Politiques récurrentes** : zero-padding, `SEQ_LENS`, `LOSS_MASK`, encodeurs LSTM/attention. Le papier mentionne le style RNN de [Mni+16] sans en donner la mécanique.
- **Deux axes de distribution** : `num_env_runners` (les $`N`$ acteurs du papier) **et** `num_learners` (parallélisme de gradient, absent du papier). Le batch total vaut `train_batch_size_per_learner × num_learners`.
- **Schedulers** sur `lr` et `entropy_coeff`.
- **Métriques** : `vf_explained_var`, `vf_loss_unclipped`, `curr_kl_coeff` — précisément les instruments qui permettent de diagnostiquer les écarts ci-dessus.
- **Drapeaux morts ou incohérents** : `use_gae` est sans effet sur le new stack, `use_critic=False` laisse un critique non entraîné piloter les avantages. Détaillé dans [`03-gae-papier-vs-rllib.md`](03-gae-papier-vs-rllib.md) §4.2-4.3.

### 4.8 Variante APPO

`rllib/algorithms/appo/appo.py` s'éloigne davantage du papier :

```python
self.vtrace = True         # correction hors-politique V-trace (IMPALA) — absente du papier
self.clip_param = 0.4      # vs 0.2
self.use_kl_loss = False   # KL désactivée ici — contrairement à PPO !
self.lambda_ = 1.0
self.target_worker_clipping = 2.0
self.use_circular_buffer = True
self.grad_clip = 40.0
```

PPO y devient un algorithme **hors-politique** : la correction V-trace remplace l'hypothèse « les données viennent de $`\pi_{old}`$ » sur laquelle repose tout l'argument du surrogate clippé. Le réseau cible et le tampon circulaire n'ont pas d'équivalent dans le papier. C'est une descendance d'IMPALA autant que de PPO.

> **En clair** : APPO n'attend plus que tous les acteurs aient fini pour apprendre — il apprend en continu sur des données légèrement périmées. Le ratio $`\pi_\theta/\pi_{old}`$ n'a alors plus la même signification, d'où V-trace pour corriger le décalage.

---

## 5. Synthèse

### Fidèle au papier
- Ratio de vraisemblance et objectif clippé, éq. (7) au caractère près.
- $`\pi_{\theta_{old}}`$ = politique de collecte, figée par itération, resynchronisée en fin d'itération.
- GAE (11)-(12), avec en plus une gestion terminated/truncated que le papier n'aborde pas.
- Perte combinée politique + valeur + entropie, éq. (9).
- Adam, $`K`$ époques de SGD par minibatch, structure de l'Algorithme 1.
- Pas de partage de paramètres politique/critique pour PPO.
- $`\gamma = 0{,}99`$, $`c_1 = 1`$, $`c_2 = 0`$.

### Écarts substantiels

| # | Écart | Gravité |
|---|---|---|
| 1 | **Clipping + pénalité KL cumulés** par défaut — objectif hybride jamais évalué dans le papier, dont la composante KL y est mesurée comme inférieure | élevée |
| 2 | **Clipping de la perte VF à 10** — hors des trois papiers ; annule le gradient du critique dès $`\|V-V^{targ}\|>3{,}16`$ | **critique** |
| 3 | **$`\lambda = 1{,}0`$** par défaut — désactive de fait GAE | élevée |
| 4 | **Règle $`\beta`$ aux constantes interverties** (2/1,5 au lieu de 1,5/2), asymétrique, estimée sur le dernier minibatch | moyenne |
| 5 | **Standardisation des avantages** — hors papier, modifie l'échelle effective de $`\epsilon`$ | moyenne |
| 6 | **$`\epsilon = 0{,}3`$** alors que le papier mesure 0,2 meilleur ; **lr $`5\times10^{-5}`$** vs $`3\times10^{-4}`$ ; **30 époques** vs 10 | moyenne |
| 7 | **Architecture** $`256\times256`$ vs $`64\times64`$ ; écart-type dépendant de l'état vs paramètre libre | faible |
| 8 | **Minibatchs circulaires** pouvant chevaucher deux époques | faible |

### Recette pour se rapprocher du papier (MuJoCo, Tab. 3)

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

config = (
    PPOConfig()
    .environment("HalfCheetah-v5")
    .env_runners(num_env_runners=1, rollout_fragment_length=2048)
    .training(
        # --- Objectif : clipping seul, comme la section 3 du papier ---
        use_kl_loss=False,
        kl_coeff=0.0,
        clip_param=0.2,          # Tab. 1 : 0.2 (0.82) > 0.3 (0.70)

        # --- GAE, Table 3 ---
        lambda_=0.95,            # au lieu de 1.0 : réactive GAE
        gamma=0.99,

        # --- Optimisation, Table 3 ---
        lr=3e-4,
        num_epochs=10,
        minibatch_size=64,
        train_batch_size_per_learner=2048,

        # --- Fonction de valeur ---
        vf_clip_param=float("inf"),   # neutralise le clipping de perte, absent du papier
        vf_loss_coeff=1.0,
        entropy_coeff=0.0,            # le papier n'utilise pas de bonus d'entropie sur MuJoCo
    )
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[64, 64],
            fcnet_activation="tanh",
            vf_share_layers=False,
            free_log_std=True,        # écart-type indépendant de l'état, comme le papier
        ),
    )
)
```

**Restent non reproductibles sans code supplémentaire** :

- la standardisation des avantages (câblée dans `GeneralAdvantageEstimation` ; il faut sous-classer le connecteur et remonter le pipeline dans `PPOLearner.build()`) ;
- le découpage exact des minibatchs (`MiniBatchCyclicIterator` n'est pas configurable) ;
- l'annealing linéaire de $`\epsilon`$ du protocole Atari (`clip_param` n'accepte pas de schedule).

**Métriques à surveiller pour vérifier que tout va bien** :

| Métrique | Ce qu'elle révèle |
|---|---|
| `vf_explained_var` | proche de 0 → le critique n'apprend pas (typiquement §4.3) |
| `vf_loss_unclipped` vs `vf_loss` | écart important → le clipping de perte mord |
| `curr_kl_coeff` | dérive vers le haut → la politique bouge trop vite |
| `mean_kl_loss` | comparer à `kl_target` |

---

## 6. Note de méthode

**Sur l'archive source du papier** : le `tar.gz` déposé sur arXiv ne contient **pas** les sources LaTeX. Il contient `ppo-min.pdf` et un fichier `ppo-arxiv.tex` de 220 octets qui se contente d'un `\includepdf[pages=1-last]{ppo-min.pdf}`. Les auteurs ont soumis un PDF pré-compilé enveloppé dans un `.tex` minimal. L'analyse a donc porté sur le PDF.

**Sur le code** : clone *sparse* de `ray-project/ray` au commit `7a5d7f1667f79a907c3106f9347d411285297219`, limité à `rllib/algorithms/ppo`, `rllib/core`, `rllib/connectors`, `rllib/utils`, `rllib/env`, `rllib/algorithms/appo`. Tous les numéros de ligne cités renvoient à ce commit. Le dépôt évoluant vite, ils dérivent : les noms de symboles restent le point d'entrée fiable.
