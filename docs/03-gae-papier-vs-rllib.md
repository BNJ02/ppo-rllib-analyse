# GAE : du papier Schulman et al. (2016) à l'implémentation Ray RLlib

**Papier** : *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, Pieter Abbeel (UC Berkeley), ICLR 2016, arXiv:1506.02438v6
**Code analysé** : `ray-project/ray` @ `7a5d7f1`, principalement `rllib/utils/postprocessing/value_predictions.py` et `rllib/connectors/learner/general_advantage_estimation.py`
**Rapport lié** : [`02-ppo-papier-vs-rllib.md`](02-ppo-papier-vs-rllib.md)

---

## 0. GAE, c'est quoi ? (la version en français)

**GAE = Generalized Advantage Estimation**, « estimation généralisée de l'avantage ».

Le problème que résout ce papier tient en une phrase : **quand un agent reçoit une récompense, comment savoir quelle action l'a méritée ?** C'est le *credit assignment problem*. Si un robot bipède tombe à la seconde 12, est-ce à cause du pas qu'il vient de faire, ou d'un déséquilibre amorcé à la seconde 8 ?

Pour répondre, on veut mesurer, pour chaque action, **à quel point elle était meilleure que la moyenne** dans la situation où elle a été prise. C'est exactement la définition de la fonction d'avantage :

$$A^\pi(s_t, a_t) := Q^\pi(s_t, a_t) - V^\pi(s_t)$$

> **En clair** : $V^\pi(s_t)$ = « ce que je gagne en moyenne à partir d'ici en jouant normalement ». $Q^\pi(s_t,a_t)$ = « ce que je gagne si je joue *cette* action-là, puis normalement ». Leur différence $A$ dit si l'action était un bon ou un mauvais choix. Positif → on augmente sa probabilité. Négatif → on la diminue. Le gradient de politique n'a besoin de rien d'autre.

Sauf qu'on ne connaît pas $A$. Il faut l'**estimer** à partir de trajectoires observées, et là on tombe sur un dilemme classique :

| Comment estimer $A$ | Biais | Variance | Intuition |
|---|---|---|---|
| Somme des récompenses observées jusqu'à la fin (Monte-Carlo) | nul | énorme | on attribue à une action tout le bruit des 2000 pas suivants |
| Une seule étape : $r_t + \gamma V(s_{t+1}) - V(s_t)$ | fort | faible | on fait confiance à $V$, qui est un réseau approximatif donc faux |

> **En clair** : soit on mesure tout et le signal est noyé dans le bruit, soit on demande son avis à un réseau de neurones qui se trompe. Les deux échouent.

**L'idée de GAE** : ne pas choisir. On construit une **moyenne pondérée exponentiellement de tous les estimateurs intermédiaires** (1 étape, 2 étapes, 3 étapes, …), avec un paramètre $\lambda \in [0,1]$ qui règle le curseur. C'est exactement l'idée de TD($\lambda$) de Sutton, mais appliquée à l'*avantage* au lieu de la *fonction de valeur*.

Résultat, une formule d'une simplicité désarmante :

$$\boxed{\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l\, \delta^V_{t+l}} \qquad \text{avec} \quad \delta^V_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

> **En clair** : on calcule à chaque pas une petite « surprise » $\delta_t$ (l'écart entre ce qu'on attendait et ce qui s'est passé), et on additionne les surprises futures en les faisant décroître géométriquement au taux $\gamma\lambda$. Plus $\lambda$ est grand, plus on regarde loin dans le futur — plus on est fidèle mais bruité. Plus $\lambda$ est petit, plus on se fie au réseau de valeur — plus c'est stable mais biaisé.

C'est **cette formule** que RLlib implémente, et c'est ce que règle le paramètre `lambda_` de `PPOConfig`.

---

## 1. Le contenu formel du papier

### 1.1 Cadre et gradient de politique (§2)

Le papier se place dans un cadre **non actualisé** : l'objectif est $\sum_{t=0}^\infty r_t$, et $\gamma$ n'est **pas** une donnée du problème mais **un paramètre de l'algorithme servant à réduire la variance**. C'est une différence de point de vue importante et souvent ignorée.

Le gradient de politique s'écrit :

$$g = \mathbb{E}\left[\sum_{t=0}^{\infty} \Psi_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right] \tag{1}$$

où $\Psi_t$ peut être : la récompense totale, la récompense future, une version avec baseline, $Q^\pi$, $A^\pi$, ou le résidu TD. Le papier montre que $\Psi_t = A^\pi(s_t,a_t)$ **minimise quasiment la variance**.

> **En clair** : l'équation (1) dit « pousse le logarithme de la probabilité de l'action dans la direction de sa qualité ». Les six choix de $\Psi_t$ sont six façons de définir « qualité » ; toutes donnent le bon gradient en moyenne, mais avec des variances radicalement différentes. Prendre l'avantage, c'est prendre la moins bruitée.

### 1.2 Estimateurs $\gamma$-*just* (§3, Déf. 1 et Prop. 1)

Le papier introduit la notion technique d'estimateur **$\gamma$-just** : un $\hat{A}_t$ est $\gamma$-just s'il n'introduit **aucun biais** lorsqu'on l'utilise à la place de $A^{\pi,\gamma}$ dans (6).

$$\mathbb{E}\left[\hat{A}_t(s_{0:\infty}, a_{0:\infty}) \nabla_\theta \log \pi_\theta(a_t|s_t)\right] = \mathbb{E}\left[A^{\pi,\gamma}(s_t,a_t)\nabla_\theta \log \pi_\theta(a_t|s_t)\right] \tag{7}$$

> **En clair** : « $\gamma$-just » = « on a le droit de le substituer sans fausser le gradient ». C'est le critère de rigueur qui permet de dire lesquels des estimateurs qui suivent sont honnêtes et lesquels trichent.

### 1.3 La construction de GAE (§3, éq. 11-16)

Résidu TD, qui est déjà un estimateur d'avantage à une étape :

$$\delta^V_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

Si $V = V^{\pi,\gamma}$ exactement, alors $\mathbb{E}[\delta_t] = A^{\pi,\gamma}(s_t,a_t)$ — il est non biaisé (éq. 10). Sinon il est biaisé.

Les estimateurs à $k$ étapes, obtenus par somme télescopique :

$$\hat{A}_t^{(k)} := \sum_{l=0}^{k-1} \gamma^l \delta^V_{t+l} = -V(s_t) + r_t + \gamma r_{t+1} + \cdots + \gamma^{k-1} r_{t+k-1} + \gamma^k V(s_{t+k}) \tag{14}$$

> **En clair** : $\hat{A}^{(k)}$ = « je regarde les $k$ vraies récompenses, puis je demande au réseau de valeur de deviner le reste ». Plus $k$ est grand, moins on fait confiance au réseau, plus on encaisse de bruit. Le biais diminue avec $k$ parce que le terme $\gamma^k V(s_{t+k})$, seul porteur d'erreur, est de plus en plus écrasé.

GAE est la **moyenne exponentiellement pondérée** de tous les $\hat{A}^{(k)}$ :

$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} := (1-\lambda)\left(\hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2 \hat{A}_t^{(3)} + \cdots\right) = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta^V_{t+l} \tag{16}$$

Les deux cas limites :

$$\text{GAE}(\gamma, 0):\quad \hat{A}_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \tag{17}$$

$$\text{GAE}(\gamma, 1):\quad \hat{A}_t = \sum_{l=0}^{\infty}\gamma^l \delta_{t+l} = \sum_{l=0}^{\infty}\gamma^l r_{t+l} - V(s_t) \tag{18}$$

Le papier est explicite : **GAE($\gamma$,1) est $\gamma$-just quelle que soit la qualité de $V$, mais a une variance élevée** ; **GAE($\gamma$,0) est $\gamma$-just seulement si $V = V^{\pi,\gamma}$, donc biaisé en pratique, mais a une variance bien plus faible**.

### 1.4 $\gamma$ et $\lambda$ ne jouent pas le même rôle (§3, fin)

Passage central du papier, et le plus souvent mal compris :

> *« $\gamma$ most importantly determines the scale of the value function $V^{\pi,\gamma}$, which does not depend on $\lambda$. Taking $\gamma < 1$ introduces bias into the policy gradient estimate, regardless of the value function's accuracy. On the other hand, $\lambda < 1$ introduces bias only when the value function is inaccurate. Empirically, we find that the best value of $\lambda$ is much lower than the best value of $\gamma$. »*

> **En clair** : $\gamma$ décide de **l'horizon temporel du problème** — il dit « au-delà de $1/(1-\gamma)$ pas, j'arrête de me soucier des conséquences ». Il biaise toujours. $\lambda$ décide de **combien on fait confiance au critique** — s'il était parfait, $\lambda$ serait gratuit. Ce sont deux boutons distincts, et c'est pour ça que l'optimum est typiquement $\gamma \approx 0.99$ mais $\lambda \approx 0.95$.

### 1.5 Interprétation par *reward shaping* (§4)

Le papier réinterprète GAE comme une somme actualisée de récompenses **remodelées**. Avec la transformation de Ng et al. (1999) :

$$\tilde{r}(s,a,s') = r(s,a,s') + \gamma \Phi(s') - \Phi(s) \tag{20}$$

qui laisse $A^{\pi,\gamma}$ inchangée. En posant $\Phi = V$, la récompense remodelée **est** le résidu de Bellman $\delta^V$, et :

$$\sum_{l=0}^{\infty}(\gamma\lambda)^l \tilde{r}(s_{t+l}, a_{t+l}, s_{t+l+1}) = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta^V_{t+l} = \hat{A}_t^{\text{GAE}(\gamma,\lambda)} \tag{25}$$

> **En clair** : GAE, c'est jouer au jeu original mais avec les récompenses réécrites par le critique, puis appliquer un second actualisateur $\gamma\lambda$ plus agressif. Le critique « rapproche » temporellement les récompenses de leurs causes ; le $\lambda$ coupe ensuite le bruit des dépendances longues devenues inutiles. C'est la meilleure justification intuitive du papier : $\lambda$ n'est pas un hack, c'est un second horizon appliqué à un problème déjà simplifié.

Le papier formalise cela avec la **fonction de réponse** :

$$\chi(l; s_t, a_t) = \mathbb{E}[r_{t+l} \mid s_t, a_t] - \mathbb{E}[r_{t+l} \mid s_t] \tag{26}$$

qui décompose l'avantage dans le temps : $A^{\pi,\gamma}(s,a) = \sum_l \gamma^l \chi(l;s,a)$. Un $\chi$ non nul pour $l \gg 0$ = un vrai problème d'attribution à long terme.

### 1.6 Estimation de la fonction de valeur (§5)

Cible Monte-Carlo, régression simple (éq. 28) :

$$\min_\phi \sum_{n=1}^{N} \left\| V_\phi(s_n) - \hat{V}_n \right\|^2, \qquad \hat{V}_t = \sum_{l=0}^{\infty}\gamma^l r_{t+l}$$

Mais le papier utilise en réalité une **région de confiance sur la fonction de valeur** (éq. 29) :

$$\min_\phi \sum_n \|V_\phi(s_n) - \hat{V}_n\|^2 \quad \text{s.c.} \quad \frac{1}{N}\sum_n \frac{\|V_\phi(s_n) - V_{\phi_{old}}(s_n)\|^2}{2\sigma^2} \le \epsilon$$

résolue par gradient conjugué avec approximation de Gauss-Newton du hessien.

> **En clair** : les auteurs ne laissent pas le critique bouger n'importe comment entre deux itérations. Ils lui imposent la même discipline qu'à la politique dans TRPO : « améliore-toi, mais reste proche de ta version précédente ». Sinon le critique se réajuste violemment sur le dernier batch (surapprentissage) et les avantages deviennent absurdes.

**Note de bas de page 2 — importante pour RLlib** : les auteurs mentionnent une cible alternative, la cible TD($\lambda$)

$$\hat{V}_t^\lambda = V_{\phi_{old}}(s_n) + \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta^V_{t+l}$$

et précisent l'avoir testée sans constater de différence de performance avec la cible $\lambda=1$ de l'éq. (28). **C'est exactement la cible qu'utilise RLlib** (§3.2 ci-dessous).

### 1.7 Algorithme complet (§6.1)

```
Initialiser θ₀ (politique) et φ₀ (valeur)
pour i = 0, 1, 2, … :
    simuler π_{θ_i} pendant N pas
    calculer δ_t^V pour tout t, en utilisant V = V_{φ_i}
    calculer Â_t = Σ_l (γλ)^l δ_{t+l}^V pour tout t
    mettre à jour θ_{i+1} par TRPO, éq. (31)
    mettre à jour φ_{i+1} par la région de confiance, éq. (30)
```

Remarque explicite du papier : la mise à jour de la politique utilise $V_{\phi_i}$ (**avant** mise à jour du critique), **pas** $V_{\phi_{i+1}}$. Raison donnée : si on surajustait d'abord le critique, les résidus de Bellman tomberaient à zéro et le gradient de politique s'annulerait.

> **En clair** : si le critique devine parfaitement le batch qu'on vient de collecter, alors « rien n'est surprenant », toutes les surprises $\delta$ valent 0, et l'agent n'apprend plus rien. Il faut donc mesurer les surprises **avant** de laisser le critique apprendre du batch.

### 1.8 Résultats expérimentaux

| Tâche | Meilleur $\gamma$ | Meilleur $\lambda$ |
|---|---|---|
| Cart-pole (21 seeds) | $[0.96,\ 0.99]$ | $[0.92,\ 0.99]$ |
| Bipède 3D (9 seeds) | $[0.99,\ 0.995]$ | $[0.96,\ 0.99]$ |
| Quadrupède 3D | 0.995 fixé | 0.96 meilleur que 1 |
| Bipède debout | 0.99 fixé | 0.96 ≈ 1 |

Conclusion du papier : *« choosing an appropriate intermediate value of $\lambda$ in the range $[0.9, 0.99]$ usually results in the best performance »*, et $\lambda = 0$ produit un **biais excessif et de mauvaises performances**.

Architecture : MLP 100-50-25, tanh, **même architecture pour politique et valeur mais réseaux séparés**.

---

## 2. Où GAE vit dans RLlib

| Fichier | Rôle |
|---|---|
| `rllib/utils/postprocessing/value_predictions.py` | `compute_value_targets()` — **la récursion GAE elle-même** |
| `rllib/connectors/learner/general_advantage_estimation.py` | connecteur : passe avant du critique, appel de la récursion, standardisation, remise dans le batch |
| `rllib/connectors/learner/add_one_ts_to_episodes_and_truncate.py` | pas de temps fantôme permettant le *bootstrap* en une seule passe |
| `rllib/algorithms/ppo/ppo_learner.py:55-74` | montage du pipeline : `AddOneTs…` en tête, `GeneralAdvantageEstimation` en queue |
| `rllib/algorithms/ppo/ppo.py:134` | `self.lambda_ = 1.0` |
| `rllib/algorithms/algorithm_config.py:384` | `self.gamma = 0.99` |

Le placement dans le pipeline n'est pas anodin : GAE est le **dernier** connecteur du Learner, après que le batch est complet, parce qu'il a besoin de toutes les observations pour faire une passe avant unique du critique.

---

## 3. Correspondance avec le papier

### 3.1 La récursion — **exacte, avec un raffinement absent du papier**

`utils/postprocessing/value_predictions.py` :

```python
non_terminal = 1.0 - terminateds
propagate = non_terminal * (1.0 - truncateds)

next_state_values = np.append(values[1:], 0.0)
td_residuals = rewards + gamma * non_terminal * next_state_values - values

advantages = np.zeros_like(rewards, dtype=np.float32)
running_advantage = 0.0
for t in reversed(range(td_residuals.shape[0])):
    running_advantage = (
        td_residuals[t] + gamma * lambda_ * propagate[t] * running_advantage
    )
    advantages[t] = running_advantage

return (advantages + values).astype(np.float32)
```

La ligne `td_residuals = ...` est $\delta^V_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, et la boucle arrière est la forme récursive de l'éq. (16) :

$$\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$$

C'est l'implémentation canonique, en $O(T)$ au lieu de $O(T^2)$.

**Ce que RLlib ajoute** : le papier se place dans un cadre où les épisodes se terminent (état absorbant) ou sont simplement découpés en batchs de $N$ pas. Il ne dit rien de la distinction entre :

- **terminaison** (`terminated`) : l'épisode est *vraiment* fini, il n'existe pas de $s_{t+1}$ ;
- **troncature** (`truncated`) : on a coupé la trajectoire (limite de temps, fin du rollout), mais $s_{t+1}$ existe bel et bien.

RLlib traite les deux différemment, et **c'est correct** :

| | $\delta_t$ garde le bootstrap $\gamma V(s_{t+1})$ ? | La récursion GAE franchit-elle la frontière ? |
|---|---|---|
| `terminated[t]` | non (`non_terminal = 0`) | non |
| `truncated[t]` | **oui** — $V(s_{t+1})$ est une prédiction légitime | non (`propagate = 0`) |

> **En clair** : quand un épisode est coupé par une limite de temps, faire comme s'il n'y avait plus rien après est une erreur classique qui apprend à l'agent que le temps qui passe est une punition. RLlib évite ce piège. Mais on ne propage pas la récursion au-delà de la coupure, car les surprises de l'épisode suivant n'ont rien à voir avec celui-ci.

Le mécanisme de bootstrap mérite un mot : `AddOneTsToEpisodesAndTruncate` allonge chaque épisode d'**un pas artificiel** (dernière observation dupliquée, récompense 0), ce qui permet de calculer $V(s_0), \ldots, V(s_T)$ **et** la valeur de bootstrap $V(s_{T+1})$ dans **une seule passe avant vectorisée**. Un `LOSS_MASK` neutralise ensuite ce pas fantôme dans la perte. Détail d'ingénierie, pas de théorie — mais c'est ce qui rend GAE bon marché sur des milliers d'épisodes.

### 3.2 La cible du critique — RLlib utilise la note de bas de page, pas le texte principal

`compute_value_targets` retourne :

```python
return (advantages + values)
```

soit :

$$\hat{V}_t^{\text{targ}} = \hat{A}_t^{\text{GAE}(\gamma,\lambda)} + V(s_t) = V_{\phi_{old}}(s_t) + \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta^V_{t+l}$$

C'est **littéralement la cible TD($\lambda$) de la note 2 du papier**, pas la cible Monte-Carlo de l'éq. (28) du texte principal.

> **En clair** : le papier régresse le critique sur « la somme réelle des récompenses observées ». RLlib le régresse sur « ta prédiction actuelle, corrigée par les surprises pondérées ». Les auteurs ont testé les deux et n'ont vu aucune différence de performance. La version RLlib est plus stable quand $\lambda < 1$ (elle hérite de la réduction de variance de GAE), et strictement équivalente à Monte-Carlo quand $\lambda = 1$.

Conséquence intéressante et rarement notée : **avec le défaut `lambda_ = 1.0` de RLlib, la cible du critique dégénère en retour Monte-Carlo pur**, donc RLlib retombe exactement sur l'éq. (28) du papier — mais par accident, pas par choix.

### 3.3 Ordre des opérations — **conforme à l'algorithme du papier**

Le papier insiste : calculer $\delta$ avec $V_{\phi_i}$, **avant** de mettre à jour le critique.

RLlib respecte cela structurellement : le connecteur `GeneralAdvantageEstimation` s'exécute dans le pipeline **avant** `compute_losses`, donc avec les poids du critique d'avant la mise à jour. Les avantages et cibles sont ensuite **figés** pour les 30 époques de SGD qui suivent.

> **En clair** : les avantages sont calculés une fois par itération, puis réutilisés tels quels pendant toutes les époques. Le critique bouge pendant ce temps, mais les avantages qu'il a produits ne bougent plus. C'est cohérent avec le papier, et c'est aussi ce qui rend légitime le ratio $\pi_\theta/\pi_{\theta_{old}}$ de PPO.

### 3.4 Deux paramètres distincts — conforme

RLlib expose bien $\gamma$ (`gamma`, dans `AlgorithmConfig`) et $\lambda$ (`lambda_`, dans `PPOConfig`) comme deux réglages indépendants, ce qui respecte l'analyse §3 du papier. La docstring RLlib de `lambda_` est d'ailleurs une reformulation correcte des éq. (17)-(18) :

> *« A `lambda_` of 0.0 makes the GAE rely only on immediate rewards (and vf predictions from there on, reducing variance, but increasing bias), while a `lambda_` of 1.0 only incorporates vf predictions at the truncation points of the given episodes or episode chunks (reducing bias but increasing variance). »*

Traduction en formules : $\lambda=0 \Rightarrow$ éq. (17) ; $\lambda=1 \Rightarrow$ éq. (18), où $V$ ne subsiste que comme baseline en $s_t$ (qui s'annule dans la cible) et comme bootstrap au point de troncature.

---

## 4. Écarts par rapport au papier GAE

### 4.1 ⚠️ `lambda_ = 1.0` par défaut — le défaut annule la contribution du papier

`ppo.py:134` :

```python
self.lambda_ = 1.0
```

$\lambda = 1$ est **précisément l'estimateur que le papier GAE a été écrit pour améliorer** : l'éq. (18), retour Monte-Carlo moins baseline, variance maximale. Toute la machinerie de l'éq. (16) est présente dans le code et **inactive** au réglage par défaut.

Rappel des mesures du papier : meilleur $\lambda \in [0.92, 0.99]$ sur cart-pole, $[0.96, 0.99]$ sur bipède 3D, et *« choosing an appropriate intermediate value of $\lambda$ in the range $[0.9, 0.99]$ usually results in the best performance »*. Le papier PPO utilise $\lambda = 0.95$ dans **toutes** ses tables.

> **En clair** : RLlib livre une voiture avec la boîte de vitesses installée mais bloquée en prise directe. Ça roule, mais on n'a rien de ce pour quoi on l'a achetée. Mettre `lambda_=0.95` est probablement le changement d'une seule ligne le plus rentable sur une config PPO RLlib par défaut.

C'est un héritage de `AlgorithmConfig` (valeur générique jamais spécialisée pour PPO), pas une décision documentée. `gamma = 0.99` est en revanche conforme.

### 4.2 ⚠️ `use_gae` est un paramètre mort sur le new API stack

La docstring annonce :

> *`use_gae` – If true, use the Generalized Advantage Estimator (GAE) with a value function, see https://arxiv.org/pdf/1506.02438.pdf*

En réalité, sur le new API stack, `use_gae` n'est lu **qu'une seule fois** dans tout le code, dans `PPOConfig.validate()` (`ppo.py:329`), pour une vérification de cohérence avec `batch_mode` :

```python
if (not self.in_evaluation
        and self.batch_mode == "truncate_episodes"
        and not self.use_gae):
    self._value_error("Episode truncation is not supported without a value function…")
```

Le connecteur `GeneralAdvantageEstimation` est ajouté **inconditionnellement** dans `PPOLearner.build()` (`ppo_learner.py:71-74`) — il ne consulte jamais `use_gae`. Poser `use_gae=False` ne désactive donc rien : cela relâche seulement une validation. Vestige de l'ancienne API stack, où le drapeau pilotait `compute_gae_for_sample_batch`.

### 4.3 ⚠️ `use_critic=False` produit un état incohérent

`use_critic` n'est consulté qu'à un seul endroit sur le new stack, `ppo_torch_learner.py:95`, où il met à zéro le terme de perte de valeur :

```python
if config.use_critic:
    value_fn_out = module.compute_values(...)
    ...
else:
    z = torch.tensor(0.0, device=surrogate_loss.device)
    value_fn_out = mean_vf_unclipped_loss = vf_loss_clipped = mean_vf_loss = z
```

Mais le connecteur GAE, lui, appelle `module.compute_values(...)` **quoi qu'il arrive**. Conséquence : avec `use_critic=False`, le réseau de valeur n'est **jamais entraîné** mais continue de **produire les avantages**. On calcule donc GAE à partir d'un critique resté à son initialisation aléatoire.

La docstring dit *« Should use a critic as a baseline (otherwise don't use value baseline; required for using GAE) »*, ce qui décrit le comportement de l'ancienne API. Le code contient d'ailleurs un `TODO (Kourosh) This is experimental. Don't forget to remove .use_critic from algorithm config.` (`ppo.py:252-254`).

> **En clair** : le drapeau censé dire « pas de critique » laisse en fait un critique non entraîné piloter tout le calcul des avantages. Ne pas l'utiliser tel quel. Le mode « sans fonction de valeur » du papier (courbes « No VF » des figures 2-4) n'a pas d'équivalent fonctionnel sur le new API stack.

### 4.4 Standardisation des avantages — absente du papier

`general_advantage_estimation.py:142-150` :

```python
module_advantages = module_value_targets - module_vf_preds
module_advantages = (module_advantages - module_advantages.mean()) / max(
    1e-4, module_advantages.std()
)
```

$$\hat{A}_t \leftarrow \frac{\hat{A}_t - \operatorname{mean}(\hat{A})}{\max(10^{-4},\ \operatorname{std}(\hat{A}))}$$

Le papier GAE n'en parle nulle part, et il utilise $\hat{A}$ brut dans le gradient. Deux remarques :

1. **Ce n'est pas anodin théoriquement.** Soustraire la moyenne est inoffensif (c'est une baseline constante, d'espérance nulle dans le gradient). **Diviser par l'écart-type l'est moins** : c'est un pas d'apprentissage adaptatif déguisé, qui change l'échelle du gradient d'un batch à l'autre. Et surtout, cela interagit avec le $\epsilon$ de PPO — le même $\epsilon=0.2$ ne veut plus dire la même chose selon que les avantages sont normalisés ou non.
2. **Elle est faite sur le batch complet**, avant découpage en minibatchs, et **inclut les pas fantômes** ajoutés par `AddOneTsToEpisodesAndTruncate` dans le calcul de moyenne/écart-type (le masque n'agit qu'au niveau de la perte). Biais faible (1 pas par épisode) mais réel.

Un effet secondaire notable : la standardisation **masque en partie l'effet de $\lambda$**. Avec $\lambda=1$, les avantages ont une variance énorme — mais après division par l'écart-type, ils ont l'air normaux. Le bruit, lui, est toujours là : ce sont les *écarts relatifs* entre échantillons qui sont bruités, et la normalisation ne les corrige pas. Cela contribue à rendre le mauvais défaut `lambda_=1.0` peu visible en pratique.

### 4.5 Pas de région de confiance sur la fonction de valeur

Le papier consacre sa §5 à optimiser le critique **sous contrainte de région de confiance** (éq. 29-30, gradient conjugué, hessien de Gauss-Newton). C'est présenté comme une des trois contributions du papier.

RLlib fait de la **descente Adam sur une MSE clippée** :

```python
vf_loss = torch.pow(value_fn_out - batch[Postprocessing.VALUE_TARGETS], 2.0)
vf_loss_clipped = torch.clamp(vf_loss, 0, config.vf_clip_param)   # défaut 10.0
```

$$L^{VF} = \min\left((V_\theta(s_t) - \hat{V}_t^{\text{targ}})^2,\ \texttt{vf\_clip\_param}\right)$$

Trois observations :

1. Le clipping de la **perte** (et non de la prédiction) n'est ni dans le papier GAE, ni dans le papier PPO, ni dans OpenAI baselines. C'est une invention RLlib.
2. Il ne remplace pas la région de confiance : une région de confiance limite le **déplacement** de $V_\phi$ par rapport à $V_{\phi_{old}}$ ; le clipping de perte limite l'**erreur prise en compte**. Ce n'est pas la même chose du tout.
3. Il crée un piège sévère : dès que $|V_\theta - \hat{V}^{targ}| > \sqrt{10} \approx 3{,}16$, le gradient du critique est **exactement nul**. Sur un environnement à récompenses d'ordre 100, le critique ne démarre jamais — et donc GAE tourne, comme au §4.3, avec un critique inutile.

> **En clair** : le papier freine le critique pour qu'il n'aille pas trop vite. RLlib, lui, lui bande les yeux dès qu'il se trompe de plus de 3,16. L'intention est la même (éviter que le critique déraille), le mécanisme est bien plus grossier, et l'effet de bord est qu'il peut ne jamais apprendre.

### 4.6 $\gamma$ : discount du problème, pas paramètre de variance

Le papier construit toute son analyse dans un cadre **non actualisé**, où $\gamma$ est un paramètre d'algorithme (« *we are not using a discount as part of the problem specification… it will appear below as an algorithm parameter that adjusts a bias-variance tradeoff* »).

RLlib traite $\gamma$ comme le facteur d'actualisation standard du MDP : il est dans `AlgorithmConfig`, partagé par tous les algorithmes (DQN, SAC, IMPALA…). Différence de cadrage conceptuel, sans conséquence sur le code, mais qui explique pourquoi la documentation RLlib ne suggère jamais de régler $\gamma$ pour réduire la variance.

### 4.7 Horizon infini vs épisodes réels

Les éq. (16)-(18) somment jusqu'à $l = \infty$. En pratique, la récursion arrière de RLlib s'arrête à la fin de chaque chunk d'épisode présent dans le batch, avec `propagate = 0` à la frontière. Autrement dit RLlib calcule un **GAE tronqué** — exactement ce que fait le papier PPO à son éq. (11), et ce que faisait déjà le papier GAE en pratique (batchs de $N$ pas). Conforme, mais il faut savoir que **la longueur du rollout borne l'horizon effectif de GAE** : avec `rollout_fragment_length` court et $\lambda$ élevé, on tronque des termes que $\lambda$ voulait garder.

---

## 5. Synthèse

### Fidèle au papier
- Éq. (16) implémentée exactement, sous forme récursive $O(T)$, avec $\delta^V_t$ conforme.
- $\gamma$ et $\lambda$ exposés comme deux paramètres indépendants, conformément à l'analyse §3.
- Avantages calculés avec le critique **d'avant** la mise à jour, comme l'exige l'algorithme §6.1.
- Cible du critique = cible TD($\lambda$) de la note 2 du papier (variante testée par les auteurs, jugée équivalente).
- $\gamma = 0.99$ par défaut, dans la plage optimale mesurée $[0.96, 0.995]$.
- Gestion terminated/truncated **plus rigoureuse** que le papier, qui n'aborde pas la question.

### Écarts substantiels
1. **`lambda_ = 1.0` par défaut** — dégénère GAE en Monte-Carlo (éq. 18), soit exactement l'estimateur que le papier cherche à remplacer. Le papier mesure l'optimum dans $[0.9, 0.99]$.
2. **`use_gae` est mort** sur le new API stack : lu uniquement dans `validate()`, sans effet sur le calcul.
3. **`use_critic=False` est incohérent** : la perte VF est neutralisée mais le connecteur GAE continue d'appeler le critique — avantages produits par un réseau jamais entraîné.
4. **Standardisation des avantages** — hors papier, modifie l'échelle du gradient, interagit avec $\epsilon$, et masque partiellement le coût d'un mauvais $\lambda$.
5. **Pas de région de confiance sur le critique** (§5 du papier) — remplacé par un clipping de la perte MSE à 10, mécanisme différent qui peut annuler complètement le gradient du critique.
6. **Cadrage de $\gamma$** : discount du MDP chez RLlib, paramètre de variance chez Schulman.

### Correctif minimal

```python
config.training(
    lambda_=0.95,               # GAE réellement actif — papier GAE + papier PPO
    gamma=0.99,                 # déjà le défaut, dans la plage optimale du papier
    vf_clip_param=float("inf"), # laisse le critique apprendre à toutes les échelles
)
# Ne pas toucher à use_gae (sans effet) ni à use_critic (incohérent sur le new stack).
```

Retirer la standardisation des avantages demande de sous-classer `GeneralAdvantageEstimation` et de remonter le pipeline à la main dans `PPOLearner.build()` — il n'y a pas de drapeau de config pour cela sur le new API stack.

---

## 6. Vue d'ensemble : comment les trois papiers s'emboîtent

```
TRPO (2015)          →  contrainte KL, garantie d'amélioration monotone
   │                     mais optimisation de second ordre, lourde
   │
GAE (2016)           →  COMMENT estimer Â_t  :  Σ (γλ)^l δ_t+l
   │                     règle le compromis biais/variance du critique
   │                     (+ région de confiance sur le critique)
   │
PPO (2017)           →  COMMENT utiliser Â_t :  min(r·Â, clip(r,1±ε)·Â)
   │                     remplace la contrainte KL par du clipping, premier ordre
   │
RLlib PPO (2026)     →  les deux, plus :
                         · pénalité KL cumulée au clipping (hors papier PPO)
                         · standardisation des avantages (hors papier GAE)
                         · clipping de la perte VF (hors des deux papiers)
                         · λ=1.0 par défaut (annule GAE)
                         · gestion terminated/truncated (meilleure que les deux)
```

> **En clair** : GAE répond à « quel chiffre mettre dans $\hat{A}_t$ », PPO répond à « quoi faire de ce chiffre ». Les deux sont orthogonaux — on peut faire du PPO sans GAE ($\lambda=1$, ce que fait RLlib par défaut) et du GAE sans PPO (le papier GAE utilise TRPO). RLlib implémente les deux correctement mais les livre configurés de façon à ce que la moitié de GAE soit inactive.

Voir [`02-ppo-papier-vs-rllib.md`](02-ppo-papier-vs-rllib.md) pour l'analyse de la partie PPO.
