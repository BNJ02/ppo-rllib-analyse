# PPO en multi-agent dans Ray RLlib

**Code analysé** : `ray-project/ray` @ `7a5d7f1` (25/08/2026), new API stack
**Cadre de lecture** : taxonomie à trois axes de [`BNJ02/marl-rllib-sota`](https://github.com/BNJ02/marl-rllib-sota) §1.1
**Rapports liés** : [`02-ppo-papier-vs-rllib.md`](02-ppo-papier-vs-rllib.md) · [`03-gae-papier-vs-rllib.md`](03-gae-papier-vs-rllib.md)

Ce document ne refait pas le SOTA. Il répond à une question plus étroite : **qu'est-ce que le code de PPO fait, exactement, quand il y a plusieurs modules ?** Quelles lignes changent, ce qui est calculé par module et ce qui reste global, et quels effets de bord cela produit.

---

## 0. La réponse en une phrase

> **Il n'y a pas de « PPO multi-agent » dans RLlib.** Il y a *un* PPO, dont la perte est calculée indépendamment pour chaque module, dont les résultats sont **sommés en un scalaire unique**, et dont un seul `backward()` alimente **N optimiseurs Adam disjoints**.

Sur les trois axes :

| Axe | Position de PPO/RLlib par défaut |
|---|---|
| **1 — structure des récompenses** | **agnostique**. Aucune ligne du code PPO ne lit la structure de récompense. Coopératif, compétitif, mixte : même code. |
| **2 — ce qui est centralisé** | **IL** (Independent Learning) = IPPO. Le CTDE n'est pas fourni, mais le new stack le rend bon marché — §7. |
| **3 — partage de paramètres** | **configurable par `policy_mapping_fn`**, du full sharing (1 module) au no sharing (N modules), sans changer une ligne de PPO. |

> **En clair** : RLlib traite le multi-agent comme un problème de *routage*, pas d'algorithme. Les agents sont routés vers des modules, les modules ont chacun leur perte et leur optimiseur, et PPO ne sait même pas qu'il y en a plusieurs. Toute la richesse du MARL est déportée dans la fonction de mapping, dans l'environnement, et dans les connecteurs qu'on écrit soi-même.

C'est cohérent avec la conclusion du SOTA (« RLlib ne contient aucun algorithme MARL, il fournit une infrastructure multi-agent »). Ce rapport en donne la contrepartie côté code PPO.

---

## 1. Axe 1 — structure des récompenses : PPO ne la voit pas

**Recherche exhaustive dans le code de PPO** : les récompenses n'apparaissent qu'à un seul endroit du chemin d'entraînement, dans le calcul de GAE (`value_predictions.py`) :

```math
\delta^V_t = r_t + \gamma\,(1-\texttt{terminated}_t)\,V(s_{t+1}) - V(s_t)
```

et cela se passe **par module, indépendamment**. Il n'existe aucun mécanisme dans PPO qui :

- agrège les récompenses entre agents ;
- compare les récompenses de deux agents ;
- décompose une récompense commune (pas de VDN, pas de QMIX — cf. SOTA §9.2) ;
- calcule une contribution marginale (pas de COMA, pas de difference rewards).

**Conséquence directe** : le *credit assignment* — difficulté dominante du régime coopératif — est **entièrement hors du périmètre de PPO**. Si l'environnement donne `+10` à tout le monde, chaque module reçoit `+10` dans ses résidus TD, et chaque module apprend que ce qu'il faisait à ce moment-là était bien. Y compris le module qui ne faisait rien.

> **En clair** : PPO ne peut pas distinguer « j'ai contribué » de « j'étais là ». Il n'a aucun organe pour ça. C'est exactement le mur décrit au §5.6 du SOTA — l'exemple du drone kamikaze où s'abstenir domine strictement frapper. Aucun réglage de `clip_param`, `lambda_` ou `num_epochs` ne répare une action dominée : PPO optimise fidèlement la récompense qu'on lui donne, y compris quand elle est fausse.

Le seul levier que PPO offre vraiment sur cet axe est **indirect** : un critique mieux informé réduit la variance de $`\hat{A}_t`$ (§7). Il ne crée pas de signal — ce que la mesure du SOTA §12.2 confirme empiriquement (`vf_explained_var` −0.006 → +0.199, retour inchangé, p = 0.38).

### Le régime compétitif, lui, touche PPO directement

En compétitif (somme nulle, self-play), le problème n'est pas le credit assignment mais la **non-stationnarité**. Et là, un mécanisme de PPO joue vraiment : le ratio

```math
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
```

n'est valide que si les données viennent bien de $`\pi_{\theta_{old}}`$. En multi-agent, **la distribution de transitions perçue par le module A dépend aussi de la politique du module B**. Or PPO ne corrige que le décalage de A par rapport à lui-même. Le décalage induit par B n'est corrigé nulle part.

> **En clair** : le clipping de PPO est un frein sur *ma* dérive, pas sur celle de mon adversaire. Si B change vite, mes données périment même quand je n'ai pas bougé — et rien dans PPO ne le détecte. C'est la raison structurelle pour laquelle le self-play RLlib gèle les adversaires (SOTA §6) : un adversaire gelé restaure la stationnarité, faute de quoi le clipping protège contre le mauvais risque.

---

## 2. Axe 2 — ce qui est centralisé : IPPO par défaut

### 2.1 Ce que fait le code

`core/learner/learner.py:943-991` — `compute_losses` :

```python
loss_per_module = {}
for module_id in fwd_out:
    loss = self.compute_loss_for_module(
        module_id=module_id,
        config=self.config.get_config_for_module(module_id),
        batch=batch[module_id],
        fwd_out=fwd_out[module_id],
    )
    loss_per_module[module_id] = loss
return loss_per_module
```

La docstring est explicite : *« If the algorithm uses independent multi-agent learning (default behavior for RLlib's multi-agent setups), also only `compute_loss_for_module()` should be overridden, but it will be called for each individual RLModule »*.

Puis `core/learner/torch/torch_learner.py:183` :

```python
total_loss = sum(loss_per_module.values())
...
total_loss.backward()
```

```math
L_{\text{total}} = \sum_{m \in \mathcal{M}} L_m^{PPO}, \qquad L_m^{PPO} = \hat{\mathbb{E}}_{t \sim \mathcal{B}_m}\!\left[-L^{CLIP} + c_1\tilde{L}^{VF} - c_2 S\right] + \beta_m \overline{\mathrm{KL}}_m
```

> **En clair** : on additionne N pertes indépendantes en un seul nombre, on rétropropage une fois, et chaque optimiseur ne récupère que les gradients de ses propres paramètres. Tant que les modules ne partagent rien, c'est **strictement équivalent** à N entraînements PPO séparés — juste plus efficace (un seul graphe, un seul `backward`).

C'est **Independent Learning** au sens de l'axe 2, c'est-à-dire **IPPO** — et le SOTA rappelle à juste titre que ce n'est pas un homme de paille : IPPO égale ou bat MAPPO/QMIX sur une grande partie de SMAC.

### 2.2 Un optimiseur par module

`core/learner/learner.py:484-487` :

```python
for module_id in self.module.keys():
    if self.rl_module_is_compatible(self.module[module_id]):
        config = self.config.get_config_for_module(module_id)
        self.configure_optimizers_for_module(module_id=module_id, config=config)
```

et `torch_learner.py:126-136` : un `torch.optim.Adam` par module, enregistré avec `lr_or_lr_schedule=config.lr` — donc **le pas d'apprentissage est réglable par module**.

### 2.3 Où la somme cesse d'être innocente

`total_loss = sum(...)` est neutre **si et seulement si les paramètres des modules sont disjoints**. Dès qu'un tronc est partagé (SOTA §5.4), les gradients de tous les modules **s'accumulent** sur l'encodeur, et deux choses arrivent :

1. La magnitude du gradient sur le tronc **croît linéairement avec le nombre de modules** — pas de moyenne, une somme.
2. Si l'encodeur est un `nn.Module` classique attribut des têtes, il apparaît dans `p0.parameters()` **et** `p1.parameters()`, donc dans deux optimiseurs → **double mise à jour**. C'est exactement le piège que la liste Python `self._encoder_ref = []` du SOTA §5.4 contourne, et que l'exemple officiel résout à l'inverse par un optimiseur unique.

> **En clair** : la somme des pertes est le mécanisme qui rend l'encodeur partagé *possible* (les gradients de toutes les têtes remontent bien dans le tronc). C'est aussi ce qui le rend piégeux, parce que rien dans RLlib ne vérifie qu'un paramètre n'est enregistré qu'une seule fois.

### 2.4 Gel de modules : la centralisation partielle

`core/learner/learner.py:1206-1212` :

```python
for module_id in list(batch.policy_batches.keys()):
    if not self.should_module_be_updated(module_id, batch):
        del batch.policy_batches[module_id]
if not batch.policy_batches:
    return {}
```

`should_module_be_updated` lit `config.policies_to_train` (liste, ensemble, ou callable). Un module gelé disparaît du batch → aucune perte, aucun gradient. Puis, dans `ppo.py:455-470` :

```python
modules_to_update = set(learner_results[0].keys()) - {ALL_MODULES}
self.env_runner_group.sync_weights(..., policies=modules_to_update, inference_only=True)
```

Le module gelé n'apparaît pas dans `learner_results` → il n'est **pas resynchronisé**. Cohérent : un adversaire gelé doit rester gelé sur les EnvRunners aussi. Le `TODO (sven)` juste au-dessus signale que déduire la liste des résultats est un contournement, pas un design.

---

## 3. Axe 3 — partage de paramètres : tout passe par le routage

### 3.1 Le mapping est résolu une fois par agent et par épisode

`env/multi_agent_episode.py:1022-1042` :

```python
def module_for(self, agent_id: AgentID) -> Optional[ModuleID]:
    if agent_id not in self._agent_to_module_mapping:
        module_id = self._agent_to_module_mapping[agent_id] = \
            self.agent_to_module_mapping_fn(agent_id, self)
        return module_id
    else:
        return self._agent_to_module_mapping[agent_id]
```

Mise en cache dans `_agent_to_module_mapping`, et ce dictionnaire est **transporté** lors des découpages d'épisode (`agent_module_ids=` aux lignes 991 et 1640). Donc le mapping est stable non seulement sur la durée d'un épisode, mais aussi à travers les chunks d'un même épisode répartis sur plusieurs batchs.

> **En clair** : c'est ce qui rend le tirage `hash(episode.id_) % 2` du self-play alterné cohérent — un agent ne change pas de camp au milieu d'une partie, ni entre deux itérations d'entraînement portant sur la même partie.

### 3.2 Ce que le partage change dans PPO, mécaniquement

Soit $`N`$ agents, $`T`$ pas d'environnement collectés, $`|\mathcal{M}|`$ modules.

| | Full sharing ($`|\mathcal{M}|=1`$) | No sharing ($`|\mathcal{M}|=N`$) |
|---|---|---|
| Lignes dans le batch du module | $`N \times T`$ | $`T`$ |
| Paramètres totaux | $`P`$ | $`N \times P`$ |
| Optimiseurs Adam | 1 | $`N`$ |
| Termes dans `sum(loss_per_module)` | 1 | $`N`$ |
| Lignes par pas de gradient | `minibatch_size` | $`N \times`$ `minibatch_size` (§4.2) |
| Minibatchs pour `num_epochs` passes | $`\frac{N \cdot T \cdot K}{M}`$ | $`\frac{T \cdot K}{M}`$ |

Le tableau se lit dans les deux sens : le partage donne **N fois plus de données par réseau**, mais coûte **N fois plus de minibatchs** pour faire les mêmes `num_epochs`. Le budget de calcul du learner, à `num_epochs` constant, est **le même**. Ce qui change, c'est la quantité de données par paramètre — et c'est là qu'est le gain 1.7× mesuré au SOTA §5.2.

### 3.3 Le partage par rôle est gratuit

`config.get_config_for_module(module_id)` (`algorithm_config.py:1412-1444`) résout des overrides par module, mis en cache dans `_per_module_overrides`. Dans la perte PPO, **tous** les hyperparamètres suivants passent par cet objet et sont donc réglables par module :

`clip_param` · `vf_clip_param` · `vf_loss_coeff` · `entropy_coeff` · `use_kl_loss` · `kl_coeff` · `kl_target` · `use_critic` · `lr`

> **En clair** : un attaquant peut avoir `entropy_coeff=0.01` et `clip_param=0.2` pendant qu'un défenseur tourne à `entropy_coeff=0.0` et `clip_param=0.3`, dans le même `algo.train()`. Rien à écrire hors config.

**Mais pas tous** — voir §5.

---

## 4. Le chemin des données, en multi-agent

### 4.1 Le pipeline réel (ordre exact)

`PPOLearner.build()` (`ppo_learner.py:55-74`) **prépend** `AddOneTsToEpisodesAndTruncate` et **appende** `GeneralAdvantageEstimation` au pipeline construit par `AlgorithmConfig.build_learner_connector` (`algorithm_config.py:1252-1324`). `prepend` est un `insert(0, …)` (`connector_pipeline_v2.py:255`). D'où l'ordre effectif :

```
 AddOneTsToEpisodesAndTruncate     ← prepend : AVANT les pièces custom
 [vos pièces custom]               ← la colonne d'état global s'ajoute ici
 AddObservationsFromEpisodesToBatch
 AddColumnsFromEpisodesToTrainBatch
 AddTimeDimToBatchAndZeroPad
 AddStatesFromEpisodesToBatch
 AgentToModuleMapping              ← AgentID → ModuleID (multi-agent uniquement)
 BatchIndividualItems
 NumpyToTensor
 GeneralAdvantageEstimation        ← appelle compute_values(), par module
```

> **Précision par rapport au SOTA §5.5** : le tableau du SOTA place `[vos pièces]` en tête. C'est exact vis-à-vis des connecteurs par défaut, mais `AddOneTsToEpisodesAndTruncate` est prépendu **par PPOLearner après coup**, donc il passe *avant* les pièces custom. Conséquence concrète pour la recette MAPPO : quand `AddGlobalObservation` s'exécute, **les `SingleAgentEpisode` ont déjà leur pas fantôme**. `len(sae)` vaut donc $`T_{\text{réel}}+1`$.
>
> Le code du SOTA est correct parce qu'il utilise `len(sae)` de façon **cohérente** des deux côtés (slice des observations *et* argument `n` de `add_n_batch_items`). C'est précisément pourquoi le « piège 2 » (`operands could not be broadcast together with shapes (3489,) (2740,)`) se déclenche dès qu'on mélange une longueur brute et une longueur post-extension.

Deux ordres de grandeur à retenir : avant `AgentToModuleMapping`, le batch est indexé par `(episode_id, agent_id)` ; après, par `module_id`. Une pièce custom qui s'exécute avant doit donc parler AgentID — ce que fait `add_n_batch_items(batch, col, items, n, single_agent_episode)`.

### 4.2 Deux sémantiques d'itération dans la même mise à jour

Détail qui surprend à la lecture :

| Appelant | Méthode | Itère sur |
|---|---|---|
| `Learner.compute_losses` → `MultiRLModule._forward_train` | `for mid in batch.keys() if mid in self` | **les modules présents dans le batch** |
| `GeneralAdvantageEstimation.__call__` | `rl_module.foreach_module(...)` | **tous les modules enregistrés** |

`multi_rl_module.py:219-232` vs `multi_rl_module.py:346-352`. D'où la garde défensive dans le connecteur GAE :

```python
vf_preds = rl_module.foreach_module(
    func=lambda mid, module: (
        module.compute_values(batch[mid])
        if mid in batch and isinstance(module, ValueFunctionAPI)
        else None
    ),
    return_dict=True,
)
```

avec le commentaire `TODO (sven): We need to check here in the pipeline already, whether a module should even be updated or not… For now, we'll just check, whether mid is in batch and skip if it isn't.`

C'est la propriété que le SOTA §5.4 exploite (« il n'itère que sur les clés présentes dans le batch ») : un sous-module encodeur, jamais présent dans le batch, ne reçoit jamais de perte tout en recevant les gradients des têtes. Elle vaut pour `_forward_train`, **pas** pour `foreach_module`.

---

## 5. Ce qui est par module, et ce qui ne l'est pas

C'est la table la plus utile de ce rapport.

### Par module ✅

| Élément | Où |
|---|---|
| Perte PPO complète (surrogate, VF, entropie, KL) | `compute_loss_for_module`, appelé par module |
| Coefficient KL adaptatif $`\beta_m`$ | `curr_kl_coeffs_per_module` (`LambdaDefaultDict`, `ppo_learner.py:44-52`) |
| Scheduler d'entropie | `entropy_coeff_schedulers_per_module` (`ppo_learner.py:31-42`) |
| Optimiseur Adam + `lr` | `configure_optimizers_for_module` (`learner.py:484-487`) |
| `clip_param`, `vf_clip_param`, `vf_loss_coeff`, `use_critic`, `use_kl_loss`, `kl_target` | `config.get_config_for_module(module_id)` |
| Calcul GAE (passe avant du critique, récursion) | boucle `for module_id, module_vf_preds in vf_preds.items()` |
| **Standardisation des avantages** | idem — voir §6.2 |
| Gel / entraînement | `should_module_be_updated` |
| Métriques (`vf_explained_var`, `curr_kl_coeff`, `mean_kl_loss`, …) | `metrics.log_dict(..., key=module_id)` |

### Global ❌

| Élément | Où | Pourquoi c'est gênant |
|---|---|---|
| **`gamma`** et **`lambda_`** | `ppo_learner.py:71-74` : `GeneralAdvantageEstimation(gamma=self.config.gamma, lambda_=self.config.lambda_)` | un seul connecteur GAE pour tous les modules → **impossible de donner un $`\lambda`$ différent à deux rôles** |
| **`num_epochs`** | `ppo.py:428-434`, passé une fois à `learner_group.update()` | même nombre de passes pour tous |
| **`minibatch_size`** | idem | et il s'applique **par module** — §6.1 |
| **`shuffle_batch_per_epoch`** | idem | |
| `train_batch_size` / `count_steps_by` | `AlgorithmConfig` | §6.3 |
| `grad_clip` | `postprocess_gradients` | appliqué globalement |

> **En clair** : on peut donner à chaque rôle son propre pas d'apprentissage, sa propre bande de clipping et sa propre pénalité KL — mais pas son propre horizon temporel. Un agent « tactique » à décisions courtes et un agent « stratège » à décisions longues doivent partager le même $`\gamma`$ et le même $`\lambda`$. Contourner cela demande d'instancier plusieurs connecteurs GAE et de sous-classer `PPOLearner.build()`.

---

## 6. Effets de bord spécifiques au multi-agent

Cinq comportements que le code produit et que rien ne documente.

### 6.1 ⚠️ Données déséquilibrées entre modules → nombre d'époques inégal

`utils/minibatch_utils.py:85-96`, condition de boucle :

```python
while (
    self._num_total_minibatches == 0
    and min(self._num_covered_epochs.values()) < self._num_epochs
) or (...):
```

et, **à l'intérieur** de chaque minibatch, une extraction par module de `minibatch_size` lignes :

```python
for module_id, module_batch in self._batch.policy_batches.items():
    ...
    n_steps = self._minibatch_size
    while s + n_steps >= get_len(module_batch):
        ...
        self._num_covered_epochs[module_id] += 1
        if self._shuffle_batch_per_epoch:
            module_batch.shuffle()
```

Deux conséquences.

**(a) La taille effective du minibatch est $`|\mathcal{M}| \times`$ `minibatch_size`.** Avec 10 modules indépendants et `minibatch_size=128`, chaque pas de gradient porte sur **1280 lignes**, pas 128. Le défaut PPO `minibatch_size=128` a donc un sens très différent selon le nombre de modules.

**(b) La boucle s'arrête quand le module le PLUS FOURNI a fait `num_epochs` passes.** Les modules moins fournis, eux, bouclent bien plus souvent. Formellement, si $`n_m`$ est le nombre de lignes du module $`m`$ :

```math
\text{époques effectives}(m) \;\approx\; K \cdot \frac{\max_{m'} n_{m'}}{n_m}
```

> **En clair** : un agent qui n'agit qu'un pas sur dix (tour par tour, agent intermittent, manager hiérarchique) voit ses données **dix fois plus souvent** que les autres, sur les mêmes 30 époques nominales. Avec le défaut PPO `num_epochs=30`, cela fait 300 passes sur un petit jeu de données, à ratio $`\pi_\theta/\pi_{old}`$ figé. C'est du surapprentissage sur l'agent rare, et il ne produit **aucun avertissement**.

Configurations concernées : tour par tour (SOTA §3.2), hiérarchique (SOTA §1.2), agents qui meurent tôt, self-play où un camp joue moins. **Métrique de contrôle** : comparer `num_module_steps_trained` entre modules ; s'ils diffèrent d'un facteur > 2, le problème est là.

### 6.2 ⚠️ Les avantages sont standardisés **par module**

`connectors/learner/general_advantage_estimation.py:142-150`, à l'intérieur de la boucle par module :

```python
module_advantages = module_value_targets - module_vf_preds
module_advantages = (module_advantages - module_advantages.mean()) / max(
    1e-4, module_advantages.std()
)
```

```math
\hat{A}^{(m)} \leftarrow \frac{\hat{A}^{(m)} - \mathrm{mean}\big(\hat{A}^{(m)}\big)}{\max\big(10^{-4},\ \mathrm{std}(\hat{A}^{(m)})\big)}
```

La normalisation est **locale à chaque module**, jamais globale. Trois conséquences, dans l'ordre de gravité :

**(a) Elle détruit l'information d'échelle entre agents.** Un agent dont les avantages valent $`\pm 0{,}01`$ et un agent dont les avantages valent $`\pm 100`$ produisent, après normalisation, **exactement le même gradient**. En coopératif, si un rôle contribue marginalement et un autre massivement, PPO les traite à égalité.

**(b) Elle interagit avec le partage.** En full sharing, tous les agents sont dans un seul module → une seule normalisation, l'information relative entre agents **survit**. En independent learning, chaque agent est normalisé chez lui → elle **disparaît**. Le choix de l'axe 3 change donc silencieusement la fonction objectif optimisée.

**(c) En compétitif, elle est plutôt bénéfique** : gagnant et perdant sont recentrés séparément, ce qui évite qu'un module au retour structurellement plus faible reçoive des gradients systématiquement plus petits.

> **En clair** : la normalisation par module fait qu'un agent qui compte peu et un agent qui compte beaucoup poussent aussi fort. C'est le contraire de ce qu'on veut en coopératif, et plutôt ce qu'on veut en compétitif. Et ce n'est écrit nulle part.

### 6.3 `total_train_batch_size` compte des pas d'**environnement**, pas d'agent

`algorithm_config.py:465` : `self.count_steps_by = "env_steps"`. Dans `ppo.py:399-419`, cela sélectionne `max_env_steps=self.config.total_train_batch_size`.

Avec le défaut PPO `train_batch_size = 4000` et $`N`$ agents actifs à chaque pas :

```math
\text{lignes réelles dans le batch} \;=\; 4000 \times N
```

Avec 10 agents en full sharing, le module unique reçoit **40 000 lignes**, et `num_epochs=30` / `minibatch_size=128` donne $`30 \times 40000/128 \approx 9\,375`$ pas de gradient par itération.

Bascule disponible : `config.multi_agent(count_steps_by="agent_steps")` → `max_agent_steps`, et le batch contient alors 4000 lignes au total, réparties entre modules.

> **En clair** : le même `train_batch_size=4000` signifie 4000 lignes en single-agent et 40 000 en multi-agent à 10 agents. Le temps par itération explose sans qu'aucun réglage n'ait changé. C'est la première chose à vérifier quand un passage en multi-agent devient dix fois plus lent.

### 6.4 Les modules gelés paient quand même GAE en entier

Ordre des opérations, vérifié dans le code :

1. le pipeline de connecteurs s'exécute, **GAE compris** → `foreach_module` fait une passe avant du critique et la récursion GAE pour **chaque module présent dans le batch** ;
2. **ensuite seulement**, `learner.py:1206-1212` retire du batch les modules non entraînables ;
3. les minibatchs sont construits sur ce qui reste.

Donc en self-play avec une league de $`k`$ adversaires gelés, chaque itération dépense une passe avant complète du critique **plus** la récursion GAE Python (boucle `for t in reversed(...)`, non vectorisée) pour chacun d'eux — puis jette tout.

Coût $`O(k \cdot T)`$ en Python pur par itération, invisible dans les métriques. Contournement : filtrer plus tôt via un `ConnectorV2` custom placé avant GAE, ou accepter le coût si $`k`$ est petit.

### 6.5 Le pas fantôme est ajouté par épisode single-agent

`add_one_ts_to_episodes_and_truncate.py` itère via `self.single_agent_episode_iterator(episodes, agents_that_stepped_only=False)` et renomme les IDs :

```python
if isinstance(episodes[0], MultiAgentEpisode):
    for i, ma_episode in enumerate(episodes):
        ma_episode.id_ += "_" + str(i)
        for sa_episode in ma_episode.agent_episodes.values():
            sa_episode.multi_agent_episode_id = ma_episode.id_
...
sa_episode.id_ += "_" + str(i)
```

Donc : **un pas artificiel par agent**, pas un par pas d'environnement. Avec $`N`$ agents, $`N`$ lignes fantômes par épisode. Elles sont masquées dans la perte par `LOSS_MASK`, mais **elles entrent dans le calcul de moyenne et d'écart-type de la standardisation** (§6.2), qui s'exécute avant l'application du masque. Biais faible, croissant avec $`N`$ et décroissant avec la longueur d'épisode.

Le renommage d'ID est nécessaire pour que deux chunks du même épisode ne soient pas fusionnés lors du zero-padding RNN — le code le documente comme *« a little bit of a hack »*.

---

## 7. CTDE / MAPPO : pourquoi c'est le code de PPO qui rend la recette possible

Le SOTA §5.5 donne la recette. Voici la raison, côté code, pour laquelle elle est **structurellement propre** sur le new stack.

### 7.1 PPO ne calcule plus la valeur pendant le rollout

Sur l'ancien stack, la value function était évaluée sur l'EnvRunner et stockée dans `SampleBatch["vf_preds"]` ; d'où le passage obligé par `postprocess_trajectory()`, et l'exemple historique `centralized_critic.py` qui ne s'applique plus.

Sur le new stack, l'unique appel à `compute_values` sur le chemin d'entraînement est dans le **connecteur learner** :

```python
# general_advantage_estimation.py
vf_preds = rl_module.foreach_module(
    func=lambda mid, module: module.compute_values(batch[mid]) ...
)
```

Et le module PPO par défaut le confirme — `default_ppo_torch_rl_module.py`, `_forward` (rollout) ne produit **que** `ACTION_DIST_INPUTS` :

```python
def _forward(self, batch, **kwargs):
    output = {}
    encoder_outs = self.encoder(batch)
    ...
    output[Columns.ACTION_DIST_INPUTS] = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
    return output
```

alors que `_forward_train` ajoute `EMBEDDINGS`, et `compute_values` est une méthode séparée de l'API `ValueFunctionAPI`.

> **En clair** : acteur et critique sont physiquement séparés dans le temps. L'acteur tourne sur l'EnvRunner, avec la seule observation locale. Le critique tourne sur le Learner, sur le batch complet, où l'on peut lui donner ce qu'on veut. **Le CTDE strict n'est pas une contrainte à faire respecter, c'est le comportement par défaut** — il suffit de nourrir le critique différemment. Le SOTA a raison de dire que le CTDE est « gratuit » ; la raison est là.

### 7.2 Le point d'injection, et pourquoi il est unique

Toute colonne ajoutée au batch **avant** `GeneralAdvantageEstimation` est visible de `compute_values`. Le pipeline (§4.1) laisse exactement une fenêtre : les pièces custom, en tête (après `AddOneTs`). C'est le seul endroit où l'on dispose à la fois des objets `MultiAgentEpisode` (donc de tous les agents) et d'un batch encore indexé par AgentID.

```math
V^{\text{centralisé}}_\phi\big(\underbrace{o^1_t \oplus o^2_t \oplus \cdots \oplus o^N_t}_{\texttt{GLOBAL\_OBS}}\big) \quad\text{vs}\quad \pi_\theta\big(a^i_t \mid \underbrace{o^i_t}_{\texttt{OBS}}\big)
```

### 7.3 Ce que le code impose, et qui explique les deux pièges du SOTA

**Piège 1 (liste d'agents figée)** — GAE exige `module_value_targets.shape[0] == sum(episode_lens)` (assertion explicite, `general_advantage_estimation.py:140`). Un vecteur global de largeur variable produit d'abord une erreur `np.concatenate`, sinon une incohérence plus loin.

**Piège 2 (alignement des longueurs)** — `add_n_batch_items(batch, col, items, n, sa_episode)` doit produire exactement $`n`$ lignes pour l'épisode single-agent concerné, avec $`n`$ **incluant le pas fantôme** (§4.1). L'erreur `operands could not be broadcast together with shapes (3489,) (2740,)` est la soustraction `module_value_targets - module_vf_preds` qui échoue, deux lignes après.

**Conséquence non mentionnée dans le SOTA** : un critique centralisé partage nécessairement `gamma` et `lambda_` avec tous les autres modules (§5), puisqu'il n'y a qu'un seul connecteur GAE.

### 7.4 Ce que la mesure du SOTA dit, et que le code corrobore

`vf_explained_var` −0.006 → +0.199, retour inchangé (p = 0.38). Le code explique pourquoi c'est le résultat attendu : le critique centralisé n'intervient **que** dans $`V(s_t)`$, donc dans $`\delta_t`$, donc dans $`\hat{A}_t`$ — et $`\hat{A}_t`$ est **immédiatement renormalisé** (§6.2). Une meilleure estimation de $`V`$ réduit la variance de $`\hat{A}`$ ; la normalisation efface ensuite le changement d'échelle. Ce qui reste, c'est un gradient moins bruité — pas un gradient qui pointe ailleurs.

> **En clair** : le critique centralisé rend le signal plus net, pas plus riche. Si le signal lui-même est faux (récompense qui rend une action dominée, SOTA §5.6), le nettoyer ne sert à rien. D'où l'ordre des trois leviers du SOTA : récompense d'abord, critique ensuite.

---

## 8. Synthèse

### Ce que PPO/RLlib fait bien en multi-agent

- **Découplage total** entre l'algorithme et l'architecture multi-agent. Passer de IL à full sharing à partage par rôle ne touche pas une ligne de PPO.
- **Hyperparamètres par module** sur presque toute la perte (`lr`, `clip_param`, `kl_coeff`, `entropy_coeff`, `vf_*`, `use_critic`).
- **Gel de modules propre** : retirés du batch, exclus du gradient, non resynchronisés — le socle du self-play.
- **CTDE gratuit par construction** : l'acteur n'a jamais besoin de l'état global, le critique est appelé côté Learner.
- **Agents hétérogènes, intermittents, hiérarchiques** supportés sans contorsion.

### Ce qu'il faut savoir avant de lancer un run

| # | Comportement | Impact | Contrôle |
|---|---|---|---|
| 1 | `num_epochs` effectif $`\propto \max_m n_m / n_m`$ | surapprentissage sur les agents rares | comparer `num_module_steps_trained` entre modules |
| 2 | Avantages standardisés **par module** | l'échelle relative entre agents disparaît en IL | passer en full sharing si les agents sont comparables |
| 3 | `train_batch_size` en pas d'**environnement** | batch $`\times N`$, itérations $`\times N`$ | `count_steps_by="agent_steps"` |
| 4 | Minibatch effectif = $`|\mathcal{M}| \times`$ `minibatch_size` | pas de gradient bien plus gros qu'annoncé | ajuster `minibatch_size` selon $`|\mathcal{M}|`$ |
| 5 | Modules gelés : GAE calculé puis jeté | $`O(k \cdot T)`$ Python gaspillé par itération | filtrer avant GAE si la league est grande |
| 6 | `gamma` / `lambda_` non réglables par module | un seul horizon pour tous les rôles | sous-classer `PPOLearner.build()` |
| 7 | `sum(loss_per_module)`, pas de moyenne | gradient $`\times |\mathcal{M}|`$ sur un tronc partagé | vérifier `len(list(p0.parameters()))` |
| 8 | $`N`$ pas fantômes par épisode | biais faible dans la standardisation | négligeable sauf $`N`$ grand, épisodes courts |

À quoi s'ajoutent les écarts **non multi-agent** documentés dans les deux autres rapports, qui frappent chaque module : `vf_clip_param=10.0` qui gèle le critique, `lambda_=1.0` qui désactive GAE, KL cumulée au clipping.

### Configuration de départ raisonnable

```python
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(MonEnvMultiAgent)
    .multi_agent(
        policy_mapping_fn=lambda aid, ep, **kw: "shared",   # axe 3 : full sharing
        policies_to_train=None,                             # tous entraînés
        count_steps_by="agent_steps",                       # §6.3
    )
    .training(
        lambda_=0.95,                 # réactive GAE (cf. rapport GAE §4.1)
        vf_clip_param=float("inf"),   # débloque le critique (cf. rapport PPO §4.3)
        use_kl_loss=False,            # clipping seul, comme le papier
        clip_param=0.2,
        num_epochs=10,
        minibatch_size=128,
    )
)
```

Avec du partage par rôle, ajouter les overrides par module :

```python
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

config.rl_module(
    algorithm_config_overrides_per_module={
        "attacker": AlgorithmConfig.overrides(lr=3e-4, entropy_coeff=0.01),
        "defender": AlgorithmConfig.overrides(lr=1e-4, entropy_coeff=0.0),
    },
)
```

> Rappel : `gamma` et `lambda_` placés dans ces overrides **seront ignorés** — le connecteur GAE lit `self.config`, pas la config par module (§5).

---

## 9. Position finale sur les trois axes

```
Axe 1 — récompenses      PPO est aveugle. Le credit assignment se règle dans
                         l'environnement, pas dans l'algorithme. Seule
                         concession : un critique mieux informé réduit la
                         variance (pas le biais, pas le signal).

Axe 2 — centralisation   IL par défaut = IPPO, un backward, N optimiseurs.
                         CTDE : non fourni mais structurellement gratuit,
                         parce que compute_values() vit côté Learner.
                         CC (politique jointe) : possible mais dégénéré —
                         un seul module, un seul agent déguisé.

Axe 3 — partage          Entièrement piloté par policy_mapping_fn. Ni PPO
                         ni le Learner ne savent combien d'agents existent —
                         ils ne voient que des ModuleIDs. Le seul endroit
                         où le nombre d'agents transparaît vraiment est
                         MiniBatchCyclicIterator (§6.1) et le comptage
                         env_steps/agent_steps (§6.3).
```

> **En clair, la phrase à retenir** : RLlib fait du multi-agent en faisant N fois du single-agent dans le même processus, avec un routage propre et des hyperparamètres par module. Tout ce que le MARL a de spécifique — décomposition de valeur, contrefactuels, communication, modélisation d'adversaire — reste à écrire. Ce que RLlib fournit, et qui est loin d'être rien, c'est un endroit propre où l'écrire : le `ConnectorV2` juste avant GAE.
