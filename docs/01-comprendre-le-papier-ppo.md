# Comprendre le papier PPO

Guide de lecture de *Proximal Policy Optimization Algorithms* (arXiv:1707.06347). Objectif : que chaque équation du papier devienne évidente, et que les pièges de lecture classiques soient désamorcés.

À lire avec le PDF à côté. Les numéros d'équation sont ceux du papier.

---

## Ordre de lecture conseillé

Le papier n'est pas fait pour être lu linéairement.

| Ordre | Section | Pourquoi |
|---|---|---|
| 1 | §2.1 (gradient de politique) | le point de départ, 10 lignes |
| 2 | **§3 + Figure 1** | le cœur — si vous ne lisez qu'une chose |
| 3 | §5 éq. (9) | l'objectif réel qu'on code |
| 4 | §5 éq. (11)-(12) + Algorithme 1 | la boucle |
| 5 | Tables 3-5 (annexe A) | les vrais hyperparamètres |
| 6 | §6.1 + Table 1 | l'ablation qui justifie les choix |
| 7 | §4 | la variante KL — utile pour comprendre RLlib |
| 8 | §2.2 (TRPO) | contexte historique, sautable en première lecture |

---

## 1. D'où on part : le gradient de politique

$$\hat{g} = \hat{\mathbb{E}}_t\big[\nabla_\theta \log \pi_\theta(a_t|s_t)\,\hat{A}_t\big] \tag{1}$$

**Comment lire ça.** $\nabla_\theta\log\pi_\theta(a_t|s_t)$ est la direction dans l'espace des poids qui **augmente le plus vite** la probabilité de l'action $a_t$. On la multiplie par $\hat{A}_t$, un nombre signé qui dit si l'action était bonne.

- $\hat{A}_t = +3$ → on pousse fort dans la direction qui rend $a_t$ plus probable.
- $\hat{A}_t = -3$ → on pousse fort dans la direction opposée.
- $\hat{A}_t = 0$ → on ne bouge pas.

**Le piège de notation.** $\hat{\mathbb{E}}_t[\dots]$ n'est pas une espérance théorique : le papier le dit explicitement, c'est *« the empirical average over a finite batch of samples »*. Une moyenne sur le batch. Rien de plus.

En pratique on n'écrit jamais (1) : on écrit une **fonction de perte** dont l'autodiff donnera (1) en la dérivant :

$$L^{PG}(\theta) = \hat{\mathbb{E}}_t\big[\log\pi_\theta(a_t|s_t)\,\hat{A}_t\big] \tag{2}$$

C'est tout le sens de la phrase *« Implementations that use automatic differentiation software work by constructing an objective function whose gradient is the policy gradient estimator »*.

---

## 2. Pourquoi ça casse

Phrase clé du §2.1 :

> *« While it is appealing to perform multiple steps of optimization on this loss $L^{PG}$ using the same trajectory, doing so is not well-justified, and empirically it often leads to destructively large policy updates. »*

**La raison.** $\hat{A}_t$ a été mesuré sous $\pi_{\theta_{old}}$. Après un pas de gradient, la politique a changé : le nombre $\hat{A}_t$ ne décrit plus rien de la politique actuelle. Faire dix pas dessus, c'est répéter dix fois une correction fondée sur une seule observation.

> **Analogie** : corriger un tir d'artillerie. Une correction fondée sur une observation, c'est légitime. Dix corrections successives fondées sur la même observation, c'est du hasard.

**Pourquoi on veut quand même le faire.** Collecter des données coûte cher (simulation, robot réel). Ne faire qu'un pas de gradient par batch de 2048 transitions est un gâchis énorme. Tout PPO existe pour rendre les passes multiples **sûres**.

---

## 3. Le ratio : la pièce à comprendre en premier

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}, \qquad r_t(\theta_{old}) = 1$$

**Trois lectures possibles, toutes correctes :**

1. **Échantillonnage d'importance.** On veut évaluer la nouvelle politique avec des données de l'ancienne ; le ratio est le poids de correction standard.
2. **Mesure de changement d'avis.** $r_t = 1{,}3$ signifie « la nouvelle politique juge cette action 30 % plus probable ».
3. **Substitut du $\log$.** Autour de $\theta_{old}$, $\nabla_\theta r_t = \nabla_\theta \log\pi_\theta$. C'est pourquoi $L^{CPI} = \hat{\mathbb{E}}[r_t\hat{A}_t]$ a le **même gradient** que $L^{PG}$ au point de départ. Le papier le dit : *« $L^{CLIP} = L^{CPI}$ to first order around $\theta_{old}$ »*.

> **Le point 3 est celui qui débloque tout.** Au démarrage de chaque itération, $r_t = 1$ partout, donc PPO fait *exactement* ce que ferait un gradient de politique classique. Le clipping ne change rien **au premier pas** ; il ne se réveille que quand la politique commence à s'éloigner. PPO n'est pas un autre algorithme, c'est un gradient de politique **avec un frein**.

---

## 4. L'équation (7), décortiquée

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\Big[\min\big(\underbrace{r_t\hat{A}_t}_{A},\ \underbrace{\operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t}_{B}\big)\Big] \tag{7}$$

Prenons $\epsilon = 0{,}2$ et déroulons quatre cas concrets.

| Cas | $\hat{A}_t$ | $r_t$ | $A$ | $B$ | $\min$ | Gradient ? |
|---|---|---|---|---|---|---|
| Bonne action, pas encore trop poussée | $+2$ | 1,1 | 2,2 | 2,2 | 2,2 | ✅ oui |
| Bonne action, **trop** poussée | $+2$ | 1,5 | 3,0 | **2,4** | 2,4 | ❌ **nul** |
| Mauvaise action, pas encore trop réduite | $-2$ | 0,9 | −1,8 | −1,8 | −1,8 | ✅ oui |
| Mauvaise action, **trop** réduite | $-2$ | 0,5 | −1,0 | **−1,6** | **−1,6** | ❌ **nul** |

**Lisez la dernière ligne deux fois.** Avec $\hat{A}<0$, le clip **augmente** la valeur de $B$ ($-1{,}6 > -1{,}0$ en valeur), et c'est le `min` qui rattrape en choisissant $B$. Sans le `min`, on aurait pris $-1$ et l'objectif aurait *récompensé* le fait d'avoir trop réduit.

**La règle générale :** le gradient s'annule **uniquement** du côté où la politique s'est déjà trop éloignée *dans le sens qui l'arrange*. S'il faut revenir en arrière (le ratio est hors bande et il faut le ramener), le gradient passe.

> **Erreur de lecture n°1 :** croire que PPO *empêche* $r_t$ de sortir de $[1-\epsilon, 1+\epsilon]$. Faux. $r_t$ peut valoir 5. Ce qui est plafonné, c'est le **bénéfice** qu'on en tire.

**La Figure 1** est exactement ce tableau, tracé. Panneau gauche ($A>0$) : rampe qui devient plate à droite de $1+\epsilon$. Panneau droit ($A<0$) : plateau qui **descend** à gauche de $1-\epsilon$ — et cette descente est ce qui ramène la politique. Le point rouge à $r=1$ est le point de départ commun.

**La Figure 2** est plus subtile et souvent survolée : on interpole linéairement entre $\theta_{old}$ et le $\theta$ obtenu après une itération de PPO, et on trace les quatre objectifs le long de ce segment. On voit $L^{CPI}$ (orange) monter sans fin, et $L^{CLIP}$ (rouge) atteindre un **maximum** puis redescendre. C'est la preuve visuelle que $L^{CLIP}$ est une borne inférieure qui **s'auto-limite**. Le maximum tombe à KL ≈ 0,02, ce qui est aussi l'ordre de grandeur du $d_{targ}$ de la variante KL — les deux mécanismes visent bien la même chose.

---

## 5. L'objectif complet, et le piège de $c_1$

$$L^{CLIP+VF+S} = \hat{\mathbb{E}}_t\big[L^{CLIP} - c_1 L^{VF} + c_2 S[\pi_\theta]\big] \tag{9}$$

- $L^{VF} = (V_\theta(s_t) - V_t^{targ})^2$ : le critique apprend à prédire les retours. On en a besoin parce que c'est **lui qui fabrique les $\hat{A}_t$**.
- $S[\pi_\theta]$ : entropie. Prime à l'indécision, pour éviter que la politique se fige trop tôt.

> **Erreur de lecture n°2 :** croire que $c_1$ est un hyperparamètre important. Le papier écrit *« we don't share parameters between the policy and value function (so coefficient $c_1$ is irrelevant) »*. Deux réseaux séparés = deux optimiseurs = deux pertes indépendantes ; multiplier l'une par une constante ne change rien à la direction de descente. $c_1$ ne compte **que** si un tronc est partagé.

> **Erreur de lecture n°3 :** croire que le bonus d'entropie fait partie de PPO. Sur MuJoCo, $c_2 = 0$ — *« we don't use an entropy bonus »*. Il n'apparaît que sur Atari (Table 5, $c_2 = 0{,}01$), parce qu'un espace d'actions discret peut s'effondrer sur une action unique.

---

## 6. Les avantages : la partie que le papier ne développe pas

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \tag{11-12}$$

C'est GAE, publié séparément (Schulman et al., ICLR 2016, arXiv:1506.02438). Le papier PPO le cite et passe.

**Le minimum vital pour lire ces deux lignes :**

- $\delta_t$ est une « surprise » : ce que j'ai eu ($r_t$ + valeur de la suite) moins ce que j'attendais ($V(s_t)$).
- $\hat{A}_t$ additionne les surprises futures avec une décroissance $\gamma\lambda$.
- $\lambda = 0$ → une seule surprise : stable, mais biaisé par les erreurs de $V$.
- $\lambda = 1$ → toutes les surprises : non biaisé, mais très bruité (éq. 10, l'estimateur de A3C).
- $\lambda = 0{,}95$ (le choix du papier) → entre les deux.

> **Erreur de lecture n°4 :** confondre $\gamma$ et $\lambda$. $\gamma$ définit l'**horizon du problème** (jusqu'où les conséquences comptent). $\lambda$ définit **la confiance accordée au critique**. Le papier GAE insiste : $\gamma$ biaise toujours, $\lambda$ ne biaise que si $V$ est inexact. D'où $\gamma = 0{,}99$ mais $\lambda = 0{,}95$.

---

## 7. L'Algorithme 1, et la ligne qui fait tout

```
pour itération = 1, 2, … :
    pour acteur = 1..N :
        exécuter π_{θ_old} pendant T pas
        calculer Â_1..Â_T
    optimiser L, K époques, minibatch M ≤ NT
    θ_old ← θ                              ← ICI
```

**La dernière ligne est la clé.** $\theta_{old}$ ne bouge **qu'une fois par itération**. Pendant les $K$ époques, le dénominateur du ratio reste figé sur la politique qui a réellement collecté les données. C'est ce qui rend $r_t$ interprétable et le clipping légitime.

> **Erreur de lecture n°5 :** croire qu'il faut garder une copie du réseau ancien en mémoire. Non — il suffit d'avoir enregistré $\log\pi_{\theta_{old}}(a_t|s_t)$ au moment où l'action a été jouée. C'est un scalaire par transition. C'est exactement ce que fait RLlib (`Columns.ACTION_LOGP`).

**Ordre de grandeur, config MuJoCo** : $N{=}1$, $T{=}2048$, $K{=}10$, $M{=}64$ → $10 \times 2048/64 = 320$ pas de gradient par itération, sur les mêmes 2048 transitions. Ce facteur 320 est précisément ce que le clipping rend possible.

---

## 8. Lire la Table 1 correctement

C'est l'ablation qui justifie tout le papier. Sept environnements MuJoCo, 3 graines, 1 M pas, scores normalisés (politique aléatoire = 0, meilleur = 1), moyennés sur 21 runs.

| Réglage | Score |
|---|---|
| Sans clipping ni pénalité | **−0,39** |
| Clipping $\epsilon=0{,}1$ | 0,76 |
| **Clipping $\epsilon=0{,}2$** | **0,82** |
| Clipping $\epsilon=0{,}3$ | 0,70 |
| KL adaptative, $d_{targ}=0{,}01$ | 0,74 |
| KL fixe, $\beta=3$ | 0,72 |

**Trois lectures :**

1. **Le score négatif est le vrai résultat.** Sans garde-fou, l'agent finit **pire qu'une politique aléatoire** (le papier précise que HalfCheetah s'effondre). Le frein n'est pas une optimisation, c'est une condition d'existence.
2. **L'optimum en $\epsilon$ est net et non monotone** : 0,1 → 0,76, 0,2 → 0,82, 0,3 → 0,70. Trop serré, on n'apprend plus ; trop lâche, on diverge.
3. **Le clipping bat la KL** (0,82 vs 0,74) — c'est ce qui rend « PPO-Clip » synonyme de « PPO ». Mais l'écart n'est pas énorme, ce qui explique que la variante KL survive dans des implémentations réelles.

> **Erreur de lecture n°6 :** ces chiffres sont des scores **normalisés et agrégés sur 7 tâches**, pas des retours. Un écart de 0,08 est significatif à cette échelle. Ne pas les comparer à des retours bruts.

---

## 9. La variante KL (§4), et pourquoi elle survit

$$L^{KLPEN} = \hat{\mathbb{E}}_t\big[r_t\hat{A}_t - \beta\,\mathrm{KL}[\pi_{\theta_{old}},\pi_\theta]\big] \tag{8}$$

avec $\beta$ ajusté après chaque mise à jour : $d < d_{targ}/1{,}5 \Rightarrow \beta \leftarrow \beta/2$ ; $d > 1{,}5\,d_{targ} \Rightarrow \beta \leftarrow 2\beta$.

**Pourquoi le papier la garde** malgré son score inférieur : c'est *« an important baseline »*, et c'est le pont direct avec TRPO (contrainte dure → pénalité molle → clipping). Comprendre (8) permet de comprendre pourquoi (7) existe.

**Pourquoi elle compte en pratique** : plusieurs implémentations largement déployées (dont RLlib) l'ajoutent **par-dessus** le clipping. Ce n'est aucune des lignes de la Table 1. Si vous lisez du code PPO et trouvez un terme KL, ce n'est pas une erreur — c'est un héritage TRPO, mais ce n'est pas non plus le papier.

---

## 10. Ce que le papier ne dit pas et que tout le monde fait

Ces quatre pratiques sont **absentes du papier** mais présentes dans presque toutes les implémentations. Ne pas les attribuer à Schulman :

| Pratique | Où elle apparaît vraiment |
|---|---|
| **Standardisation des avantages** $(\hat{A}-\mu)/\sigma$ | OpenAI baselines |
| **Value clipping** (clipper $V_\theta$ autour de $V_{old}$) | OpenAI baselines |
| **Clipping de la *perte* de valeur** | invention RLlib, ni papier ni baselines |
| **Clipping de la norme du gradient** | pratique générale de deep learning |

Le papier n'utilise **aucune** des quatre. Il y a une littérature entière sur ce sujet (*« Implementation Matters in Deep RL »*, Engstrom et al. 2020), qui montre que certaines de ces astuces comptent autant que l'objectif clippé lui-même.

---

## 11. Auto-test

Si vous répondez aux six, le papier est acquis.

1. Pourquoi le titre est-il au pluriel ?
2. Que vaut $r_t(\theta)$ juste avant le premier pas de gradient d'une itération, et qu'est-ce que ça implique ?
3. Avec $\hat{A}_t < 0$ et $r_t = 0{,}5$, le gradient est-il nul ? Lequel des deux termes du `min` est sélectionné ?
4. Pourquoi le papier dit-il que $c_1$ est « irrelevant » dans ses expériences MuJoCo ?
5. Quelle différence de rôle entre $\gamma$ et $\lambda$ ?
6. À quel moment exact $\theta_{old}$ est-il mis à jour, et que se passerait-il s'il l'était après chaque minibatch ?

<details>
<summary>Réponses</summary>

1. Le papier propose une **famille** : sans garde-fou, clipping, KL adaptative, KL fixe. §6.1 les compare. « PPO » = la famille ; « PPO-Clip » = le vainqueur.
2. $r_t = 1$ exactement. Donc $L^{CLIP} = L^{CPI}$ et le gradient est **identique** à celui d'un gradient de politique classique. Le clipping ne se réveille qu'ensuite.
3. **Non, le gradient n'est pas nul** — c'est le terme **clippé** ($B = -1{,}6$) qui est sélectionné par le `min`, et il porte un gradient qui **ramène** $r_t$ vers la bande. Le gradient ne s'annule que du côté où la politique s'éloignerait *à son avantage*.
4. Parce que politique et critique sont deux réseaux **séparés**, avec des paramètres disjoints. Multiplier une perte indépendante par une constante ne change pas la direction de descente. $c_1$ ne compte que sous partage de paramètres.
5. $\gamma$ = horizon du problème, biaise **toujours** le gradient. $\lambda$ = confiance dans le critique, ne biaise **que si** $V$ est inexact. D'où $\gamma$ proche de 1 et $\lambda$ plus bas.
6. Une seule fois, **en fin d'itération**, après les $K$ époques. S'il était mis à jour après chaque minibatch, $r_t$ vaudrait toujours ≈ 1, le clipping ne mordrait jamais, et on retomberait sur $L^{PG}$ avec passes multiples — c'est-à-dire la ligne à −0,39 de la Table 1.

</details>

---

## Pour aller plus loin

- **GAE** — Schulman et al., ICLR 2016, [arXiv:1506.02438](https://arxiv.org/abs/1506.02438) : d'où viennent les $\hat{A}_t$.
- **TRPO** — Schulman et al., ICML 2015, [arXiv:1502.05477](https://arxiv.org/abs/1502.05477) : ce que PPO simplifie.
- **Implementation Matters in Deep Policy Gradients** — Engstrom et al., ICLR 2020, [arXiv:2005.12729](https://arxiv.org/abs/2005.12729) : quelle part des performances vient de l'objectif, quelle part des astuces d'implémentation.
- **The 37 Implementation Details of PPO** — Huang et al., ICLR blog 2022 : l'inventaire exhaustif de l'écart papier/code.
