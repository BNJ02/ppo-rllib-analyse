# PPO — résumé en une page

> Schulman, Wolski, Dhariwal, Radford, Klimov (OpenAI), *Proximal Policy Optimization **Algorithms***, arXiv:1707.06347, 2017.

Le papier propose **une nouvelle famille de méthodes de gradient de politique**. Elle se comprend à partir de celle qu'elle remplace : TRPO.

## D'où vient TRPO

L'estimateur de gradient de politique

```math
\hat{g} = \hat{\mathbb{E}}_t\big[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\hat{A}_t\big]
```

est la dérivée de $`L^{PG} = \hat{\mathbb{E}}_t[\log\pi_\theta(a_t \mid s_t)\hat{A}_t]`$ — un objet dont la seule fonction est d'être dérivé. Faire **plusieurs** pas dessus avec le même lot n'est pas justifié et produit des mises à jour destructrices.

On remplace donc $`L^{PG}`$ par l'**avantage de substitution** (*surrogate advantage*), noté $`L^{CPI}`$ dans le papier (d'après *conservative policy iteration*, Kakade & Langford 2002), obtenu par échantillonnage préférentiel des données de $`\pi_{\theta_{old}}`$ :

```math
L^{CPI}(\theta) = \hat{\mathbb{E}}_t\big[r_t(\theta)\hat{A}_t\big],
\qquad r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}
```

$`r_t`$ est le **ratio de probabilité**, ou de vraisemblance. Il vaut 1 avant toute mise à jour, et c'est là que $`\nabla L^{CPI} = \hat g`$ : les deux substituts ont le **même gradient au point de départ**, mais $`L^{CPI}`$ garde un sens quand on s'éloigne. D'où la possibilité de plusieurs pas.

Maximiser $`L^{CPI}`$ sans garde-fou donne un pas trop grand. TRPO ajoute une **contrainte dure** :

```math
\max_{\theta}\; \hat{\mathbb{E}}_t\big[r_t(\theta)\hat{A}_t\big]
\quad \text{sous} \quad
\hat{\mathbb{E}}_t\big[\mathrm{KL}[\pi_{\theta_{old}}, \pi_\theta]\big] \le \delta
```

## Ce que TRPO coûte

La contrainte est approchée **au second ordre** : sa hessienne est la matrice d'information de Fisher $`F`$. TRPO ne la calcule jamais.

- **Gradient conjugué** : on résout $`Fx = g`$ sans former ni stocker $`F`$ ($`N^2`$ cases), avec seulement des produits $`Fv`$ obtenus par produit hessien-vecteur.
- **Recherche linéaire par rebroussement** : le pas est calculé, puis **testé avant d'être appliqué**, et réduit exponentiellement tant que les deux conditions ne tiennent pas — KL exacte $`\le \delta`$, **et** substitut réellement amélioré.

Il reste lourd : gradients conjugués et recherche linéaire à chaque mise à jour, et la machinerie du second ordre **interdit en pratique le dropout et le partage de paramètres acteur/critique**. Le dropout, parce que la KL et les produits $`Fv`$ exigent que $`\pi_\theta`$ soit une fonction déterministe de $`\theta`$ — deux passes tirent deux masques, et le test « KL ≤ δ ? » ne mesure plus rien. Le partage, parce que la contrainte ne porte que sur la politique : la perte de valeur n'a pas de place dans le programme, et la régression du critique rebougerait le tronc en défaisant la région de confiance qu'on venait d'obtenir.

## Ce que PPO change

**Premier ordre seulement**, ce qui permet une **montée de gradient stochastique** ordinaire, plusieurs époques par minibatch sur le même lot. Et pour cela, la contrainte passe *dans* la fonction objectif, sous forme d'un **ratio de probabilité clipé** :

```math
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta)\hat{A}_t,\ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat{A}_t\big)\Big] \tag{7}
```

Le premier terme du `min` est exactement $`L^{CPI}`$. Le second le plafonne : au-delà de $`\pm\epsilon`$, l'objectif devient **plat**, le gradient s'annule. Le `min` fait que ce plafonnement ne joue que dans le sens qui *avantagerait* la politique — d'où une **estimation pessimiste**, borne inférieure de l'objectif non clipé. On n'interdit pas les grands pas, on les rend sans intérêt.

**Ce qu'on perd au passage :** le théorème d'amélioration monotone de TRPO. La seule inégalité que PPO démontre est $`L^{CLIP} \le L^{CPI}`$ — triviale, c'est un `min` — et rien ne relie $`L^{CLIP}`$ au rendement vrai.

## Pourquoi « Algorithms » au pluriel

Le papier ne propose pas *un* algorithme mais **une famille**. L'abstract l'annonce : *« a new family of policy gradient methods »*. Quatre objectifs sont construits et comparés :

| # | Objectif | Statut dans le papier |
|---|---|---|
| 1 | $`L^{PG}`$ — sans garde-fou | référence négative (score −0,39) |
| 2 | $`L^{CLIP}`$ — **clipping du ratio** (§3) | **recommandé** (score 0,82) |
| 3 | $`L^{KLPEN}`$ — pénalité KL **adaptative** (§4) | alternative, moins bonne (0,74) |
| 4 | $`L^{KLPEN}`$ à $`\beta`$ **fixe** | baseline, moins bonne (0,72) |

« PPO » désigne la famille ; « PPO-Clip » (n°2) est le membre qui a gagné et que tout le monde appelle aujourd'hui « PPO ». C'est ce pluriel qui explique que RLlib implémente **les deux** — et les active simultanément par défaut, ce qui ne correspond à aucune des quatre lignes.

## L'objectif complet (§5)

```math
L^{CLIP+VF+S} = \hat{\mathbb{E}}_t\big[L^{CLIP} - c_1\underbrace{(V_\theta(s_t)-V_t^{targ})^2}_{\text{critique}} + c_2\underbrace{S[\pi_\theta](s_t)}_{\text{entropie}}\big] \tag{9}
```

Une seule perte scalaire, un seul pas d'Adam — exactement ce que la contrainte de TRPO rendait impossible. Avantages estimés par GAE tronqué : $`\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots`$, $`\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)`$.

## L'algorithme (§5, Algorithme 1)

```
pour itération = 1, 2, … :
    N acteurs exécutent π_{θ_old} pendant T pas chacun    → NT échantillons
    calculer Â_1..Â_T par GAE
    K époques de SGD (Adam) par minibatch M ≤ NT sur L
    θ_old ← θ                                             ← une seule fois par itération
```

Le point qui fait tout : $`\theta_{old}`$ n'est mis à jour **qu'en fin d'itération**. Pendant les K époques, le dénominateur du ratio reste la politique qui a réellement produit les données.

## Résultats

- **MuJoCo** (7 tâches, 1 M pas) : $`\epsilon=0{,}2`$ meilleur réglage (0,82) ; bat TRPO, A2C, CEM, vanilla PG sur presque toutes les tâches.
- **Atari** (49 jeux) : bat A2C largement ; bat ACER en vitesse d'apprentissage (30 jeux sur 49), légèrement derrière en performance finale (19 contre 28) — mais **bien plus simple**.
- **Roboschool** : locomotion humanoïde 3D, course, redressement.

Hyperparamètres MuJoCo : $`T`$=2048, Adam $`3\!\times\!10^{-4}`$, $`K`$=10, $`M`$=64, $`\gamma`$=0,99, $`\lambda`$=0,95, MLP 2×64 tanh, réseaux politique/critique **séparés**, pas de bonus d'entropie.

## Ce qu'il faut retenir

PPO remplace une contrainte mathématiquement élégante par un **truc d'ingénieur qui marche mieux** : quelques lignes de différence avec un gradient de politique classique, du premier ordre, compatible avec n'importe quelle architecture. Meilleur compromis simplicité / efficacité en échantillons / temps de calcul de 2017 — et il tient toujours.

> Le détail — figures 1 et 2 du papier, clipping cas par cas, tableau comparatif des trois algorithmes, garanties perdues une par une — est dans [**De TRPO à PPO**](06-de-trpo-a-ppo.md).

**Notation :** $`\delta`$ = rayon de la région de confiance (TRPO, 0,01) ; $`\epsilon`$ = clipping (PPO, 0,2). La [fiche de cours](fiche-policy-gradient.md) note les deux $`\epsilon`$.
