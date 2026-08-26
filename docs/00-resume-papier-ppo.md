# PPO — résumé en une page

> Schulman, Wolski, Dhariwal, Radford, Klimov (OpenAI), *Proximal Policy Optimization **Algorithms***, arXiv:1707.06347, 2017.

## Pourquoi « Algorithms » au pluriel ?

Parce que le papier ne propose pas *un* algorithme mais **une famille**. L'abstract l'annonce : *« a new family of policy gradient methods »*. Le papier construit et compare quatre objectifs distincts :

| # | Objectif | Statut dans le papier |
|---|---|---|
| 1 | $`L^{PG} = \hat{\mathbb{E}}_t[\log\pi_\theta(a_t \mid s_t)\hat{A}_t]`$ — sans garde-fou | référence négative (score −0,39) |
| 2 | $`L^{CLIP}`$ — **clipping du ratio** (§3) | **recommandé** (score 0,82) |
| 3 | $`L^{KLPEN}`$ — pénalité KL **adaptative** (§4) | alternative, moins bonne (0,74) |
| 4 | $`L^{KLPEN}`$ à $`\beta`$ **fixe** | baseline, moins bonne (0,72) |

« PPO » désigne donc la famille ; « PPO-Clip » (n°2) est le membre qui a gagné et que tout le monde appelle aujourd'hui « PPO ». Le §6.1 est entièrement consacré à les départager. C'est aussi ce pluriel qui explique que RLlib implémente **les deux** — et les active simultanément par défaut, ce qui ne correspond à aucune des quatre lignes du tableau.

## Le problème

On veut améliorer une politique à partir de données collectées par la politique **précédente**. Un grand pas rend ces données non représentatives et fait diverger l'entraînement. TRPO (2015) l'évitait par une contrainte KL dure, au prix d'une optimisation de second ordre lourde et incompatible avec dropout ou paramètres partagés.

**Objectif du papier** : la stabilité de TRPO, avec du premier ordre seulement.

## L'idée centrale (§3)

Soit le ratio de vraisemblance $`r_t(\theta) = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}`$, qui vaut 1 avant toute mise à jour.

```math
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta)\hat{A}_t,\ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat{A}_t\big)\Big] \tag{7}
```

Au-delà de $`\pm\epsilon`$, l'objectif devient **plat** : le gradient s'annule, il n'y a plus rien à gagner à s'éloigner. Le `min` fait que ce plafonnement ne joue que dans le sens qui *avantagerait* la politique — d'où une **borne inférieure pessimiste** sur l'objectif réel. On n'interdit pas les grands pas, on les rend sans intérêt.

## L'objectif complet (§5)

```math
L^{CLIP+VF+S} = \hat{\mathbb{E}}_t\big[L^{CLIP} - c_1\underbrace{(V_\theta(s_t)-V_t^{targ})^2}_{\text{critique}} + c_2\underbrace{S[\pi_\theta](s_t)}_{\text{entropie}}\big] \tag{9}
```

avec des avantages estimés par GAE tronqué : $`\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots`$, $`\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)`$.

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
- **Atari** (49 jeux) : bat A2C largement ; bat ACER en vitesse d'apprentissage (30 jeux gagnés sur 49), légèrement derrière ACER en performance finale (19 contre 28) — mais **bien plus simple**.
- **Roboschool** : locomotion humanoïde 3D, course, redressement.

Hyperparamètres MuJoCo : $`T`$=2048, Adam $`3\!\times\!10^{-4}`$, $`K`$=10, $`M`$=64, $`\gamma`$=0,99, $`\lambda`$=0,95, MLP 2×64 tanh, réseaux politique/critique **séparés**, pas de bonus d'entropie.

## Ce qu'il faut retenir

PPO remplace une contrainte mathématiquement élégante par un **truc d'ingénieur qui marche mieux** : quelques lignes de différence avec un gradient de politique classique, du premier ordre, compatible avec n'importe quelle architecture. C'est le meilleur compromis simplicité / efficacité en échantillons / temps de calcul de 2017 — et il tient toujours.
