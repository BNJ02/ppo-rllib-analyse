# PPO : résumé en une page

> Schulman, Wolski, Dhariwal, Radford, Klimov (OpenAI), *Proximal Policy Optimization **Algorithms***, arXiv:1707.06347, 2017.

Le papier propose **une nouvelle famille de méthodes de gradient de politique**. Elle se comprend à partir de celle qu'elle remplace : TRPO.

## D'où vient TRPO

L'estimateur de gradient de politique

```math
\hat{g} = \hat{\mathbb{E}}_t\big[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\hat{A}_t\big]
```

est la dérivée de $`L^{PG} = \hat{\mathbb{E}}_t[\log\pi_\theta(a_t \mid s_t)\hat{A}_t]`$. Cette expression n'a pas d'interprétation propre : elle est construite pour que sa dérivée soit $`\hat g`$, et sa valeur ne mesure rien. L'optimiser sur plusieurs pas avec un seul lot de données n'a donc aucune justification, et conduit en pratique à des mises à jour de politique excessivement grandes.

On la remplace par l'**avantage de substitution** (*surrogate advantage*), noté $`L^{CPI}`$ dans le papier (d'après *conservative policy iteration*, Kakade & Langford 2002), obtenu par échantillonnage préférentiel des données de $`\pi_{\theta_{old}}`$ :

```math
L^{CPI}(\theta) = \hat{\mathbb{E}}_t\big[r_t(\theta)\hat{A}_t\big],
\qquad r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}
```

$`r_t`$ est le **ratio de probabilité**, ou de vraisemblance. Il vaut 1 avant toute mise à jour, et c'est en ce point que $`\nabla L^{CPI} = \hat g`$ : les deux expressions ont le même gradient au départ, mais elles se séparent ensuite. La valeur de $`L^{PG}`$ reste dénuée de sens dès qu'on quitte $`\theta_{old}`$, alors que $`L^{CPI}`$ demeure une estimation de la performance de $`\pi_\theta`$ pour tout $`\theta`$, puisque le ratio repondère les données au fur et à mesure que la politique s'éloigne. C'est ce qui autorise à continuer de l'optimiser après le premier pas, donc à réutiliser un même lot sur plusieurs époques.

Maximiser $`L^{CPI}`$ sans garde-fou donne un pas trop grand. TRPO ajoute une **contrainte dure** :

```math
\max_{\theta}\; \hat{\mathbb{E}}_t\big[r_t(\theta)\hat{A}_t\big]
\quad \text{sous} \quad
\hat{\mathbb{E}}_t\big[\mathrm{KL}[\pi_{\theta_{old}}, \pi_\theta]\big] \le \delta
```

## Ce que TRPO coûte

La contrainte est approchée **au second ordre** : sa hessienne est la matrice d'information de Fisher $`F`$. TRPO ne la calcule jamais.

- **Gradient conjugué** : on résout $`Fx = g`$ sans former ni stocker $`F`$ ($`N^2`$ cases), avec seulement des produits $`Fv`$ obtenus par produit hessien-vecteur.
- **Recherche linéaire par rebroussement** : le pas est calculé, puis **testé avant d'être appliqué**, et réduit exponentiellement tant que les deux conditions ne tiennent pas, à savoir une KL exacte $`\le \delta`$ **et** un substitut réellement amélioré.

Il reste lourd : gradients conjugués et recherche linéaire à chaque mise à jour, et la machinerie du second ordre **interdit en pratique le dropout et le partage de paramètres acteur/critique**. Le dropout, parce que la KL et les produits $`Fv`$ exigent que $`\pi_\theta`$ soit une fonction déterministe de $`\theta`$ : deux passes tirent deux masques, et le test « KL ≤ δ ? » ne mesure plus rien. Le partage, parce que la contrainte ne porte que sur la politique ; la perte de valeur n'a pas de place dans le programme, et la régression du critique rebougerait le tronc en défaisant la région de confiance qu'on venait d'obtenir.

## Ce que PPO change

Le papier construit et compare plusieurs objectifs, d'où le pluriel du titre, mais tous reposent sur le même principe : la contrainte de proximité passe **directement dans la fonction objectif**, au lieu d'être posée à côté d'elle.

L'optimisation redevient alors du **premier ordre**, ce qui permet une **montée de gradient stochastique** ordinaire, sur plusieurs époques par minibatch avec le même lot. Le membre recommandé de la famille écrit cette contrainte sous forme d'un **ratio de probabilité clipé** :

```math
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta)\hat{A}_t,\ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat{A}_t\big)\Big]
```

Le premier terme du `min` est exactement $`L^{CPI}`$. Le second le plafonne : au-delà de $`\pm\epsilon`$, l'objectif devient **plat** et le gradient s'annule. Le `min` fait que ce plafonnement ne joue que dans le sens qui *avantagerait* la politique, ce qui donne une **estimation pessimiste**, borne inférieure de l'objectif non clipé. On n'interdit pas les grands pas, on les rend sans intérêt.

**Ce qu'on perd au passage :** le théorème d'amélioration monotone de TRPO. La seule inégalité que PPO démontre est $`L^{CLIP} \le L^{CPI}`$, triviale puisque c'est un `min`, et rien ne relie $`L^{CLIP}`$ au rendement vrai.

## Ce qu'il faut retenir

PPO échange la garantie théorique de TRPO contre une contrainte implicite, portée par l'objectif lui-même. Le coût d'implémentation retombe à celui d'un gradient de politique classique augmenté de quelques lignes, l'optimisation reste du premier ordre, et aucune restriction d'architecture ne subsiste. C'est ce rapport entre simplicité, efficacité d'échantillonnage et temps de calcul qui a fait de PPO une méthode de référence, et qui l'y maintient.

> Le détail (figures 1 et 2 du papier, clipping cas par cas, tableau comparatif des trois algorithmes, garanties perdues une par une) est dans [**De TRPO à PPO**](06-de-trpo-a-ppo.md).

**Notation :** $`\delta`$ = rayon de la région de confiance (TRPO, 0,01) ; $`\epsilon`$ = clipping (PPO, 0,2). La [fiche de cours](fiche-policy-gradient.md) note les deux $`\epsilon`$.
