# PPO : résumé en une page

> Schulman, Wolski, Dhariwal, Radford, Klimov (OpenAI), *Proximal Policy Optimization **Algorithms***, arXiv:1707.06347, 2017.

Le papier propose **une nouvelle famille de méthodes de gradient de politique**. Elle se comprend à partir de celle qu'elle remplace : TRPO.

## D'où vient TRPO

On ne peut pas dériver le rendement $`J(\theta)`$ directement : c'est une espérance sur des trajectoires dont la loi dépend elle-même de $`\theta`$. Le théorème du gradient de politique fournit à la place un estimateur calculable sur un lot d'échantillons :

```math
\hat{g} = \hat{\mathbb{E}}_t\big[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\hat{A}_t\big]
```

Comme les bibliothèques de différentiation automatique attendent un scalaire à dériver, on écrit

```math
L^{PG}(\theta) = \hat{\mathbb{E}}_t\big[\log\pi_\theta(a_t \mid s_t)\,\hat{A}_t\big]
```

dont la dérivée vaut $`\hat g`$ par construction. C'est un artifice de calcul, pas une mesure de performance : $`L^{PG}`$ n'est pas une estimation de $`J`$, et $`\hat A_t`$ y est traité comme une constante.

Deux raisons interdisent d'enchaîner plusieurs pas dessus avec le même lot. D'abord, l'identité $`\nabla L^{PG} = \hat g`$ n'est valable qu'en $`\theta = \theta_{old}`$, la politique qui a produit les données ; passé le premier pas, plus rien ne corrige le fait qu'elles sont périmées. Ensuite, $`L^{PG}`$ n'est pas borné : le maximiser revient à pousser vers 1 la probabilité de toute action d'avantage positif, sans contrepoids. D'où les mises à jour excessivement grandes que constate le papier.

On remplace donc $`L^{PG}`$ par l'**avantage de substitution** (*surrogate advantage*), noté $`L^{CPI}`$ dans le papier (d'après *conservative policy iteration*, Kakade & Langford 2002). Il s'obtient par **échantillonnage préférentiel** (*importance sampling*) : pour estimer une espérance sous une loi $`p`$ alors qu'on ne dispose que de tirages sous une loi $`q`$, on repondère chaque échantillon par le rapport des deux densités,

```math
\mathbb{E}_{x \sim p}\big[f(x)\big] = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)}\,f(x)\right]
```

Ici $`p = \pi_\theta`$, la politique qu'on veut évaluer, et $`q = \pi_{\theta_{old}}`$, celle qui a collecté les données :

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

PPO échange la garantie théorique de TRPO contre une contrainte implicite, portée par l'objectif lui-même. Partir d'un gradient de politique classique pour arriver à PPO ne demande qu'une modification mineure du code : calculer le ratio, le clipper, prendre le minimum. Là où TRPO exigeait un gradient conjugué, des produits hessien-vecteur et une recherche linéaire, l'optimisation reste ici du premier ordre, et aucune restriction d'architecture ne subsiste. C'est ce rapport entre simplicité, efficacité d'échantillonnage et temps de calcul qui a fait de PPO une méthode de référence, et qui l'y maintient.
