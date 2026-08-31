# PPO : résumé en une page

> Schulman, Wolski, Dhariwal, Radford, Klimov (OpenAI), *Proximal Policy Optimization **Algorithms***, arXiv:1707.06347, 2017.

Le papier propose **une nouvelle famille de méthodes de gradient de politique**. Elle se comprend à partir de celle qu'elle remplace : TRPO.

## D'où vient TRPO

On ne peut pas dériver le rendement $`J(\theta)`$ directement : c'est une espérance sur des trajectoires dont la loi dépend elle-même de $`\theta`$. Le théorème du gradient de politique fournit à la place un estimateur calculable sur un lot d'échantillons :

```math
\hat{g} = \hat{\mathbb{E}}_t\big[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\hat{A}_t\big]
```

En pratique, $`\hat g`$ ne se code pas directement : les frameworks comme PyTorch ou TensorFlow calculent un gradient en appelant `.backward()` sur un scalaire, jamais en l'écrivant à la main. On construit donc un scalaire dont la dérivée redonne $`\hat g`$ :

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

## Annexe : tous les symboles

### Grandeurs de base

| Symbole | Anglais | Français | Définition |
|---|---|---|---|
| $`J(\theta)`$ | *(expected) return*, *objective* | rendement, récompense espérée | Ce qu'on maximise : espérance de la somme actualisée des récompenses quand on suit $`\pi_\theta`$. Non calculable directement. |
| $`\theta`$ | *policy parameters* | paramètres de la politique | Vecteur des poids du réseau acteur. |
| $`\theta_{old}`$ | *old policy parameters* | anciens paramètres | Valeur de $`\theta`$ **avant** la mise à jour, celle qui a collecté le lot de données. Figée pendant les époques. |
| $`\pi_\theta(a \mid s)`$ | *policy* | politique | Loi de probabilité sur les actions $`a`$ sachant l'état $`s`$, paramétrée par $`\theta`$. |
| $`s_t`$ | *state* | état | Observation de l'environnement au pas de temps $`t`$. |
| $`a_t`$ | *action* | action | Action tirée par la politique en $`s_t`$. |
| $`t`$ | *timestep* | pas de temps | Indice dans la trajectoire. |
| $`\hat A_t`$ | *(estimated) advantage* | avantage (estimé) | $`Q(s_t,a_t) - V(s_t)`$ estimé : de combien l'action prise vaut mieux que la moyenne de l'état. Positif = à renforcer. Le chapeau marque l'estimation (par GAE en pratique). |

### Opérateurs

| Symbole | Anglais | Français | Définition |
|---|---|---|---|
| $`\mathbb{E}`$ | *expectation* | espérance | Moyenne théorique sous une loi. |
| $`\hat{\mathbb{E}}_t`$ | *empirical average over timesteps* | espérance empirique | Moyenne **effectivement calculée** sur les échantillons du lot : $`\tfrac{1}{T}\sum_t`$. Le chapeau signale un estimateur, pas la vraie espérance. |
| $`\nabla_\theta`$ | *gradient w.r.t. θ* | gradient par rapport à $`\theta`$ | Vecteur des dérivées partielles selon chaque paramètre. |
| $`\log`$ | *log-likelihood* | logarithme (log-vraisemblance) | Ici $`\log\pi_\theta(a_t\mid s_t)`$ : log de la probabilité de l'action prise. |
| $`\min`$ | *minimum* | minimum | Prend le plus petit des deux termes, terme à terme. |
| $`\mathrm{clip}(r, a, b)`$ | *clipping* | écrêtage, troncature | Ramène $`r`$ dans $`[a,b]`$ : renvoie $`a`$ si $`r<a`$, $`b`$ si $`r>b`$, sinon $`r`$. |
| $`\mathrm{KL}[p, q]`$ | *Kullback-Leibler divergence* | divergence de Kullback-Leibler | Mesure de dissemblance entre deux lois. Vaut 0 si elles sont identiques, croît quand elles s'écartent. Ce n'est pas une distance : elle n'est pas symétrique. |

### Objectifs

| Symbole | Anglais | Français | Définition |
|---|---|---|---|
| $`\hat g`$ | *policy gradient estimator* | estimateur du gradient de politique | Estimation de $`\nabla_\theta J`$ calculable sur un lot. La quantité qu'on veut vraiment. |
| $`L^{PG}`$ | *policy gradient (surrogate) loss* | objectif de gradient de politique | Scalaire fabriqué pour que la différentiation automatique redonne $`\hat g`$. **PG** pour *policy gradient*. Sa valeur ne mesure rien. |
| $`L^{CPI}`$ | *conservative policy iteration objective*, *surrogate advantage* | avantage de substitution | Objectif repondéré par le ratio. Reste une estimation de la performance de $`\pi_\theta`$ même loin de $`\theta_{old}`$. **CPI** pour *conservative policy iteration* (Kakade & Langford 2002). |
| $`L^{CLIP}`$ | *clipped surrogate objective* | objectif de substitution écrêté | Objectif principal de PPO : $`L^{CPI}`$ plafonné par le `min` et le `clip`. Estimation pessimiste. |
| $`r_t(\theta)`$ | *probability ratio*, *likelihood ratio* | ratio de probabilité, ratio de vraisemblance | $`\pi_\theta(a_t\mid s_t)/\pi_{\theta_{old}}(a_t\mid s_t)`$. Vaut 1 avant toute mise à jour. Mesure de combien la nouvelle politique a changé d'avis sur cette action précise. |
| $`\epsilon`$ | *clipping parameter* | paramètre d'écrêtage | Demi-largeur de la zone autorisée du ratio : $`[1-\epsilon, 1+\epsilon]`$. Papier PPO : 0,2. |
| $`\delta`$ | *trust region radius*, *KL bound* | rayon de la région de confiance | Borne sur la KL moyenne dans TRPO. Papier TRPO : 0,01. |

### Échantillonnage préférentiel

| Symbole | Anglais | Français | Définition |
|---|---|---|---|
| $`p`$ | *target distribution* | loi cible | Loi sous laquelle on veut l'espérance. Ici $`\pi_\theta`$. |
| $`q`$ | *proposal distribution*, *behaviour distribution* | loi de proposition, loi de comportement | Loi sous laquelle on a réellement tiré. Ici $`\pi_{\theta_{old}}`$. |
| $`x`$ | *random variable* | variable aléatoire | Échantillon générique. |
| $`f(x)`$ | *integrand* | fonction à moyenner | Quantité dont on veut l'espérance. Ici c'est $`\hat A_t`$ qui joue ce rôle. |
| $`p(x)/q(x)`$ | *importance weight* | poids d'importance | Facteur de correction appliqué à chaque échantillon. Se spécialise en $`r_t`$. |

### Second ordre (TRPO)

| Symbole | Anglais | Français | Définition |
|---|---|---|---|
| $`F`$ | *Fisher information matrix* | matrice d'information de Fisher | Hessienne de la KL en $`\theta_{old}`$, c'est-à-dire la courbure locale de l'espace des politiques. Taille $`N \times N`$, jamais formée. |
| $`g`$ | *gradient of the surrogate* | gradient du substitut | Membre de droite de $`Fx = g`$ : le gradient de $`L^{CPI}`$, soit $`\hat g`$. |
| $`x`$ (dans $`Fx = g`$) | *search direction* | direction de recherche | Inconnue résolue par gradient conjugué : la direction du pas naturel. Sans rapport avec le $`x`$ de l'échantillonnage préférentiel. |
| $`v`$ | *arbitrary vector* | vecteur quelconque | Argument des produits $`Fv`$ (produits hessien-vecteur) qu'exige le gradient conjugué. |
| $`N`$ | *number of parameters* | nombre de paramètres | Dimension de $`\theta`$. D'où le coût $`N^2`$ pour stocker $`F`$. |

Deux collisions de notation à garder en tête : $`x`$ désigne une variable aléatoire dans l'échantillonnage préférentiel et la direction de recherche dans $`Fx = g`$ ; $`\epsilon`$ (écrêtage, PPO) et $`\delta`$ (région de confiance, TRPO) sont deux seuils distincts, souvent notés tous les deux $`\epsilon`$ ailleurs dans ce dépôt.
