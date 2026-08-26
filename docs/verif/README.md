# Vérification de `fiche-policy-gradient.md`

La fiche est destinée à de l'apprentissage par cœur : une coquille dans une
formule est mémorisée telle quelle. Ces scripts prouvent que les démonstrations
tiennent, au lieu de l'affirmer.

```bash
cd rapport/verif
python3 verify_fiche.py             # 50 assertions numériques exactes
python3 verify_symbolic.py          # 22 assertions symboliques (sympy)
python3 mutation_test.py            # le banc sait-il détecter une faute ?
python3 check_wikipedia_coverage.py # couverture des 85 formules de la page
```

Chaque script sort en code 1 s'il échoue. Aucune dépendance à installer :
numpy, scipy, sympy et torch du système suffisent.

## Principe

`mdp.py` construit un PDM fini (3 états, 3 actions, horizon 3, γ = 0,9) et
**énumère ses 2187 trajectoires** avec leur probabilité exacte. Toute espérance
devient une somme finie : pas d'échantillonnage, donc pas de bruit. Une identité
vraie tient à 1e-15, une identité fausse saute immédiatement.

Le gradient de référence est calculé par **deux voies indépendantes** — autodiff
PyTorch sur `J(θ)` énuméré, et différences finies centrées — qui doivent
coïncider avant tout le reste.

La politique est un softmax dont la dernière composante des logits est fixée à 0.
Sans cette réduction, la matrice de Fisher est singulière (ajouter une constante
à `θ[s, :]` ne change pas `π`) et `F⁻¹` n'existe pas.

## Fichiers

| Fichier | Rôle |
|---|---|
| `mdp.py` | banc d'essai : PDM, énumération, fonctions de valeur, les Ψ_t du tableau §3.5 |
| `verify_fiche.py` | étage 1 — identités en espérance, sections 2 à 7 de la fiche |
| `verify_symbolic.py` | étage 2 — Taylor de la KL, lagrangien du QP, échange min/max, convexité |
| `mutation_test.py` | injecte 14 fautes réalistes et exige qu'elles soient détectées |
| `check_wikipedia_coverage.py` | apparie chaque formule de la page Wikipédia à son équivalent dans la fiche |

`../figures/make_figures.py` importe lui aussi `mdp.py` : toutes les figures de
[`fiche-policy-gradient-intuition.md`](../fiche-policy-gradient-intuition.md) sont donc
tracées sur le même banc d'essai, avec les mêmes espérances exactes.

## Trois résultats trouvés par ces scripts

1. **La pondération GAE de Wikipédia est fautive.** La page écrit
   `λ^(n-1)/(1-λ)` ; la somme des poids vaut alors `1/(1-λ)²` au lieu de 1, et
   l'estimateur vaut `∇J/(1-λ)²`. Le papier GAE (éq. 16) donne `(1-λ)λ^(n-1)`.
   Facteur mesuré : 11,111 pour λ = 0,7, soit exactement `1/(1-λ)²`.
2. **L'identité k3 exige d'échantillonner sous `π_θ`**, pas sous la politique de
   collecte `π_{θ_t}` comme l'écrit la page. Exacte seulement en `θ = θ_t`.
3. **Le baseline `μ` de GRPO n'est pas admissible au sens strict** (il contient
   la récompense de l'action qu'il pondère) : le gradient obtenu vaut `(1-1/G)`
   fois le vrai gradient. Biais d'échelle, pas de direction.

## Ce que le test de mutation apprend

Quatre variantes qui *semblent* fausses restent exactement non biaisées :
baseline ajouté au lieu d'être soustrait, `V₀` à la place de `V_t`, résidu TD
avec le mauvais signe sur `V(S_t)`, récursion GAE en `λ` au lieu de `γλ`.

La raison : (a) toute fonction `b(t,s)` est un baseline admissible quelle que
soit sa qualité ; (b) `E[δ_{t+l} | S_{t+l}] = 0` pour `l ≥ 1`, donc n'importe
quel coefficient sur les résidus TD futurs laisse l'espérance inchangée. Seul le
rôle d'**amorçage** exige la bonne valeur de `V`.
