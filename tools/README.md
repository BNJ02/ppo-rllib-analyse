# Outils

## `md2pdf.mjs` — markdown GitHub → PDF

```bash
npm install --prefix tools          # marked, katex, mermaid
node tools/md2pdf.mjs docs/fiche-policy-gradient-intuition.md build/fiche.pdf
```

Ni pandoc ni LaTeX ne sont installés sur le Jetson : la chaîne est
`marked` → HTML, KaTeX rendu côté serveur, mermaid rendu par le navigateur,
puis `chromium --headless --print-to-pdf`.

Deux points valent d'être connus :

- **Les maths sont extraites avant `marked`.** Le dépôt écrit aux délimiteurs
  GitHub (` ```math ` et `` $`…`$ ``), pas aux délimiteurs LaTeX. Sans
  extraction préalable, `` `…` `` serait vu comme du code littéral.
- **Les `graph LR` deviennent `graph TD`** dans le PDF. Un graphe horizontal de
  8 nœuds ramené à la largeur d'une A4 portrait tombe à ~13 mm de haut et
  devient illisible. Le markdown n'est pas modifié — LR reste le bon choix sur
  GitHub, où la page est large. `--keep-direction` désactive la bascule.

Sous snap, chromium ne lit pas `/tmp` : garder entrée et sortie sous `$HOME`.

## `check_links.py` — liens et ancres internes

```bash
python3 tools/check_links.py .
```

Reproduit la slugification GitHub, y compris le détail qui fait échouer les
vérificateurs naïfs : GitHub ne collapse **pas** les espaces multiples, donc un
titre à tiret cadratin (`3.5 La forme générale — le par cœur central`) produit
une ancre à **double** tiret. Ignore les blocs math, où `](a_t|s_t)` ressemble à
un lien.
