#!/usr/bin/env python3
"""Etage 3.1 : chaque formule de la page Wikipedia a-t-elle sa contrepartie
dans la fiche ?

La fiche reformate les formules (\\mid au lieu de |, espacements \;, retours a
la ligne), donc une comparaison exacte est impossible. On normalise agressivement
puis on cherche la meilleure correspondance par difflib. Les formules sous le
seuil sortent dans une liste A RELIRE A LA MAIN -- le script ne conclut pas seul.

    python3 check_wikipedia_coverage.py [chemin_wikitext] [chemin_fiche]
"""
import difflib
import pathlib
import re
import sys

SEUIL = float(__import__("os").environ.get("SEUIL", 0.72))

wiki_path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                         "/tmp/claude-1000/-home-bnj-ppo-ray-analyse/"
                         "0316b2ee-3f82-47c0-be29-d2853ccaf232/scratchpad/pg.wiki")
fiche_path = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else
                          pathlib.Path(__file__).parent.parent / "fiche-policy-gradient.md")

wiki = wiki_path.read_text()
fiche = fiche_path.read_text()


def normalise(f):
    f = re.sub(r'\\(?:display)?style', '', f)
    f = re.sub(r'\\[Bb]ig[gl]?\|', '|', f)
    f = f.replace(r'\mid', '|').replace(r'\vert', '|')
    f = re.sub(r'\\(?:qquad|quad|;|,|!|:|>)', '', f)
    f = re.sub(r'\\(?:left|right)', '', f)
    f = re.sub(r'\\mathbb\s*\{?E\}?', 'E', f)
    f = re.sub(r'\\dfrac', r'\\frac', f)
    f = re.sub(r'\\text(?:rm|bf|it)?\s*\{([^}]*)\}', r'\1', f)
    f = re.sub(r'\\operatorname\{([^}]*)\}', r'\1', f)
    f = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', f)
    f = re.sub(r'\s+', '', f)
    f = f.replace('{', '').replace('}', '').replace('\\', '')
    return f.lower()


# formules de la page
wiki_forms = []
for m in re.finditer(r'<math(?P<attrs>[^>]*)>(?P<body>.*?)</math>', wiki, re.S):
    kind = 'block' if 'block' in m.group('attrs') else 'inline'
    body = m.group('body').strip()
    if len(normalise(body)) < 12:          # symboles isoles : hors sujet
        continue
    wiki_forms.append((kind, body))

# formules de la fiche : blocs ```math et inline $`...`$
fiche_forms = re.findall(r'```math\n(.*?)\n```', fiche, re.S)
fiche_forms += re.findall(r'\$`(.*?)`\$', fiche, re.S)
fiche_norm = [normalise(f) for f in fiche_forms]

manquantes, faibles = [], []
for kind, body in wiki_forms:
    tgt = normalise(body)
    best, ratio = "", 0.0
    for fn, orig in zip(fiche_norm, fiche_forms):
        if tgt in fn or fn in tgt:
            ratio, best = 1.0, orig
            break
        r = difflib.SequenceMatcher(None, tgt, fn).ratio()
        if r > ratio:
            ratio, best = r, orig
    if ratio >= SEUIL:
        continue
    (manquantes if ratio < 0.5 else faibles).append((kind, body, best, ratio))

n_block = sum(1 for k, _ in wiki_forms if k == 'block')
print(f"Page Wikipedia : {len(wiki_forms)} formules retenues "
      f"({n_block} display, {len(wiki_forms) - n_block} inline)")
print(f"Fiche          : {len(fiche_forms)} formules")
print(f"Couvertes (ratio >= {SEUIL}) : "
      f"{len(wiki_forms) - len(manquantes) - len(faibles)}/{len(wiki_forms)}")

for titre, lot in (("SANS CORRESPONDANCE (ratio < 0.5)", manquantes),
                   ("CORRESPONDANCE FAIBLE -- a relire", faibles)):
    if lot:
        print(f"\n--- {titre} : {len(lot)}")
        for kind, body, best, ratio in lot:
            print(f"\n  [{kind}] ratio={ratio:.2f}")
            print(f"    wiki  : {' '.join(body.split())[:190]}")
            print(f"    fiche : {' '.join(best.split())[:190]}")

sys.exit(1 if manquantes else 0)
