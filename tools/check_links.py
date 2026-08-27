"""Verifie les liens internes et les ancres d'un depot markdown, a la maniere
de GitHub. La slugification GitHub NE collapse PAS les espaces multiples :
"A — B" donne "a--b". Un verificateur qui les collapse produit de faux
negatifs sur tous les titres a tiret cadratin."""
import re, os, sys

def slug(t):
    t = re.sub(r'`', '', t).strip().lower()
    t = re.sub(r'[^\w\s\-]', '', t)
    return t.replace(' ', '-')          # PAS \s+ -> '-'

root = sys.argv[1] if len(sys.argv) > 1 else "."
anch = {}
SKIP = {".git", ".venv", "node_modules", "build", "__pycache__", "ray", "sota"}

for d, dirs, fs in os.walk(root):
    dirs[:] = [x for x in dirs if x not in SKIP]   # elague : os.walk relit `dirs`
    for f in fs:
        if f.endswith(".md"):
            p = os.path.normpath(os.path.join(d, f))
            anch[p] = {slug(l.lstrip('#')) for l in open(p) if l.startswith('#')}

bad = []
for p in sorted(anch):
    base = os.path.dirname(p)
    txt = open(p).read()
    # Ignore tout ce qui est du code ou des maths : "](a_t|s_t)" y ressemble a
    # un lien sans en etre un. L'ordre compte -- les blocs d'abord, sinon un
    # backtick simple a l'interieur d'un bloc coupe la paire.
    txt = re.sub(r'```.*?```', '', txt, flags=re.S)
    txt = re.sub(r'\$`.*?`\$', '', txt, flags=re.S)
    txt = re.sub(r'`[^`\n]*`', '', txt)
    for m in re.finditer(r'\]\(([^)\s]+)\)', txt):
        t = m.group(1)
        if t.startswith(('http', 'mailto', '#!')):
            continue
        f, _, a = t.partition('#')
        tgt = os.path.normpath(os.path.join(base, f)) if f else p
        if f and not os.path.exists(tgt):
            bad.append((p, t, "fichier absent"))
        elif a and tgt in anch and a not in anch[tgt]:
            bad.append((p, t, "ancre absente"))

for p, t, why in bad:
    print(f"{p}: {t}  <- {why}")
print(f"{len(bad)} lien(s) casse(s)")
sys.exit(1 if bad else 0)
