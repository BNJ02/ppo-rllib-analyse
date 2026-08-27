/**
 * Markdown GitHub -> PDF, sans pandoc ni LaTeX (aucun des deux n'est installe
 * sur le Jetson).
 *
 *   node tools/md2pdf.mjs docs/fiche-policy-gradient-intuition.md [sortie.pdf]
 *
 * Chaine : marked -> HTML, KaTeX rendu cote serveur, mermaid rendu dans le
 * navigateur, puis chromium --headless --print-to-pdf.
 *
 * Les maths sont ecrites aux delimiteurs GitHub (```math et $`...`$), pas aux
 * delimiteurs LaTeX : c'est la contrainte du depot (skill markdown-math-github).
 * Elles sont donc extraites AVANT marked -- sinon `...` est vu comme du code --
 * et reinjectees apres, deja rendues.
 */

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { dirname, resolve, basename } from "node:path";
import { marked } from "marked";
import katex from "katex";

const src = process.argv[2];
if (!src) {
  console.error("usage: node tools/md2pdf.mjs <fichier.md> [sortie.pdf]");
  process.exit(1);
}
const out = (process.argv[3] && !process.argv[3].startsWith("--"))
  ? process.argv[3] : src.replace(/\.md$/, ".pdf");
const base = resolve(dirname(src));
let md = readFileSync(src, "utf8");

// --- 1. sortir les maths et les diagrammes du flux markdown -----------------
const slots = [];
const park = (html) => `%%SLOT${slots.push(html) - 1}%%`;

const tex = (src, display) => {
  try {
    return katex.renderToString(src, { displayMode: display, throwOnError: false });
  } catch (e) {
    return `<code class="katex-error">${src}</code>`;
  }
};

// Un `graph LR` de 8 noeuds fait ~2000 px de large : ramene a la largeur d'une
// A4 portrait, il tombe a ~13 mm de haut et devient illisible. On le bascule en
// vertical POUR LE PDF seulement -- le markdown n'est pas touche, et LR reste le
// bon choix sur GitHub, ou la page est large et defile.
const keepDir = process.argv.includes("--keep-direction");
md = md.replace(/```mermaid\n([\s\S]*?)```/g, (_, code) => {
  if (!keepDir) code = code.replace(/^(\s*(?:graph|flowchart))\s+LR\b/m, "$1 TD");
  return park(`<pre class="mermaid">${code.replace(/</g, "&lt;")}</pre>`);
});
md = md.replace(/```math\n([\s\S]*?)```/g, (_, code) => park(tex(code.trim(), true)));
md = md.replace(/\$`([^`]+?)`\$/g, (_, code) => park(tex(code, false)));

// --- 2. markdown -> HTML ---------------------------------------------------
marked.setOptions({ gfm: true, breaks: false });
let body = marked.parse(md);
body = body.replace(/%%SLOT(\d+)%%/g, (_, i) => slots[+i]);

// Les images sont relatives au markdown : file:// absolu pour chromium.
body = body.replace(/src="(?!https?:|file:|data:)([^"]+)"/g,
  (_, p) => `src="file://${resolve(base, p)}"`);

const katexCss = readFileSync(
  resolve("tools/node_modules/katex/dist/katex.min.css"), "utf8")
  .replace(/url\((.*?)\)/g, (m, u) =>
    `url(file://${resolve("tools/node_modules/katex/dist", u.replace(/["']/g, ""))})`);
const mermaidJs = readFileSync(
  resolve("tools/node_modules/mermaid/dist/mermaid.min.js"), "utf8");

const html = `<!doctype html><html lang="fr"><head><meta charset="utf-8">
<title>${basename(src)}</title>
<style>${katexCss}</style>
<style>
  @page { size: A4; margin: 18mm 16mm; }
  body { font: 10.5pt/1.55 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
         color: #1a1a1a; max-width: none; }
  h1 { font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: .3em; }
  h2 { font-size: 15pt; margin-top: 1.6em; border-bottom: 1px solid #ddd;
       padding-bottom: .2em; break-after: avoid; }
  h3 { font-size: 12.5pt; margin-top: 1.3em; break-after: avoid; }
  h2, h3, h4 { break-inside: avoid; }
  p, li { orphans: 2; widows: 2; }
  img, pre, table, blockquote, .katex-display { break-inside: avoid; }
  img { max-width: 100%; display: block; margin: 1em auto; }
  code { font-family: "DejaVu Sans Mono", Menlo, Consolas, monospace; font-size: .88em;
         background: #f4f4f4; padding: .12em .35em; border-radius: 3px; }
  pre { background: #f6f8fa; padding: .8em 1em; border-radius: 5px; overflow-x: auto;
        font-size: .82em; line-height: 1.4; }
  pre code { background: none; padding: 0; }
  pre.mermaid { background: none; text-align: center; break-inside: avoid; }
  /* A4 utile = 297 - 36 mm de marges. Sans plafond, un graphe vertical de
     8 noeuds depasse la page : le bas est coupe et, avec break-inside: avoid,
     il pousse une page blanche devant lui. */
  pre.mermaid svg { max-width: 100%; max-height: 225mm; height: auto; }
  blockquote { margin: 1em 0; padding: .1em 1.1em; border-left: 4px solid #b8c4d0;
               background: #f7f9fb; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: .88em; }
  th, td { border: 1px solid #d0d7de; padding: .38em .6em; text-align: left;
           vertical-align: top; }
  th { background: #f0f3f6; }
  a { color: #0550ae; text-decoration: none; }
  .katex-display { margin: .9em 0; }
  .katex { font-size: 1.04em; }
  .katex-error { color: #b00; }
</style></head><body>
${body}
<script>${mermaidJs}</script>
<script>
  mermaid.initialize({ startOnLoad: false, theme: "neutral",
                       flowchart: { padding: 12 } });
  mermaid.run().then(() => { document.title = "READY " + document.title; });
</script>
</body></html>`;

const tmp = resolve(out.replace(/\.pdf$/, "") + ".render.html");
writeFileSync(tmp, html);

const chrome = ["/usr/bin/chromium-browser", "/snap/bin/chromium"]
  .find((p) => existsSync(p));
if (!chrome) { console.error("chromium introuvable"); process.exit(1); }

execFileSync(chrome, [
  "--headless", "--disable-gpu", "--no-sandbox",
  "--run-all-compositor-stages-before-draw",
  "--virtual-time-budget=20000",          // laisse mermaid finir son rendu
  "--no-pdf-header-footer",
  `--print-to-pdf=${resolve(out)}`,
  `file://${tmp}`,
], { stdio: ["ignore", "ignore", "pipe"] });

console.log(`${out}  (HTML intermediaire : ${tmp})`);
