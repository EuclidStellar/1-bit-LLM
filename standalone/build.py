#!/usr/bin/env python3
"""Re-inline web/*.js into standalone/index.html.

The four modules have no cross-imports, so concatenation is enough --
stripping the `export` keyword is the only transform needed. Edit the
modules in web/, then run this; don't hand-edit the inlined copy.
"""
import pathlib, re

HERE = pathlib.Path(__file__).resolve().parent
WEB  = HERE.parent / "web"
PAGE = HERE / "index.html"

MODULES = ["heap.js", "bitllm.js", "tokenizer.js", "wasmlm.js"]

libs = []
for f in MODULES:
    src = re.sub(r'^export\s+', '', WEB.joinpath(f).read_text(), flags=re.M)
    libs.append(f"// ===== {f} " + "=" * (62 - len(f)) + "\n" + src)

app = re.search(r'<script type="module">(.*?)</script>',
                WEB.joinpath("index.html").read_text(), re.S).group(1)
app = re.sub(r'^\s*import\s+.*?from\s+".*?";\s*$', '', app, flags=re.M | re.S)
assert "import " not in app.split("const HF")[0], "an import survived"

page = PAGE.read_text()
head, _, tail = page.partition('<script type="module">')
extras = tail.split("// --- live readouts")[1]          # keep page-only wiring
body = "\n\n".join(libs) + "\n\n// ===== app " + "=" * 60 + "\n" + app \
     + "\n// --- live readouts" + extras
PAGE.write_text(head + '<script type="module">\n' + body)
print(f"rebuilt {PAGE} -> {PAGE.stat().st_size:,} bytes")
