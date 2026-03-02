"""
Generate Reference/ pages for every Python module in the theseus package.

Run automatically by mkdocs-gen-files at `mkdocs serve` / `mkdocs build` time.
The script DFS-walks theseus/, creates one .md stub per module with a
`:::` autodoc directive, and writes a SUMMARY.md for mkdocs-literate-nav.

Doc paths strip the top-level `theseus` prefix so that e.g.
`theseus/dispatch/solve.py` appears at `Reference/dispatch/solve` rather than
`Reference/theseus/dispatch/solve`.  The mkdocstrings identifier still uses the
full dotted module name so imports resolve correctly.

Skipped:
  - __pycache__ directories
  - files/directories whose name starts with _ (except __init__.py)
"""

from pathlib import Path
import mkdocs_gen_files

ROOT = Path("theseus")
nav = mkdocs_gen_files.Nav()

for path in sorted(ROOT.rglob("*.py")):
    if "__pycache__" in path.parts:
        continue

    # Full module identifier: theseus.dispatch.solve
    module_parts = path.relative_to(ROOT.parent).with_suffix("").parts

    # Doc path parts: strip the leading 'theseus' component
    doc_parts = path.relative_to(ROOT).with_suffix("").parts

    # Skip private modules but keep __init__
    if any(p.startswith("_") and p != "__init__" for p in doc_parts):
        continue

    if doc_parts[-1] == "__init__":
        # theseus/dispatch/__init__.py → Reference/dispatch/index.md
        module_parts = module_parts[:-1]
        doc_parts = doc_parts[:-1]
        if not doc_parts:
            # Root theseus/__init__.py — skip; nothing useful and it would
            # turn the "Reference" nav section into a clickable page.
            continue
        doc_path = Path(*doc_parts, "index.md")
    else:
        doc_path = path.relative_to(ROOT).with_suffix(".md")

    full_doc_path = Path("Reference", doc_path)
    identifier = ".".join(module_parts)

    nav[doc_parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("Reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
