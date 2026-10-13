# -*- coding: utf-8 -*-
"""Сборка analysis.py (формат # %%-ячеек) в volve_ml_forecast.ipynb."""
import nbformat as nbf

SRC = "analysis.py"
OUT = "volve_ml_forecast.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

cell_src, cell_kind = [], []
for line in open(SRC, encoding="utf-8"):
    if line.startswith("# %%"):
        kind = "md" if "[markdown]" in line else "code"
        cell_kind.append(kind)
        cell_src.append([])
    elif cell_kind:
        cell_src[-1].append(line)

for kind, lines in zip(cell_kind, cell_src):
    src = "".join(lines).strip("\n")
    if kind == "md":
        src = "\n".join(l[2:] if l.startswith("# ") else l for l in src.splitlines())
        nb.cells.append(nbf.v4.new_markdown_cell(src))
    else:
        nb.cells.append(nbf.v4.new_code_cell(src))

nbf.write(nb, OUT)
print(f"ноутбук собран: {OUT}, ячеек: {len(nb.cells)} "
      f"(md: {cell_kind.count('md')}, code: {cell_kind.count('code')})")
