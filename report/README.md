# Report

Paper writing for the course project. Figures are read from `../figures/`.

| File | What it is |
| --- | --- |
| `midterm.tex` | Midterm report, LaTeX source |
| `midterm.pdf` | Compiled midterm report, ready to submit |
| `drafts/midterm_D_sections.md` | First drafts of Related work, Experiments, Analysis |
| `drafts/midterm_S_sections.md` | First drafts of Introduction, Methodology |

Editable online version (Claude doc): https://claude.ai/artifact/M3EhwostLA7Hm5wWeLeTgC

## Build

From this folder, with any LaTeX install:

```bash
pdflatex midterm.tex      # or: tectonic midterm.tex
```

For Overleaf, upload `midterm.tex` plus `figures/fig1_eta_reliability.pdf`,
`figures/fig2_null_hierarchy.pdf` and `figures/fig3_additive_vs_eta.pdf`, keeping the
`figures/` folder one level up or editing the `\includegraphics` paths.

## Before submitting

- References 5–11 are marked `% (verify)` in `midterm.tex`; check titles, venues and years.
