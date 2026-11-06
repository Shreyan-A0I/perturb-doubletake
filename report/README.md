# Report

Paper writing for the course project.

| Path | What it is |
| --- | --- |
| `midterm/` | Midterm report on the course template (CMU `preprint.cls`), self-contained Overleaf project |
| `midterm/main.pdf` | Compiled midterm report, ready to submit |
| `midterm/main.tex` | Title, authors and section layout; sections live in `midterm/sec/` |
| `midterm/ref.bib` | References |
| `midterm/figs/` | Copies of Figs. 1-3 from `../figures/` |
| `drafts/` | First drafts of the sections, as Markdown |

Editable online version (Claude doc): https://claude.ai/artifact/M3EhwostLA7Hm5wWeLeTgC

## Build

The template needs pdfLaTeX (Overleaf's default compiler). XeLaTeX and Tectonic fail on
the class's microtype settings.

- **Overleaf:** zip the `midterm/` folder, then New Project, Upload Project.
- **Locally:** from `midterm/`, run `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

If figures in `../figures/` are regenerated, copy the PDFs into `midterm/figs/` again.

## Before submitting

- `ref.bib` entries marked `% verify` were written from memory; check titles, venues and years.
