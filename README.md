# TEAM NTUA Thesis Code

Post-processing and plotting scripts for the MSc thesis

**Development of a Saliva Collecting Micro-Device**
Anastasios Kamenides — MSc *Translational Engineering in Health and Medicine*,
School of Electrical and Computer Engineering, National Technical University of Athens, 2026.

The thesis proposes a microvalve in which a dielectrophoretically actuated polystyrene sphere
gates the admission of saliva into a sealed collection chamber, for autonomous overnight sampling.
The device was modelled in COMSOL Multiphysics 5.2; these scripts turn the exported simulation
data into the figures and tables of the thesis.

## Scripts

| Script | Input | Output | Thesis |
|---|---|---|---|
| `depplot1.py` | — (analytical) | `dep_induced_dipoles.pdf/.png` | Fig. 2.1 — induced dipoles in uniform vs non-uniform field |
| `plot_cm.py` | — (analytical) | `cm_factor.pdf/.png` | Fig. 2.2 — Clausius–Mossotti factor of saliva |
| `plots.py` | `PEA_10X10.ods`, `PEA_20X20.ods`, `PEA_30X30.ods`, `PEA_50X50.ods` | `Yr_log_r{r}.png`, `Ucs_log_r{r}.png`, `DEP_R{r}.png` per pitch | Fig. 4.1 and Appendix A — 2D parametric sweep |
| `plots2.py` | `3dchr2.ods` | `*_r30_3Dvs2D.pdf/.png` | Fig. 5.2, Table 5.1 — selected cell, 3D vs 2D |
| `plot_3cell.py` | `3CELL.ods` | `3cell_sequence`, `3cell_cycle`, `3cell_verification` (.pdf/.png), `3cell_metrics.txt`, `3cell_processed.csv` | Figs 4.3–4.5, Tables 4.8–4.9 — three-cell assembly |
| `plot_electrodes.py` | — (values transcribed from COMSOL tables) | `electrodes_force_sweeps.pdf/.png` | Fig. B.4 — screened electrode geometries |

### Notes on individual scripts

- **`plots.py`** — each workbook holds one sheet per sphere radius (`R10`, `R20`, …) with
  side-by-side voltage blocks (2, 4, 6, 8, 16, 24 V). Produces normalised displacement and
  normalised velocity against log-time and DEP force against displacement, and prints the
  voltage and radius power-law fits to the console.
- **`plots2.py`** — columns A–E hold the 3D run and F–J the 2D run. Prints equilibrium height,
  inferred drag coefficient, force decay length and the 2D/3D comparison against the 3D mesh
  uncertainty. Options: `NORMALISE_TIME`, `SHOW_RESTARTS`, `SAVE_PDF`, `SAVE_PNG`.
- **`plot_3cell.py`** — implements the trajectory correction described in Section 4.5.4 of the
  thesis. The equation of motion of an inactive cell has no seat, so its state sinks below the
  seat before activation and the cell starts with a delay. The script removes these artefacts,
  fits the delay independently and verifies it against the predicted sink-recovery time, and
  builds the operating sequence. All verification figures are written to `3cell_metrics.txt`.

## Running

```bash
pip install numpy pandas matplotlib odfpy
python plot_3cell.py 3CELL.ods out/3cell
python plot_cm.py out/cm
python plot_electrodes.py out/electrodes
```

`plot_3cell.py`, `plot_cm.py` and `plot_electrodes.py` take input and output paths as
command-line arguments. `depplot1.py`, `plots.py` and `plots2.py` contain absolute Windows
paths from the author's machine; edit the path constants at the top of each file before
running them.

Tested with Python 3, pandas, matplotlib and odfpy.

## Data

The exported COMSOL results (`.ods`) are in `data/`. The COMSOL model files (`.mph`) are not
included because of their size; they are available from the author on request.

## Citation

If you use this code, please cite the thesis:

> A. Kamenides, *Development of a Saliva Collecting Micro-Device*, MSc thesis,
> National Technical University of Athens, 2026.

## AI assistance

Parts of this code were developed with AI assistance (Claude, Anthropic), as declared in the
thesis. All results were verified by the author.

## License

MIT — see `LICENSE`.
