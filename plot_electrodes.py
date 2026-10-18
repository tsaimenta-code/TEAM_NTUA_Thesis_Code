"""Appendix B: static force sweeps of the 3D conical and concentric-ring electrode screens
(10 Vp2p, 10 MHz), transcribed from the COMSOL tables of the topology-study slides."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#3D3D3A", "#6E6D68", "#E4E3DF"
COLS = ["#185FA5", "#D85A30", "#2E8B57", "#7A5195"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})

z6 = [0, 10, 20, 30, 40, 50]
cone = {  # z (um) -> F_z (N)
    r"A: $(b,g) = (25,25)$ $\mu$m, y-paired":        (z6, [6.524e-10, 1.183e-8, 1.970e-8, 2.468e-8, 2.660e-8, 2.616e-8]),
    r"A: $(b,g) = (50,50)$ $\mu$m, y-paired":        (z6, [2.544e-9, 1.781e-8, 2.880e-8, 3.304e-8, 3.174e-8, 2.811e-8]),
    r"B: $(30,30,30)$ $\mu$m":        (z6, [-5.144e-9, 4.271e-9, 1.198e-8, 1.844e-8, 2.121e-8, 2.222e-8]),
    r"B: $(b,g) = (50,50)$ $\mu$m, x-paired": (z6, [1.324e-7, 8.419e-7, 5.613e-7, 1.003e-6, 7.943e-7, 1.844e-8]),
}
ring = {
    r"A: $(\Delta R,g) = (10,10)$ $\mu$m":      ([55, 60, 65, 70], [1.427e-7, 1.122e-7, 8.539e-8, 6.281e-8]),
    r"A: $(\Delta R,g) = (50,50)$ $\mu$m":      ([5, 15, 25, 35, 45], [-1.898e-8, -1.012e-8, -1.372e-9, 5.991e-9, 1.103e-8]),
    r"B: $(\Delta R,g,R) = (30,30,30)$ $\mu$m": ([5, 15, 25, 35, 45], [-7.021e-10, 1.645e-8, 1.978e-8, 1.611e-8, 1.027e-8]),
    r"B: $(\Delta R,g,R) = (50,50,50)$ $\mu$m": ([55, 60, 65, 70], [9.851e-8, 8.723e-8, 7.498e-8, 6.239e-8]),
}

def panel(ax, data, title, inset_last=False):
    for (lab, (z, F)), c, mk in zip(data.items(), COLS, "osD^"):
        if inset_last and lab == list(data)[-1]:
            continue
        ax.plot(z, np.array(F) * 1e9, color=c, marker=mk, ms=4.5, lw=1.5, label=lab)
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel(r"sphere-centre height $z$  ($\mu$m)"); ax.set_ylabel(r"$F_z$  (nN)")
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)

fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw=dict(wspace=0.32))
panel(a, cone, "(a) conical electrodes", inset_last=True)
z, F = list(cone.values())[-1]
ins = a.inset_axes([0.6, 0.1, 0.38, 0.3])
ins.plot(z, np.array(F) * 1e9, color=COLS[3], marker="^", ms=3.5, lw=1.2)
ins.set_title("x-paired (50,50)", fontsize=7, color=INK); ins.tick_params(labelsize=6.5)
ins.grid(True, color=GRID, lw=0.5)
panel(b, ring, "(b) concentric-ring electrodes")
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"electrodes_force_sweeps.{ext}")
