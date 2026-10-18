"""Clausius-Mossotti figure at the thesis values: eps_m = 78 eps0, sigma_m = 0.2-2 S/m.
(a) Re[K] map over (sigma_p/sigma_m, eps_p/eps_m) at the operating band, (b) Re[K](f) for polystyrene."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT.mkdir(parents=True, exist_ok=True)
E0 = 8.854e-12
EPS_M = 78.0
SIG_M = [0.2, 0.35, 0.6, 1.3, 2.0]          # S/m: measured low, median, high, renal/dehydrated, adopted bound
BAND = (1e5, 1e7)
INK, MUTED, GRID, BAND_C = "#3D3D3A", "#6E6D68", "#E4E3DF", "#DCE6F2"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})

def reK(ep, sp, em, sm, f):
    w = 2 * np.pi * f
    P = ep * E0 - 1j * sp / w
    M = em * E0 - 1j * sm / w
    return ((P - M) / (P + 2 * M)).real

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.55))

# (a) map at the centre of the band, sigma_m = 2 S/m (Omega = w eps_m / sigma_m = 2.2e-3)
f_op, sm0 = 1e6, 2.0
x = np.linspace(-15, 3, 600)          # log10(sigma_p/sigma_m)
y = np.linspace(-2, 3, 400)           # log10(eps_p/eps_m)
X, Y = np.meshgrid(x, y)
K = reK(10**Y * EPS_M, 10**X * sm0, EPS_M, sm0, f_op)
im = ax.pcolormesh(X, Y, K, cmap="RdBu_r", norm=TwoSlopeNorm(0, -0.5, 1.0), shading="auto", rasterized=True)
ax.contour(X, Y, K, levels=[0], colors="white", linewidths=1.2, linestyles="--")
cb = fig.colorbar(im, ax=ax, pad=0.02, ticks=[-0.5, -0.25, 0, 0.5, 1.0])
cb.set_label(r"Re$[K^*]$", color=INK)
mats = [("polystyrene", 2.55, 1e-12), ("silica", 3.8, 1e-14), ("rutile", 85, 1e-10)]
for (name, ep, sp), dy, dx in zip(mats, (-0.35, 0.3, 0.3), (0, 2.2, 0)):
    xs = np.log10(sp / np.array([2.0, 0.2]))      # span over sigma_m = 0.2-2 S/m
    yv = np.log10(ep / EPS_M)
    ax.plot(xs, [yv, yv], color="#F2C14E", lw=3, solid_capstyle="round")
    ax.text(xs.mean() + dx, yv + dy, name, color="white", ha="center", fontsize=7.5, fontweight="bold")
ax.text(-7, 2.4, "nDEP", color="white", ha="center", fontsize=9, fontweight="bold")
ax.text(1.7, 0.5, "pDEP", color="white", ha="center", va="center", rotation=90, fontsize=9, fontweight="bold")
ax.set_xlabel(r"$\log_{10}(\sigma_p/\sigma_m)$"); ax.set_ylabel(r"$\log_{10}(\epsilon_p/\epsilon_m)$")
ax.set_title(r"(a) $f$ = 1 MHz, $\epsilon_m = 78\epsilon_0$", loc="left", fontsize=9, color=INK)

# (b) polystyrene against frequency, across the conductivity range
f = np.logspace(4, 9.5, 600)
bx.axvspan(*BAND, color=BAND_C, lw=0)
bx.text(np.sqrt(BAND[0] * BAND[1]), -0.437, "operating band", ha="center", color=MUTED, fontsize=7.5)
cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(SIG_M)))
for sm, c in zip(SIG_M, cols):
    bx.semilogx(f, reK(2.55, 1e-12, EPS_M, sm, f), color=c, lw=1.6, label=rf"{sm:g} S/m")
hi = (2.55 - EPS_M) / (2.55 + 2 * EPS_M)
bx.axhline(-0.5, color=MUTED, lw=0.6, ls=":"); bx.axhline(hi, color=MUTED, lw=0.6, ls=":")
bx.text(1.3e4, -0.497, r"$-0.500$ (conductivity limit)", color=MUTED, fontsize=7.5, va="bottom")
bx.text(1.3e4, hi + 0.002, rf"${hi:.3f}$ ($f \gg f_c$)", color=MUTED, fontsize=7.5, va="bottom")
bx.set_ylim(-0.51, -0.43); bx.set_xlim(1e4, 3e9)
bx.grid(True, which="major", color=GRID, lw=0.7); bx.grid(True, which="minor", color=GRID, lw=0.35)
bx.set_xlabel(r"$f$  (Hz)"); bx.set_ylabel(r"Re$[K^*]$")
bx.legend(title=r"$\sigma_m$", frameon=False, loc="upper right", title_fontsize=7.5)
bx.set_title(r"(b) polystyrene in saliva", loc="left", fontsize=9, color=INK)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"cm_factor.{ext}")
print("hi-f limit", hi)
