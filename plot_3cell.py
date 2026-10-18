"""
Three-cell assembly (2D): post-processing and thesis figures.

Input : 3CELL.ods  (columns: time, Ycs, Ycs2, Ycs3 [um], Ucs.. [m/s], DEP.. [N])
Output: Images/results/3cell/*.pdf|png  and  3cell_metrics.txt

Why the data need treatment
---------------------------
The equation of motion of each cell is a free global ODE, (F - W)/b = U, with no
seat in it: the seat exists only in the mesh, through the flc1hs mask in the
prescribed displacement. An inactive cell (F = 0) therefore keeps integrating its
own sedimentation, U = -W/b, *through* the seat while its mesh stays pinned.
  - before activation : cell k's ODE state sinks by (t_on - 0) * W/b below the seat;
  - at activation     : the mesh stays seated (force held at its seat value) until
                        the ODE state has climbed back, which shows up as a force /
                        velocity plateau of duration tau_k;
  - after reseating   : U stays at -W/b although the sphere is at rest on the seat.
None of this is physical. The genuine physics of cells 2 and 3 is the part after
the plateau. The plotted operating sequence uses cell 1's cycle (which starts from
the true initial state and is not affected) shifted to each cell's activation
time; the genuine cell-2/3 output is used only to verify that this is legitimate.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

# ----------------------------------------------------------------------------
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("3CELL.ods")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("Images/results/3cell")
OUT.mkdir(parents=True, exist_ok=True)

R_UM   = 30.0                 # sphere radius (um)
T_ON   = [0.0, 2.0, 4.0]      # LPEA_k energised  (h1, h3, h5)
T_HOLD = 1.0                  # LPEA on-time; UPEA_k then energised for t_1 = 1 s
T_END  = 6.5                  # plotted span

BLUE, CORAL, GREEN = "#185FA5", "#D85A30", "#2E8B57"
AMBER, GRID, INK, MUTED = "#BA7517", "#E4E3DF", "#3D3D3A", "#6E6D68"
CELL_COL = [BLUE, CORAL, GREEN]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.8, "savefig.dpi": 300, "savefig.bbox": "tight",
})

def style(ax):
    ax.grid(True, which="major", color=GRID, lw=0.7)
    ax.grid(True, which="minor", color=GRID, lw=0.35, alpha=0.6)
    ax.set_axisbelow(True)

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)

# ----------------------------------------------------------------------------
# Load
raw = pd.read_excel(SRC, engine="odf")
raw.columns = ["t", "Y1", "Y2", "Y3", "U1", "U2", "U3", "F1", "F2", "F3"]
d = (raw.dropna(subset=["t"])
        .drop_duplicates(subset="t", keep="last")      # restart rows are written twice
        .sort_values("t").reset_index(drop=True))

# Drag coefficient and effective weight from the solved trajectory itself:
# closing transit of cell 1, where (F - W) = b U holds with U well resolved.
m = (d.t > 1.005) & (d.t < 2.0) & (d.Y1 > d.Y1.iloc[0] + 0.01)
b, W = np.linalg.lstsq(np.c_[d.U1[m], np.ones(m.sum())], d.F1[m], rcond=None)[0]
Y_SEAT = d.Y1.iloc[0]

# ----------------------------------------------------------------------------
# Canonical cycle: cell 1, from activation until it is back on the seat
c = d[d.t <= T_ON[1]][["t", "Y1", "U1", "F1"]].copy()
c.columns = ["t", "Y", "U", "F"]
seated = (c.t > T_HOLD) & (c.Y <= Y_SEAT + 1e-9)
c.loc[seated, ["Y", "U", "F"]] = [Y_SEAT, 0.0, 0.0]
# the seated return of cell 1 lies just past t = 2 s in the file
tail = d[(d.t > T_ON[1]) & (d.Y1 > Y_SEAT + 1e-9)][["t", "Y1", "U1", "F1"]]
tail.columns = c.columns
c = pd.concat([c, tail]).sort_values("t").reset_index(drop=True)
t_reseat = d.t[(d.t > T_HOLD) & (d.Y1 <= Y_SEAT + 1e-9)].min()

# Assemble the operating sequence on a common time base
tb = np.unique(np.r_[np.linspace(0, T_END, 6501),
                     *[c.t.values + t0 for t0 in T_ON]])
tb = tb[tb <= T_END]
seq = {"t": tb}
for k, t0 in enumerate(T_ON, 1):
    tl = tb - t0
    act = (tl >= 0) & (tl <= c.t.max())
    for q, rest in (("Y", Y_SEAT), ("U", 0.0), ("F", 0.0)):
        v = np.full_like(tb, rest)
        v[act] = np.interp(tl[act], c.t, c[q])
        seq[f"{q}{k}"] = v
seq = pd.DataFrame(seq)
seq.to_csv(OUT / "3cell_processed.csv", index=False)

# ----------------------------------------------------------------------------
# Verification from the genuine cell-2/3 output
def plateau(k, t0):
    s = d[(d.t > t0) & (d.t < t0 + T_HOLD)]
    F = s[f"F{k}"].values; t = s.t.values
    Fp = d.F1.iloc[0] if k == 1 else np.median(F[:3])
    i = np.argmax(F < 0.98 * Fp)
    return Fp, t[i] - t0

L_fit_mask = (c.t > 0.05) & (c.t < T_HOLD) & (c.F > 0)
# decay length over the outer half of the travel, as in Section 5.4
dY = c.Y - Y_SEAT
outer = L_fit_mask & (dY > 0.5 * dY[c.t <= T_HOLD].max())
L_um = -1.0 / np.polyfit(dY[outer], np.log(c.F[outer]), 1)[0]

ref_F_seat = d.F1.iloc[0]   # seated, full-amplitude force (frequency-domain initial state)
lines = []
lines.append(f"b      = {b:.4e} N s/m   (inferred from (F-W)/U, cell 1 closing)")
lines.append(f"W_eff  = {W:.4e} N       ({W*1e12:.1f} pN)")
lines.append(f"W/b    = {W/b*1e6:.2f} um/s   (sedimentation speed)")
lines.append(f"Y_seat = {Y_SEAT:.4f} um")
dYmax = c.Y[c.t <= T_HOLD].max() - Y_SEAT
lines.append(f"dY at end of hold = {dYmax:.3f} um ; dY/r = {dYmax/R_UM:.3f}")
F_end = c.F[(c.t - (T_HOLD - 0.01)).abs().idxmin()]
lines.append(f"F at end of hold  = {F_end/W:.4f} W_eff (not yet at rest)")
lines.append(f"F at seat (t=0+)  = {ref_F_seat/W:.2f} W_eff")
lines.append(f"L (outer half)    = {L_um:.2f} um")
for q, lab in ((0.5, '50'), (0.9, '90'), (0.99, '99')):
    tq = c.t[(c.t <= T_HOLD) & (c.Y - Y_SEAT >= q * dYmax)].min()
    lines.append(f"t to {lab}% of dY(1 s) = {tq:.4f} s")
lines.append(f"closing: F at start = {c.F[c.t > T_HOLD].iloc[0]/W:+.3f} W_eff; "
             f"reseated at t = {t_reseat:.3f} s ({t_reseat - T_HOLD:.3f} s after UPEA on)")
lines.append("")
lines.append("cell  F_seat/F_seat,1   sink depth (um)   tau predicted (s)   tau fitted (s)   rms dlnF   F_end/F_end,1   dY implied (um)")
taus = {}
s1 = d[(d.t > 0) & (d.t < T_HOLD - 0.005)]
def align(k, t0):
    """lag tau_k minimising the log-force mismatch with cell 1 over the motion."""
    s = d[(d.t > t0) & (d.t < t0 + T_HOLD - 0.005)]
    best = (np.inf, 0.0)
    for tau in np.arange(0.0, 0.30, 0.0005):
        ts = s.t.values - t0 - tau
        mk = (ts > 0.01) & (ts < 0.6)
        if mk.sum() < 10: continue
        ref = np.interp(ts[mk], s1.t.values, np.log(s1.F1.values))
        e = np.sqrt(np.mean((np.log(s[f"F{k}"].values[mk]) - ref) ** 2))
        if e < best[0]: best = (e, tau)
    return best
for k, t0 in enumerate(T_ON, 1):
    Fp, _ = plateau(k, t0)
    sink = t0 * W / b * 1e6
    Up = (Fp - W) / b
    tau_pred = sink * 1e-6 / Up
    rms, tau = align(k, t0) if k > 1 else (0.0, 0.0)
    taus[k] = tau
    Fe = d[f"F{k}"][(d.t - (t0 + T_HOLD - 0.01)).abs().idxmin()]
    lines.append(f"{k:>3}   {Fp/ref_F_seat:14.4f}   {sink:13.1f}   {tau_pred:15.4f}   {tau:13.4f}   "
                 f"{rms:8.4f}   {Fe/F_end:13.4f}   {L_um*np.log(Fe/F_end):+13.3f}")
(OUT / "3cell_metrics.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ----------------------------------------------------------------------------
# Figure 1 - operating sequence: drive schedule + normalised displacement
fig, (a0, a1) = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 3], hspace=0.08))
for k, (t0, col) in enumerate(zip(T_ON, CELL_COL), 1):
    yk = 3 - k
    a0.broken_barh([(t0, T_HOLD)], (yk - 0.32, 0.64), facecolors=col, edgecolor="none")
    a0.broken_barh([(t0 + T_HOLD, T_HOLD)], (yk - 0.32, 0.64), facecolors="white",
                   edgecolor=col, hatch="////", lw=1.0)
a0.set_yticks([2, 1, 0], ["cell 1", "cell 2", "cell 3"])
a0.set_ylim(-0.6, 2.6)
a0.tick_params(axis="y", length=0)
a0.spines["left"].set_visible(False)
a0.grid(True, axis="x", color=GRID, lw=0.7)
from matplotlib.patches import Patch
a0.legend(handles=[Patch(fc=INK, label="LPEA on (opening)"),
                   Patch(fc="white", ec=INK, hatch="////", label="UPEA on (closing)")],
          loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(1, 1.42))

for k, col in enumerate(CELL_COL, 1):
    a1.plot(seq.t, (seq[f"Y{k}"] - Y_SEAT) / R_UM, color=col, label=f"cell {k}")
    t0 = T_ON[k - 1]
    a1.text(t0 + 0.5, dYmax / R_UM + 0.07, f"cell {k}", color=INK, ha="center", fontsize=8)
for t0 in T_ON:
    a1.axvspan(t0, t0 + T_HOLD, color=GRID, alpha=0.45, lw=0)
a1.set_ylabel(r"$dY/r$  (1)")
a1.set_xlabel(r"$t$  (s)")
a1.set_xlim(0, T_END)
a1.set_ylim(-0.05, dYmax / R_UM + 0.25)
style(a1)
a1.legend(loc="center right", frameon=False)
save(fig, "3cell_sequence")

# ----------------------------------------------------------------------------
# Figure 2 - one cycle: (a) force landscape of the two transitions, (b) speed
fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw=dict(wspace=0.32))
op = c[(c.t > 0) & (c.t < T_HOLD - 0.005)]
cl = c[(c.t > T_HOLD + 0.005) & (c.t < 2 * T_HOLD - 0.005) & (c.Y > Y_SEAT + 1e-9)]
ax.semilogy(op.Y - Y_SEAT, op.F * 1e12, color=BLUE, label="opening (LPEA)")
ax.semilogy(cl.Y - Y_SEAT, -cl.F * 1e12, color=CORAL, label="closing (UPEA)")
ax.axhline(W * 1e12, color=AMBER, lw=1.2)
ax.text(1, W * 1e12 * 1.15, rf"$W_{{\mathrm{{eff}}}}$ = {W*1e12:.1f} pN", color=AMBER, fontsize=8)
ax.set_xlabel(r"$dY$  ($\mu$m)"); ax.set_ylabel(r"$|F_{\mathrm{DEP}}|$  (pN)")
style(ax); ax.legend(frameon=False, loc="lower right")
ax.set_title("(a)", loc="left", fontsize=9, color=INK)

o2 = c[(c.t > 0) & (c.t < T_HOLD - 0.005)]
c2 = c[(c.t > T_HOLD + 0.005) & (c.U < 0)]
bx.semilogy(o2.t, o2.U * 1e6, color=BLUE, label="opening, $U>0$")
bx.semilogy(c2.t, -c2.U * 1e6, color=CORAL, label="closing, $U<0$")
bx.axhline(W / b * 1e6, color=AMBER, lw=1.0, ls="--")
bx.text(1.05, W / b * 1e6 * 0.62, r"$W_{\mathrm{eff}}/b$", color=AMBER, fontsize=8)
bx.axvspan(0, T_HOLD, color=GRID, alpha=0.45, lw=0)
bx.set_xlim(0, t_reseat + 0.05)
bx.set_xlabel(r"$t$  (s)"); bx.set_ylabel(r"$|U|$  ($\mu$m/s)")
style(bx); bx.legend(frameon=False, loc="upper right")
bx.set_title("(b)", loc="left", fontsize=9, color=INK)
save(fig, "3cell_cycle")

# ----------------------------------------------------------------------------
# Figure 3 - verification: genuine solver force of each cell vs time since motion began
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for k, (t0, col) in enumerate(zip(T_ON, CELL_COL), 1):
    s = d[(d.t > t0) & (d.t < t0 + T_HOLD - 0.005)]
    ts = s.t - t0 - taus[k]
    keep = ts > 0
    ax.loglog(ts[keep], s[f"F{k}"][keep] * 1e12, color=col,
              label=f"cell {k}" + ("" if k == 1 else rf"  ($\tau_{k}$ = {taus[k]*1e3:.0f} ms removed)"),
              ls="-" if k == 1 else ("--" if k == 2 else ":"), lw=1.8 if k == 1 else 1.6)
ax.axhline(W * 1e12, color=AMBER, lw=1.0)
ax.text(0.25, W * 1e12 * 1.08, r"$W_{\mathrm{eff}}$", color=AMBER, fontsize=8)
ax.set_xlabel(r"$t - t_{\mathrm{on},k} - \tau_k$  (s)")
ax.set_ylabel(r"$F_{\mathrm{DEP}}$  (pN)")
style(ax); ax.legend(frameon=False, fontsize=7.5, loc="lower left")
save(fig, "3cell_verification")
print("written to", OUT.resolve())
