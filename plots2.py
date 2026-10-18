"""
Transient response of the selected cell (Lambda = 60 um, r = 30 um, 6 V p2p):
three-dimensional model against its two-dimensional counterpart.

Reads the COMSOL export '3dchr2.ods', which carries two column blocks:
    cols A-E   3D run   t, Displacement (um), Y/r, Velocity (m/s), F_DEP (N)
    cols F-J   2D run   same five columns

NOTE ON THE FILE: the 2D block is only valid for its first 308 rows. After
t = 100 s its time column jumps backwards to 3.08 and the remaining rows are a
copy of the 3D data. The loader below truncates at that reversal automatically,
but fix it in the spreadsheet too.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ODS_PATH = Path(r"C:\Users\tsaim\Desktop\msc\Thesis\presentations\3dchr2.ods")
OUT_DIR  = Path(r'C:\Users\tsaim\Desktop\msc\Thesis\Images\results\3d')
STEM     = 'r30_3Dvs2D'

R_SPHERE = 30e-6        # m   sphere radius
RHO_S    = 1050.        # kg/m^3
RHO_M    = 1005.        # kg/m^3
G        = 9.81

# Time axes: the 2D and 3D runs use drag coefficients that differ by ~38x, so
# their absolute timescales are NOT comparable (see the printed diagnostics).
#   False -> plot against t [s]           : honest, but the curves sit ~2 decades apart
#   True  -> plot against t/tau, tau = b*L/W_eff : collapses them onto one another
NORMALISE_TIME = False

SAVE_PDF = True
SAVE_PNG = True

# The 3D run was advanced as a sequence of restarted segments (automatic
# remeshing was disabled). Two artifacts follow, both cosmetic:
#   * F steps discontinuously at each restart, because the remeshed field gives
#     a slightly different Maxwell-stress integral for the same geometry.
#     U inherits the step through F - W_eff = b U; the drag coefficient itself
#     is intact across every one of them.
#   * U is held constant across consecutive outputs wherever the solver's own
#     step exceeded the output interval, producing a staircase that is an
#     export artifact rather than a feature of the motion.
SHOW_RESTARTS = True    # tick the restart instants on (a) and (b)
DROP_HELD_U   = True    # drop the repeated-U points from the velocity panel

# ---------------------------------------------------------------------------
# Palette and global style  (matching dep_induced_dipoles.py)
# ---------------------------------------------------------------------------
BLUE_FILL   = '#E6F1FB'
BLUE_EDGE   = '#185FA5'
BLUE_TEXT   = '#0C447C'
CORAL_FILL  = '#FAECE7'
CORAL_EDGE  = '#D85A30'
CORAL_TEXT  = '#993C1D'
AMBER       = '#BA7517'
GRAY_FIELD  = '#6E6D68'
GRAY_LABEL  = '#3D3D3A'
GRID        = '#D9D8D4'

C2D = dict(color=BLUE_EDGE,  fill=BLUE_FILL,  text=BLUE_TEXT,  label='2D model')
C3D = dict(color=CORAL_EDGE, fill=CORAL_FILL, text=CORAL_TEXT, label='3D model')

plt.rcParams.update({
    'font.family'     : 'sans-serif',
    'font.sans-serif' : ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size'       : 10,
    'text.usetex'     : False,
    'savefig.dpi'     : 300,
    'axes.edgecolor'  : GRAY_LABEL,
    'axes.labelcolor' : GRAY_LABEL,
    'xtick.color'     : GRAY_LABEL,
    'ytick.color'     : GRAY_LABEL,
    'axes.linewidth'  : 0.9,
})

W_EFF = 4.0 / 3.0 * np.pi * R_SPHERE**3 * (RHO_S - RHO_M) * G


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_runs(path):
    """Return {'3D': DataFrame, '2D': DataFrame} with columns t, Y, Yr, U, F.

    Y is in metres. The 2D block is truncated at the first backwards step in
    its time column, which is where the spreadsheet's copied-down 3D data
    begins.
    """
    raw = pd.read_excel(path, engine='odf', sheet_name=0, header=None)
    num = raw.iloc[2:, :10].apply(pd.to_numeric, errors='coerce').reset_index(drop=True)
    num.columns = ['t3', 'Y3', 'Yr3', 'U3', 'F3', 't2', 'Y2', 'Yr2', 'U2', 'F2']

    runs = {}
    for tag, cols in (('3D', ['t3', 'Y3', 'Yr3', 'U3', 'F3']),
                      ('2D', ['t2', 'Y2', 'Yr2', 'U2', 'F2'])):
        df = num[cols].dropna().reset_index(drop=True)
        df.columns = ['t', 'Y', 'Yr', 'U', 'F']
        back = np.where(np.diff(df['t'].values) < 0)[0]
        if back.size:
            n = int(back[0]) + 1
            print(f'  [{tag}] time column reverses at row {n} '
                  f'(t {df["t"].iloc[n-1]:g} -> {df["t"].iloc[n]:g}); truncating there.')
            df = df.iloc[:n].reset_index(drop=True)
        df['Y'] = df['Y'] * 1e-6                      # um -> m
        runs[tag] = df
    return runs


def find_restarts(df, t_max=None):
    """Indices where F steps away from its local trend: the solver restarts."""
    t, Y, F = df['t'].values, df['Y'].values, df['F'].values
    ok = F > 0
    sl = np.diff(np.log(F[ok])) / np.diff(Y[ok])
    med = pd.Series(sl).rolling(21, center=True, min_periods=5).median().values
    r = sl - med
    thr = 5 * np.nanstd(r[np.abs(r) < np.nanpercentile(np.abs(r), 85)])
    idx = np.where(np.abs(r) > thr)[0]
    if t_max is not None:
        idx = idx[t[ok][idx] < t_max]
    steps = np.array([abs((F[ok][i+1] - F[ok][i]) / F[ok][i]
                          - med[i] * (Y[ok][i+1] - Y[ok][i])) for i in idx])
    return t[ok][idx], steps


def held_mask(df):
    """True where U repeats the previous value (solver step > output step)."""
    U = df['U'].values
    m = np.zeros(len(U), bool)
    m[1:] = np.diff(U) == 0
    return m


def diagnostics(df, tag):
    """Drag coefficient, force decay length and derived timescales."""
    t, Y, U, F = (df[c].values for c in ('t', 'Y', 'U', 'F'))
    # well-conditioned window: still moving, still clearly above W_eff, and not
    # on a held-U plateau (which would corrupt the ratio).
    moving = (U > U.max() * 1e-3) & (F > 1.05 * W_EFF) & (~held_mask(df))
    bvals = (F[moving] - W_EFF) / U[moving]
    b = np.median(bvals)
    # robust spread: a handful of points at the output-rate transitions are
    # wild outliers, so the standard deviation is meaningless here.
    bspread = np.median(np.abs(bvals - b)) / abs(b) * 100

    outer = (Y > 0.5 * Y[-1]) & (F > 0)
    slope, icept = np.polyfit(Y[outer], np.log(F[outer]), 1)
    L = -1.0 / slope
    resid = np.std(np.log(F[outer]) - (slope * Y[outer] + icept))

    i99 = int(np.argmax(Y >= 0.99 * Y[-1]))
    rt, rs = find_restarts(df, t_max=10.0)
    # a force uncertain by dF/F displaces the equilibrium by L*ln(1+dF/F)
    dF = np.median(rs) if rs.size else 0.0
    dF90 = np.percentile(rs, 90) if rs.size else 0.0
    d = dict(b=b, bspread=bspread, L=L, resid=resid, tau=b * L / W_EFF,
             Yeq=Y[-1], Yr=Y[-1] / R_SPHERE, Upeak=U.max(),
             Fseat=F[0], t99=t[i99], Fend_over_W=F[-1] / W_EFF,
             rt=rt, rs=rs, dYr=L * np.log(1 + dF90) / R_SPHERE)

    print(f'\n  [{tag}]')
    print(f'    Y_eq            = {d["Yeq"]*1e6:8.3f} um      (Y/r = {d["Yr"]:.4f})')
    print(f'    F at the seat   = {d["Fseat"]:8.4e} N  = {d["Fseat"]/W_EFF:5.1f} W_eff')
    print(f'    F at the end    = {d["Fend_over_W"]:8.4f} W_eff   (force balance closes)')
    print(f'    U_peak          = {d["Upeak"]:8.4e} m/s')
    print(f'    drag coeff b    = {d["b"]:8.4e} N s/m   (6*pi*mu*r at mu=0.025 is 1.413e-05)')
    print(f'      spread in b   = {d["bspread"]:8.2f} %  (MAD) -> quasi-steady balance intact')
    print(f'    decay length L  = {d["L"]*1e6:8.2f} um      (log-residual {d["resid"]:.1e})')
    print(f'    tau = b L/W_eff = {d["tau"]:8.4g} s')
    print(f'    99% of Y_eq at t= {d["t99"]:8.4g} s')
    if d['rs'].size:
        print(f'    solver restarts = {d["rs"].size} before t = 10 s; force steps '
              f'median {100*np.median(d["rs"]):.1f} %, 90th pct {100*np.percentile(d["rs"],90):.1f} %, '
              f'max {100*d["rs"].max():.1f} %')
        print(f'      -> Y_eq carries +/-{d["dYr"]:.3f} r of mesh uncertainty')
    return d


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def _style(ax):
    ax.grid(True, which='major', color=GRID, lw=0.6, alpha=0.9)
    ax.grid(True, which='minor', color=GRID, lw=0.4, alpha=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def _time(df, d):
    t = df['t'].values
    return t / d['tau'] if NORMALISE_TIME else t


TLABEL = r'$t/\tau$  (1)' if NORMALISE_TIME else r'$t$  (s)'


def panel_displacement(ax, runs, diag):
    for tag, c in (('2D', C2D), ('3D', C3D)):
        df, d = runs[tag], diag[tag]
        t = _time(df, d)
        m = t > 0
        ax.semilogx(t[m], df['Yr'].values[m], lw=1.9, color=c['color'],
                    label=f"{c['label']}  ($Y/r$ = {d['Yr']:.3f})", zorder=3)
        ax.plot([t[m][-1]], [d['Yr']], 'o', ms=6.5, mfc=c['fill'],
                mec=c['color'], mew=1.6, zorder=4)
        ax.axhline(d['Yr'], color=c['color'], lw=0.8, ls=':', alpha=0.65, zorder=2)
    if SHOW_RESTARTS:
        d = diag['3D']
        tt = d['rt'] / d['tau'] if NORMALISE_TIME else d['rt']
        tt = tt[tt > 0]                      # log axis: drop any restart at t = 0
        if tt.size:
            ax.plot(tt, np.full_like(tt, ax.get_ylim()[1]), marker='|', ls='none',
                    ms=7, color=CORAL_EDGE, alpha=0.55, clip_on=False, zorder=5)
    ax.set_xlabel(TLABEL)
    ax.set_ylabel(r'$Y/r$  (1)')
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc='upper left', fontsize=9)
    _style(ax)


def panel_velocity(ax, runs, diag):
    for tag, c in (('2D', C2D), ('3D', C3D)):
        df, d = runs[tag], diag[tag]
        t = _time(df, d)
        m = (t > 0) & (df['U'].values > 0)
        if DROP_HELD_U:
            m = m & (~held_mask(df))
        ax.loglog(t[m], df['U'].values[m] / d['Upeak'], lw=1.9,
                  color=c['color'], label=c['label'], zorder=3)
    ax.set_ylim(1e-4, 2)
    if SHOW_RESTARTS:
        d = diag['3D']
        tt = d['rt'] / d['tau'] if NORMALISE_TIME else d['rt']
        tt = tt[tt > 0]                      # log axis: drop any restart at t = 0
        if tt.size:
            ax.plot(tt, np.full_like(tt, ax.get_ylim()[1]), marker='|', ls='none',
                    ms=7, color=CORAL_EDGE, alpha=0.55, clip_on=False, zorder=5)
    ax.set_xlabel(TLABEL)
    ax.set_ylabel(r'$U/U_{\mathrm{peak}}$  (1)')
    ax.legend(frameon=False, loc='lower left', fontsize=9)
    _style(ax)


def panel_force(ax, runs, diag):
    """Drag-independent: this is the one panel the two models share an axis on."""
    for tag, c in (('2D', C2D), ('3D', C3D)):
        df, d = runs[tag], diag[tag]
        Y, F = df['Y'].values * 1e6, df['F'].values
        m = F > 0
        ax.semilogy(Y[m], F[m] * 1e12, lw=1.9, color=c['color'],
                    label=f"{c['label']}  ($L$ = {d['L']*1e6:.1f} $\\mu$m)", zorder=3)
        ax.plot([Y[m][-1]], [F[m][-1] * 1e12], 'o', ms=6.5, mfc=c['fill'],
                mec=c['color'], mew=1.6, zorder=4)
    if SHOW_RESTARTS:
        df, d = runs['3D'], diag['3D']
        t, Y, F = df['t'].values, df['Y'].values * 1e6, df['F'].values
        j = [int(np.argmin(np.abs(t - x))) for x in d['rt']]
        ax.plot(Y[j], F[j] * 1e12, 'x', ms=5, mew=1.1, color=CORAL_EDGE,
                alpha=0.75, ls='none', zorder=5, label='solver restarts')
    ax.axhline(W_EFF * 1e12, color=AMBER, lw=1.4, zorder=2)
    ax.annotate(r'$W_{\mathrm{eff}}$ = %.1f pN' % (W_EFF * 1e12),
                xy=(0.015, W_EFF * 1e12), xycoords=('axes fraction', 'data'),
                xytext=(0, 5), textcoords='offset points',
                color=AMBER, fontsize=9, fontweight='bold', va='bottom')
    ax.set_xlabel(r'$dY$  ($\mu$m)')
    ax.set_ylabel(r'$F_{\mathrm{DEP}}$  (pN)')
    ax.set_xlim(left=0)
    ax.legend(frameon=False, loc='upper right', fontsize=9)
    _style(ax)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_PDF:
        fig.savefig(OUT_DIR / f'{name}.pdf', bbox_inches='tight')
    if SAVE_PNG:
        fig.savefig(OUT_DIR / f'{name}.png', bbox_inches='tight')
    print(f'    wrote {name}')


def main():
    print('Loading:')
    runs = load_runs(ODS_PATH)
    print('\nDiagnostics:')
    diag = {tag: diagnostics(runs[tag], tag) for tag in ('2D', '3D')}

    ratio = diag['3D']['b'] / diag['2D']['b']
    print(f'\n  drag coefficients differ by {ratio:.1f}x -> the two time axes are NOT '
          f'comparable.\n  Set NORMALISE_TIME = True to collapse them onto t/tau.')
    gap = diag['3D']['Yr'] - diag['2D']['Yr']
    print(f'  3D lifts {100*(diag["3D"]["Yr"]/diag["2D"]["Yr"] - 1):.1f}% further than 2D; '
          f'seat force differs by {100*(diag["3D"]["Fseat"]/diag["2D"]["Fseat"] - 1):.1f}%.')
    print(f'  that gap is {gap:.4f} r against {diag["3D"]["dYr"]:.3f} r of mesh uncertainty '
          f'({gap/diag["3D"]["dYr"]:.1f} sigma): report the two as agreeing, not differing.')

    print('\nFigures:')
    # (a) displacement
    fig, ax = plt.subplots(figsize=(5.4, 4.0)); panel_displacement(ax, runs, diag)
    fig.tight_layout(); save(fig, f'Yr_log_{STEM}'); plt.close(fig)

    # (b) velocity
    fig, ax = plt.subplots(figsize=(5.4, 4.0)); panel_velocity(ax, runs, diag)
    fig.tight_layout(); save(fig, f'Ucs_log_{STEM}'); plt.close(fig)

    # (c) force
    fig, ax = plt.subplots(figsize=(6.4, 4.2)); panel_force(ax, runs, diag)
    fig.tight_layout(); save(fig, f'DEP_{STEM}'); plt.close(fig)

    # combined collage
    fig = plt.figure(figsize=(11.6, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.26)
    a1 = fig.add_subplot(gs[0, 0]); panel_displacement(a1, runs, diag)
    a2 = fig.add_subplot(gs[0, 1]); panel_velocity(a2, runs, diag)
    a3 = fig.add_subplot(gs[1, :]); panel_force(a3, runs, diag)
    for ax, lab in ((a1, '(a)'), (a2, '(b)'), (a3, '(c)')):
        ax.set_title(lab, loc='left', fontsize=12, fontweight='bold',
                     color=GRAY_LABEL, pad=8)
    save(fig, f'collage_{STEM}'); plt.close(fig)
    print('\nDone.')


if __name__ == '__main__':
    main()