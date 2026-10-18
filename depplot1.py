import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BLUE_FILL   = '#E6F1FB'
BLUE_EDGE   = '#185FA5'
BLUE_TEXT   = '#0C447C'
CORAL_FILL  = '#FAECE7'
CORAL_EDGE  = '#D85A30'
CORAL_TEXT  = '#993C1D'
AMBER       = '#BA7517'
GRAY_FIELD  = '#6E6D68'
GRAY_SPHERE = '#5F5E5A'
GRAY_LABEL  = '#3D3D3A'

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family'     : 'sans-serif',
    'font.sans-serif' : ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size'       : 10,
    'text.usetex'     : False,
    'savefig.dpi'     : 300,
})

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def force_arrow(ax, x_tail, x_head, color, lw=2.2, ms=11, zorder=6):
    """Draw a horizontal force arrow along y = 0."""
    ann = ax.annotate(
        '', xy=(x_head, 0), xytext=(x_tail, 0),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                        mutation_scale=ms)
    )
    ann.arrow_patch.set_zorder(zorder)


# ---------------------------------------------------------------------------
# Panel drawing function
# ---------------------------------------------------------------------------
def draw_panel(ax: plt.Axes, uniform: bool, title: str) -> None:
    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-2.9, 2.6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    R  = 1.00   # sphere radius  (data units)
    rc = 0.13   # charge-circle radius

    # ── field lines ────────────────────────────────────────────────────────
    y_vals  = np.linspace(-1.8, 1.8, 7)
    x0, x1  = -3.0, 3.0
    focal_x = 5.0           # off-screen convergence point (non-uniform field)

    for y in y_vals:
        y0 = y
        y1 = y if uniform else y * (focal_x - x1) / (focal_x - x0)
        ann = ax.annotate(
            '', xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle='->', color=GRAY_FIELD,
                            lw=0.85, mutation_scale=8)
        )
        ann.arrow_patch.set_zorder(1)

    # ── sphere ─────────────────────────────────────────────────────────────
    ax.add_patch(Circle((0, 0), R, facecolor='#F9F9F8',
                         edgecolor=GRAY_SPHERE, linewidth=1.6, zorder=3))

    # ── surface charges ────────────────────────────────────────────────────
    for ang in np.radians([150, 180, 210]):        # negative / left pole
        cx, cy = np.cos(ang), np.sin(ang)
        ax.add_patch(Circle((cx, cy), rc, facecolor=BLUE_FILL,
                             edgecolor=BLUE_EDGE, linewidth=1.2, zorder=4))
        ax.text(cx, cy, '\u2212', ha='center', va='center',
                fontsize=11.5, fontweight='bold', color=BLUE_TEXT, zorder=5)

    for ang in np.radians([330, 0, 30]):           # positive / right pole
        cx, cy = np.cos(ang), np.sin(ang)
        ax.add_patch(Circle((cx, cy), rc, facecolor=CORAL_FILL,
                             edgecolor=CORAL_EDGE, linewidth=1.2, zorder=4))
        ax.text(cx, cy, '+', ha='center', va='center',
                fontsize=11.5, fontweight='bold', color=CORAL_TEXT, zorder=5)

    # ── force arrows ───────────────────────────────────────────────────────
    x_lft = -(R + rc)
    x_rgt =  (R + rc)

    if uniform:
        L = 0.68
        force_arrow(ax, x_lft, x_lft - L, BLUE_EDGE)
        ax.text(x_lft - L - 0.10, 0.19, 'F\u2212',
                ha='right', fontsize=10.5, color=BLUE_EDGE, fontweight='bold')
        force_arrow(ax, x_rgt, x_rgt + L, CORAL_EDGE)
        ax.text(x_rgt + L + 0.10, 0.19, 'F+',
                ha='left', fontsize=10.5, color=CORAL_EDGE, fontweight='bold')
        ax.text(0, -2.55, '|F+| = |F\u2212|  \u2192  no net force',
                ha='center', fontsize=10, color=GRAY_LABEL)

    else:
        Ls, Ll = 0.38, 0.80          # short (weak-field) / long (strong-field)
        force_arrow(ax, x_lft, x_lft - Ls, BLUE_EDGE)
        ax.text(x_lft - Ls - 0.10, 0.19, 'F\u2212',
                ha='right', fontsize=10.5, color=BLUE_EDGE, fontweight='bold')
        force_arrow(ax, x_rgt, x_rgt + Ll, CORAL_EDGE)
        ax.text(x_rgt + Ll + 0.10, 0.19, 'F\u208A',
                ha='left', fontsize=10.5, color=CORAL_EDGE, fontweight='bold')

        # net DEP force arrow
        dep_y = -1.72
        ann = ax.annotate(
            '', xy=(0.72, dep_y), xytext=(-0.72, dep_y),
            arrowprops=dict(arrowstyle='->', color=AMBER, lw=2.6,
                            mutation_scale=12)
        )
        ann.arrow_patch.set_zorder(6)
        ax.text(0, dep_y - 0.26, r'$F_{\mathrm{DEP}}$',
                ha='center', fontsize=11, color=AMBER, fontweight='bold')
        ax.text(0, -2.55, r'|F+| > |F-|  $\rightarrow$  $F_{\mathrm{DEP}} \neq 0$',
                ha='center', fontsize=10, color=GRAY_LABEL)


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.8))

draw_panel(ax1, uniform=True,  title='A \u2014 uniform field')
draw_panel(ax2, uniform=False, title='B \u2014 non-uniform field')

# E-field labels
kw = dict(fontsize=10, color=GRAY_FIELD, style='italic')
ax1.text(-2.9,  2.3, '$E$  \u2192', **kw)
ax2.text(-2.9,  2.3, 'weak $E$  \u2192', **kw)
ax2.text( 2.9,  0.5, '\u2192 strong $E$', ha='right', **kw)

plt.tight_layout(pad=2.5)
plt.savefig(r'C:\Users\tsaim\Desktop\msc\Thesis\Images\dep_induced_dipoles.pdf', bbox_inches='tight')
plt.savefig(r'C:\Users\tsaim\Desktop\msc\Thesis\Images\dep_induced_dipoles.png', bbox_inches='tight')
print("Done. Saved PDF and PNG.")