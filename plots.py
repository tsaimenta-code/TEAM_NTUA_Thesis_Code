import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# ---------- Files to process ----------
# Map each input workbook to its own output subfolder.
DATASETS = {
    r"C:\Users\tsaim\Desktop\msc\Thesis\presentations\PEA_10X10.ods": r"C:\Users\tsaim\Desktop\msc\Thesis\presentations\pres\10x10",
    r"C:\Users\tsaim\Desktop\msc\Thesis\presentations\PEA_20X20.ods": r"C:\Users\tsaim\Desktop\msc\Thesis\presentations\pres\20x20",
    r"C:\Users\tsaim\Desktop\msc\Thesis\presentations\PEA_30X30.ods": r"C:\Users\tsaim\Desktop\msc\Thesis\presentations\pres\30x30",
    r"C:\Users\tsaim\Desktop\msc\Thesis\presentations\PEA_50X50.ods": r"C:\Users\tsaim\Desktop\msc\Thesis\presentations\pres\50x50",
}

col_names = ["t", "dY", "Y/r", "U", "DEP"]
EXPECTED_VOLTAGES = [2.0, 4.0, 6.0, 8.0, 16.0, 24.0]

# ---------- Column-label normalization ----------
def normalize_label(s):
    s = str(s).strip().lower().replace(" ", "")
    if s.startswith("t(s)"):
        return "t"
    if s.startswith("dy"):
        return "dY"
    if s.startswith("y/r"):
        return "Y/r"
    if s.startswith("ucs"):
        return "U"
    if s.startswith("f_dep") or s.startswith("fdep"):
        return "DEP"
    return None

# ---------- Voltage block boundaries per sheet ----------
def find_blocks(raw):
    """Locate each 'VP2P = xV' marker in row 0 and its start column.
    Block width is inferred from the gap to the next marker (or sheet
    edge for the last block) rather than assumed to be a fixed 5."""
    row0 = raw.iloc[0]
    starts = []
    for i in range(raw.shape[1]):
        cell = str(row0[i]).strip()
        if cell.lower() in ("nan", ""):
            continue
        clean = cell.split("=")[1].strip() if "=" in cell else cell
        clean = clean.replace("V", "").strip()
        try:
            v = float(clean)
        except ValueError:
            continue
        starts.append((i, v))
    return starts

# ---------- Header row detection (not always row 2!) ----------
def find_header_row(raw, start_col, search_rows=range(0, 6)):
    for r in search_rows:
        if r >= raw.shape[0]:
            continue
        if normalize_label(raw.iloc[r, start_col]) == "t":
            return r
    return None

# ---------- Block extraction (label-based, with positional fallback) ----------
def get_block(raw, start_col, end_col, voltage, r_um, sheet_name, warnings):
    header_row = find_header_row(raw, start_col)
    if header_row is None:
        warnings.append(f"{sheet_name} {voltage}V: could not locate header row near col {start_col}; skipping block")
        return pd.DataFrame(columns=col_names + ["r_um", "Vp2p"])

    data_start = header_row + 1
    sub = raw.iloc[:, start_col:end_col]
    header = sub.iloc[header_row]
    names = [normalize_label(header.iloc[j]) for j in range(sub.shape[1])]

    cnt = Counter([n for n in names if n is not None])
    dupes = [k for k, c in cnt.items() if c > 1]
    missing_names = [n for n in col_names if n not in names]
    width = end_col - start_col

    use_positional = False
    if (dupes or missing_names) and width == 5:
        # Full-width block with a labeling error (e.g. a column header typo
        # duplicating another column's label) rather than a genuinely
        # missing column. Trust the sheet's universal column order instead.
        use_positional = True
        warnings.append(
            f"{sheet_name} {voltage}V: header labels inconsistent {header.tolist()} "
            f"(dupes={dupes}, missing={missing_names}) -> using positional column order"
        )

    data = {}
    if use_positional:
        for j, name in enumerate(col_names):
            data[name] = pd.to_numeric(sub.iloc[data_start:, j], errors="coerce").reset_index(drop=True)
    else:
        for j in range(sub.shape[1]):
            name = names[j]
            if name is None:
                continue
            data[name] = pd.to_numeric(sub.iloc[data_start:, j], errors="coerce").reset_index(drop=True)
        n = len(next(iter(data.values()))) if data else 0
        for name in col_names:
            if name not in data:
                # genuinely absent in this block
                data[name] = pd.Series([np.nan] * n)

    block = pd.DataFrame(data)[col_names]
    block = block.dropna(how="all", subset=col_names)
    block["r_um"] = r_um
    block["Vp2p"] = voltage
    return block

def transf_block(block, voltage, r_um, fallback_t):
    if block.empty or block["t"].isna().all():
        block = pd.DataFrame({"t": fallback_t, "dY": 0.0, "Y/r": 0.0, "U": 0.0, "DEP": 0.0})
        block["r_um"] = r_um
        block["Vp2p"] = voltage
        status = "Inadequate"
    else:
        status = "OK"
    block["status"] = status
    return block

# ---------- Shading helper ----------
def add_region_shading(ax, t_max, edges=None, colors=None):
    if edges is None:
        edges = [1e-4, 0.001, 0.1, 1, t_max]
    if colors is None:
        colors = ["#bebebe", "#9f9f9f", "#7e7e7e", "#4b4b4b"]
    for i in range(len(edges) - 1):
        ax.axvspan(edges[i], edges[i+1], color=colors[i], alpha=0.3)
    return ax

# ---------- Y/r vs t (log-x, shaded, steady-state marker) ----------
def Y2R_log_over_time(ax, tidy, r_um2):
    r_data2 = tidy[tidy.r_um == r_um2]
    t_max = r_data2["t"].max()
    ax.set_xscale("log")
    add_region_shading(ax, t_max)
    for voltage in sorted(r_data2["Vp2p"].unique()):
        v_data = r_data2[r_data2["Vp2p"] == voltage]
        ax.plot(v_data["t"], v_data["Y/r"], label=f"{voltage}V")
        steady = v_data[v_data["t"] >= 1]
        if not steady.empty:
            mean_yr = steady["Y/r"].mean()
            t_marker = steady["t"].max()
            ax.plot(t_marker, mean_yr, marker="o", color="black", markersize=4)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    return ax

# ---------- Ucs vs t (log-x, shaded, normalized to each curve's own peak) ----------
def Ucs_over_time(ax, tidy, r_um2):
    r_data2 = tidy[tidy.r_um == r_um2]
    t_max = r_data2["t"].max()
    ax.set_xscale("log")
    add_region_shading(ax, t_max)

    for voltage in sorted(r_data2["Vp2p"].unique()):
        v_data = r_data2[r_data2["Vp2p"] == voltage].sort_values("t")
        u_max = v_data["U"].max()
        if u_max == 0 or pd.isna(u_max):
            ax.plot(v_data["t"], v_data["U"] * 0.0, label=f"{voltage}V")
            continue

        # The first 1-2 recorded timesteps are a solver dead-time/noise
        # artifact (U near 0, sometimes non-monotonically so, before motion
        # actually starts) rather than a real physical ramp-up. Trim
        # everything before the curve reaches its plateau/peak, so each
        # curve's first plotted point IS that peak. Use a small tolerance
        # (99% of the true max) rather than the exact numerical maximum:
        # low-voltage curves plateau almost immediately and then stay flat
        # for a long stretch, where floating-point noise on the order of
        # ~0.1-1% can make the strict maximum land far into that plateau
        # instead of at its start, making the curve appear to begin much
        # later than it physically does.
        near_peak = (v_data["U"].to_numpy() >= 0.99 * u_max)
        peak_pos = int(near_peak.argmax()) if near_peak.any() else 0
        v_data = v_data.iloc[peak_pos:]

        u_norm = v_data["U"] / u_max
        ax.plot(v_data["t"], u_norm, label=f"{voltage}V")

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    return ax

# ---------- DEP vs dY ----------
def DEP_vs_dY(ax, tidy, r_um2):
    r_data2 = tidy[tidy.r_um == r_um2]
    ax.set_yscale("log")
    for voltage in sorted(r_data2["Vp2p"].unique()):
        v_data = r_data2[r_data2["Vp2p"] == voltage]
        ax.plot(v_data["dY"], v_data["DEP"], label=f"{voltage}V")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    return ax

# ---------- Power-law fits (appendix) ----------
def fit_power_law(t0_data, group_col, x_col, y_col="DEP"):
    results = []
    for group_value, group_df in t0_data.groupby(group_col):
        group_df = group_df[group_df[y_col] > 0]
        if len(group_df) < 2:
            continue
        logx = np.log(group_df[x_col])
        logy = np.log(group_df[y_col])
        slope, intercept = np.polyfit(logx, logy, 1)
        results.append((group_value, slope, intercept))
    return pd.DataFrame(results, columns=[group_col, "slope", "intercept"])

# ---------- Process one workbook end-to-end ----------
def process_dataset(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    warnings = []

    xl = pd.ExcelFile(input_path, engine="odf")
    sheets = [s for s in xl.sheet_names if re.fullmatch(r"R\d+", s)]
    radii = {s: int(s[1:]) for s in sheets}

    all_blocks = []
    for sheet in sheets:
        raw = xl.parse(sheet, header=None)
        r_um = radii[sheet]
        starts = find_blocks(raw)

        found_voltages = [v for _, v in starts]
        missing_v = [v for v in EXPECTED_VOLTAGES if v not in found_voltages]
        if missing_v:
            warnings.append(f"{sheet}: voltage block(s) {missing_v} not found in source data (not fabricated, simply excluded)")

        raw_blocks, fallback_t = [], None
        for idx, (start_col, voltage) in enumerate(starts):
            end_col = starts[idx + 1][0] if idx + 1 < len(starts) else raw.shape[1]
            b = get_block(raw, start_col, end_col, voltage, r_um, sheet, warnings)
            raw_blocks.append((voltage, b))
            if fallback_t is None and not b.empty and not b["t"].isna().all():
                fallback_t = b["t"]
        for voltage, b in raw_blocks:
            all_blocks.append(transf_block(b, voltage, r_um, fallback_t))

    tidy = pd.concat(all_blocks, ignore_index=True)

    # Initial-condition artifact: at t=0 the solver's dY and Y/r readings are
    # numerical noise and should be ~0 (particle starts at rest). DEP force at
    # t=0 is a real physical starting value (the initial driving force) and is
    # left untouched.
    tidy.loc[tidy["t"] == 0, ["dY", "Y/r"]] = 0.0

    # Known solver artifact: final saved U step is spurious -> carry forward previous value
    for (r, v), grp in tidy.groupby(["r_um", "Vp2p"]):
        idx = grp.sort_values("t").index
        if len(idx) >= 2:
            tidy.loc[idx[-1], "U"] = tidy.loc[idx[-2], "U"]

    # ---------- Figures ----------
    for r in radii.values():
        fig, ax = plt.subplots()
        ax.set_title(f"Response of R = {r} µm")
        ax.set_xlabel("Time (s) - Logarithmic Scale")
        ax.set_ylabel("Relative Displacement [1]")
        Y2R_log_over_time(ax, tidy, r)
        fig.savefig(f"{output_dir}/Yr_log_r{r}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    for r in radii.values():
        fig, ax = plt.subplots()
        ax.set_title(f"Response of R = {r} µm")
        ax.set_xlabel("Time (s) - Logarithmic Scale")
        ax.set_ylabel("Normalized Velocity [U / U_peak]")
        Ucs_over_time(ax, tidy, r)
        fig.savefig(f"{output_dir}/Ucs_log_r{r}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    for r in radii.values():
        fig, ax = plt.subplots()
        ax.set_title(f"DEP Force applied on R = {r}")
        ax.set_xlabel("Displacement (um)")
        ax.set_ylabel("DEP Force")
        DEP_vs_dY(ax, tidy, r)
        fig.savefig(f"{output_dir}/DEP_R{r}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---------- Appendix: power-law fits ----------
    f0 = tidy[(tidy["t"] == 0) & (tidy["DEP"] > 0)]
    voltage_scaling = fit_power_law(f0, group_col="r_um", x_col="Vp2p")
    radius_scaling = fit_power_law(f0, group_col="Vp2p", x_col="r_um")

    print(f"\n===== {os.path.basename(input_path)} =====")
    if warnings:
        print("Data issues found and handled:")
        for w in warnings:
            print("  -", w)
    print("Voltage-scaling fit (per radius):")
    print(voltage_scaling)
    print("Radius-scaling fit (per voltage):")
    print(radius_scaling)

    return tidy, warnings

if __name__ == "__main__":
    for input_path, output_dir in DATASETS.items():
        process_dataset(input_path, output_dir)