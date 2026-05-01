from __future__ import annotations

import argparse
import shutil
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


PT_TO_INCH = 1.0 / 72.27
NIELSEN_THEME_ORDER = [
    "1. Visibility of system status",
    "2. Match between system and the real world",
    "3. User control and freedom",
    "4. Consistency and standards",
    "5. Error prevention",
    "6. Recognition rather than recall",
    "7. Flexibility and efficiency of use",
    "8. Aesthetic and minimalist design",
    "9. Help users recognize, diagnose, and recover from errors",
    "10. Help and documentation",
]


def _wrap_label(label: str, width: int = 20) -> str:
    s = str(label).replace("/", "/ ").replace(" / ", " / ")
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=True, break_on_hyphens=True)) or str(label)


def _short_heatmap_label(label: str) -> str:
    exact = {
        "Build/CI/installation/distribution release issues": "Build/CI/install/distrib release",
        "User confusion / unclear documentation": "User confusion / unclear docs",
        "Missing feature / enhancement request": "Missing feature / enhancement req",
        "Notification/Logging /Web UI Issues": "Notif/logging / Web UI",
        "Authentication/Authorization tools": "Auth/Authz tools",
        "Key Management Core / Secrets Backend": "Secrets Engine",
        "Policy/configuration": "Policy/config",
        "1. Visibility of system status": "1. Visibility of system status",
        "2. Match between system and the real world": "2. Match between system & real-world",
        "3. User control and freedom": "3. User control",
        "4. Consistency and standards": "4. consistency & standards",
        "5. Error prevention": "5. Error Prevention",
        "6. Recognition rather than recall": "6. Recognition not Recall",
        "7. Flexibility and efficiency of use": "7. Flexibility & Efficiency",
        "8. Aesthetic and minimalist design": "8. Aesthetic & Min. design",
        "9. Help users recognize, diagnose, and recover from errors": "9. Diagnose & Recognize errors",
        "10. Help and documentation": "10. Help and docs",
    }
    if label in exact:
        return exact[label]
    # Light generic shortening for long labels.
    out = str(label)
    out = out.replace("documentation", "docs")
    out = out.replace("configuration", "config")
    out = out.replace("Authentication", "Auth")
    out = out.replace("Authorization", "Authz")
    out = out.replace("Management", "Mgmt")
    out = out.replace("Distribution", "Distrib")
    out = out.replace("distribution", "distrib")
    out = out.replace("enhancement request", "enhancement req")
    return out


def _long_l1_heatmap_label(label: str) -> str:
    exact = {
        "Authentication friction": "Authentication friction",
        "Build/CI/installation/distribution release issues": "Build/CI/Install/release",
        "Configuration friction": "Configuration Friction",
        "Integration failure/issues": "Integration Issues",
        "Missing feature / enhancement request": "Missing feature/enhancement",
        "Notification/Logging /Web UI Issues": "Notification/logging/web ui",
        "Performance issue": "Perfromance Issues",
        "Security concerns": "Security Concerns",
        "Tedious Workflows": "Tedious workflows",
        "Unexpected behavior": "unexpected behaviour",
        "User confusion / unclear documentation": "User confusion/unclear documentation",
        "User confusion / unclear documentation/documentation improvements": "User confusion/unclear documentation",
    }
    return exact.get(str(label), str(label))


def _read_matrix(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.set_index(df.columns[0])


def _set_pub_style(style_path: Path, use_tex: str):
    plt.style.use(str(style_path))
    if use_tex == "on":
        matplotlib.rcParams["text.usetex"] = True
    elif use_tex == "off":
        matplotlib.rcParams["text.usetex"] = False
    else:
        matplotlib.rcParams["text.usetex"] = shutil.which("latex") is not None


def _figure_size(width_pt: float, ratio: float = (5**0.5 - 1) / 2.0) -> tuple[float, float]:
    width_in = width_pt * PT_TO_INCH
    return (width_in, width_in * ratio)


def _save_dual(fig: plt.Figure, base_path: Path):
    fig.savefig(base_path.with_suffix(".pdf"))
    fig.savefig(base_path.with_suffix(".png"), dpi=220)


def plot_sig_heatmap(
    beta: pd.DataFrame,
    pvals: pd.DataFrame,
    out_base: Path,
    alpha: float,
    width_pt: float,
    title: str,
    xlabel: str,
    shorten_dense_labels: bool = True,
    xlabels_override: list[str] | None = None,
    x_tick_weight: str = "normal",
    wrap_width_override: int | None = None,
    font_scale: float = 1.0,
):
    pvals = pvals.reindex(index=beta.index, columns=beta.columns)

    sig = pvals < alpha
    arr = beta.where(sig).to_numpy(dtype=float)
    masked = np.ma.masked_invalid(arr)
    vmax = np.nanmax(np.abs(beta.to_numpy(dtype=float)))
    n_cols = len(beta.columns)
    dense = n_cols >= 10
    ratio = 0.74 if dense else 0.48
    wrap_width = 10 if dense else 17
    if wrap_width_override is not None:
        wrap_width = wrap_width_override
    x_tick_fs = (5.8 if dense else 7.5) * font_scale
    y_tick_fs = (7.6 if dense else 8.4) * font_scale
    ann_fs = (5.8 if dense else 6.3) * font_scale
    title_fs = (10.5 if dense else 11.5) * font_scale
    axis_label_fs = (9.5 if dense else 10.5) * font_scale
    cbar_label_fs = 10.0 * font_scale
    cbar_tick_fs = 8.8 * font_scale
    bottom_margin = 0.43 if dense else 0.24

    fig, ax = plt.subplots(figsize=_figure_size(width_pt, ratio=ratio))
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="#efefef")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_title(title, fontsize=title_fs)
    ax.set_xlabel(xlabel, fontsize=axis_label_fs)
    ax.set_ylabel("Tool", fontsize=axis_label_fs)
    if xlabels_override is not None and len(xlabels_override) == len(beta.columns):
        xlabels = xlabels_override
    else:
        xlabels = [_short_heatmap_label(c) if (dense and shorten_dense_labels) else c for c in beta.columns]
    ax.set_xticks(np.arange(len(beta.columns)))
    ax.set_xticklabels(
        [_wrap_label(c, wrap_width) for c in xlabels],
        rotation=50 if dense else 35,
        ha="right",
        fontweight=x_tick_weight,
    )
    ax.set_yticks(np.arange(len(beta.index)))
    ax.set_yticklabels(beta.index.tolist())
    ax.tick_params(axis="x", labelsize=x_tick_fs)
    ax.tick_params(axis="y", labelsize=y_tick_fs)
    fig.subplots_adjust(bottom=bottom_margin)

    for i, tool in enumerate(beta.index):
        for j, theme in enumerate(beta.columns):
            if bool(sig.loc[tool, theme]):
                ax.text(j, i, f"{beta.loc[tool, theme]:.3f}", ha="center", va="center", fontsize=ann_fs)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(r"Poisson slope ($\beta_1$)", fontsize=cbar_label_fs)
    cbar.ax.tick_params(labelsize=cbar_tick_fs)

    _save_dual(fig, out_base)
    plt.close(fig)


def _trend_long_to_matrices(
    trends_csv: Path,
    theme_order: list[str] | None = None,
):
    df = pd.read_csv(trends_csv)
    needed = {"tool", "theme", "beta", "p_value"}
    miss = needed.difference(df.columns)
    if miss:
        raise KeyError(f"Missing required columns in {trends_csv}: {sorted(miss)}")

    beta = df.pivot_table(index="tool", columns="theme", values="beta", aggfunc="mean")
    pvals = df.pivot_table(index="tool", columns="theme", values="p_value", aggfunc="mean")

    beta = beta.sort_index()
    pvals = pvals.reindex(index=beta.index, columns=beta.columns)

    if theme_order is None:
        ordered_cols = sorted(beta.columns.tolist())
    else:
        ordered_cols = [c for c in theme_order if c in beta.columns] + [c for c in beta.columns if c not in theme_order]
    beta = beta.reindex(columns=ordered_cols)
    pvals = pvals.reindex(columns=ordered_cols)
    sig = (pvals < 0.05).astype(bool)
    return beta, pvals, sig


def _plot_curve_panel(
    wide_csv: Path,
    out_base: Path,
    title: str,
    width_pt: float,
    legend_loc: tuple[float, float] = (0.98, 0.97),
    font_scale: float = 1.0,
):
    df = pd.read_csv(wide_csv)
    month_col = df.columns[0]
    df[month_col] = pd.to_datetime(df[month_col], errors="coerce")
    df = df.sort_values(month_col)
    series_cols = [c for c in df.columns if c != month_col]

    fig, ax = plt.subplots(figsize=_figure_size(width_pt, ratio=0.62))
    for col in series_cols:
        ax.plot(df[month_col], df[col], label=col)

    title_fs = 11.5 * font_scale
    axis_label_fs = 10.5 * font_scale
    tick_fs = 9.0 * font_scale
    legend_fs = 8.0 * font_scale

    ax.set_title(title, fontsize=title_fs)
    ax.set_xlabel("Calendar month", fontsize=axis_label_fs)
    ax.set_ylabel("Expected monthly issue count", fontsize=axis_label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    # In-panel legend (inside plot whitespace)
    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=legend_loc,
        ncol=1,
        frameon=True,
        borderpad=0.3,
        labelspacing=0.25,
        handlelength=1.8,
        fontsize=legend_fs,
    )
    legend.get_frame().set_alpha(0.88)
    legend.get_frame().set_linewidth(0.5)

    _save_dual(fig, out_base)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-style RQ3 figures from fig_rq3_* CSV tables."
    )
    parser.add_argument("--tables-dir", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/plots/trend/pub_latex"))
    parser.add_argument(
        "--style",
        type=Path,
        default=Path("scripts/pubfigs/paper.mplstyle"),
        help="Matplotlib style file.",
    )
    parser.add_argument(
        "--width-pt",
        type=float,
        default=510.0,
        help="Figure width in LaTeX points (e.g., ~240 one-column, ~510 two-column).",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for heatmap mask.")
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Global font-size scale for pub_latex figures (1.0 keeps prior sizing).",
    )
    parser.add_argument(
        "--curve-font-scale",
        type=float,
        default=None,
        help="Optional font-size scale for curve panels only. If unset, uses --font-scale.",
    )
    parser.add_argument(
        "--use-tex",
        choices=["auto", "on", "off"],
        default="auto",
        help="Enable LaTeX rendering for figure text.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _set_pub_style(args.style, args.use_tex)

    l1_beta = _read_matrix(args.tables_dir / "fig_rq3_heatmap_l1_sig_beta_matrix.csv")
    l1_p = _read_matrix(args.tables_dir / "fig_rq3_heatmap_l1_sig_p_matrix.csv")
    l1_full_labels = [_long_l1_heatmap_label(c) for c in l1_beta.columns]

    # Non-bold, full written labels.
    curve_font_scale = args.curve_font_scale if args.curve_font_scale is not None else args.font_scale

    plot_sig_heatmap(
        beta=l1_beta,
        pvals=l1_p,
        out_base=args.out_dir / "heatmap_beta_sig_L1_Theme_pub_latex",
        alpha=args.alpha,
        width_pt=args.width_pt,
        title="Poisson Time Slopes by Tool and Primary Usability Theme",
        xlabel="Inductively Generated Primary Theme",
        shorten_dense_labels=False,
        xlabels_override=l1_full_labels,
        x_tick_weight="normal",
        wrap_width_override=14,
        font_scale=args.font_scale,
    )

    # Bold-label variant, full written labels.
    plot_sig_heatmap(
        beta=l1_beta,
        pvals=l1_p,
        out_base=args.out_dir / "heatmap_beta_sig_L1_Theme_pub_latex_boldlabels",
        alpha=args.alpha,
        width_pt=args.width_pt,
        title="Poisson Time Slopes by Tool and Primary Usability Theme",
        xlabel="Inductively Generated Primary Theme",
        shorten_dense_labels=False,
        xlabels_override=l1_full_labels,
        x_tick_weight="bold",
        wrap_width_override=14,
        font_scale=args.font_scale,
    )

    # Build Nielsen matrices from trend-long CSV, save for reproducibility, and plot.
    nielsen_beta, nielsen_p, nielsen_sig = _trend_long_to_matrices(
        args.tables_dir / "poisson_trends_Nielsen_theme.csv",
        theme_order=NIELSEN_THEME_ORDER,
    )
    nielsen_beta.to_csv(args.tables_dir / "fig_rq3_heatmap_nielsen_sig_beta_matrix.csv")
    nielsen_p.to_csv(args.tables_dir / "fig_rq3_heatmap_nielsen_sig_p_matrix.csv")
    nielsen_sig.to_csv(args.tables_dir / "fig_rq3_heatmap_nielsen_sig_mask_matrix.csv")

    plot_sig_heatmap(
        beta=nielsen_beta,
        pvals=nielsen_p,
        out_base=args.out_dir / "heatmap_beta_sig_Nielsen_theme_pub_latex",
        alpha=args.alpha,
        width_pt=args.width_pt,
        title="Poisson Time Slopes by Tool and Nielsen Theme",
        xlabel="Nielsen Theme",
        font_scale=args.font_scale,
    )

    # Build affected-component matrices from trend-long CSV, save, and plot.
    comp_beta, comp_p, comp_sig = _trend_long_to_matrices(
        args.tables_dir / "poisson_trends_Associated_Component_Theme.csv",
    )
    comp_beta.to_csv(args.tables_dir / "fig_rq3_heatmap_associated_component_sig_beta_matrix.csv")
    comp_p.to_csv(args.tables_dir / "fig_rq3_heatmap_associated_component_sig_p_matrix.csv")
    comp_sig.to_csv(args.tables_dir / "fig_rq3_heatmap_associated_component_sig_mask_matrix.csv")

    plot_sig_heatmap(
        beta=comp_beta,
        pvals=comp_p,
        out_base=args.out_dir / "heatmap_beta_sig_Associated_Component_Theme_pub_latex",
        alpha=args.alpha,
        width_pt=args.width_pt,
        title="Poisson Time Slopes by Tool and Affected Component Theme",
        xlabel="Affected Component Theme",
        font_scale=args.font_scale,
    )

    _plot_curve_panel(
        wide_csv=args.tables_dir / "fig_rq3_aggregate_l1_curves_wide.csv",
        out_base=args.out_dir / "aggregate_poisson_expected_counts_l1_theme_pub_latex",
        title="Aggregate Poisson fitted means by inductively generated primary theme",
        width_pt=args.width_pt,
        font_scale=curve_font_scale,
    )

    _plot_curve_panel(
        wide_csv=args.tables_dir / "fig_rq3_aggregate_components_curves_wide.csv",
        out_base=args.out_dir / "aggregate_poisson_expected_counts_associated_component_pub_latex",
        title="Aggregate Poisson fitted means by affected component",
        width_pt=args.width_pt,
        font_scale=curve_font_scale,
    )

    print("Wrote publication-style figures to:", args.out_dir.resolve())


if __name__ == "__main__":
    main()
