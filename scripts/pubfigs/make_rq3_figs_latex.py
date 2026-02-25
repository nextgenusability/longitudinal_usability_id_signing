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
    return "\n".join(textwrap.wrap(str(label), width=width)) or str(label)


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
):
    pvals = pvals.reindex(index=beta.index, columns=beta.columns)

    sig = pvals < alpha
    arr = beta.where(sig).to_numpy(dtype=float)
    masked = np.ma.masked_invalid(arr)
    vmax = np.nanmax(np.abs(beta.to_numpy(dtype=float)))

    fig, ax = plt.subplots(figsize=_figure_size(width_pt, ratio=0.48))
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="#efefef")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tool")
    ax.set_xticks(np.arange(len(beta.columns)))
    ax.set_xticklabels([_wrap_label(c, 17) for c in beta.columns], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(beta.index)))
    ax.set_yticklabels(beta.index.tolist())

    for i, tool in enumerate(beta.index):
        for j, theme in enumerate(beta.columns):
            if bool(sig.loc[tool, theme]):
                ax.text(j, i, f"{beta.loc[tool, theme]:.3f}", ha="center", va="center", fontsize=6.3)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(r"Poisson slope ($\beta_1$)")

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
):
    df = pd.read_csv(wide_csv)
    month_col = df.columns[0]
    df[month_col] = pd.to_datetime(df[month_col], errors="coerce")
    df = df.sort_values(month_col)
    series_cols = [c for c in df.columns if c != month_col]

    fig, ax = plt.subplots(figsize=_figure_size(width_pt, ratio=0.62))
    for col in series_cols:
        ax.plot(df[month_col], df[col], label=col)

    ax.set_title(title)
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("Expected monthly issue count")
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
    plot_sig_heatmap(
        beta=l1_beta,
        pvals=l1_p,
        out_base=args.out_dir / "heatmap_beta_sig_L1_Theme_pub_latex",
        alpha=args.alpha,
        width_pt=args.width_pt,
        title="Poisson Time Slopes by Tool and Primary Usability Theme",
        xlabel="Inductively Generated Primary Theme",
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
    )

    _plot_curve_panel(
        wide_csv=args.tables_dir / "fig_rq3_aggregate_l1_curves_wide.csv",
        out_base=args.out_dir / "aggregate_poisson_expected_counts_l1_theme_pub_latex",
        title="Aggregate Poisson fitted means by inductively generated primary theme",
        width_pt=args.width_pt,
    )

    _plot_curve_panel(
        wide_csv=args.tables_dir / "fig_rq3_aggregate_components_curves_wide.csv",
        out_base=args.out_dir / "aggregate_poisson_expected_counts_associated_component_pub_latex",
        title="Aggregate Poisson fitted means by affected component",
        width_pt=args.width_pt,
    )

    print("Wrote publication-style figures to:", args.out_dir.resolve())


if __name__ == "__main__":
    main()
