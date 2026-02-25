from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _wrap_label(label: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(str(label), width=width)) or str(label)


def _prep_matrix(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    idx_col = df.columns[0]
    return df.set_index(idx_col)


def plot_l1_sig_heatmap(
    beta_csv: Path,
    p_csv: Path,
    out_path: Path,
    alpha: float = 0.05,
):
    beta = _prep_matrix(beta_csv)
    pvals = _prep_matrix(p_csv)
    pvals = pvals.reindex(index=beta.index, columns=beta.columns)

    sig = pvals < alpha
    arr = beta.where(sig).to_numpy(dtype=float)
    masked = np.ma.masked_invalid(arr)

    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="#f2f2f2")

    vmax = np.nanmax(np.abs(beta.to_numpy(dtype=float)))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_title("Poisson Time Slopes by Tool and Inductively Generated Primary Theme (p < 0.05)")
    ax.set_xlabel("Inductively Generated Primary Theme")
    ax.set_ylabel("Tool")
    ax.set_xticks(np.arange(len(beta.columns)))
    ax.set_xticklabels([_wrap_label(c, 18) for c in beta.columns], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(beta.index)))
    ax.set_yticklabels(beta.index.tolist())

    # Annotate significant cells only.
    for i, tool in enumerate(beta.index):
        for j, theme in enumerate(beta.columns):
            if bool(sig.loc[tool, theme]):
                ax.text(j, i, f"{beta.loc[tool, theme]:.3f}", ha="center", va="center", fontsize=7, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Poisson slope (beta)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_curves(
    wide_csv: Path,
    out_path: Path,
    title: str,
    y_label: str = "Expected monthly issue count (Poisson mean)",
):
    df = pd.read_csv(wide_csv)
    month_col = df.columns[0]
    df[month_col] = pd.to_datetime(df[month_col], errors="coerce")
    df = df.sort_values(month_col)

    fig, ax = plt.subplots(figsize=(12.5, 6.2))

    series_cols = [c for c in df.columns if c != month_col]
    for col in series_cols:
        ax.plot(df[month_col], df[col], linewidth=2.1, label=col)

    ax.set_title(title)
    ax.set_xlabel("Calendar month")
    ax.set_ylabel(y_label)

    # Legend intentionally inside whitespace of plot area.
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        fontsize=7,
        ncol=1,
        frameon=True,
        framealpha=0.9,
    )

    ax.grid(alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_pub_rq3_figs(tables_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_l1_sig_heatmap(
        beta_csv=tables_dir / "fig_rq3_heatmap_l1_sig_beta_matrix.csv",
        p_csv=tables_dir / "fig_rq3_heatmap_l1_sig_p_matrix.csv",
        out_path=out_dir / "heatmap_beta_sig_L1_Theme_pub.png",
    )

    plot_aggregate_curves(
        wide_csv=tables_dir / "fig_rq3_aggregate_l1_curves_wide.csv",
        out_path=out_dir / "aggregate_poisson_expected_counts_l1_theme_pub.png",
        title="Aggregate Poisson Fitted Means by Inductively Generated Primary Theme",
    )

    plot_aggregate_curves(
        wide_csv=tables_dir / "fig_rq3_aggregate_components_curves_wide.csv",
        out_path=out_dir / "aggregate_poisson_expected_counts_associated_component_pub.png",
        title="Aggregate Poisson Fitted Means by Affected Component",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create publication-style RQ3 figures from fig_rq3_* tables.")
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("outputs/tables"),
        help="Directory containing fig_rq3_* CSV tables.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/plots/trend/pub"),
        help="Directory to write publication-style PNG figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_pub_rq3_figs(args.tables_dir, args.out_dir)
    print("Wrote publication-style RQ3 figures to:", args.out_dir.resolve())


if __name__ == "__main__":
    main()

