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
    s = str(label).replace("/", "/ ").replace(" / ", " / ")
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=True, break_on_hyphens=True)) or str(label)


def _short_heatmap_label(label: str) -> str:
    exact = {
        "Build/CI/installation/distribution release issues": "Build/CI/install/distrib release",
        "User confusion / unclear documentation": "User confusion / unclear docs",
        "Missing feature / enhancement request": "Missing feature / enhancement req",
        "Notification/Logging /Web UI Issues": "Notif/logging / Web UI",
        "Authentication/Authorization tools": "Auth/Authz tools",
        "Key Management Core / Secrets Backend": "Key Mgmt Core / Secrets Backend",
        "Policy/configuration": "Policy/config",
    }
    if label in exact:
        return exact[label]
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


def _l1_display_label(label: str) -> str:
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


def _prep_matrix(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    idx_col = df.columns[0]
    return df.set_index(idx_col)


def plot_l1_sig_heatmap(
    beta_csv: Path,
    p_csv: Path,
    out_path: Path,
    alpha: float = 0.05,
    font_scale: float = 1.0,
):
    beta = _prep_matrix(beta_csv)
    pvals = _prep_matrix(p_csv)
    pvals = pvals.reindex(index=beta.index, columns=beta.columns)

    sig = pvals < alpha
    arr = beta.where(sig).to_numpy(dtype=float)
    masked = np.ma.masked_invalid(arr)
    n_cols = len(beta.columns)
    dense = n_cols >= 10
    wrap_width = 10 if dense else 18
    x_tick_fs = (7.6 if dense else 9.0) * font_scale
    y_tick_fs = (8.6 if dense else 9.2) * font_scale
    ann_fs = (7.2 if dense else 8.0) * font_scale
    title_fs = 15.0 * font_scale
    axis_label_fs = 14.0 * font_scale
    cbar_label_fs = 13.0 * font_scale
    cbar_tick_fs = 11.0 * font_scale
    fig_h = 6.6 if dense else 4.8

    fig, ax = plt.subplots(figsize=(12.5, fig_h))
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="#f2f2f2")

    vmax = np.nanmax(np.abs(beta.to_numpy(dtype=float)))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_title(
        "Poisson Time Slopes by Tool and Inductively Generated Primary Theme (p < 0.05)",
        fontsize=title_fs,
    )
    ax.set_xlabel("Inductively Generated Primary Theme", fontsize=axis_label_fs)
    ax.set_ylabel("Tool", fontsize=axis_label_fs)
    xlabels = [_l1_display_label(c) for c in beta.columns]
    ax.set_xticks(np.arange(len(beta.columns)))
    ax.set_xticklabels([_wrap_label(c, wrap_width) for c in xlabels], rotation=50 if dense else 35, ha="right")
    ax.set_yticks(np.arange(len(beta.index)))
    ax.set_yticklabels(beta.index.tolist())
    ax.tick_params(axis="x", labelsize=x_tick_fs)
    ax.tick_params(axis="y", labelsize=y_tick_fs)
    fig.subplots_adjust(bottom=0.43 if dense else 0.24)

    # Annotate significant cells only.
    for i, tool in enumerate(beta.index):
        for j, theme in enumerate(beta.columns):
            if bool(sig.loc[tool, theme]):
                ax.text(j, i, f"{beta.loc[tool, theme]:.3f}", ha="center", va="center", fontsize=ann_fs, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Poisson slope (beta)", fontsize=cbar_label_fs)
    cbar.ax.tick_params(labelsize=cbar_tick_fs)

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


def build_pub_rq3_figs(tables_dir: Path, out_dir: Path, heatmap_font_scale: float = 1.0):
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_l1_sig_heatmap(
        beta_csv=tables_dir / "fig_rq3_heatmap_l1_sig_beta_matrix.csv",
        p_csv=tables_dir / "fig_rq3_heatmap_l1_sig_p_matrix.csv",
        out_path=out_dir / "heatmap_beta_sig_L1_Theme_pub.png",
        font_scale=heatmap_font_scale,
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
    parser.add_argument(
        "--heatmap-font-scale",
        type=float,
        default=1.0,
        help="Font-size scale for pub heatmap text.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_pub_rq3_figs(args.tables_dir, args.out_dir, heatmap_font_scale=args.heatmap_font_scale)
    print("Wrote publication-style RQ3 figures to:", args.out_dir.resolve())


if __name__ == "__main__":
    main()
