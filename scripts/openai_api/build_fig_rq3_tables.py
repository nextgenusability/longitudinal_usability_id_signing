from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_required_csv(path: Path, required_cols: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    df = pd.read_csv(path)
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"{path} missing required columns: {sorted(missing)}")
    return df


def _build_heatmap_helpers(trends_csv: Path, out_stem: str, out_dir: Path) -> None:
    df = _load_required_csv(
        trends_csv,
        {"tool", "theme", "beta", "p_value"},
    )
    beta = df.pivot_table(index="tool", columns="theme", values="beta", aggfunc="mean").sort_index()
    pmat = (
        df.pivot_table(index="tool", columns="theme", values="p_value", aggfunc="mean")
        .reindex(index=beta.index, columns=beta.columns)
        .sort_index()
    )
    mask = (pmat < 0.05).astype(bool)

    beta.to_csv(out_dir / f"fig_rq3_heatmap_{out_stem}_beta_matrix.csv")
    pmat.to_csv(out_dir / f"fig_rq3_heatmap_{out_stem}_p_matrix.csv")
    mask.to_csv(out_dir / f"fig_rq3_heatmap_{out_stem}_mask_matrix.csv")

    long_df = df[["tool", "theme", "beta", "p_value"]].copy()
    long_df["sig_0_05"] = long_df["p_value"] < 0.05
    long_df.to_csv(out_dir / f"fig_rq3_heatmap_{out_stem}_long.csv", index=False)


def _build_curve_helpers(curves_csv: Path, out_stem: str, out_dir: Path) -> None:
    c = _load_required_csv(
        curves_csv,
        {"category", "month_start", "expected_count"},
    )
    c["month_start"] = pd.to_datetime(c["month_start"], errors="coerce")
    long_df = c[["category", "month_start", "expected_count"]].sort_values(["month_start", "category"]).copy()
    wide_df = (
        long_df.pivot_table(index="month_start", columns="category", values="expected_count", aggfunc="mean")
        .sort_index()
        .reset_index()
    )

    long_df.to_csv(out_dir / f"fig_rq3_{out_stem}_curves_long.csv", index=False)
    wide_df.to_csv(out_dir / f"fig_rq3_{out_stem}_curves_wide.csv", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build fig_rq3_* helper CSV tables from trend outputs.")
    p.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("outputs/tables"),
        help="Directory containing poisson_trends_* and aggregate_poisson_* input tables.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tables_dir = args.tables_dir
    tables_dir.mkdir(parents=True, exist_ok=True)

    _build_heatmap_helpers(
        tables_dir / "poisson_trends_L1_Theme.csv",
        "l1_sig",
        tables_dir,
    )
    _build_heatmap_helpers(
        tables_dir / "poisson_trends_Nielsen_theme.csv",
        "nielsen_sig",
        tables_dir,
    )
    _build_heatmap_helpers(
        tables_dir / "poisson_trends_Associated_Component_Theme.csv",
        "associated_component_sig",
        tables_dir,
    )

    _build_curve_helpers(
        tables_dir / "aggregate_poisson_expected_counts_l1_theme_curves.csv",
        "aggregate_l1",
        tables_dir,
    )
    _build_curve_helpers(
        tables_dir / "aggregate_poisson_expected_counts_associated_component_curves.csv",
        "aggregate_components",
        tables_dir,
    )

    print("Wrote fig_rq3 helper tables to:", tables_dir.resolve())


if __name__ == "__main__":
    main()

