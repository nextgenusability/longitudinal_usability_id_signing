from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


KEY_CANDIDATES = [
    "repo",
    "tool",
    "tool_family",
    "theme_family",
    "category",
    "theme",
    "metric",
    "phase",
    "phase12_subset",
    "month",
    "month_start",
]


def _is_numeric_or_bool(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)


def _stable_key_cols(cols: Iterable[str]) -> list[str]:
    cset = set(cols)
    keys = [k for k in KEY_CANDIDATES if k in cset]
    return keys


def _coerce_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s, errors="coerce")


def _entity_key(row: pd.Series, keys: list[str]) -> str:
    if not keys:
        return f"row_id={int(row['__row_id'])}"
    parts = []
    for k in keys:
        v = row.get(k, "")
        if pd.isna(v):
            v = ""
        parts.append(f"{k}={v}")
    return "|".join(parts)


def build_diff_report(old_dir: Path, new_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    old_files = {p.name: p for p in old_dir.glob("*.csv")}
    new_files = {p.name: p for p in new_dir.glob("*.csv")}
    common = sorted(set(old_files) & set(new_files))

    detail_rows: list[dict] = []
    summary_rows: list[dict] = []

    for fname in common:
        old_df = pd.read_csv(old_files[fname])
        new_df = pd.read_csv(new_files[fname])

        # Stabilize unnamed index columns if present.
        if old_df.columns[0].startswith("Unnamed:"):
            old_df = old_df.rename(columns={old_df.columns[0]: "__index_col"})
        if new_df.columns[0].startswith("Unnamed:"):
            new_df = new_df.rename(columns={new_df.columns[0]: "__index_col"})

        key_cols = _stable_key_cols(old_df.columns.intersection(new_df.columns))
        if not key_cols:
            old_df = old_df.copy()
            new_df = new_df.copy()
            old_df["__row_id"] = np.arange(len(old_df))
            new_df["__row_id"] = np.arange(len(new_df))
            key_cols = ["__row_id"]

        merged = old_df.merge(
            new_df,
            on=key_cols,
            how="outer",
            suffixes=("_old", "_new"),
            indicator=True,
        )

        old_only = int((merged["_merge"] == "left_only").sum())
        new_only = int((merged["_merge"] == "right_only").sum())

        shared_cols = [c for c in old_df.columns if c in new_df.columns and c not in key_cols]
        metric_cols = [c for c in shared_cols if _is_numeric_or_bool(old_df[c]) and _is_numeric_or_bool(new_df[c])]

        changed_cells = 0
        compared_cells = 0

        for metric in metric_cols:
            a = _coerce_numeric(merged[f"{metric}_old"])
            b = _coerce_numeric(merged[f"{metric}_new"])
            delta = b - a
            abs_delta = delta.abs()

            for i in range(len(merged)):
                status = merged.iloc[i]["_merge"]
                old_v = a.iloc[i]
                new_v = b.iloc[i]
                d = delta.iloc[i]
                ad = abs_delta.iloc[i]

                if pd.isna(old_v) and pd.isna(new_v):
                    continue

                compared_cells += 1
                is_changed = False
                if status != "both":
                    is_changed = True
                elif pd.isna(old_v) != pd.isna(new_v):
                    is_changed = True
                elif pd.notna(ad) and float(ad) > 0:
                    is_changed = True

                if is_changed:
                    changed_cells += 1

                detail_rows.append(
                    {
                        "table": fname,
                        "entity_key": _entity_key(merged.iloc[i], key_cols),
                        "metric": metric,
                        "old_value": old_v,
                        "new_value": new_v,
                        "delta_new_minus_old": d,
                        "abs_delta": ad,
                        "row_status": status,
                        "changed": "yes" if is_changed else "no",
                    }
                )

        summary_rows.append(
            {
                "table": fname,
                "n_rows_old": int(len(old_df)),
                "n_rows_new": int(len(new_df)),
                "n_rows_left_only": old_only,
                "n_rows_right_only": new_only,
                "n_numeric_metrics_compared": int(len(metric_cols)),
                "n_cells_compared": int(compared_cells),
                "n_cells_changed": int(changed_cells),
                "pct_cells_changed": (100.0 * changed_cells / compared_cells) if compared_cells else np.nan,
            }
        )

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows).sort_values(["pct_cells_changed", "table"], ascending=[False, True])
    return detail, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two analysis table directories and emit machine-readable diffs.")
    p.add_argument("--old-dir", type=Path, required=True, help="Baseline tables directory (paper version).")
    p.add_argument("--new-dir", type=Path, required=True, help="New tables directory (prompt v4).")
    p.add_argument("--out-csv", type=Path, required=True, help="Output long-form diff CSV.")
    p.add_argument(
        "--out-summary-csv",
        type=Path,
        default=None,
        help="Optional per-table summary CSV path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    detail, summary = build_diff_report(args.old_dir, args.new_dir)
    detail.to_csv(args.out_csv, index=False)

    summary_path = args.out_summary_csv or args.out_csv.with_name(args.out_csv.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("Wrote diff detail:", args.out_csv.resolve())
    print("Wrote diff summary:", summary_path.resolve())
    print("Tables compared:", len(summary))


if __name__ == "__main__":
    main()

