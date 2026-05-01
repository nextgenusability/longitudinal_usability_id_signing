from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _normalize_label(label: str) -> str:
    s = str(label).strip().strip("'\"")
    s = s.lstrip("[](){} \t\r\n")
    s = s.rstrip("[](){} \t\r\n")
    return s.strip()


def _is_core(label: str) -> bool:
    return re.sub(r"[^a-z0-9]+", "", label.lower()) == "core"


def remove_core_if_mixed(cell):
    if pd.isna(cell):
        return cell, False
    s = str(cell).strip()
    if not s:
        return cell, False

    parts = [_normalize_label(p) for p in s.split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return cell, False

    has_core = any(_is_core(p) for p in parts)
    has_non_core = any(not _is_core(p) for p in parts)
    if not (has_core and has_non_core):
        return cell, False

    filtered = [p for p in parts if not _is_core(p)]
    if not filtered:
        return cell, False

    out = ", ".join(filtered)
    return out, out != s


def process_issues_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    stats: dict[str, int] = {}

    target_cols = [
        "Associated Component",
        "Associated component",
        "Associated Component Theme",
        "Associated component Theme",
    ]

    for col in target_cols:
        if col not in out.columns:
            continue
        changed = 0
        new_vals = []
        for v in out[col]:
            nv, did = remove_core_if_mixed(v)
            new_vals.append(nv)
            if did:
                changed += 1
        out[col] = new_vals
        stats[col] = changed

    return out, stats


def process_workbook(src: Path, dst: Path) -> dict[str, int]:
    sheets = pd.read_excel(src, sheet_name=None, engine="openpyxl")
    stats: dict[str, int] = {}

    if "issues" in sheets:
        issues_out, issues_stats = process_issues_df(sheets["issues"])
        sheets["issues"] = issues_out
        stats.update(issues_stats)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(dst, engine="openpyxl") as writer:
        for name, sdf in sheets.items():
            sdf.to_excel(writer, sheet_name=name, index=False)
    return stats


def main():
    ap = argparse.ArgumentParser(
        description="Duplicate labeled corpus and remove 'Core' from mixed component labels (e.g., 'Core, CLI tooling' -> 'CLI tooling')."
    )
    ap.add_argument(
        "--src-dir",
        default="outputs/data/phase3_prompt_v4/recode_full/copied_gh_issues_with_openai_labels",
        help="Source directory containing labeled xlsx files.",
    )
    ap.add_argument(
        "--dst-dir",
        default="outputs/data/phase3_prompt_v4/recode_full_core_normalized/copied_gh_issues_with_openai_labels",
        help="Destination directory for normalized xlsx files.",
    )
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_files = sorted(p for p in src_dir.glob("*.xlsx") if not p.name.startswith("~$"))
    if not src_files:
        raise FileNotFoundError(f"No .xlsx files found in {src_dir}")

    rows = []
    for src in src_files:
        dst = dst_dir / src.name
        stats = process_workbook(src, dst)
        row = {"file": src.name}
        row.update(stats)
        row["total_changed"] = int(sum(stats.values()))
        rows.append(row)

    report = pd.DataFrame(rows).fillna(0)
    report_path = dst_dir.parent / "core_normalization_report.csv"
    report.to_csv(report_path, index=False)

    print(f"Wrote normalized workbooks to: {dst_dir}")
    print(f"Wrote report: {report_path}")
    if not report.empty:
        print("Summary changed cells:")
        print(report[["file", "total_changed"]].to_string(index=False))


if __name__ == "__main__":
    main()
