from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path("data/phase3")
TABLE_DIR = Path("outputs/tables")
PLOT_DIR = Path("outputs/plots/trend")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

COL_TOOL = "repo"
COL_CREATED = "created_at"
COL_USABILITY = "usability_non-usability_type"


def norm_usability(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"1", "usability", "true", "yes"}:
        return 1
    if s in {"0", "non-usability", "nonusability", "false", "no"}:
        return 0
    try:
        return int(float(s))
    except Exception:
        return None


def load_phase3_data(data_dir: Path) -> pd.DataFrame:
    paths = sorted(data_dir.glob("*.xlsx"))
    if not paths:
        raise FileNotFoundError(f"No .xlsx files found in {data_dir.resolve()}")
    frames = []
    for fp in paths:
        d = pd.read_excel(fp)
        d["__source_file__"] = fp.name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    for col in [COL_TOOL, COL_CREATED, COL_USABILITY]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
    df[COL_CREATED] = pd.to_datetime(df[COL_CREATED], errors="coerce", utc=True)
    df[COL_USABILITY] = df[COL_USABILITY].apply(norm_usability)
    df = df[df[COL_CREATED].notna()].copy()
    return df


def main():
    df = load_phase3_data(DATA_DIR)
    df_u = df[df[COL_USABILITY] == 1].copy()
    if df_u.empty:
        raise ValueError("No usability rows found where usability_non-usability_type == 1")

    df_u["month"] = df_u[COL_CREATED].dt.to_period("M")
    all_months = pd.period_range(df_u["month"].min(), df_u["month"].max(), freq="M")

    counts = (
        df_u.groupby([COL_TOOL, "month"])
        .size()
        .reset_index(name="usability_issue_count")
    )

    tools_by_total = (
        counts.groupby(COL_TOOL)["usability_issue_count"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    wide = counts.pivot_table(
        index="month",
        columns=COL_TOOL,
        values="usability_issue_count",
        aggfunc="sum",
        fill_value=0,
    ).reindex(index=all_months, columns=tools_by_total, fill_value=0)
    wide = wide.astype(int)

    wide_out = wide.copy()
    wide_out.insert(0, "month", wide_out.index.astype(str))
    wide_out.insert(1, "month_start", wide_out.index.to_timestamp())
    wide_out.to_csv(TABLE_DIR / "raw_monthly_usability_counts_by_tool_wide.csv", index=False)

    long_base = wide.reset_index().rename(columns={wide.index.name or "index": "month_period"})
    long_out = (
        long_base
        .melt(id_vars=["month_period"], var_name=COL_TOOL, value_name="usability_issue_count")
        .sort_values(["month_period", COL_TOOL])
    )
    long_out["month"] = long_out["month_period"].astype(str)
    long_out["month_start"] = long_out["month_period"].dt.to_timestamp()
    long_out = long_out[[COL_TOOL, "month", "month_start", "usability_issue_count"]]
    long_out.to_csv(TABLE_DIR / "raw_monthly_usability_counts_by_tool_long.csv", index=False)

    plt.figure(figsize=(12, 6))
    x = wide.index.to_timestamp()
    for tool in tools_by_total:
        plt.plot(x, wide[tool].values, linewidth=1.8, label=tool)
    plt.title("Raw monthly usability issue counts by tool")
    plt.xlabel("Calendar month")
    plt.ylabel("Usability issue count (raw)")
    plt.grid(alpha=0.25, linewidth=0.6)
    plt.legend(fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "raw_monthly_usability_counts_by_tool.png", dpi=180, bbox_inches="tight")
    plt.close()

    print("Wrote:", TABLE_DIR / "raw_monthly_usability_counts_by_tool_wide.csv")
    print("Wrote:", TABLE_DIR / "raw_monthly_usability_counts_by_tool_long.csv")
    print("Wrote:", PLOT_DIR / "raw_monthly_usability_counts_by_tool.png")


if __name__ == "__main__":
    main()
