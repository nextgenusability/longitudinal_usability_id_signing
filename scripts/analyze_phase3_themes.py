from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Config ----------
DATA_DIR = Path("data/phase3")
OUT_DIR = Path("outputs")
TABLE_DIR = OUT_DIR / "tables"
PLOT_DIR = OUT_DIR / "plots"
PIE_DIR = PLOT_DIR / "pies"
BAR_DIR = PLOT_DIR / "bars"
STACKED_DIR = PLOT_DIR / "stacked"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
PIE_DIR.mkdir(parents=True, exist_ok=True)
BAR_DIR.mkdir(parents=True, exist_ok=True)
STACKED_DIR.mkdir(parents=True, exist_ok=True)

COL_TOOL = "repo"
COL_USABILITY = "usability_non-usability_type"

THEME_COLS = [
    "Associated Component Theme",
    "L1_Theme",
    "L1_Theme Secondary",
    "Nielsen_theme",
]

# Keep pies readable
PIE_MAX_SLICES = 10

# Keep stacked bars readable (top categories + Other)
STACKED_TOPK = 20

# Optional: per-tool top-k bars for each theme (can be slow if you have lots of labels)
MAKE_PER_TOOL_BARS = False
PER_TOOL_BAR_TOPK = 20

L1_SECONDARY_GROUPS = [
    (
        "Operational Friction",
        [
            "Configuration friction",
            "Authentication friction",
            "Build/CI/installation/distribution release issues",
            "Integration failure/issues",
            "Tedious Workflows",
        ],
    ),
    (
        "Cognitive Friction",
        [
            "User confusion / unclear documentation",
            "Notification/Logging Issues",
        ],
    ),
    (
        "Functional Reliability",
        [
            "Unexpected behavior",
            "Performance issue",
            "Security concerns",
        ],
    ),
    (
        "Functional Gap",
        [
            "Missing feature / enhancement request",
        ],
    ),
]

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


# ---------- Helpers ----------
def norm_usability(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"1", "usability", "true", "yes"}:
        return 1
    if s in {"0", "non-usability", "nonusability", "false", "no"}:
        return 0
    try:
        return int(float(s))
    except Exception:
        return np.nan


def split_multi(cell) -> list[str]:
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def split_multi_nielsen(cell) -> list[str]:
    """
    Split Nielsen labels while preserving commas inside a single heuristic name.
    Uses numbered pattern boundaries like '1. ...', '2. ...'.
    """
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    matches = re.findall(r"(?:^|,\s*)(\d+\.\s.*?)(?=(?:,\s*\d+\.|$))", s)
    if matches:
        return [m.strip() for m in matches if m.strip()]
    return split_multi(s)


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def _strip_nielsen_number_prefix(s: str) -> str:
    return re.sub(r"^\s*\d+\.\s*", "", str(s).strip())


def _nielsen_alias_to_canonical() -> dict[str, str]:
    out: dict[str, str] = {}
    for canonical in NIELSEN_THEME_ORDER:
        out[norm_key(canonical)] = canonical
        out[norm_key(_strip_nielsen_number_prefix(canonical))] = canonical
    # Common shortened/variant spellings
    out[norm_key("9. Help users recognize and recover from errors")] = NIELSEN_THEME_ORDER[8]
    out[norm_key("Help users recognize, diagnose, and recover from errors")] = NIELSEN_THEME_ORDER[8]
    return out


def canonicalize_nielsen_theme(x: str, alias_map: dict[str, str]) -> str | None:
    key = norm_key(x)
    return alias_map.get(key)


def latex_escape(s: str) -> str:
    rep = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = str(s)
    for k, v in rep.items():
        out = out.replace(k, v)
    return out


def split_multi_for_col(cell, col_name: str) -> list[str]:
    if col_name == "Nielsen_theme":
        return split_multi_nielsen(cell)
    return split_multi(cell)


def _l1_alias_to_canonical() -> dict[str, str]:
    aliases = {
        "Configuration friction": ["Configuration friction"],
        "Authentication friction": ["Authentication friction"],
        "Build/CI/installation/distribution release issues": [
            "Build/CI/installation/distribution release issues",
        ],
        "Integration failure/issues": ["Integration failure/issues"],
        "Tedious Workflows": ["Tedious Workflows"],
        "User confusion / unclear documentation": [
            "User confusion / unclear documentation",
            "User confusion / unclear documentation/documentation improvements",
        ],
        "Notification/Logging Issues": [
            "Notification/Logging Issues",
            "Notification/Logging /Web UI Issues",
        ],
        "Unexpected behavior": ["Unexpected behavior"],
        "Performance issue": ["Performance issue"],
        "Security concerns": ["Security concerns"],
        "Missing feature / enhancement request": ["Missing feature / enhancement request"],
    }
    out: dict[str, str] = {}
    for canonical, vals in aliases.items():
        for v in vals:
            out[norm_key(v)] = canonical
    return out


def build_l1_grouped_pct_latex(
    pct_df: pd.DataFrame,
    caption: str,
    label: str,
    groups: list[tuple[str, list[str]]],
) -> str:
    # Add vertical separators between L1_Theme Secondary groups.
    colfmt = "l" + "".join("|" + ("c" * len(cols)) for _, cols in groups)
    header_group_cells = []
    for group_name, cols in groups:
        header_group_cells.append(f"\\multicolumn{{{len(cols)}}}{{c|}}{{{latex_escape(group_name)}}}")
    header_group = "Tool & " + " & ".join(header_group_cells) + r" \\"

    subcols = []
    for _, cols in groups:
        subcols.extend(cols)
    header_sub = " & " + " & ".join(latex_escape(c) for c in subcols) + r" \\"

    body_lines = []
    for tool, row in pct_df.iterrows():
        vals = [f"{float(row[c]):.1f}\\%" for c in subcols]
        body_lines.append(latex_escape(tool) + " & " + " & ".join(vals) + r" \\")

    table = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{{colfmt}}}",
        "\\hline",
        header_group,
        "\\hline",
        header_sub,
        "\\hline",
        *body_lines,
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(table)


def build_l1_grouped_percent_table(df: pd.DataFrame):
    if "L1_Theme" not in df.columns:
        raise KeyError("Missing theme column 'L1_Theme'")

    alias_map = _l1_alias_to_canonical()
    ordered_l1 = [theme for _, themes in L1_SECONDARY_GROUPS for theme in themes]
    theme_to_group = {theme: group for group, themes in L1_SECONDARY_GROUPS for theme in themes}

    src = df[df[COL_USABILITY] == 1][[COL_TOOL, "L1_Theme", "__source_file__"]].copy()
    src["l1_theme_raw"] = src["L1_Theme"].apply(split_multi)
    src = src.explode("l1_theme_raw")
    src = src[src["l1_theme_raw"].notna() & (src["l1_theme_raw"].astype(str).str.strip() != "")]
    src["l1_theme_raw"] = src["l1_theme_raw"].astype(str).str.strip()
    src["l1_theme_canonical"] = src["l1_theme_raw"].map(lambda x: alias_map.get(norm_key(x)))
    src["l1_theme_secondary_group"] = src["l1_theme_canonical"].map(theme_to_group)

    # Save the exact exploded data used as input.
    src.to_csv(TABLE_DIR / "l1_theme_grouped_table_source_exploded_usability.csv", index=False)

    used = src[src["l1_theme_canonical"].notna()].copy()
    if used.empty:
        return

    unmapped = src[src["l1_theme_canonical"].isna()].copy()
    if not unmapped.empty:
        unmapped.to_csv(TABLE_DIR / "l1_theme_grouped_table_unmapped_labels.csv", index=False)

    denom = used.groupby(COL_TOOL).size().rename("denom_theme_assignments")
    counts = (
        used.groupby([COL_TOOL, "l1_theme_secondary_group", "l1_theme_canonical"])
        .size()
        .reset_index(name="count")
    )
    counts.to_csv(TABLE_DIR / "l1_theme_grouped_table_counts_long.csv", index=False)

    tools = denom.sort_values(ascending=False).index.tolist()
    wide_counts = counts.pivot_table(
        index=COL_TOOL,
        columns="l1_theme_canonical",
        values="count",
        fill_value=0,
        aggfunc="sum",
    ).reindex(index=tools, columns=ordered_l1, fill_value=0)
    wide_counts = wide_counts.astype(int)

    wide_pct = wide_counts.div(denom.reindex(tools), axis=0).fillna(0.0) * 100.0
    wide_pct = wide_pct.round(2)

    counts_out = wide_counts.copy()
    counts_out.insert(0, "denom_theme_assignments", denom.reindex(tools).astype(int).values)
    counts_out.to_csv(TABLE_DIR / "l1_theme_grouped_table_counts_by_tool.csv")
    wide_pct.to_csv(TABLE_DIR / "l1_theme_grouped_table_percent_by_tool.csv")

    latex = build_l1_grouped_pct_latex(
        wide_pct,
        caption=(
            "L1 theme distribution by tool (usability-only), normalized by total "
            "L1 theme assignments per tool and grouped by L1_Theme Secondary."
        ),
        label="tab:l1-theme-grouped-percent-by-tool",
        groups=L1_SECONDARY_GROUPS,
    )
    (TABLE_DIR / "l1_theme_grouped_table_percent_by_tool.tex").write_text(latex)


def build_nielsen_percent_table(df: pd.DataFrame):
    if "Nielsen_theme" not in df.columns:
        raise KeyError("Missing theme column 'Nielsen_theme'")

    alias_map = _nielsen_alias_to_canonical()
    src = df[df[COL_USABILITY] == 1][[COL_TOOL, "Nielsen_theme", "__source_file__"]].copy()
    src["nielsen_theme_raw"] = src["Nielsen_theme"].apply(split_multi_nielsen)
    src = src.explode("nielsen_theme_raw")
    src = src[src["nielsen_theme_raw"].notna() & (src["nielsen_theme_raw"].astype(str).str.strip() != "")]
    src["nielsen_theme_raw"] = src["nielsen_theme_raw"].astype(str).str.strip()
    src["nielsen_theme_canonical"] = src["nielsen_theme_raw"].map(lambda x: canonicalize_nielsen_theme(x, alias_map))
    src.to_csv(TABLE_DIR / "nielsen_theme_table_source_exploded_usability.csv", index=False)

    used = src[src["nielsen_theme_canonical"].notna()].copy()
    if used.empty:
        return

    unmapped = src[src["nielsen_theme_canonical"].isna()].copy()
    if not unmapped.empty:
        unmapped.to_csv(TABLE_DIR / "nielsen_theme_table_unmapped_labels.csv", index=False)

    denom = used.groupby(COL_TOOL).size().rename("denom_theme_assignments")
    counts = (
        used.groupby([COL_TOOL, "nielsen_theme_canonical"])
        .size()
        .reset_index(name="count")
    )
    counts.to_csv(TABLE_DIR / "nielsen_theme_counts_long.csv", index=False)

    tools = denom.sort_values(ascending=False).index.tolist()
    wide_counts = counts.pivot_table(
        index=COL_TOOL,
        columns="nielsen_theme_canonical",
        values="count",
        fill_value=0,
        aggfunc="sum",
    ).reindex(index=tools, columns=NIELSEN_THEME_ORDER, fill_value=0)
    wide_counts = wide_counts.astype(int)

    wide_pct = wide_counts.div(denom.reindex(tools), axis=0).fillna(0.0) * 100.0
    wide_pct = wide_pct.round(2)

    counts_out = wide_counts.copy()
    counts_out.insert(0, "denom_theme_assignments", denom.reindex(tools).astype(int).values)
    counts_out.to_csv(TABLE_DIR / "nielsen_theme_counts_by_tool.csv")
    wide_pct.to_csv(TABLE_DIR / "nielsen_theme_percent_by_tool.csv")

    pct_tex = wide_pct.reset_index().rename(columns={COL_TOOL: "tool"})
    for c in pct_tex.columns[1:]:
        pct_tex[c] = pct_tex[c].map(lambda x: f"{x:.1f}\\%")
    (TABLE_DIR / "nielsen_theme_percent_by_tool.tex").write_text(
        to_latex_table(
            pct_tex.set_index("tool"),
            "Nielsen theme distribution by tool (usability-only), normalized by total Nielsen theme assignments per tool.",
            "tab:nielsen-theme-percent-by-tool",
        )
    )


def build_rq2_top3_components_table(df: pd.DataFrame):
    comp_col = "Associated Component Theme"
    if comp_col not in df.columns:
        raise KeyError(f"Missing theme column '{comp_col}'")

    tmp = df[df[COL_USABILITY] == 1][[COL_TOOL, comp_col]].copy()
    tmp[comp_col] = tmp[comp_col].apply(split_multi)
    tmp = tmp.explode(comp_col)
    tmp = tmp[tmp[comp_col].notna() & (tmp[comp_col].astype(str).str.strip() != "")]
    tmp[comp_col] = tmp[comp_col].astype(str).str.strip()
    if tmp.empty:
        return

    counts = tmp.groupby([COL_TOOL, comp_col]).size().reset_index(name="count")
    totals = counts.groupby(COL_TOOL)["count"].sum().rename("total_component_assignments")

    long_rows = []
    wide_rows = []
    tool_order = totals.sort_values(ascending=False).index.tolist()
    for tool in tool_order:
        g = counts[counts[COL_TOOL] == tool].copy()
        g = g.sort_values(["count", comp_col], ascending=[False, True]).head(3).reset_index(drop=True)
        total = int(totals.loc[tool])

        wide = {"tool": tool}
        for i, row in g.iterrows():
            rank = i + 1
            pct = (float(row["count"]) / total * 100.0) if total else 0.0
            comp = row[comp_col]
            count = int(row["count"])
            long_rows.append(
                {
                    "tool": tool,
                    "rank": rank,
                    "component": comp,
                    "count": count,
                    "pct_within_tool_component_assignments": round(pct, 2),
                    "tool_total_component_assignments": total,
                }
            )
            wide[f"top{rank}"] = f"{comp} ({pct:.1f})"
        for rank in [1, 2, 3]:
            wide.setdefault(f"top{rank}", "")
        wide_rows.append(wide)

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)
    long_df.to_csv(TABLE_DIR / "rq2_top3_components_by_tool_long.csv", index=False)
    wide_df.to_csv(TABLE_DIR / "rq2_top3_components_by_tool.csv", index=False)

    # Compact manuscript table: one row per tool with top-3 components and within-tool percentages.
    wide_tex = wide_df.set_index("tool")
    (TABLE_DIR / "rq2_top3_components_by_tool.tex").write_text(
        to_latex_table(
            wide_tex,
            "Top-3 affected components per tool (usability-only). Values are within-tool percentages in parentheses.",
            "tab:rq2-top3-components-by-tool",
        )
    )


def load_phase3_excels(data_dir: Path) -> pd.DataFrame:
    paths = sorted(data_dir.glob("*.xlsx"))
    if not paths:
        raise FileNotFoundError(f"No .xlsx files found in {data_dir.resolve()}")
    dfs = []
    for fp in paths:
        df = pd.read_excel(fp)
        df["__source_file__"] = fp.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def exploded_counts(df: pd.DataFrame, theme_col: str) -> pd.DataFrame:
    tmp = df[[COL_TOOL, theme_col]].copy()
    tmp[theme_col] = tmp[theme_col].apply(lambda x: split_multi_for_col(x, theme_col))
    tmp = tmp.explode(theme_col)
    tmp = tmp[tmp[theme_col].notna() & (tmp[theme_col].astype(str).str.strip() != "")]
    return tmp.groupby([COL_TOOL, theme_col]).size().reset_index(name="count")


def make_pie(counts: pd.Series, title: str, out_path: Path, max_slices: int = PIE_MAX_SLICES):
    s = counts.sort_values(ascending=False)
    if len(s) > max_slices:
        top = s.iloc[: max_slices - 1]
        other = s.iloc[max_slices - 1 :].sum()
        s = pd.concat([top, pd.Series({"Other": other})])

    plt.figure(figsize=(7, 5))
    plt.pie(
        s.values,
        labels=s.index,
        autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
    )
    plt.title(title)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def make_bar(counts: pd.Series, title: str, out_path: Path, topk: int = PER_TOOL_BAR_TOPK):
    s = counts.sort_values(ascending=False).head(topk)
    plt.figure(figsize=(10, 5.5))
    plt.bar(range(len(s)), s.values)
    plt.xticks(range(len(s)), s.index, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("Count (multi-label occurrences)")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def make_stacked_bar_themes_by_tool(mat: pd.DataFrame, title: str, out_path: Path, topk: int = STACKED_TOPK):
    """
    mat: index=theme, columns=tool, values=count
    Plot: x-axis=tool, stacked segments=themes
    """
    totals = mat.sum(axis=1).sort_values(ascending=False)
    keep = totals.head(topk).index
    other = totals.index.difference(keep)

    plot_df = mat.loc[keep].copy()
    if len(other) > 0:
        plot_df.loc["Other"] = mat.loc[other].sum(axis=0)

    plot_df = plot_df.loc[plot_df.sum(axis=1).sort_values(ascending=False).index]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(plot_df.columns))
    bottom = np.zeros(len(plot_df.columns))

    for theme in plot_df.index:
        vals = plot_df.loc[theme].values
        plt.bar(x, vals, bottom=bottom, label=theme)
        bottom += vals

    plt.xticks(x, plot_df.columns, rotation=30, ha="right")
    plt.title(title)
    plt.ylabel("Count (multi-label occurrences)")
    plt.legend(fontsize=7, ncol=2)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def make_stacked_bar_themes_by_tool_pct(mat: pd.DataFrame, title: str, out_path: Path, topk: int = STACKED_TOPK):
    """
    mat: index=theme, columns=tool, values=count
    Plot: x-axis=tool, stacked segments=theme percent within tool (100% stacked)
    """
    totals = mat.sum(axis=1).sort_values(ascending=False)
    keep = totals.head(topk).index
    other = totals.index.difference(keep)

    plot_df = mat.loc[keep].copy()
    if len(other) > 0:
        plot_df.loc["Other"] = mat.loc[other].sum(axis=0)

    # Normalize each tool column to percentages.
    col_totals = plot_df.sum(axis=0).replace(0, np.nan)
    plot_pct = plot_df.div(col_totals, axis=1).fillna(0.0) * 100.0
    plot_pct = plot_pct.loc[plot_pct.sum(axis=1).sort_values(ascending=False).index]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(plot_pct.columns))
    bottom = np.zeros(len(plot_pct.columns))

    for theme in plot_pct.index:
        vals = plot_pct.loc[theme].values
        plt.bar(x, vals, bottom=bottom, label=theme)
        bottom += vals

    plt.xticks(x, plot_pct.columns, rotation=30, ha="right")
    plt.ylim(0, 100)
    plt.title(title)
    plt.ylabel("Share within tool (%)")
    plt.legend(fontsize=7, ncol=2)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    latex = df.to_latex(index=True, escape=True)
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\scriptsize\n"
        f"{latex}\n"
        "\\end{table}\n"
    )


# ---------- Main ----------
def main():
    df = load_phase3_excels(DATA_DIR)

    # Normalize usability labels
    if COL_USABILITY in df.columns:
        df[COL_USABILITY] = df[COL_USABILITY].apply(norm_usability)
    else:
        raise KeyError(f"Missing '{COL_USABILITY}' column")

    # 1) Usability vs non-usability counts
    us = (
        df.groupby([COL_TOOL, COL_USABILITY]).size()
        .unstack(fill_value=0)
        .rename(columns={0: "non-usability", 1: "usability"})
    )
    for c in ["non-usability", "usability"]:
        if c not in us.columns:
            us[c] = 0
    us["total"] = us["non-usability"] + us["usability"]
    us["usability_pct"] = (us["usability"] / us["total"] * 100).round(1)
    us = us.sort_values("total", ascending=False)

    us.to_csv(TABLE_DIR / "usability_vs_nonusability_by_tool.csv")
    (TABLE_DIR / "usability_vs_nonusability_by_tool.tex").write_text(
        to_latex_table(
            us[["non-usability", "usability", "total", "usability_pct"]],
            "Phase 3 issue counts by tool split by usability vs non-usability.",
            "tab:phase3-usability-counts",
        )
    )

    # 2) L1_Theme table grouped by your L1_Theme Secondary mapping
    build_l1_grouped_percent_table(df)

    # 3) Nielsen-theme normalized table by your explicit mapping order
    build_nielsen_percent_table(df)

    # 3b) Compact RQ2 table: top-3 affected components per tool
    build_rq2_top3_components_table(df)

    # 4) Theme distributions (multi-label) + plots + tables
    tools = sorted(df[COL_TOOL].dropna().unique().tolist())

    for col in THEME_COLS:
        if col not in df.columns:
            raise KeyError(f"Missing theme column '{col}'")

        counts_long = exploded_counts(df, col)
        counts_long.to_csv(TABLE_DIR / f"counts_{col.replace(' ', '_')}_by_tool_long.csv", index=False)

        pivot = counts_long.pivot_table(
            index=col,
            columns=COL_TOOL,
            values="count",
            fill_value=0,
            aggfunc="sum",
        )
        pivot["Total"] = pivot.sum(axis=1)
        pivot = pivot.sort_values("Total", ascending=False)
        pivot.to_csv(TABLE_DIR / f"counts_{col.replace(' ', '_')}_by_tool_pivot.csv")

        # Stacked bars by tool with themes stacked (top categories + Other)
        make_stacked_bar_themes_by_tool(
            pivot.drop(columns=["Total"]),
            f"{col}: themes stacked by tool (top categories + Other)",
            STACKED_DIR / f"stacked_{col.replace(' ', '_')}.png",
            topk=STACKED_TOPK,
        )
        make_stacked_bar_themes_by_tool_pct(
            pivot.drop(columns=["Total"]),
            f"{col}: normalized themes stacked by tool (100%)",
            STACKED_DIR / f"stacked_pct_{col.replace(' ', '_')}.png",
            topk=STACKED_TOPK,
        )

        # Per-tool pies (and optional bars)
        for tool in tools:
            tdf = df[df[COL_TOOL] == tool]
            vc = tdf[col].apply(lambda x: split_multi_for_col(x, col)).explode()
            vc = vc[vc.notna() & (vc.astype(str).str.strip() != "")]
            counts = vc.value_counts()
            if len(counts) == 0:
                continue

            make_pie(
                counts,
                f"{tool} | {col}",
                PIE_DIR / f"{safe_filename(tool)}__{col.replace(' ', '_')}.png",
                max_slices=PIE_MAX_SLICES,
            )

            if MAKE_PER_TOOL_BARS:
                make_bar(
                    counts,
                    f"{tool} | {col} (top {PER_TOOL_BAR_TOPK})",
                    BAR_DIR / f"{safe_filename(tool)}__{col.replace(' ', '_')}.png",
                    topk=PER_TOOL_BAR_TOPK,
                )

        # LaTeX: top-20 categories (wide table)
        top = pivot.head(20)
        (TABLE_DIR / f"top20_{col.replace(' ', '_')}_by_tool.tex").write_text(
            to_latex_table(
                top,
                f"Top {len(top)} {col} categories (multi-label counts) by tool.",
                f"tab:phase3-{col.replace(' ', '-').lower()}",
            )
        )

    print("Done. Outputs written to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
