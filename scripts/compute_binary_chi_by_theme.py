from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


OUT_DIR = Path("outputs/tables")


def load_primary() -> pd.DataFrame:
    df = pd.read_csv(OUT_DIR / "l1_theme_grouped_table_counts_by_tool.csv").set_index("repo")
    if "denom_theme_assignments" in df.columns:
        df = df.drop(columns=["denom_theme_assignments"])
    return df


def load_secondary_mapped(primary: pd.DataFrame) -> pd.DataFrame:
    sec_map = {
        "Operational Friction": [
            "Configuration friction",
            "Authentication friction",
            "Build/CI/installation/distribution release issues",
            "Integration failure/issues",
            "Tedious Workflows",
        ],
        "Cognitive Friction": [
            "User confusion / unclear documentation",
            "Notification/Logging Issues",
        ],
        "Functional Reliability": [
            "Unexpected behavior",
            "Performance issue",
            "Security concerns",
        ],
        "Functional Gap": [
            "Missing feature / enhancement request",
        ],
    }
    sec = pd.DataFrame(index=primary.index)
    for name, cols in sec_map.items():
        sec[name] = primary[cols].sum(axis=1)
    return sec


def load_secondary_coded() -> pd.DataFrame:
    df = pd.read_csv(OUT_DIR / "counts_L1_Theme_Secondary_by_tool_pivot.csv")
    df = df.set_index(df.columns[0])
    if "Total" in df.columns:
        df = df.drop(columns=["Total"])
    return df.T


def load_nielsen() -> pd.DataFrame:
    df = pd.read_csv(OUT_DIR / "nielsen_theme_counts_by_tool.csv")
    df = df.set_index(df.columns[0])
    if "denom_theme_assignments" in df.columns:
        df = df.drop(columns=["denom_theme_assignments"])
    return df


def load_components() -> pd.DataFrame:
    df = pd.read_csv(OUT_DIR / "counts_Associated_Component_Theme_by_tool_pivot.csv")
    df = df.set_index(df.columns[0])
    if "Total" in df.columns:
        df = df.drop(columns=["Total"])
    return df.T


def binary_chi_for_category(df: pd.DataFrame, category: str) -> tuple[dict, pd.DataFrame]:
    pos = df[category].astype(float)
    total = df.sum(axis=1).astype(float)
    neg = total - pos

    obs = np.vstack([pos.values, neg.values])
    chi2, p, dof, exp = chi2_contingency(obs)
    v = np.sqrt(chi2 / (obs.sum() * min(obs.shape[0] - 1, obs.shape[1] - 1)))

    stats = {
        "category": category,
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p),
        "cramers_v": float(v),
        "n_assignments": int(obs.sum()),
        "min_expected": float(exp.min()),
        "cells_expected_lt5": int((exp < 5).sum()),
    }

    shares = pd.DataFrame(
        {
            "tool": df.index,
            "count_in_category": pos.values.astype(int),
            "total_assignments_in_family": total.values.astype(int),
            "share_pct": (pos / total * 100.0).values,
        }
    )
    shares["rank_within_category"] = shares["share_pct"].rank(ascending=False, method="min").astype(int)
    shares = shares.sort_values(["share_pct", "tool"], ascending=[False, True])
    return stats, shares


def latex_escape(x: str) -> str:
    s = str(x)
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def write_latex_summary(df: pd.DataFrame, out_path: Path):
    cols = [
        "theme_family",
        "category",
        "chi2",
        "dof",
        "p_value",
        "cramers_v",
        "n_assignments",
    ]
    d = df[cols].copy()
    d["chi2"] = d["chi2"].map(lambda x: f"{x:.2f}")
    d["p_value"] = d["p_value"].map(lambda x: f"{x:.2e}")
    d["cramers_v"] = d["cramers_v"].map(lambda x: f"{x:.3f}")

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Binary chi-square association tests by category (category vs. all-other categories) across tools.}"
    )
    lines.append(r"\label{tab:rq-binary-chi-all-categories}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{llrrrrrr}")
    lines.append(r"\hline")
    lines.append(r"Theme family & Category & $\chi^2$ & dof & $p$ & Cramer's $V$ & $N$ \\")
    lines.append(r"\hline")
    for _, r in d.iterrows():
        row = [
            latex_escape(r["theme_family"]),
            latex_escape(r["category"]),
            r["chi2"],
            str(r["dof"]),
            r["p_value"],
            r["cramers_v"],
            str(r["n_assignments"]),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    out_path.write_text("\n".join(lines))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    primary = load_primary()
    families = {
        "Primary themes (inductively generated)": primary,
        "Secondary themes (inductively generated; mapped)": load_secondary_mapped(primary),
        "Secondary themes (as coded field)": load_secondary_coded(),
        "Nielsen themes": load_nielsen(),
        "Affected-component themes": load_components(),
    }

    summary_rows: list[dict] = []
    share_rows: list[pd.DataFrame] = []

    for family_name, mat in families.items():
        mat = mat.sort_index()
        for category in mat.columns:
            stats, shares = binary_chi_for_category(mat, category)
            stats["theme_family"] = family_name
            summary_rows.append(stats)
            shares["theme_family"] = family_name
            shares["category"] = category
            share_rows.append(shares)

    summary = pd.DataFrame(summary_rows).sort_values(["theme_family", "p_value", "category"])
    shares_long = pd.concat(share_rows, ignore_index=True)

    summary_csv = OUT_DIR / "rq_binary_chi_all_categories_summary.csv"
    shares_csv = OUT_DIR / "rq_binary_chi_all_categories_shares_long.csv"
    summary_tex = OUT_DIR / "rq_binary_chi_all_categories_summary.tex"

    summary.to_csv(summary_csv, index=False)
    shares_long.to_csv(shares_csv, index=False)
    write_latex_summary(summary, summary_tex)

    # Convenience extracts for the two most-cited families.
    summary[summary["theme_family"] == "Primary themes (inductively generated)"].to_csv(
        OUT_DIR / "rq_binary_chi_primary_theme_summary.csv", index=False
    )
    summary[summary["theme_family"] == "Affected-component themes"].to_csv(
        OUT_DIR / "rq_binary_chi_component_theme_summary.csv", index=False
    )
    shares_long[shares_long["theme_family"] == "Primary themes (inductively generated)"].to_csv(
        OUT_DIR / "rq_binary_chi_primary_theme_shares_long.csv", index=False
    )
    shares_long[shares_long["theme_family"] == "Affected-component themes"].to_csv(
        OUT_DIR / "rq_binary_chi_component_theme_shares_long.csv", index=False
    )

    print("Wrote:", summary_csv)
    print("Wrote:", shares_csv)
    print("Wrote:", summary_tex)


if __name__ == "__main__":
    main()

