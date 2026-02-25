from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.discrete.discrete_model import NegativeBinomial


# ---------------- Config ----------------
DATA_DIR = Path("data/phase3")
OUT_DIR = Path("outputs")
TABLE_DIR = OUT_DIR / "tables"
PLOT_DIR = OUT_DIR / "plots" / "trend"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

COL_TOOL = "repo"
COL_CREATED = "created_at"
COL_USABILITY = "usability_non-usability_type"

THEME_COLS = [
    "Associated Component Theme",
    "L1_Theme",
    "L1_Theme Secondary",
    "Nielsen_theme",
]

# Heatmap size control: pick top themes by total occurrences (across all tools)
TOPK_THEMES_PER_COL = 15
ALPHA = 0.05
DISPERSION_OVER_THRESHOLD = 1.5
DISPERSION_UNDER_THRESHOLD = 0.8

# Aggregate curve outputs (across all tools)
AGG_USABILITY_THEME_COL = "L1_Theme"
AGG_COMPONENT_COL = "Associated Component Theme"
AGG_TOPK_CATEGORIES: int | None = None  # Set an int to limit the number of plotted curves.
TOOL_GROUP_ORDER = ["Sigstore", "Vault", "Notary", "Keyfactor", "OpenPubKey"]
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

# If you want to restrict to a date window:
DATE_MIN = None  # e.g., "2021-01-01"
DATE_MAX = None  # e.g., "2025-12-31"


# ---------------- Helpers ----------------
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
    # Capture tokens from one numbered label up to the next numbered label or end.
    matches = re.findall(r"(?:^|,\s*)(\d+\.\s.*?)(?=(?:,\s*\d+\.|$))", s)
    if matches:
        return [m.strip() for m in matches if m.strip()]
    return split_multi(s)


def split_multi_for_col(cell, col_name: str) -> list[str]:
    if col_name == "Nielsen_theme":
        return split_multi_nielsen(cell)
    return split_multi(cell)


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
    out[norm_key("9. Help users recognize and recover from errors")] = NIELSEN_THEME_ORDER[8]
    out[norm_key("Help users recognize, diagnose, and recover from errors")] = NIELSEN_THEME_ORDER[8]
    return out


def load_phase3_excels(data_dir: Path) -> pd.DataFrame:
    paths = sorted(data_dir.glob("*.xlsx"))
    if not paths:
        raise FileNotFoundError(f"No .xlsx files found in {data_dir.resolve()}")
    dfs = []
    for fp in paths:
        df = pd.read_excel(fp)
        df["__source_file__"] = fp.name
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)

    # Basic checks
    needed = {COL_TOOL, COL_CREATED, COL_USABILITY}
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Parse dates
    out[COL_CREATED] = pd.to_datetime(out[COL_CREATED], errors="coerce", utc=True)
    out[COL_USABILITY] = out[COL_USABILITY].apply(norm_usability)

    # Optional date filter
    if DATE_MIN is not None:
        out = out[out[COL_CREATED] >= pd.to_datetime(DATE_MIN, utc=True)]
    if DATE_MAX is not None:
        out = out[out[COL_CREATED] <= pd.to_datetime(DATE_MAX, utc=True)]

    # Drop rows without created_at
    out = out[out[COL_CREATED].notna()].copy()
    return out


def monthly_index(series_dt: pd.Series) -> pd.Series:
    # Convert timestamps to monthly Period
    return series_dt.dt.to_period("M")


def complete_month_grid(month_counts: pd.DataFrame, months: pd.PeriodIndex) -> pd.DataFrame:
    # Ensure every month exists with count=0
    month_counts = month_counts.set_index("month").reindex(months, fill_value=0).reset_index()
    month_counts.rename(columns={"index": "month"}, inplace=True)
    return month_counts


def fit_poisson_slope(month_df: pd.DataFrame) -> dict:
    """
    month_df: columns = ["month", "count"]
    Model: count ~ month_index (integer)
    Returns slope beta, RR, pct change per month, CI, p-value.
    """
    month_df = month_df.sort_values("month").copy()
    month_df["t"] = np.arange(len(month_df), dtype=float)
    y = month_df["count"].astype(float).values
    X = sm.add_constant(month_df["t"].values)
    res = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    return _poisson_stats_from_result(res, y, int(len(month_df)))


def _poisson_stats_from_result(res, y: np.ndarray, n_months: int) -> dict:
    beta = float(res.params[1])
    se = float(res.bse[1])
    p = float(res.pvalues[1])

    z = 1.96
    lo, hi = beta - z * se, beta + z * se
    rr = float(np.exp(beta))
    rr_lo, rr_hi = float(np.exp(lo)), float(np.exp(hi))

    pct = (rr - 1.0) * 100.0
    pct_lo, pct_hi = (rr_lo - 1.0) * 100.0, (rr_hi - 1.0) * 100.0

    # Basic diagnostics for Poisson model adequacy.
    df_resid = float(res.df_resid) if pd.notna(res.df_resid) else np.nan
    pearson_chi2 = float(res.pearson_chi2) if pd.notna(res.pearson_chi2) else np.nan
    deviance = float(res.deviance) if pd.notna(res.deviance) else np.nan
    if pd.notna(df_resid) and df_resid > 0:
        dispersion_pearson = pearson_chi2 / df_resid if pd.notna(pearson_chi2) else np.nan
        dispersion_deviance = deviance / df_resid if pd.notna(deviance) else np.nan
    else:
        dispersion_pearson = np.nan
        dispersion_deviance = np.nan

    y_hat = res.predict()
    y_hat = np.asarray(y_hat, dtype=float)
    zero_frac_obs = float(np.mean(y == 0.0))
    # Expected zero-probability under Poisson is exp(-mu_t); average over t.
    zero_frac_pred = float(np.mean(np.exp(-np.clip(y_hat, 0.0, None))))

    # Residual serial dependence check (lag-1 Ljung-Box on Pearson residuals).
    lb_pvalue_lag1 = np.nan
    try:
        resid = np.asarray(res.resid_pearson, dtype=float)
        if len(resid) >= 4:
            lb = acorr_ljungbox(resid, lags=[1], return_df=True)
            lb_pvalue_lag1 = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        lb_pvalue_lag1 = np.nan

    return {
        "beta": beta,
        "se": se,
        "p_value": p,
        "rr_per_month": rr,
        "rr_ci_low": rr_lo,
        "rr_ci_high": rr_hi,
        "pct_change_per_month": pct,
        "pct_ci_low": pct_lo,
        "pct_ci_high": pct_hi,
        "n_months": n_months,
        "total_count": int(np.sum(y)),
        "n_obs": int(len(y)),
        "df_resid": float(df_resid) if pd.notna(df_resid) else np.nan,
        "pearson_chi2": pearson_chi2,
        "deviance": deviance,
        "dispersion_pearson": dispersion_pearson,
        "dispersion_deviance": dispersion_deviance,
        "overdispersed_1_5": bool(pd.notna(dispersion_pearson) and dispersion_pearson > DISPERSION_OVER_THRESHOLD),
        "underdispersed_0_8": bool(pd.notna(dispersion_pearson) and dispersion_pearson < DISPERSION_UNDER_THRESHOLD),
        "zero_frac_obs": zero_frac_obs,
        "zero_frac_pred_poisson": zero_frac_pred,
        "ljungbox_p_lag1": lb_pvalue_lag1,
        "autocorr_sig_0_05": bool(pd.notna(lb_pvalue_lag1) and lb_pvalue_lag1 < ALPHA),
    }


def _coef_to_rate_stats(beta: float, se: float, p: float) -> dict:
    z = 1.96
    lo, hi = beta - z * se, beta + z * se
    rr = float(np.exp(beta))
    rr_lo, rr_hi = float(np.exp(lo)), float(np.exp(hi))
    pct = (rr - 1.0) * 100.0
    pct_lo, pct_hi = (rr_lo - 1.0) * 100.0, (rr_hi - 1.0) * 100.0
    return {
        "beta": float(beta),
        "se": float(se),
        "p_value": float(p),
        "rr_per_month": rr,
        "rr_ci_low": rr_lo,
        "rr_ci_high": rr_hi,
        "pct_change_per_month": pct,
        "pct_ci_low": pct_lo,
        "pct_ci_high": pct_hi,
    }


def fit_poisson_robust_slope(month_df: pd.DataFrame) -> dict:
    """
    Poisson slope with HC0 robust covariance (same mean structure as Poisson GLM).
    """
    month_df = month_df.sort_values("month").copy()
    month_df["t"] = np.arange(len(month_df), dtype=float)
    y = month_df["count"].astype(float).values
    X = sm.add_constant(month_df["t"].values)
    res = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type="HC0")
    beta = float(res.params[1])
    se = float(res.bse[1])
    p = float(res.pvalues[1])
    out = _coef_to_rate_stats(beta, se, p)
    out["sig_0_05"] = bool(p < ALPHA)
    return out


def fit_negative_binomial_slope(month_df: pd.DataFrame) -> dict:
    """
    Negative-binomial sensitivity model for monthly count trend.
    """
    month_df = month_df.sort_values("month").copy()
    month_df["t"] = np.arange(len(month_df), dtype=float)
    y = month_df["count"].astype(float).values
    X = sm.add_constant(month_df["t"].values)
    try:
        res = NegativeBinomial(y, X).fit(disp=0, maxiter=200)
        beta = float(res.params[1])
        se = float(res.bse[1])
        p = float(res.pvalues[1])
        out = _coef_to_rate_stats(beta, se, p)
        alpha_nb = np.nan
        if hasattr(res, "lnalpha") and pd.notna(res.lnalpha):
            alpha_nb = float(np.exp(res.lnalpha))
        elif len(res.params) > X.shape[1]:
            alpha_raw = float(res.params[-1])
            alpha_nb = float(np.exp(alpha_raw)) if alpha_raw < 0 else alpha_raw
        out.update(
            {
                "nb_alpha": alpha_nb,
                "nb_converged": bool(getattr(res, "mle_retvals", {}).get("converged", True)),
                "nb_fit_error": "",
                "sig_0_05": bool(p < ALPHA),
            }
        )
        return out
    except Exception as e:
        return {
            "beta": np.nan,
            "se": np.nan,
            "p_value": np.nan,
            "rr_per_month": np.nan,
            "rr_ci_low": np.nan,
            "rr_ci_high": np.nan,
            "pct_change_per_month": np.nan,
            "pct_ci_low": np.nan,
            "pct_ci_high": np.nan,
            "nb_alpha": np.nan,
            "nb_converged": False,
            "nb_fit_error": str(e)[:180],
            "sig_0_05": False,
        }


def fit_poisson_curve(month_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    month_df: columns = ["month", "count"]
    Returns per-month fitted means plus slope summary stats.
    """
    fit_df = month_df.sort_values("month").copy()
    fit_df["t"] = np.arange(len(fit_df), dtype=float)

    y = fit_df["count"].astype(float).values
    X = sm.add_constant(fit_df["t"].values)
    res = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    fit_df["expected_count"] = res.predict(X)

    stats = _poisson_stats_from_result(res, y, int(len(fit_df)))
    return fit_df, stats


def latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    def esc(x: str) -> str:
        out = str(x)
        repl = {
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
        for k, v in repl.items():
            out = out.replace(k, v)
        return out

    colfmt = "l" * len(df.columns)
    header = " & ".join(esc(c) for c in df.columns) + r" \\"
    rows = []
    for _, row in df.iterrows():
        rows.append(" & ".join(esc(v) for v in row.tolist()) + r" \\")

    body = "\n".join(
        [
            f"\\begin{{tabular}}{{{colfmt}}}",
            "\\hline",
            header,
            "\\hline",
            *rows,
            "\\hline",
            "\\end{tabular}",
        ]
    )
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\scriptsize\n"
        f"{body}\n"
        "\\end{table}\n"
    )


def sig_mark(p: float, alpha: float = ALPHA) -> str:
    return "*" if pd.notna(p) and float(p) < alpha else ""


def tool_family_from_repo(repo: str) -> str:
    s = str(repo).strip()
    org, _, name = s.partition("/")
    org_l = org.lower()
    name_l = name.lower()
    if org_l == "sigstore":
        return "Sigstore"
    if org_l == "hashicorp" and name_l == "vault":
        return "Vault"
    if org_l == "notaryproject" and name_l == "notation":
        return "Notary"
    if org_l == "keyfactor":
        return "Keyfactor"
    if org_l == "openpubkey":
        return "OpenPubKey"
    return org if org else s


def build_avg_slope_tables_by_tool_and_theme(
    out_df: pd.DataFrame,
    output_stem: str,
    theme_name_for_caption: str,
    latex_label_base: str,
    theme_order: list[str] | None = None,
):
    """
    Build the requested summary table:
    average Poisson slopes (beta1) by tool family and L1 usability theme.
    """
    base = out_df[["tool", "theme", "beta"]].copy()
    if base.empty:
        return

    base["tool_family"] = base["tool"].map(tool_family_from_repo)
    avg = (
        base.groupby(["tool_family", "theme"], as_index=False)["beta"]
        .mean()
        .rename(columns={"beta": "beta1_avg"})
    )
    avg.to_csv(TABLE_DIR / f"{output_stem}_long.csv", index=False)

    # Wide table for manuscript-ready presentation.
    wide = avg.pivot_table(
        index="tool_family",
        columns="theme",
        values="beta1_avg",
        aggfunc="mean",
    )
    ordered_rows = [x for x in TOOL_GROUP_ORDER if x in wide.index] + [x for x in wide.index if x not in TOOL_GROUP_ORDER]
    wide = wide.reindex(ordered_rows)
    if theme_order is None:
        wide = wide.reindex(columns=sorted(wide.columns))
    else:
        ordered_cols = [c for c in theme_order if c in wide.columns] + [c for c in wide.columns if c not in theme_order]
        wide = wide.reindex(columns=ordered_cols)
    wide.to_csv(TABLE_DIR / f"{output_stem}.csv")

    wide_tex = wide.reset_index().copy()
    for c in wide_tex.columns[1:]:
        wide_tex[c] = wide_tex[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    (TABLE_DIR / f"{output_stem}.tex").write_text(
        latex_table(
            wide_tex,
            f"Average Poisson time-trend slopes ($\\beta_1$) by tool and {theme_name_for_caption}. Negative values indicate decreasing expected issue frequency over time.",
            latex_label_base,
        )
    )

    # Compact interpretation helper: broad negativity vs mixed directionality.
    summary = avg.groupby("tool_family", as_index=False).agg(
        avg_beta=("beta1_avg", "mean"),
        min_beta=("beta1_avg", "min"),
        max_beta=("beta1_avg", "max"),
        n_themes=("beta1_avg", "size"),
        n_negative=("beta1_avg", lambda s: int((s < 0).sum())),
        n_positive=("beta1_avg", lambda s: int((s > 0).sum())),
    )
    summary["pattern"] = np.where(
        (summary["n_negative"] > 0) & (summary["n_positive"] > 0),
        "mixed",
        np.where(summary["n_positive"] == 0, "all_negative_or_zero", "all_positive_or_zero"),
    )
    summary = summary.set_index("tool_family").reindex(ordered_rows).reset_index()
    summary.to_csv(TABLE_DIR / f"{output_stem}_summary.csv", index=False)

    summary_tex = summary.copy()
    for c in ["avg_beta", "min_beta", "max_beta"]:
        summary_tex[c] = summary_tex[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    (TABLE_DIR / f"{output_stem}_summary.tex").write_text(
        latex_table(
            summary_tex,
            f"Summary of average Poisson slope patterns by tool family across {theme_name_for_caption}.",
            f"{latex_label_base}-summary",
        )
    )


def plot_heatmap(matrix: pd.DataFrame, title: str, out_path: Path):
    """
    matrix: index=tool, columns=theme, values=beta (slope).
    """
    tools = list(matrix.index)
    themes = list(matrix.columns)
    arr = matrix.values.astype(float)

    plt.figure(figsize=(max(8, 0.6 * len(themes)), max(4, 0.5 * len(tools))))
    im = plt.imshow(arr, aspect="auto")  # default colormap
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Poisson slope (beta per month)")
    plt.xticks(np.arange(len(themes)), themes, rotation=45, ha="right")
    plt.yticks(np.arange(len(tools)), tools)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close()


def plot_heatmap_significant_only(
    beta_mat: pd.DataFrame,
    p_mat: pd.DataFrame,
    title: str,
    out_path: Path,
    alpha: float = ALPHA,
):
    """
    Show only statistically significant beta values (p < alpha).
    Non-significant cells are masked (light gray).
    """
    tools = list(beta_mat.index)
    themes = list(beta_mat.columns)
    p_aligned = p_mat.reindex(index=tools, columns=themes)
    sig = p_aligned < alpha
    arr = beta_mat.where(sig).values.astype(float)
    masked = np.ma.masked_invalid(arr)

    plt.figure(figsize=(max(8, 0.6 * len(themes)), max(4, 0.5 * len(tools))))
    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="#e5e7eb")
    im = plt.imshow(masked, aspect="auto", cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04, label=f"Poisson slope (beta), p < {alpha:g}")
    plt.xticks(np.arange(len(themes)), themes, rotation=45, ha="right")
    plt.yticks(np.arange(len(tools)), tools)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close()


def aggregate_poisson_expected_curves(
    df_u: pd.DataFrame,
    all_months: pd.PeriodIndex,
    category_col: str,
    stem: str,
    title: str,
    latex_caption: str,
    latex_label: str,
):
    if category_col not in df_u.columns:
        raise KeyError(f"Missing category column: {category_col}")

    tmp = df_u[["month", category_col]].copy()
    tmp[category_col] = tmp[category_col].apply(lambda x: split_multi_for_col(x, category_col))
    tmp = tmp.explode(category_col)
    tmp = tmp[tmp[category_col].notna() & (tmp[category_col].astype(str).str.strip() != "")].copy()
    if tmp.empty:
        return pd.DataFrame()
    tmp[category_col] = tmp[category_col].astype(str).str.strip()

    totals = tmp[category_col].value_counts()
    if AGG_TOPK_CATEGORIES is not None:
        totals = totals.head(AGG_TOPK_CATEGORIES)
        tmp = tmp[tmp[category_col].isin(totals.index)].copy()

    categories = totals.index.tolist()
    curve_parts = []
    stat_rows = []

    for cat in categories:
        g = tmp[tmp[category_col] == cat]
        mc = g.groupby("month").size().reset_index(name="count")
        mc = complete_month_grid(mc, all_months)

        fit_df, stats = fit_poisson_curve(mc)
        fit_df["category"] = cat
        fit_df["month_start"] = fit_df["month"].dt.to_timestamp()
        curve_parts.append(fit_df[["category", "month", "month_start", "count", "expected_count"]])
        stat_rows.append({"category": cat, **stats})

    curves = pd.concat(curve_parts, ignore_index=True)
    stats_df = pd.DataFrame(stat_rows).sort_values("beta")
    stats_df["sig_0_05"] = stats_df["p_value"] < ALPHA

    curves.to_csv(TABLE_DIR / f"{stem}_curves.csv", index=False)
    stats_df.to_csv(TABLE_DIR / f"{stem}_slopes.csv", index=False)

    plt.figure(figsize=(12.5, 7))
    for cat in categories:
        sub = curves[curves["category"] == cat].sort_values("month")
        plt.plot(sub["month_start"], sub["expected_count"], linewidth=2, label=cat)

    plt.title(title)
    plt.xlabel("Calendar month")
    plt.ylabel("Expected monthly issue count (Poisson mean)")
    plt.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{stem}.png", dpi=170, bbox_inches="tight")
    plt.close()

    stats_tbl = stats_df[[
        "category",
        "beta",
        "rr_per_month",
        "pct_change_per_month",
        "p_value",
        "n_months",
        "total_count",
    ]].copy()
    stats_tbl["sig_0_05"] = stats_tbl["p_value"].map(lambda p: "yes" if float(p) < ALPHA else "no")
    stats_tbl["beta"] = stats_tbl["beta"].map(lambda x: f"{x:.4f}")
    stats_tbl["rr_per_month"] = stats_tbl["rr_per_month"].map(lambda x: f"{x:.3f}")
    stats_tbl["pct_change_per_month"] = stats_tbl["pct_change_per_month"].map(lambda x: f"{x:.2f}")
    stats_tbl["p_value"] = stats_tbl["p_value"].map(lambda x: f"{x:.3g}{sig_mark(x)}")

    (TABLE_DIR / f"{stem}_slopes.tex").write_text(
        latex_table(
            stats_tbl,
            latex_caption,
            latex_label,
        )
    )
    return stats_df


# ---------------- Main analysis ----------------
def main():
    df = load_phase3_excels(DATA_DIR)

    # Focus: monthly counts of usability issues
    df_u = df[df[COL_USABILITY] == 1].copy()
    if df_u.empty:
        raise ValueError("No usability issues found (usability_non-usability_type == 1).")

    df_u["month"] = monthly_index(df_u[COL_CREATED])
    diagnostic_rows: list[dict] = []

    # Global month range (so slopes are comparable across tools/themes)
    all_months = pd.period_range(df_u["month"].min(), df_u["month"].max(), freq="M")

    # ---- A) Overall per-tool (supports H2 directly) ----
    overall_rows = []
    for tool, g in df_u.groupby(COL_TOOL):
        mc = g.groupby("month").size().reset_index(name="count")
        mc = complete_month_grid(mc, all_months)
        stats = fit_poisson_slope(mc)
        robust_stats = fit_poisson_robust_slope(mc)
        nb_stats = fit_negative_binomial_slope(mc)
        overall_rows.append(
            {
                "tool": tool,
                **stats,
                "poisson_robust_se": robust_stats["se"],
                "poisson_robust_p_value": robust_stats["p_value"],
                "poisson_robust_rr_per_month": robust_stats["rr_per_month"],
                "poisson_robust_sig_0_05": robust_stats["sig_0_05"],
                "nb_beta": nb_stats["beta"],
                "nb_se": nb_stats["se"],
                "nb_p_value": nb_stats["p_value"],
                "nb_rr_per_month": nb_stats["rr_per_month"],
                "nb_alpha": nb_stats["nb_alpha"],
                "nb_converged": nb_stats["nb_converged"],
                "nb_fit_error": nb_stats["nb_fit_error"],
                "nb_sig_0_05": nb_stats["sig_0_05"],
            }
        )
        diagnostic_rows.append(
            {
                "model_set": "overall_usability_by_tool",
                "tool": tool,
                "theme_col": "(all)",
                "theme_or_category": "(all usability)",
                **stats,
            }
        )

    overall = pd.DataFrame(overall_rows).sort_values("beta")
    overall["sig_0_05"] = overall["p_value"] < ALPHA
    overall.to_csv(TABLE_DIR / "poisson_overall_usability_trends_by_tool.csv", index=False)

    # A compact LaTeX table (beta + rr + pct change)
    overall_tbl = overall[[
        "tool", "beta", "rr_per_month", "pct_change_per_month", "p_value", "n_months", "total_count"
    ]].copy()
    overall_tbl["sig_0_05"] = overall_tbl["p_value"].map(lambda p: "yes" if float(p) < ALPHA else "no")
    overall_tbl["beta"] = overall_tbl["beta"].map(lambda x: f"{x:.4f}")
    overall_tbl["rr_per_month"] = overall_tbl["rr_per_month"].map(lambda x: f"{x:.3f}")
    overall_tbl["pct_change_per_month"] = overall_tbl["pct_change_per_month"].map(lambda x: f"{x:.2f}")
    overall_tbl["p_value"] = overall_tbl["p_value"].map(lambda x: f"{x:.3g}{sig_mark(x)}")

    (TABLE_DIR / "poisson_overall_usability_trends_by_tool.tex").write_text(
        latex_table(
            overall_tbl,
            "Poisson regression time trends for monthly counts of usability issues (all themes combined).",
            "tab:poisson-overall-trends",
        )
    )

    # Robust SE sensitivity table (Poisson mean, HC0 covariance).
    robust_tbl = overall[
        [
            "tool",
            "beta",
            "se",
            "p_value",
            "poisson_robust_se",
            "poisson_robust_p_value",
            "rr_per_month",
            "poisson_robust_rr_per_month",
            "sig_0_05",
            "poisson_robust_sig_0_05",
        ]
    ].copy()
    robust_tbl.to_csv(TABLE_DIR / "poisson_overall_usability_trends_by_tool_robust_se.csv", index=False)
    robust_tex = robust_tbl.copy()
    for c in ["beta", "se", "poisson_robust_se", "rr_per_month", "poisson_robust_rr_per_month"]:
        robust_tex[c] = robust_tex[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    for c in ["p_value", "poisson_robust_p_value"]:
        robust_tex[c] = robust_tex[c].map(lambda x: f"{x:.3g}" if pd.notna(x) else "")
    for c in ["sig_0_05", "poisson_robust_sig_0_05"]:
        robust_tex[c] = robust_tex[c].map(lambda v: "yes" if bool(v) else "no")
    (TABLE_DIR / "poisson_overall_usability_trends_by_tool_robust_se.tex").write_text(
        latex_table(
            robust_tex,
            "Sensitivity check for overall usability trends: Poisson standard errors vs HC0 robust standard errors by tool.",
            "tab:poisson-overall-trends-robust-se",
        )
    )

    # Negative-binomial sensitivity table.
    nb_tbl = overall[
        [
            "tool",
            "beta",
            "p_value",
            "rr_per_month",
            "nb_beta",
            "nb_se",
            "nb_p_value",
            "nb_rr_per_month",
            "nb_alpha",
            "nb_converged",
            "nb_sig_0_05",
            "nb_fit_error",
        ]
    ].copy()
    nb_tbl.to_csv(TABLE_DIR / "negative_binomial_sensitivity_overall_by_tool.csv", index=False)
    nb_tex = nb_tbl.copy()
    for c in ["beta", "nb_beta", "nb_se", "rr_per_month", "nb_rr_per_month", "nb_alpha"]:
        nb_tex[c] = nb_tex[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    for c in ["p_value", "nb_p_value"]:
        nb_tex[c] = nb_tex[c].map(lambda x: f"{x:.3g}" if pd.notna(x) else "")
    for c in ["nb_converged", "nb_sig_0_05"]:
        nb_tex[c] = nb_tex[c].map(lambda v: "yes" if bool(v) else "no")
    (TABLE_DIR / "negative_binomial_sensitivity_overall_by_tool.tex").write_text(
        latex_table(
            nb_tex,
            "Negative-binomial sensitivity check for overall monthly usability-count trends by tool.",
            "tab:nb-sensitivity-overall-by-tool",
        )
    )

    # ---- B) Aggregate expected-count curves across all tools ----
    agg_l1 = aggregate_poisson_expected_curves(
        df_u=df_u,
        all_months=all_months,
        category_col=AGG_USABILITY_THEME_COL,
        stem="aggregate_poisson_expected_counts_l1_theme",
        title=(
            "Aggregate Poisson fitted means: expected monthly usability issue counts "
            "by L1 theme (all tools)"
        ),
        latex_caption=(
            "Aggregate Poisson regression slopes for expected monthly usability issue "
            "counts by L1 theme across all tools."
        ),
        latex_label="tab:poisson-aggregate-l1-theme",
    )
    if not agg_l1.empty:
        for _, r in agg_l1.iterrows():
            diagnostic_rows.append(
                {
                    "model_set": "aggregate_by_l1_theme",
                    "tool": "ALL",
                    "theme_col": AGG_USABILITY_THEME_COL,
                    "theme_or_category": str(r["category"]),
                    **r.to_dict(),
                }
            )

    agg_comp = aggregate_poisson_expected_curves(
        df_u=df_u,
        all_months=all_months,
        category_col=AGG_COMPONENT_COL,
        stem="aggregate_poisson_expected_counts_associated_component",
        title=(
            "Aggregate Poisson fitted means: expected monthly usability issue counts "
            "by affected component (all tools)"
        ),
        latex_caption=(
            "Aggregate Poisson regression slopes for expected monthly usability issue "
            "counts by affected component across all tools."
        ),
        latex_label="tab:poisson-aggregate-component",
    )
    if not agg_comp.empty:
        for _, r in agg_comp.iterrows():
            diagnostic_rows.append(
                {
                    "model_set": "aggregate_by_component_theme",
                    "tool": "ALL",
                    "theme_col": AGG_COMPONENT_COL,
                    "theme_or_category": str(r["category"]),
                    **r.to_dict(),
                }
            )

    # ---- C) Theme-by-tool slopes for each Phase 3 theme column ----
    nielsen_alias_map = _nielsen_alias_to_canonical()
    l1_theme_out: pd.DataFrame | None = None
    nielsen_theme_out: pd.DataFrame | None = None
    for theme_col in THEME_COLS:
        if theme_col not in df_u.columns:
            raise KeyError(f"Missing theme column: {theme_col}")

        tmp = df_u[[COL_TOOL, "month", theme_col]].copy()
        tmp[theme_col] = tmp[theme_col].apply(lambda x: split_multi_for_col(x, theme_col))
        tmp = tmp.explode(theme_col)
        tmp = tmp[tmp[theme_col].notna() & (tmp[theme_col].astype(str).str.strip() != "")].copy()
        if theme_col == "Nielsen_theme":
            tmp[theme_col] = tmp[theme_col].astype(str).str.strip()
            tmp[theme_col] = tmp[theme_col].map(lambda x: nielsen_alias_map.get(norm_key(x)))
            tmp = tmp[tmp[theme_col].notna()].copy()

        # Choose top themes globally (by total occurrences)
        if theme_col == "Nielsen_theme":
            top_themes = [x for x in NIELSEN_THEME_ORDER if x in set(tmp[theme_col].unique())]
            if TOPK_THEMES_PER_COL is not None:
                top_themes = top_themes[:TOPK_THEMES_PER_COL]
        else:
            top_themes = (
                tmp[theme_col].value_counts()
                .head(TOPK_THEMES_PER_COL)
                .index.tolist()
            )
        tmp = tmp[tmp[theme_col].isin(top_themes)].copy()

        rows = []
        for (tool, theme), g in tmp.groupby([COL_TOOL, theme_col]):
            mc = g.groupby("month").size().reset_index(name="count")
            mc = complete_month_grid(mc, all_months)

            stats = fit_poisson_slope(mc)
            rows.append({
                "tool": tool,
                "theme_col": theme_col,
                "theme": theme,
                **stats
            })

        out = pd.DataFrame(rows)
        out = out.sort_values(["tool", "beta"])
        out["sig_0_05"] = out["p_value"] < ALPHA
        out.to_csv(TABLE_DIR / f"poisson_trends_{safe_filename(theme_col)}.csv", index=False)
        for _, r in out.iterrows():
            diagnostic_rows.append(
                {
                    "model_set": f"tool_by_{safe_filename(theme_col)}",
                    "tool": str(r["tool"]),
                    "theme_col": theme_col,
                    "theme_or_category": str(r["theme"]),
                    **r.to_dict(),
                }
            )
        if theme_col == AGG_USABILITY_THEME_COL:
            l1_theme_out = out.copy()
        if theme_col == "Nielsen_theme":
            nielsen_theme_out = out.copy()

        # Pivot into tool × theme beta for heatmap
        heat = out.pivot_table(index="tool", columns="theme", values="beta", aggfunc="mean").fillna(0.0)
        heat_p = out.pivot_table(index="tool", columns="theme", values="p_value", aggfunc="mean")
        if theme_col == "Nielsen_theme":
            heat = heat.reindex(columns=[x for x in NIELSEN_THEME_ORDER if x in heat.columns], fill_value=0.0)
            heat_p = heat_p.reindex(columns=[x for x in NIELSEN_THEME_ORDER if x in heat_p.columns])
        plot_heatmap(
            heat,
            title=f"Poisson slopes by tool × {theme_col} (top {TOPK_THEMES_PER_COL} themes)",
            out_path=PLOT_DIR / f"heatmap_beta_{safe_filename(theme_col)}.png",
        )
        plot_heatmap_significant_only(
            heat,
            heat_p,
            title=f"Poisson slopes by tool × {theme_col} (significant only, p < {ALPHA:g})",
            out_path=PLOT_DIR / f"heatmap_beta_sig_{safe_filename(theme_col)}.png",
            alpha=ALPHA,
        )

        # Optional LaTeX table (top themes only, long format)
        out_tbl = out[["tool", "theme", "beta", "rr_per_month", "pct_change_per_month", "p_value"]].copy()
        out_tbl["sig_0_05"] = out_tbl["p_value"].map(lambda p: "yes" if float(p) < ALPHA else "no")
        out_tbl["beta"] = out_tbl["beta"].map(lambda x: f"{x:.4f}")
        out_tbl["rr_per_month"] = out_tbl["rr_per_month"].map(lambda x: f"{x:.3f}")
        out_tbl["pct_change_per_month"] = out_tbl["pct_change_per_month"].map(lambda x: f"{x:.2f}")
        out_tbl["p_value"] = out_tbl["p_value"].map(lambda x: f"{x:.3g}{sig_mark(x)}")

        (TABLE_DIR / f"poisson_trends_{safe_filename(theme_col)}.tex").write_text(
            latex_table(
                out_tbl,
                f"Poisson regression time trends for monthly counts of usability issues by tool and {theme_col} (top {TOPK_THEMES_PER_COL} themes).",
                f"tab:poisson-{safe_filename(theme_col).lower()}",
            )
        )

    # ---- D) Diagnostics tables (separate from trend result tables) ----
    if diagnostic_rows:
        diag = pd.DataFrame(diagnostic_rows)
        # Remove accidental duplicates from merged dict keys while preserving preferred identifier columns.
        keep_cols = [
            "model_set",
            "tool",
            "theme_col",
            "theme_or_category",
            "beta",
            "p_value",
            "n_obs",
            "n_months",
            "df_resid",
            "dispersion_pearson",
            "dispersion_deviance",
            "overdispersed_1_5",
            "underdispersed_0_8",
            "zero_frac_obs",
            "zero_frac_pred_poisson",
            "ljungbox_p_lag1",
            "autocorr_sig_0_05",
            "total_count",
        ]
        existing_cols = [c for c in keep_cols if c in diag.columns]
        diag = diag[existing_cols].copy()
        diag = diag.sort_values(["model_set", "tool", "theme_or_category"], na_position="last")
        diag.to_csv(TABLE_DIR / "poisson_diagnostics_all_models.csv", index=False)

        # Compact summary by model set for manuscript-ready reporting.
        summary = (
            diag.groupby("model_set", as_index=False)
            .agg(
                n_models=("model_set", "size"),
                median_dispersion_pearson=("dispersion_pearson", "median"),
                p90_dispersion_pearson=("dispersion_pearson", lambda s: float(np.nanpercentile(s, 90)) if s.notna().any() else np.nan),
                n_overdispersed_1_5=("overdispersed_1_5", lambda s: int(pd.Series(s).fillna(False).sum())),
                n_underdispersed_0_8=("underdispersed_0_8", lambda s: int(pd.Series(s).fillna(False).sum())),
                n_autocorr_sig_0_05=("autocorr_sig_0_05", lambda s: int(pd.Series(s).fillna(False).sum())),
            )
        )
        summary["pct_overdispersed_1_5"] = np.where(
            summary["n_models"] > 0,
            summary["n_overdispersed_1_5"] / summary["n_models"] * 100.0,
            np.nan,
        )
        summary.to_csv(TABLE_DIR / "poisson_diagnostics_summary_by_model_set.csv", index=False)

        summary_tbl = summary.copy()
        for c in ["median_dispersion_pearson", "p90_dispersion_pearson", "pct_overdispersed_1_5"]:
            if c in summary_tbl.columns:
                summary_tbl[c] = summary_tbl[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        (TABLE_DIR / "poisson_diagnostics_summary_by_model_set.tex").write_text(
            latex_table(
                summary_tbl,
                "Poisson model diagnostic checks by model set (dispersion and residual autocorrelation).",
                "tab:poisson-diagnostics-summary",
            )
        )

        overall_diag = diag[diag["model_set"] == "overall_usability_by_tool"].copy()
        if not overall_diag.empty:
            overall_diag.to_csv(TABLE_DIR / "poisson_diagnostics_overall_by_tool.csv", index=False)
            overall_tbl = overall_diag[
                [
                    "tool",
                    "dispersion_pearson",
                    "dispersion_deviance",
                    "overdispersed_1_5",
                    "underdispersed_0_8",
                    "ljungbox_p_lag1",
                    "autocorr_sig_0_05",
                ]
            ].copy()
            for c in ["dispersion_pearson", "dispersion_deviance", "ljungbox_p_lag1"]:
                overall_tbl[c] = overall_tbl[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
            overall_tbl["overdispersed_1_5"] = overall_tbl["overdispersed_1_5"].map(lambda v: "yes" if bool(v) else "no")
            overall_tbl["underdispersed_0_8"] = overall_tbl["underdispersed_0_8"].map(lambda v: "yes" if bool(v) else "no")
            overall_tbl["autocorr_sig_0_05"] = overall_tbl["autocorr_sig_0_05"].map(lambda v: "yes" if bool(v) else "no")
            (TABLE_DIR / "poisson_diagnostics_overall_by_tool.tex").write_text(
                latex_table(
                    overall_tbl,
                    "Poisson diagnostic checks for overall monthly usability-count trends by tool.",
                    "tab:poisson-diagnostics-overall-by-tool",
                )
            )

    if l1_theme_out is not None and not l1_theme_out.empty:
        build_avg_slope_tables_by_tool_and_theme(
            out_df=l1_theme_out,
            output_stem="poisson_average_slopes_by_tool_and_l1_theme",
            theme_name_for_caption="usability theme (L1_Theme)",
            latex_label_base="tab:poisson-avg-beta-tool-l1-theme",
        )
    if nielsen_theme_out is not None and not nielsen_theme_out.empty:
        build_avg_slope_tables_by_tool_and_theme(
            out_df=nielsen_theme_out,
            output_stem="poisson_average_slopes_by_tool_and_nielsen_theme",
            theme_name_for_caption="Nielsen theme",
            latex_label_base="tab:poisson-avg-beta-tool-nielsen-theme",
            theme_order=NIELSEN_THEME_ORDER,
        )

    print("Done.")
    print("Tables:", TABLE_DIR.resolve())
    print("Heatmaps:", PLOT_DIR.resolve())


if __name__ == "__main__":
    main()
