from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


INPUT_DIR = Path("LLM Prompts/gh-issues")
OUT_DIR = Path("outputs/tables")


def norm_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def first_col(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    by_key = {norm_key(c): c for c in columns}
    for alias in aliases:
        col = by_key.get(norm_key(alias))
        if col is not None:
            return col
    return None


def word_count(value: object) -> int:
    if value is None:
        return 0
    try:
        if pd.isna(value):
            return 0
    except Exception:
        pass
    return len(re.findall(r"\b\w+\b", str(value)))


def safe_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def read_sheet(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    if sheet_name not in xls.sheet_names:
        return pd.DataFrame()
    df = xls.parse(sheet_name)
    if len(df.columns) > 0 and all(str(c).startswith("Unnamed:") for c in df.columns):
        header = df.iloc[0].astype(str).tolist()
        df = df.iloc[1:].copy()
        df.columns = header
    return df


def quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if len(values) else np.nan


def median_iqr(series: pd.Series) -> str:
    med = quantile(series, 0.50)
    q1 = quantile(series, 0.25)
    q3 = quantile(series, 0.75)
    if pd.isna(med):
        return ""
    return f"{med:.0f} [{q1:.0f}, {q3:.0f}]"


def load_issue_context_metrics() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(INPUT_DIR.glob("*___non-human.xlsx")):
        xls = pd.ExcelFile(path)
        issues = read_sheet(xls, "issues")
        comments = read_sheet(xls, "comments")
        if issues.empty:
            continue

        repo_col = first_col(issues.columns, ["repo"])
        issue_col = first_col(issues.columns, ["issue_number", "issue number"])
        author_col = first_col(issues.columns, ["author"])
        title_col = first_col(issues.columns, ["title"])
        body_col = first_col(issues.columns, ["body_text", "body"])
        if not (repo_col and issue_col):
            continue

        base = issues[[repo_col, issue_col]].copy()
        base.columns = ["repo", "issue_number"]
        base["repo"] = base["repo"].astype(str)
        base["issue_number"] = pd.to_numeric(base["issue_number"], errors="coerce").astype("Int64")
        base = base[base["issue_number"].notna()].copy()

        issue_author = (
            issues[author_col].map(safe_text).str.strip()
            if author_col
            else pd.Series([""] * len(issues), index=issues.index)
        )
        issue_title = issues[title_col].map(safe_text) if title_col else pd.Series([""] * len(issues), index=issues.index)
        issue_body = issues[body_col].map(safe_text) if body_col else pd.Series([""] * len(issues), index=issues.index)
        base["issue_author"] = issue_author.loc[base.index].values
        base["issue_text_words"] = (
            issue_title.loc[base.index].map(word_count).to_numpy()
            + issue_body.loc[base.index].map(word_count).to_numpy()
        )
        base["issue_text_chars"] = (
            issue_title.loc[base.index].map(len).to_numpy()
            + issue_body.loc[base.index].map(len).to_numpy()
        )

        if comments.empty:
            comment_summary = pd.DataFrame(
                columns=[
                    "repo",
                    "issue_number",
                    "comment_count",
                    "unique_commenters",
                    "comment_text_words",
                    "comment_text_chars",
                    "comment_authors_joined",
                ]
            )
        else:
            c_repo = first_col(comments.columns, ["repo"])
            c_issue = first_col(comments.columns, ["issue_number", "issue number"])
            c_author = first_col(comments.columns, ["author"])
            c_body = first_col(comments.columns, ["body_text", "body"])
            c = comments.copy()
            if c_repo is None:
                c["__repo__"] = base["repo"].iloc[0] if len(base) else ""
                c_repo = "__repo__"
            if c_issue is None:
                continue
            c = c[[c_repo, c_issue] + ([c_author] if c_author else []) + ([c_body] if c_body else [])].copy()
            rename = {c_repo: "repo", c_issue: "issue_number"}
            if c_author:
                rename[c_author] = "comment_author"
            if c_body:
                rename[c_body] = "comment_body"
            c = c.rename(columns=rename)
            c["repo"] = c["repo"].astype(str)
            c["issue_number"] = pd.to_numeric(c["issue_number"], errors="coerce").astype("Int64")
            c = c[c["issue_number"].notna()].copy()
            c["comment_author"] = c.get("comment_author", "").map(safe_text).str.strip()
            c["comment_body"] = c.get("comment_body", "").map(safe_text)
            c["comment_words"] = c["comment_body"].map(word_count)
            c["comment_chars"] = c["comment_body"].map(len)

            comment_summary = (
                c.groupby(["repo", "issue_number"], dropna=False)
                .agg(
                    comment_count=("comment_body", "size"),
                    unique_commenters=("comment_author", lambda s: s[s != ""].nunique()),
                    comment_text_words=("comment_words", "sum"),
                    comment_text_chars=("comment_chars", "sum"),
                    comment_authors_joined=("comment_author", lambda s: "|".join(sorted(set(x for x in s if x)))),
                )
                .reset_index()
            )

        merged = base.merge(comment_summary, on=["repo", "issue_number"], how="left")
        for col in ["comment_count", "unique_commenters", "comment_text_words", "comment_text_chars"]:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)
        merged["comment_authors_joined"] = merged["comment_authors_joined"].fillna("")
        merged["unique_thread_participants"] = merged.apply(
            lambda row: len(
                {
                    x
                    for x in [row["issue_author"], *str(row["comment_authors_joined"]).split("|")]
                    if str(x).strip()
                }
            ),
            axis=1,
        )
        merged["total_text_words"] = merged["issue_text_words"] + merged["comment_text_words"]
        merged["total_text_chars"] = merged["issue_text_chars"] + merged["comment_text_chars"]
        rows.append(merged)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).drop_duplicates(["repo", "issue_number"], keep="first")


def summarize_by_repo(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "comment_count",
        "unique_commenters",
        "unique_thread_participants",
        "total_text_words",
    ]

    def summarize_group(g: pd.DataFrame) -> dict[str, object]:
        return {
            "n_issues": int(len(g)),
            "comments_median_iqr": median_iqr(g["comment_count"]),
            "comments_mean": round(float(g["comment_count"].mean()), 2),
            "comments_p90": round(quantile(g["comment_count"], 0.90), 1),
            "unique_commenters_median_iqr": median_iqr(g["unique_commenters"]),
            "unique_thread_participants_median_iqr": median_iqr(g["unique_thread_participants"]),
            "text_words_median_iqr": median_iqr(g["total_text_words"]),
            "text_words_mean": round(float(g["total_text_words"].mean()), 1),
            "text_words_p90": round(quantile(g["total_text_words"], 0.90), 1),
        }

    rows = []
    for repo, group in metrics.groupby("repo", sort=True):
        row = {"repo": repo}
        row.update(summarize_group(group))
        rows.append(row)

    overall = {"repo": "Overall"}
    overall.update(summarize_group(metrics))
    rows.append(overall)

    summary = pd.DataFrame(rows)
    for col in numeric_cols:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="ignore")
    return summary


def latex_escape(value: object) -> str:
    text = str(value)
    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ]:
        text = text.replace(old, new)
    return text


def write_latex(summary: pd.DataFrame, out_path: Path) -> None:
    display = summary[
        [
            "repo",
            "n_issues",
            "comments_median_iqr",
            "unique_commenters_median_iqr",
            "unique_thread_participants_median_iqr",
            "text_words_median_iqr",
        ]
    ].copy()
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Per-repository issue context distributions in the full corpus. Values are median [IQR]. Text length is approximate total issue-thread word count, including issue title, body, and comments.}",
        r"\label{tab:issue-context-distributions-by-repo}",
        r"\scriptsize",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Repository & Issues & Comments & Commenters & Thread participants & Text words \\",
        r"\hline",
    ]
    for _, row in display.iterrows():
        lines.append(
            " & ".join(
                [
                    latex_escape(row["repo"]),
                    str(int(row["n_issues"])),
                    latex_escape(row["comments_median_iqr"]),
                    latex_escape(row["unique_commenters_median_iqr"]),
                    latex_escape(row["unique_thread_participants_median_iqr"]),
                    latex_escape(row["text_words_median_iqr"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_issue_context_metrics()
    if metrics.empty:
        raise RuntimeError(f"No issue metrics could be loaded from {INPUT_DIR}")

    summary = summarize_by_repo(metrics)
    metrics_path = OUT_DIR / "issue_context_metrics_full_corpus.csv"
    summary_path = OUT_DIR / "issue_context_distributions_by_repo.csv"
    latex_path = OUT_DIR / "issue_context_distributions_by_repo.tex"

    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_latex(summary, latex_path)

    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {latex_path}")
    print(f"Loaded {len(metrics)} issues across {metrics['repo'].nunique()} repositories.")


if __name__ == "__main__":
    main()
