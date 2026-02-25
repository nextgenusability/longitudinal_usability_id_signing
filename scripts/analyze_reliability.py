from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AGREE_DIR = Path("data/agreement")
OUT_DIR = Path("outputs")
TABLE_DIR = OUT_DIR / "tables"
PLOT_DIR = OUT_DIR / "plots"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def rate(n_bad: int, n_total: int) -> float:
    return (n_bad / n_total * 100.0) if n_total else 0.0


def main():
    rows = []

    # Phase 2: usability agreement
    p2 = AGREE_DIR / "Phase_2_agreement__batch_60___my_labels_vs_human_Usability_non-usability_type.csv"
    if p2.exists():
        df = pd.read_csv(p2)
        n = len(df)
        # If "agree" exists, trust it, else compute
        if "agree" in df.columns:
            err = int((~df["agree"].astype(bool)).sum())
        else:
            err = int((df["human_usability"] != df["my_usability"]).sum())
        rows.append(
            {"phase": "Phase 2", "label": "Usability", "n": n, "errors": err, "error_pct": round(rate(err, n), 1)}
        )

    # Phase 1-ish agreement check: component + code_primary (60)
    p1 = AGREE_DIR / "Agreement_check__Error_Rate_60__-_UPDATED_after_manual_semantic_matches.csv"
    if p1.exists():
        df = pd.read_csv(p1)
        n = len(df)
        if "component_agree" in df.columns:
            err = int((~df["component_agree"].astype(bool)).sum())
            rows.append(
                {"phase": "Agreement 60", "label": "Associated Component", "n": n, "errors": err, "error_pct": round(rate(err, n), 1)}
            )
        if "code_agree" in df.columns:
            err = int((~df["code_agree"].astype(bool)).sum())
            rows.append(
                {"phase": "Agreement 60", "label": "codes_primary", "n": n, "errors": err, "error_pct": round(rate(err, n), 1)}
            )

    # Phase 3 reliability (notation usability issues)
    p3 = AGREE_DIR / "Phase_3_reliability__notation_usability_issues_.csv"
    if p3.exists():
        df = pd.read_csv(p3)
        n = len(df)
        for label, col in [
            ("Nielsen_theme", "Nielsen error"),
            ("L1_Theme", "L1 error"),
            ("Assoc Component Theme", "AssocComp error"),
        ]:
            if col in df.columns:
                err = int(df[col].fillna(0).astype(int).sum())
                rows.append(
                    {"phase": "Phase 3 (notation)", "label": label, "n": n, "errors": err, "error_pct": round(rate(err, n), 1)}
                )

    summary = pd.DataFrame(rows).sort_values(["phase", "label"])
    summary.to_csv(TABLE_DIR / "reliability_error_rates_summary.csv", index=False)

    # Simple bar plot
    if not summary.empty:
        plt.figure(figsize=(10, 5.5))
        x = np.arange(len(summary))
        plt.bar(x, summary["error_pct"].values)
        plt.xticks(x, (summary["phase"] + " | " + summary["label"]).tolist(), rotation=35, ha="right")
        plt.ylabel("Error rate (%)")
        plt.title("Interrater error rates by phase and label type")
        plt.savefig(PLOT_DIR / "reliability_error_rates_bar.png", dpi=160, bbox_inches="tight")
        plt.close()

    print("Wrote:")
    print(" -", (TABLE_DIR / "reliability_error_rates_summary.csv").resolve())
    print(" -", (PLOT_DIR / "reliability_error_rates_bar.png").resolve())


if __name__ == "__main__":
    main()
