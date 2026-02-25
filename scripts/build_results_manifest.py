from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
import csv


ROOT = Path(".")
OUT_DIR = ROOT / "outputs"
MANIFEST = OUT_DIR / "RESULTS_MANIFEST.csv"


RULES: list[dict[str, str]] = [
    {
        "pattern": "outputs/RESULTS_MANIFEST.csv",
        "generated_by": "scripts/build_results_manifest.py",
        "command": "python scripts/build_results_manifest.py",
        "description": "Per-file output manifest (provenance and regeneration command).",
    },
    # Reliability
    {
        "pattern": "outputs/tables/reliability_error_rates_summary.csv",
        "generated_by": "scripts/analyze_reliability.py",
        "command": "idtools_usability run reliability",
        "description": "Reliability/inter-rater error summary table.",
    },
    {
        "pattern": "outputs/plots/reliability_error_rates_bar.png",
        "generated_by": "scripts/analyze_reliability.py",
        "command": "idtools_usability run reliability",
        "description": "Reliability/inter-rater error bar chart.",
    },
    # Phase 3 descriptive themes
    {
        "pattern": "outputs/tables/usability_vs_nonusability_*",
        "generated_by": "scripts/analyze_phase3_themes.py",
        "command": "idtools_usability run phase3-themes",
        "description": "Usability vs non-usability counts (per tool and overall).",
    },
    {
        "pattern": "outputs/tables/l1_theme_grouped_*",
        "generated_by": "scripts/analyze_phase3_themes.py",
        "command": "idtools_usability run phase3-themes",
        "description": "Primary/secondary theme grouped tables and source expansions.",
    },
    {
        "pattern": "outputs/tables/nielsen_theme_*",
        "generated_by": "scripts/analyze_phase3_themes.py",
        "command": "idtools_usability run phase3-themes",
        "description": "Nielsen theme counts/percent tables and source expansions.",
    },
    {
        "pattern": "outputs/tables/rq2_top3_components_by_tool*",
        "generated_by": "scripts/analyze_phase3_themes.py",
        "command": "idtools_usability run phase3-themes",
        "description": "Top-3 affected component summary per tool.",
    },
    {
        "pattern": "outputs/tables/counts_*_by_tool_long.csv",
        "generated_by": "scripts/analyze_phase3_themes.py",
        "command": "idtools_usability run phase3-themes",
        "description": "Long-format per-tool counts for each theme family.",
    },
    {
        "pattern": "outputs/tables/counts_*_by_tool_pivot.csv",
        "generated_by": "scripts/analyze_phase3_themes.py",
        "command": "idtools_usability run phase3-themes",
        "description": "Pivot-format per-tool counts for each theme family.",
    },
    {
        "pattern": "outputs/tables/top20_*_by_tool.tex",
        "generated_by": "scripts/analyze_phase3_themes.py",
        "command": "idtools_usability run phase3-themes",
        "description": "Top-20 LaTeX tables by theme family.",
    },
    {
        "pattern": "outputs/plots/stacked/*",
        "generated_by": "scripts/analyze_phase3_themes.py",
        "command": "idtools_usability run phase3-themes",
        "description": "Stacked and normalized stacked bar charts.",
    },
    {
        "pattern": "outputs/plots/pies/*",
        "generated_by": "scripts/analyze_phase3_themes.py",
        "command": "idtools_usability run phase3-themes",
        "description": "Per-tool pie charts for theme distributions.",
    },
    {
        "pattern": "outputs/plots/bars/*",
        "generated_by": "scripts/analyze_phase3_themes.py",
        "command": "idtools_usability run phase3-themes",
        "description": "Per-tool bar charts (when enabled).",
    },
    # Poisson trends
    {
        "pattern": "outputs/tables/poisson_*",
        "generated_by": "scripts/trend_poisson_phase3.py",
        "command": "python scripts/trend_poisson_phase3.py",
        "description": "Poisson trend tables (overall, per-theme, summaries).",
    },
    {
        "pattern": "outputs/tables/aggregate_poisson_expected_counts_*",
        "generated_by": "scripts/trend_poisson_phase3.py",
        "command": "python scripts/trend_poisson_phase3.py",
        "description": "Aggregate Poisson curves and slope tables.",
    },
    {
        "pattern": "outputs/tables/negative_binomial_sensitivity_*",
        "generated_by": "scripts/trend_poisson_phase3.py",
        "command": "python scripts/trend_poisson_phase3.py",
        "description": "Negative-binomial sensitivity tables for monthly trend estimates.",
    },
    {
        "pattern": "outputs/plots/trend/heatmap_beta*",
        "generated_by": "scripts/trend_poisson_phase3.py",
        "command": "python scripts/trend_poisson_phase3.py",
        "description": "Poisson slope heatmaps (full/significant).",
    },
    {
        "pattern": "outputs/plots/trend/aggregate_poisson_expected_counts_*",
        "generated_by": "scripts/trend_poisson_phase3.py",
        "command": "python scripts/trend_poisson_phase3.py",
        "description": "Aggregate Poisson fitted mean curve plots.",
    },
    # Binary chi-square
    {
        "pattern": "outputs/tables/rq_binary_chi_*",
        "generated_by": "scripts/compute_binary_chi_by_theme.py",
        "command": "python scripts/compute_binary_chi_by_theme.py",
        "description": "Binary chi-square association tables and per-tool shares.",
    },
    {
        "pattern": "outputs/tables/rq_chi_square_tool_theme_association*",
        "generated_by": "legacy/manual chi-square output",
        "command": "n/a",
        "description": "Earlier omnibus chi-square tables retained for reference.",
    },
    # Raw monthly counts
    {
        "pattern": "outputs/tables/raw_monthly_usability_counts_by_tool_*",
        "generated_by": "scripts/plot_raw_monthly_usability_by_tool.py",
        "command": "python scripts/plot_raw_monthly_usability_by_tool.py",
        "description": "Raw monthly usability counts by tool (wide and long).",
    },
    {
        "pattern": "outputs/plots/trend/raw_monthly_usability_counts_by_tool.png",
        "generated_by": "scripts/plot_raw_monthly_usability_by_tool.py",
        "command": "python scripts/plot_raw_monthly_usability_by_tool.py",
        "description": "Raw monthly usability trend lines by tool (non-Poisson).",
    },
    # Codebook table
    {
        "pattern": "outputs/tables/phase3_theme_codebook_with_examples.*",
        "generated_by": "scripts/build_phase3_theme_codebook_table.py",
        "command": "python scripts/build_phase3_theme_codebook_table.py",
        "description": "Theme codebook table (primary/secondary/Nielsen/component definitions).",
    },
    # Publication figures
    {
        "pattern": "outputs/plots/trend/pub/*",
        "generated_by": "scripts/make_pub_rq3_figs.py",
        "command": "python scripts/make_pub_rq3_figs.py",
        "description": "Publication-style PNG figures from fig_rq3_* tables.",
    },
    {
        "pattern": "outputs/plots/trend/pub_latex/*",
        "generated_by": "scripts/pubfigs/make_rq3_figs_latex.py",
        "command": "python scripts/pubfigs/make_rq3_figs_latex.py --tables-dir outputs/tables --out-dir outputs/plots/trend/pub_latex --use-tex off",
        "description": "Publication-style LaTeX-ready PNG/PDF figures.",
    },
    {
        "pattern": "outputs/tables/fig_rq3_*",
        "generated_by": "scripts/pubfigs/make_rq3_figs_latex.py",
        "command": "python scripts/pubfigs/make_rq3_figs_latex.py --tables-dir outputs/tables --out-dir outputs/plots/trend/pub_latex --use-tex off",
        "description": "Helper matrices/curves for publication RQ3 figures.",
    },
    # Legacy/archived
    {
        "pattern": "outputs/phase3_theme_outputs*",
        "generated_by": "legacy artifact",
        "command": "n/a",
        "description": "Legacy bundled outputs from earlier runs.",
    },
    {
        "pattern": "outputs/plots/trend/pub copy/*",
        "generated_by": "legacy artifact",
        "command": "n/a",
        "description": "Legacy copied publication figures.",
    },
    {
        "pattern": "outputs/tables/_warmup.png",
        "generated_by": "legacy artifact",
        "command": "n/a",
        "description": "Legacy warmup figure.",
    },
]


def classify(path: str) -> tuple[str, str, str]:
    for r in RULES:
        if fnmatch(path, r["pattern"]):
            return r["generated_by"], r["command"], r["description"]
    return ("unknown/legacy", "n/a", "Not mapped by current artifact rules.")


def main():
    files = sorted(
        [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != ".DS_Store"]
    )
    rows = []
    for p in files:
        rel = p.as_posix()
        generated_by, command, description = classify(rel)
        rows.append(
            {
                "file_path": rel,
                "generated_by": generated_by,
                "regen_command": command,
                "description": description,
            }
        )

    with MANIFEST.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file_path", "generated_by", "regen_command", "description"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {MANIFEST}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
