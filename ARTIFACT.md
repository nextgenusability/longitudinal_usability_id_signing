# Research Artifact Guide

This repository is structured as a reproducible artifact for the identity-based signing usability study.

## Environment

1. Create and activate a virtual environment.
2. Install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
```

## Inputs

- `data/phase3/*.xlsx`: labeled Phase 3 issue datasets.
- `data/agreement/*.csv`: inter-rater agreement/reliability inputs.

## Reproduce Everything

Run the one-shot pipeline:

```bash
bash scripts/regenerate_artifact.sh
```

This also writes:

- `outputs/RESULTS_MANIFEST.csv` (one row per output file with producer script and regeneration command)

Or run step-by-step:

```bash
idtools_usability run reliability
idtools_usability run phase3-themes
python scripts/trend_poisson_phase3.py
python scripts/compute_binary_chi_by_theme.py
python scripts/plot_raw_monthly_usability_by_tool.py
python scripts/build_phase3_theme_codebook_table.py
python scripts/make_pub_rq3_figs.py
python scripts/pubfigs/make_rq3_figs_latex.py --tables-dir outputs/tables --out-dir outputs/plots/trend/pub_latex --use-tex off
python scripts/build_results_manifest.py
```

## Script-to-Output Map

### 1) Reliability analysis
Script: `scripts/analyze_reliability.py`  
CLI: `idtools_usability run reliability`

Outputs:
- `outputs/tables/reliability_error_rates_summary.csv`
- `outputs/plots/reliability_error_rates_bar.png`

### 2) Phase 3 descriptive theme analysis
Script: `scripts/analyze_phase3_themes.py`  
CLI: `idtools_usability run phase3-themes`

Core outputs:
- `outputs/tables/usability_vs_nonusability_by_tool.csv`
- `outputs/tables/usability_vs_nonusability_by_tool.tex`
- `outputs/tables/l1_theme_grouped_table_counts_by_tool.csv`
- `outputs/tables/l1_theme_grouped_table_percent_by_tool.csv`
- `outputs/tables/l1_theme_grouped_table_percent_by_tool.tex`
- `outputs/tables/nielsen_theme_counts_by_tool.csv`
- `outputs/tables/nielsen_theme_percent_by_tool.csv`
- `outputs/tables/nielsen_theme_percent_by_tool.tex`
- `outputs/tables/rq2_top3_components_by_tool.csv`
- `outputs/tables/rq2_top3_components_by_tool.tex`

Per-theme-family count tables:
- `outputs/tables/counts_Associated_Component_Theme_by_tool_long.csv`
- `outputs/tables/counts_Associated_Component_Theme_by_tool_pivot.csv`
- `outputs/tables/counts_L1_Theme_by_tool_long.csv`
- `outputs/tables/counts_L1_Theme_by_tool_pivot.csv`
- `outputs/tables/counts_L1_Theme_Secondary_by_tool_long.csv`
- `outputs/tables/counts_L1_Theme_Secondary_by_tool_pivot.csv`
- `outputs/tables/counts_Nielsen_theme_by_tool_long.csv`
- `outputs/tables/counts_Nielsen_theme_by_tool_pivot.csv`

Top-20 LaTeX tables:
- `outputs/tables/top20_Associated_Component_Theme_by_tool.tex`
- `outputs/tables/top20_L1_Theme_by_tool.tex`
- `outputs/tables/top20_L1_Theme_Secondary_by_tool.tex`
- `outputs/tables/top20_Nielsen_theme_by_tool.tex`

Plots:
- `outputs/plots/stacked/stacked_*.png`
- `outputs/plots/stacked/stacked_pct_*.png`
- `outputs/plots/pies/*.png`
- `outputs/plots/bars/*.png` (if enabled in script)

### 3) Poisson trend analysis (RQ3)
Script: `scripts/trend_poisson_phase3.py`

Overall and aggregate trend tables:
- `outputs/tables/poisson_overall_usability_trends_by_tool.csv`
- `outputs/tables/poisson_overall_usability_trends_by_tool.tex`
- `outputs/tables/poisson_overall_usability_trends_by_tool_robust_se.csv`
- `outputs/tables/poisson_overall_usability_trends_by_tool_robust_se.tex`
- `outputs/tables/negative_binomial_sensitivity_overall_by_tool.csv`
- `outputs/tables/negative_binomial_sensitivity_overall_by_tool.tex`
- `outputs/tables/aggregate_poisson_expected_counts_l1_theme_curves.csv`
- `outputs/tables/aggregate_poisson_expected_counts_l1_theme_slopes.csv`
- `outputs/tables/aggregate_poisson_expected_counts_l1_theme_slopes.tex`
- `outputs/tables/aggregate_poisson_expected_counts_associated_component_curves.csv`
- `outputs/tables/aggregate_poisson_expected_counts_associated_component_slopes.csv`
- `outputs/tables/aggregate_poisson_expected_counts_associated_component_slopes.tex`

Per-tool/per-theme trend tables:
- `outputs/tables/poisson_trends_L1_Theme.csv`
- `outputs/tables/poisson_trends_L1_Theme_Secondary.csv`
- `outputs/tables/poisson_trends_Nielsen_theme.csv`
- `outputs/tables/poisson_trends_Associated_Component_Theme.csv`
- matching `.tex` versions for each above

Average-slope summaries:
- `outputs/tables/poisson_average_slopes_by_tool_summary.csv`
- `outputs/tables/poisson_average_slopes_by_tool_summary.tex`
- `outputs/tables/poisson_average_slopes_by_tool_and_l1_theme*.{csv,tex}`
- `outputs/tables/poisson_average_slopes_by_tool_and_nielsen_theme*.{csv,tex}`

Trend plots:
- `outputs/plots/trend/heatmap_beta_*.png`
- `outputs/plots/trend/heatmap_beta_sig_*.png`
- `outputs/plots/trend/aggregate_poisson_expected_counts_l1_theme.png`
- `outputs/plots/trend/aggregate_poisson_expected_counts_associated_component.png`

Poisson assumption/diagnostic tables:
- `outputs/tables/poisson_diagnostics_all_models.csv`
- `outputs/tables/poisson_diagnostics_summary_by_model_set.csv`
- `outputs/tables/poisson_diagnostics_summary_by_model_set.tex`
- `outputs/tables/poisson_diagnostics_overall_by_tool.csv`
- `outputs/tables/poisson_diagnostics_overall_by_tool.tex`

### 4) Chi-square association tests
Script: `scripts/compute_binary_chi_by_theme.py`

Outputs:
- `outputs/tables/rq_binary_chi_all_categories_summary.csv`
- `outputs/tables/rq_binary_chi_all_categories_summary.tex`
- `outputs/tables/rq_binary_chi_all_categories_shares_long.csv`
- `outputs/tables/rq_binary_chi_primary_theme_summary.csv`
- `outputs/tables/rq_binary_chi_component_theme_summary.csv`
- `outputs/tables/rq_binary_chi_primary_theme_shares_long.csv`
- `outputs/tables/rq_binary_chi_component_theme_shares_long.csv`

Related omnibus chi-square tables (generated previously and kept in outputs):
- `outputs/tables/rq_chi_square_tool_theme_association.csv`
- `outputs/tables/rq_chi_square_tool_theme_association.tex`
- `outputs/tables/rq_chi_square_tool_theme_association_all.csv`
- `outputs/tables/rq_chi_square_tool_theme_association_all.tex`

### 5) Raw monthly trends (non-Poisson)
Script: `scripts/plot_raw_monthly_usability_by_tool.py`

Outputs:
- `outputs/tables/raw_monthly_usability_counts_by_tool_wide.csv`
- `outputs/tables/raw_monthly_usability_counts_by_tool_long.csv`
- `outputs/plots/trend/raw_monthly_usability_counts_by_tool.png`

### 6) Theme codebook table
Script: `scripts/build_phase3_theme_codebook_table.py`

Outputs:
- `outputs/tables/phase3_theme_codebook_with_examples.csv`
- `outputs/tables/phase3_theme_codebook_with_examples.tex`

### 7) Publication-style RQ3 figures
Scripts:
- `scripts/make_pub_rq3_figs.py`
- `scripts/pubfigs/make_rq3_figs_latex.py`

Outputs:
- `outputs/plots/trend/pub/*.png`
- `outputs/plots/trend/pub_latex/*.{pdf,png}`
- helper matrices under `outputs/tables/fig_rq3_*`

### 8) Output manifest
Script: `scripts/build_results_manifest.py`

Output:
- `outputs/RESULTS_MANIFEST.csv` (file path, generating script, regeneration command, description)

## Notes

- RQ3 Poisson models use monthly bins from `2021-11` to `2025-11` inclusive (`n_months=49`; elapsed time 48 months).
- Some legacy artifacts are preserved under `outputs/phase3_theme_outputs/` and `outputs/phase3_theme_outputs.zip`.
