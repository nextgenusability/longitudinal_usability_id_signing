# Research Artifact Guide

This document explains how to regenerate the main result sets and how the important output files relate to the scripts that created them.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
```

Some scripts write LaTeX tables through pandas and require `jinja2`; it is included in the project environment used for the current outputs.

## Input Data

`LLM Prompts/gh-issues/` contains the full 3,900-issue corpus as one workbook per repository. These workbooks include `issues` and `comments` sheets and are used by the OpenAI API recoding scripts.

`LLM Prompts/human issues/` contains human-labeled files and `issue_sample_180_50.xlsx`, the calibration/validation sample sheet.

`outputs/data/phase3_paper_version/` contains the labeled workbooks used by the original paper-version analysis.

`outputs/data/phase3_prompt_v4/recode_full/` contains prompt-v4 full-corpus OpenAI API predictions.

`outputs/data/phase3_prompt_v4/recode_full_core_normalized/` contains the prompt-v4 full-corpus data after component normalization, especially removal of generic `Core` when it co-occurs with a more specific component.

## Main Output Trees

`outputs/analysis_out_paper_version/` is the original manuscript output tree.

`outputs/analysis_out_prompt_v4/` is the prompt-v4 rerun before core normalization.

`outputs/analysis_out_prompt_v4_core_normalized/` is the current prompt-v4 analysis tree after core normalization. Use this tree for the latest prompt-v4 sensitivity outputs.

Each analysis tree mirrors the same basic structure:

- `tables/`: CSV and LaTeX tables.
- `plots/`: generated plots.
- `plots/trend/pub/`: publication-oriented PNG figures.
- `plots/trend/pub_latex/`: publication-oriented PDF/PNG figures.
- `phase3_theme_outputs/`: mirrored legacy-style copy of tables/plots for compatibility with earlier manuscript references.

## Regenerate Original Paper-Version Outputs

```bash
bash scripts/regenerate_artifact.sh
```

This runs the original package entrypoints and writes outputs under the default `outputs/tables` and `outputs/plots` paths. The curated paper-version tree is preserved under `outputs/analysis_out_paper_version/`.

Step-by-step equivalent:

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

## Regenerate Prompt-v4 Full-Corpus Recoding

Set the API key:

```bash
export OPENAI_API_KEY="sk-..."
```

Run full-corpus recoding:

```bash
python scripts/openai_api/recode_full_corpus.py \
  --model gpt-5.1 \
  --master-prompt "LLM Prompts/Master_Classifier_Prompt_v4.txt" \
  --out-dir outputs/data/phase3_prompt_v4/recode_full
```

The script is resumable. It writes `full_predictions_cache.csv` as it proceeds; rerunning the same command skips already cached rows unless `--overwrite` is passed.

The API wrapper uses:

- endpoint: `/chat/completions`
- JSON response format
- `temperature: 0`
- model string supplied by `--model` (latest recorded prompt-v4 run used `gpt-5.1`)

## Create Core-Normalized Prompt-v4 Corpus

```bash
python scripts/openai_api/make_core_normalized_corpus.py \
  --input-dir outputs/data/phase3_prompt_v4/recode_full \
  --output-dir outputs/data/phase3_prompt_v4/recode_full_core_normalized
```

This creates a duplicated result corpus for sensitivity analysis. When `Core` appears together with more specific component labels, `Core` is removed so that component-level tables are less dominated by generic labels.

## Run Isolated Prompt-v4 Analysis

```bash
bash scripts/openai_api/run_isolated_phase3_analysis.sh \
  outputs/data/phase3_prompt_v4 \
  outputs/data/phase3_prompt_v4/recode_full_core_normalized \
  outputs/analysis_out_prompt_v4_core_normalized
```

This script:

1. Builds an isolated workspace under `outputs/data/phase3_prompt_v4/run`.
2. Copies the selected recoded workbooks into `run/data/phase3`.
3. Runs descriptive theme analysis.
4. Runs Poisson trend models and diagnostics.
5. Runs binary chi-square follow-up tests.
6. Runs raw monthly trend plotting.
7. Rebuilds the codebook table.
8. Generates RQ3 helper tables and publication figures.
9. Copies results into the requested `outputs/analysis_out_*` directory.

## Script-to-Output Map

### Reliability

Script: `scripts/analyze_reliability.py`  
CLI: `idtools_usability run reliability`

Outputs:

- `reliability_error_rates_summary.csv`
- `reliability_error_rates_bar.png`

### Phase 3 Descriptive Theme Analysis

Script: `scripts/analyze_phase3_themes.py`  
CLI: `idtools_usability run phase3-themes`

Key outputs:

- `usability_vs_nonusability_by_tool.csv`
- `l1_theme_grouped_table_counts_by_tool.csv`
- `l1_theme_grouped_table_percent_by_tool.csv`
- `nielsen_theme_counts_by_tool.csv`
- `nielsen_theme_percent_by_tool.csv`
- `counts_Associated_Component_Theme_by_tool_long.csv`
- `counts_Associated_Component_Theme_by_tool_pivot.csv`
- `counts_L1_Theme_by_tool_long.csv`
- `counts_L1_Theme_by_tool_pivot.csv`
- `counts_L1_Theme_Secondary_by_tool_long.csv`
- `counts_L1_Theme_Secondary_by_tool_pivot.csv`
- `counts_Nielsen_theme_by_tool_long.csv`
- `counts_Nielsen_theme_by_tool_pivot.csv`
- `rq2_top3_components_by_tool.csv`
- `top20_*_by_tool.tex`

### Poisson Trends and Robustness

Script: `scripts/trend_poisson_phase3.py`

Key outputs:

- `poisson_overall_usability_trends_by_tool.csv`
- `poisson_overall_usability_trends_by_tool_robust_se.csv`
- `negative_binomial_sensitivity_overall_by_tool.csv`
- `aggregate_poisson_expected_counts_l1_theme_slopes.csv`
- `aggregate_poisson_expected_counts_associated_component_slopes.csv`
- `poisson_trends_L1_Theme.csv`
- `poisson_trends_L1_Theme_Secondary.csv`
- `poisson_trends_Nielsen_theme.csv`
- `poisson_trends_Associated_Component_Theme.csv`
- `poisson_diagnostics_all_models.csv`
- `poisson_diagnostics_summary_by_model_set.csv`
- `poisson_diagnostics_overall_by_tool.csv`

Interpretation note: overdispersion diagnostics are reported to qualify model assumptions. Robust standard errors and negative-binomial sensitivity checks are included for the overall monthly usability-count trends by tool.

### Chi-Square Association Tests

Script: `scripts/compute_binary_chi_by_theme.py`

Outputs:

- `rq_binary_chi_all_categories_summary.csv`
- `rq_binary_chi_all_categories_shares_long.csv`
- `rq_binary_chi_primary_theme_summary.csv`
- `rq_binary_chi_primary_theme_shares_long.csv`
- `rq_binary_chi_component_theme_summary.csv`
- `rq_binary_chi_component_theme_shares_long.csv`

These are category-specific one-vs-rest chi-square tests across tools.

Related omnibus tool-by-theme chi-square tables are preserved in the paper-version output tree:

- `outputs/analysis_out_paper_version/tables/rq_chi_square_tool_theme_association.csv`
- `outputs/analysis_out_paper_version/tables/rq_chi_square_tool_theme_association_all.csv`

Use the omnibus tables for claims about whether overall theme distributions differ by tool. Use the binary tables for follow-up category-specific claims.

### Raw Monthly Counts

Script: `scripts/plot_raw_monthly_usability_by_tool.py`

Outputs:

- `raw_monthly_usability_counts_by_tool_long.csv`
- `raw_monthly_usability_counts_by_tool_wide.csv`
- `raw_monthly_usability_counts_by_tool.png`

### Theme Codebook

Script: `scripts/build_phase3_theme_codebook_table.py`

Outputs:

- `phase3_theme_codebook_with_examples.csv`
- `phase3_theme_codebook_with_examples.tex`

### Publication RQ3 Figures

Scripts:

- `scripts/openai_api/build_fig_rq3_tables.py`
- `scripts/make_pub_rq3_figs.py`
- `scripts/pubfigs/make_rq3_figs_latex.py`

Outputs:

- `fig_rq3_*` helper matrices under `tables/`
- `plots/trend/pub/*.png`
- `plots/trend/pub_latex/*.{pdf,png}`

### Output Comparisons

Script: `scripts/compare_analysis_outputs.py`

Key outputs:

- `outputs/analysis_out_prompt_v4/tables/diff_report_paper_vs_prompt_v4.csv`
- `outputs/analysis_out_prompt_v4/tables/diff_report_paper_vs_prompt_v4_table_guide.txt`

These compare paper-version outputs to prompt-v4 outputs so changes can be inspected table by table.

### Validation Sample and Confidence Interval

Filled 180-sample validation file:

- `outputs/tables/issue_sample_180_50_with_remaining80_analystA.csv`
- `outputs/tables/issue_sample_180_50_with_remaining80_analystA.xlsx`

These files preserve the original sample fields and add validation-label provenance:

- `Usability_agreed`: consensus usability label for the first 100 human-human calibration issues.
- `Usability_Non-Usability Type- Analyst A`: filled for all 180 rows; the last 80 labels are appended from the human issue files.
- `Usability_Non-Usability Type- Analyst A source`: provenance for the Analyst A column (`original_sample_sheet` or `human_issue_file`).
- `Validation usability label`: final label used for the combined 180-sample validation estimate; it uses `Usability_agreed` where available and otherwise uses the appended single-human label.
- `Validation usability label source`: provenance for the final validation label (`consensus_Usability_agreed` or `human_issue_file`).

Current CI summary:

- `outputs/tables/stratified_sampling_ci_from_pipeline_extract.csv`
- `outputs/tables/stratified_sampling_ci_from_updated_validation_sample.csv`

Important columns:

- `N_full`: full corpus size (`3,900` issues).
- `n_labeled`: validation sample size (`180` issues).
- `n_usability`, `n_non_usability`: usability/non-usability counts in the combined validation labels.
- `usability_rate_pct`: observed validation usability rate.
- `moe95_observed_rate_fpc_pct_points`: 95% margin of error at the observed validation rate, with finite-population correction.
- `ci95_low_pct`, `ci95_high_pct`: finite-population-corrected 95% confidence interval bounds.
- `moe95_max_variance_p05_fpc_pct_points`: conservative 95% margin of error under the maximum-variance assumption (`p=0.5`).

Current values:

- combined usability rate: `131/180 = 72.78%`
- 95% finite-population-corrected CI at observed rate: `66.43%` to `79.13%`
- maximum-variance finite-population-corrected margin of error: `+-7.13 percentage points`

The current CI files were generated from `outputs/tables/issue_sample_180_50_with_remaining80_analystA.csv`. The original CI file was preserved as `outputs/tables/stratified_sampling_ci_from_pipeline_extract_before_updated_180_labels.csv`.

### Full-Corpus Issue Context

Script: `scripts/summarize_issue_context_by_repo.py`

Outputs:

- `outputs/tables/issue_context_metrics_full_corpus.csv`
- `outputs/tables/issue_context_distributions_by_repo.csv`
- `outputs/tables/issue_context_distributions_by_repo.tex`

Regenerate:

```bash
python scripts/summarize_issue_context_by_repo.py
```

`issue_context_metrics_full_corpus.csv` is the issue-level table. Important columns:

- `repo`, `issue_number`: issue identifier.
- `comment_count`: number of issue comments.
- `unique_commenters`: number of unique comment authors.
- `unique_thread_participants`: issue opener plus unique comment authors.
- `issue_text_words`: approximate word count for issue title plus issue body.
- `comment_text_words`: approximate word count across all comments.
- `total_text_words`: approximate issue-thread word count (`issue_text_words + comment_text_words`).

`issue_context_distributions_by_repo.csv` and `.tex` summarize the issue-level table by repository. Important columns:

- `n_issues`: number of issues in that repository.
- `comments_median_iqr`: median `[IQR]` issue comment count.
- `comments_mean`, `comments_p90`: mean and 90th percentile issue comment count.
- `unique_commenters_median_iqr`: median `[IQR]` unique comment authors.
- `unique_thread_participants_median_iqr`: median `[IQR]` issue opener plus comment authors.
- `text_words_median_iqr`: median `[IQR]` approximate total thread word count.
- `text_words_mean`, `text_words_p90`: mean and 90th percentile approximate total thread word count.

Current corpus context summary:

- full corpus: `3,900` issues across `8` repositories.
- overall median comments: `2 [1, 4]`.
- overall median unique commenters: `2 [1, 3]`.
- overall median thread participants: `2 [1, 3]`.
- overall median approximate text length: `323 [164, 609]` words.

## Study Window

The issue collection window runs from November 5, 2021 through November 5, 2025. Monthly trend models use calendar-month bins from `2021-11` through `2025-11` inclusive (`n_months=49`), representing 48 elapsed months.

## Current Rebuttal-Relevant Numbers

- Rerun pipeline usability issue rate: `2,965/3,900 = 76%`.
- Combined human validation usability rate: `131/180 = 72.8%`.
- Human validation 95% CI with finite-population correction: `66.4%` to `79.1%`.
- Since `76%` falls inside this interval, the rerun corpus-level rate is consistent with the human-labeled validation sample at the aggregate prevalence level.
