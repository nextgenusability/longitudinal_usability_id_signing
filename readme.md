# Identity-Based Signing Usability Artifact

This repository contains the data collection scripts, annotation prompts, analysis code, and generated outputs for a longitudinal study of developer-reported usability issues in identity-based software signing tools.

The project currently keeps two major result families:

- **Paper-version analysis**: the original analysis outputs used for the manuscript.
- **Prompt-v4 analysis**: a rerun using the OpenAI API and `Master_Classifier_Prompt_v4.txt`, with a core-normalized variant used for the most recent sensitivity checks.

For the detailed script-to-output map, see `ARTIFACT.md`.

## Quick Start

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
```

Regenerate the original artifact outputs:

```bash
bash scripts/regenerate_artifact.sh
```

Regenerate the current isolated prompt-v4/core-normalized analysis:

```bash
bash scripts/openai_api/run_isolated_phase3_analysis.sh \
  outputs/data/phase3_prompt_v4 \
  outputs/data/phase3_prompt_v4/recode_full_core_normalized \
  outputs/analysis_out_prompt_v4_core_normalized
```

## Repository Layout

`issue_collections/` stores the GitHub collection layer. The main script, `export_issues_discussions.py`, exports issues, issue comments, discussions, discussion comments, and README content into one workbook per repository. It reads `github_access_token` from the environment when available.

`LLM Prompts/` stores the coding protocol and LLM prompt files. The original phase-based workflow uses `Specific_Instructions.txt` and `Usability_Coding_Heuristics.txt`; newer API reruns use master prompts such as `Master_Classifier_Prompt_v4.txt` and `Master_Classifier_Prompt_v5.txt`.

`LLM Prompts/gh-issues/` stores the full 3,900-issue corpus workbooks used as API recoding inputs.

`LLM Prompts/human issues/` stores human-labeled issue files and the validation sample workbook `issue_sample_180_50.xlsx`.

`data/agreement/` stores reliability and agreement inputs from earlier manual checks.

`outputs/data/phase3_paper_version/` stores the labeled workbooks behind the original paper-version analysis.

`outputs/data/phase3_prompt_v4/` stores the OpenAI API full-corpus recoding artifacts. The key subfolders are:

- `recode_full/`: prompt-v4 full-corpus predictions and copied labeled workbooks.
- `recode_full_core_normalized/`: component-normalized prompt-v4 corpus used for the current core-normalized analysis.
- `run/`: temporary isolated analysis workspace created by `run_isolated_phase3_analysis.sh`.

`outputs/analysis_out_paper_version/` stores the original manuscript tables and figures.

`outputs/analysis_out_prompt_v4/` stores the prompt-v4 analysis outputs before core normalization.

`outputs/analysis_out_prompt_v4_core_normalized/` stores the current prompt-v4 core-normalized analysis outputs. This is the preferred output tree for the latest prompt-v4 sensitivity results.

`outputs/tables/` stores shared support tables that are not tied to one analysis tree, including validation-sample CI summaries and issue-context distributions.

## Current Validation Sample Files

The combined 180-issue validation sample is saved here:

- `outputs/tables/issue_sample_180_50_with_remaining80_analystA.csv`
- `outputs/tables/issue_sample_180_50_with_remaining80_analystA.xlsx`

This file uses `Usability_agreed` consensus labels for the first 100 issues and single-human labels for the additional 80 issues. The resulting combined usability rate is `131/180 = 72.78%`.
The key final column is `Validation usability label`; its provenance is recorded in `Validation usability label source`.

The corresponding finite-population corrected CI summary is:

- `outputs/tables/stratified_sampling_ci_from_pipeline_extract.csv`
- `outputs/tables/stratified_sampling_ci_from_updated_validation_sample.csv`

Current values:

- usability rate: `72.78%`
- 95% CI at observed rate with finite-population correction: `66.43%` to `79.13%`
- maximum-variance 95% margin of error: `+-7.13 percentage points`

## Full-Corpus Context Table

To describe the 3,900-issue corpus context, run:

```bash
python scripts/summarize_issue_context_by_repo.py
```

Outputs:

- `outputs/tables/issue_context_metrics_full_corpus.csv`
- `outputs/tables/issue_context_distributions_by_repo.csv`
- `outputs/tables/issue_context_distributions_by_repo.tex`

The table reports per-repository median `[IQR]` for comment count, unique commenters, thread participants, and approximate issue-thread word count.
The compact manuscript-ready table is `issue_context_distributions_by_repo.tex`; the issue-level source table is `issue_context_metrics_full_corpus.csv`.

## OpenAI API Recoding

Phase-sample recoding:

```bash
python scripts/openai_api/recode_phase_samples.py \
  --model gpt-5.1 \
  --phase12-range "2-161" \
  --phase35-range "102-120,158-161,162-169,170-181" \
  --master-prompt "LLM Prompts/Master_Classifier_Prompt_v4.txt" \
  --out-dir outputs/llm_api_recode_phase_newflow_v4
```

Full-corpus recoding:

```bash
python scripts/openai_api/recode_full_corpus.py \
  --model gpt-5.1 \
  --master-prompt "LLM Prompts/Master_Classifier_Prompt_v4.txt" \
  --out-dir outputs/data/phase3_prompt_v4/recode_full
```

Both scripts use the shared `OpenAIClient` in `scripts/openai_api/recode_phase_samples.py`; API calls set `temperature` to `0` and request JSON output.

## Important Generated Tables

Main chi-square association tables:

- `outputs/analysis_out_paper_version/tables/rq_chi_square_tool_theme_association_all.csv`
- `outputs/analysis_out_prompt_v4_core_normalized/tables/rq_binary_chi_all_categories_summary.csv`

Poisson trend diagnostics and robustness checks:

- `poisson_diagnostics_summary_by_model_set.csv`
- `poisson_diagnostics_all_models.csv`
- `poisson_overall_usability_trends_by_tool_robust_se.csv`
- `negative_binomial_sensitivity_overall_by_tool.csv`

Prompt-v4 vs paper-version comparison reports:

- `outputs/analysis_out_prompt_v4/tables/diff_report_paper_vs_prompt_v4.csv`
- `outputs/analysis_out_prompt_v4/tables/diff_report_paper_vs_prompt_v4_table_guide.txt`

## Notes

- The study window is November 5, 2021 through November 5, 2025. Monthly trend outputs use 49 calendar month bins (`2021-11` through `2025-11` inclusive), representing a 48-month elapsed window.
- The latest corpus-level rerun yielded `2,965/3,900 = 76%` usability-related issues. This falls inside the human-validation CI above.
- `.DS_Store`, virtual environments, and local caches should remain ignored by `.gitignore`.
