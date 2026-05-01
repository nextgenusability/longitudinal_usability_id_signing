# Identity-Based Signing Usability Artifact

This repository contains the data collection scripts, LLM coding prompts, human-labeled validation files, analysis code, and generated outputs for a longitudinal study of developer-reported usability issues in identity-based software signing tools.

The repository is organized around two main result families:

- **Paper-version analysis**: the original analysis outputs used for the manuscript.
- **Prompt-v4 analysis**: the OpenAI API rerun using `LLM Prompts/Master_Classifier_Prompt_v4.txt`, including the **core-normalized** variant used for the latest sensitivity checks.

For the detailed script-to-output regeneration map, see `ARTIFACT.md`. For a generated per-file inventory, see `ARTIFACT_FILE_INVENTORY.md`.

## Table of Contents

- [Quick Start](#quick-start)
- [Security and Redaction Note](#security-and-redaction-note)
- [Repository Layout at a Glance](#repository-layout-at-a-glance)
- [Complete Directory Inventory](#complete-directory-inventory)
- [Complete File Inventory](#complete-file-inventory)
- [Data Collection Inputs](#data-collection-inputs)
- [LLM Prompt and Human Labeling Inputs](#llm-prompt-and-human-labeling-inputs)
- [Validation Sample and Confidence Intervals](#validation-sample-and-confidence-intervals)
- [Full-Corpus Context Table](#full-corpus-context-table)
- [OpenAI API Recoding](#openai-api-recoding)
- [Analysis Output Families](#analysis-output-families)
- [Important Generated Tables](#important-generated-tables)
- [Regeneration Commands](#regeneration-commands)
- [Notes](#notes)

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

## Security and Redaction Note

Some generated CSV files contain raw public GitHub issue text. Public issue text can include token-looking examples, test credentials, logs, or accidentally posted secrets. GitHub push protection correctly blocks such patterns even when they appear inside public issue text.

We therefore **redact token-like strings in generated text artifacts** rather than relying on ZIP files to hide them. Zipping files is useful for convenience packaging, but it is not an appropriate privacy or security boundary for a public research artifact. The redacted full-corpus files retain the issue metadata, labels, and analysis-relevant text while replacing token-like strings with placeholders such as `[REDACTED_VAULT_TOKEN]`, `[REDACTED_AWS_ACCESS_KEY_ID]`, and `[REDACTED_GOOGLE_OAUTH_CLIENT_ID]`.

Existing ZIP archives in this repository, such as `issue_collections/gh_issue_exports.zip` and `outputs/phase3_theme_outputs.zip`, are convenience snapshots only. They should not be used as a substitute for redaction.

## Repository Layout at a Glance

| Path | What it contains | Most important files |
|---|---|---|
| `issue_collections/` | GitHub issue/discussion collection layer and archived exports. | `export_issues_discussions.py`, `gh_issue_exports.zip` |
| `LLM Prompts/` | Coding prompts, heuristic instructions, raw non-human issue workbooks, and human-labeled validation workbooks. | `Master_Classifier_Prompt_v4.txt`, `Specific_Instructions.txt`, `Usability_Coding_Heuristics.txt` |
| `LLM Prompts/gh-issues/` | Per-repository GitHub issue workbooks used as LLM/API inputs. | `sigstore__cosign___non-human.xlsx`, `hashicorp__vault___non-human.xlsx`, etc. |
| `LLM Prompts/human issues/` | Human-coded calibration/validation workbooks. | `issue_sample_180_50.xlsx`, per-repository human label workbooks |
| `data/agreement/` | Earlier manual agreement and reliability-check source tables. | Agreement/error-rate CSV files |
| `data/phase3/` | Original phase-3 labeled workbooks. | `*_phase3_labeled.xlsx` |
| `scripts/` | Analysis, plotting, reliability, trend, comparison, and artifact-maintenance scripts. | `analyze_phase3_themes.py`, `trend_poisson_phase3.py`, `compute_binary_chi_by_theme.py` |
| `scripts/openai_api/` | OpenAI API recoding and isolated rerun scripts. | `recode_phase_samples.py`, `recode_full_corpus.py`, `run_isolated_phase3_analysis.sh` |
| `outputs/data/phase3_paper_version/` | Isolated input data for the original paper-version run. | Per-repository labeled workbooks |
| `outputs/data/phase3_prompt_v4/` | Prompt-v4 full-corpus recoding data and normalized variants. | `recode_full/`, `recode_full_core_normalized/`, `run/` |
| `outputs/analysis_out_paper_version/` | Original manuscript analysis outputs. | `tables/`, `plots/`, `phase3_theme_outputs/` |
| `outputs/analysis_out_prompt_v4/` | Prompt-v4 analysis outputs before core normalization. | `tables/`, `plots/`, comparison reports |
| `outputs/analysis_out_prompt_v4_core_normalized/` | Latest preferred prompt-v4 analysis outputs after core normalization. | `tables/`, `plots/trend/pub_latex/` |
| `outputs/tables/` | Shared support tables for validation CIs and corpus context. | `stratified_sampling_ci_from_pipeline_extract.csv`, `issue_context_distributions_by_repo.tex` |
| `outputs copy/` | Legacy backup/snapshot of earlier generated outputs. | Historical comparison files only |

## Complete Directory Inventory

This table lists every tracked directory that contains artifact files, with direct and recursive file counts. A direct count means files immediately in that directory; recursive count includes files in nested subdirectories.

| Directory | Direct files | Recursive files | Contents |
|---|---:|---:|---|
| `.` | 6 | 1799 | Repository root with README, artifact guide, file inventory, Python project metadata, and requirements. |
| `LLM Prompts` | 11 | 30 | Prompt/protocol sources and sampled issue workbooks used for human and API coding. |
| `LLM Prompts/gh-issues` | 9 | 9 | Non-human GitHub issue workbooks used as LLM/API input. |
| `LLM Prompts/human issues` | 10 | 10 | Human-coded workbooks and validation sample files. |
| `data` | 0 | 11 | Original/manual data inputs used before isolated output layouts were added. |
| `data/agreement` | 3 | 3 | Manual agreement and reliability check inputs. |
| `data/phase3` | 8 | 8 | Original phase-3 labeled per-repository workbooks. |
| `idtools_usability` | 3 | 3 | Local installable helper package and CLI. |
| `issue_collections` | 2 | 2 | GitHub data-collection script and archived collected exports. |
| `outputs` | 2 | 1459 | Generated tables, figures, recoding outputs, comparison reports, and data snapshots. |
| `outputs copy` | 2 | 266 | Legacy output backup snapshot retained for comparison only. |
| `outputs copy/phase3_theme_outputs` | 0 | 78 | Theme-specific count/plot outputs generated by the descriptive phase-3 analysis. |
| `outputs copy/phase3_theme_outputs/plots` | 0 | 62 | Theme-specific count/plot outputs generated by the descriptive phase-3 analysis. |
| `outputs copy/phase3_theme_outputs/plots/bars` | 26 | 26 | Theme-specific count/plot outputs generated by the descriptive phase-3 analysis. |
| `outputs copy/phase3_theme_outputs/plots/pies` | 32 | 32 | Theme-specific count/plot outputs generated by the descriptive phase-3 analysis. |
| `outputs copy/phase3_theme_outputs/plots/stacked` | 4 | 4 | Theme-specific count/plot outputs generated by the descriptive phase-3 analysis. |
| `outputs copy/phase3_theme_outputs/tables` | 16 | 16 | Theme-specific count/plot outputs generated by the descriptive phase-3 analysis. |
| `outputs copy/plots` | 1 | 94 | Generated PNG/PDF figures for the corresponding analysis tree. |
| `outputs copy/plots/bars` | 26 | 26 | Generated PNG/PDF figures for the corresponding analysis tree. |
| `outputs copy/plots/pies` | 32 | 32 | Generated PNG/PDF figures for the corresponding analysis tree. |
| `outputs copy/plots/stacked` | 8 | 8 | Generated PNG/PDF figures for the corresponding analysis tree. |
| `outputs copy/plots/trend` | 11 | 27 | Generated PNG/PDF figures for the corresponding analysis tree. |
| `outputs copy/plots/trend/pub` | 3 | 3 | Generated PNG/PDF figures for the corresponding analysis tree. |
| `outputs copy/plots/trend/pub copy` | 3 | 3 | Generated PNG/PDF figures for the corresponding analysis tree. |
| `outputs copy/plots/trend/pub_latex` | 10 | 10 | Generated PNG/PDF figures for the corresponding analysis tree. |
| `outputs copy/tables` | 92 | 92 | Generated CSV/LaTeX tables for the corresponding analysis tree. |
| `outputs/analysis_out_paper_version` | 0 | 268 | Original manuscript analysis outputs: tables, plots, trend figures, and phase-3 summaries. |
| `outputs/analysis_out_paper_version/phase3_theme_outputs` | 0 | 78 | Original manuscript analysis outputs: tables, plots, trend figures, and phase-3 summaries. |
| `outputs/analysis_out_paper_version/phase3_theme_outputs/plots` | 0 | 62 | Original manuscript analysis outputs: tables, plots, trend figures, and phase-3 summaries. |
| `outputs/analysis_out_paper_version/phase3_theme_outputs/plots/bars` | 26 | 26 | Original manuscript analysis outputs: tables, plots, trend figures, and phase-3 summaries. |
| `outputs/analysis_out_paper_version/phase3_theme_outputs/plots/pies` | 32 | 32 | Original manuscript analysis outputs: tables, plots, trend figures, and phase-3 summaries. |
| `outputs/analysis_out_paper_version/phase3_theme_outputs/plots/stacked` | 4 | 4 | Original manuscript analysis outputs: tables, plots, trend figures, and phase-3 summaries. |
| `outputs/analysis_out_paper_version/phase3_theme_outputs/tables` | 16 | 16 | Original manuscript analysis outputs: tables, plots, trend figures, and phase-3 summaries. |
| `outputs/analysis_out_paper_version/plots` | 1 | 94 | Original manuscript analysis outputs: tables, plots, trend figures, and phase-3 summaries. |
| `outputs/analysis_out_paper_version/plots/bars` | 26 | 26 | Original manuscript analysis outputs: tables, plots, trend figures, and phase-3 summaries. |
| `outputs/analysis_out_paper_version/plots/pies` | 32 | 32 | Original manuscript analysis outputs: tables, plots, trend figures, and phase-3 summaries. |
| `outputs/analysis_out_paper_version/plots/stacked` | 8 | 8 | Original manuscript analysis outputs: tables, plots, trend figures, and phase-3 summaries. |
| `outputs/analysis_out_paper_version/plots/trend` | 11 | 27 | Original manuscript analysis outputs: tables, plots, trend figures, and phase-3 summaries. |
| `outputs/analysis_out_paper_version/plots/trend/pub` | 3 | 3 | Original manuscript analysis outputs: tables, plots, trend figures, and phase-3 summaries. |
| `outputs/analysis_out_paper_version/plots/trend/pub copy` | 3 | 3 | Original manuscript analysis outputs: tables, plots, trend figures, and phase-3 summaries. |
| `outputs/analysis_out_paper_version/plots/trend/pub_latex` | 10 | 10 | Original manuscript analysis outputs: tables, plots, trend figures, and phase-3 summaries. |
| `outputs/analysis_out_paper_version/tables` | 96 | 96 | Original manuscript analysis outputs: tables, plots, trend figures, and phase-3 summaries. |
| `outputs/analysis_out_prompt_v4` | 0 | 312 | Prompt-v4 analysis outputs before core-label normalization. |
| `outputs/analysis_out_prompt_v4/phase3_theme_outputs` | 0 | 156 | Prompt-v4 analysis outputs before core-label normalization. |
| `outputs/analysis_out_prompt_v4/phase3_theme_outputs/plots` | 0 | 66 | Prompt-v4 analysis outputs before core-label normalization. |
| `outputs/analysis_out_prompt_v4/phase3_theme_outputs/plots/pies` | 32 | 32 | Prompt-v4 analysis outputs before core-label normalization. |
| `outputs/analysis_out_prompt_v4/phase3_theme_outputs/plots/stacked` | 8 | 8 | Prompt-v4 analysis outputs before core-label normalization. |
| `outputs/analysis_out_prompt_v4/phase3_theme_outputs/plots/trend` | 11 | 26 | Prompt-v4 analysis outputs before core-label normalization. |
| `outputs/analysis_out_prompt_v4/phase3_theme_outputs/plots/trend/pub` | 3 | 3 | Prompt-v4 analysis outputs before core-label normalization. |
| `outputs/analysis_out_prompt_v4/phase3_theme_outputs/plots/trend/pub_latex` | 12 | 12 | Prompt-v4 analysis outputs before core-label normalization. |
| `outputs/analysis_out_prompt_v4/phase3_theme_outputs/tables` | 90 | 90 | Prompt-v4 analysis outputs before core-label normalization. |
| `outputs/analysis_out_prompt_v4/plots` | 0 | 66 | Prompt-v4 analysis outputs before core-label normalization. |
| `outputs/analysis_out_prompt_v4/plots/pies` | 32 | 32 | Prompt-v4 analysis outputs before core-label normalization. |
| `outputs/analysis_out_prompt_v4/plots/stacked` | 8 | 8 | Prompt-v4 analysis outputs before core-label normalization. |
| `outputs/analysis_out_prompt_v4/plots/trend` | 11 | 26 | Prompt-v4 analysis outputs before core-label normalization. |
| `outputs/analysis_out_prompt_v4/plots/trend/pub` | 3 | 3 | Prompt-v4 analysis outputs before core-label normalization. |
| `outputs/analysis_out_prompt_v4/plots/trend/pub_latex` | 12 | 12 | Prompt-v4 analysis outputs before core-label normalization. |
| `outputs/analysis_out_prompt_v4/tables` | 90 | 90 | Prompt-v4 analysis outputs before core-label normalization. |
| `outputs/analysis_out_prompt_v4_core_normalized` | 0 | 313 | Latest preferred prompt-v4 analysis outputs after core-label normalization. |
| `outputs/analysis_out_prompt_v4_core_normalized/phase3_theme_outputs` | 0 | 156 | Latest preferred prompt-v4 analysis outputs after core-label normalization. |
| `outputs/analysis_out_prompt_v4_core_normalized/phase3_theme_outputs/plots` | 0 | 66 | Latest preferred prompt-v4 analysis outputs after core-label normalization. |
| `outputs/analysis_out_prompt_v4_core_normalized/phase3_theme_outputs/plots/pies` | 32 | 32 | Latest preferred prompt-v4 analysis outputs after core-label normalization. |
| `outputs/analysis_out_prompt_v4_core_normalized/phase3_theme_outputs/plots/stacked` | 8 | 8 | Latest preferred prompt-v4 analysis outputs after core-label normalization. |
| `outputs/analysis_out_prompt_v4_core_normalized/phase3_theme_outputs/plots/trend` | 11 | 26 | Latest preferred prompt-v4 analysis outputs after core-label normalization. |
| `outputs/analysis_out_prompt_v4_core_normalized/phase3_theme_outputs/plots/trend/pub` | 3 | 3 | Latest preferred prompt-v4 analysis outputs after core-label normalization. |
| `outputs/analysis_out_prompt_v4_core_normalized/phase3_theme_outputs/plots/trend/pub_latex` | 12 | 12 | Latest preferred prompt-v4 analysis outputs after core-label normalization. |
| `outputs/analysis_out_prompt_v4_core_normalized/phase3_theme_outputs/tables` | 90 | 90 | Latest preferred prompt-v4 analysis outputs after core-label normalization. |
| `outputs/analysis_out_prompt_v4_core_normalized/plots` | 0 | 66 | Latest preferred prompt-v4 analysis outputs after core-label normalization. |
| `outputs/analysis_out_prompt_v4_core_normalized/plots/pies` | 32 | 32 | Latest preferred prompt-v4 analysis outputs after core-label normalization. |
| `outputs/analysis_out_prompt_v4_core_normalized/plots/stacked` | 8 | 8 | Latest preferred prompt-v4 analysis outputs after core-label normalization. |
| `outputs/analysis_out_prompt_v4_core_normalized/plots/trend` | 11 | 26 | Latest preferred prompt-v4 analysis outputs after core-label normalization. |
| `outputs/analysis_out_prompt_v4_core_normalized/plots/trend/pub` | 3 | 3 | Latest preferred prompt-v4 analysis outputs after core-label normalization. |
| `outputs/analysis_out_prompt_v4_core_normalized/plots/trend/pub_latex` | 12 | 12 | Latest preferred prompt-v4 analysis outputs after core-label normalization. |
| `outputs/analysis_out_prompt_v4_core_normalized/tables` | 91 | 91 | Latest preferred prompt-v4 analysis outputs after core-label normalization. |
| `outputs/data` | 0 | 358 | Isolated data trees for paper-version and prompt-v4 analyses. |
| `outputs/data/phase3_paper_version` | 8 | 8 | Data inputs used to regenerate the original paper-version analysis. |
| `outputs/data/phase3_prompt_v4` | 0 | 350 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/recode_full` | 5 | 13 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/recode_full/copied_gh_issues_with_openai_labels` | 8 | 8 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/recode_full_core_normalized` | 1 | 9 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/recode_full_core_normalized/copied_gh_issues_with_openai_labels` | 8 | 8 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/result_core_normalized` | 0 | 164 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/result_core_normalized/run` | 0 | 164 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/result_core_normalized/run/data` | 0 | 8 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/result_core_normalized/run/data/phase3` | 8 | 8 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/result_core_normalized/run/outputs` | 0 | 156 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/result_core_normalized/run/outputs/plots` | 0 | 66 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/result_core_normalized/run/outputs/plots/pies` | 32 | 32 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/result_core_normalized/run/outputs/plots/stacked` | 8 | 8 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/result_core_normalized/run/outputs/plots/trend` | 11 | 26 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/result_core_normalized/run/outputs/plots/trend/pub` | 3 | 3 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/result_core_normalized/run/outputs/plots/trend/pub_latex` | 12 | 12 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/result_core_normalized/run/outputs/tables` | 90 | 90 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/run` | 0 | 164 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/run/data` | 0 | 8 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/run/data/phase3` | 8 | 8 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/run/outputs` | 0 | 156 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/run/outputs/plots` | 0 | 66 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/run/outputs/plots/pies` | 32 | 32 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/run/outputs/plots/stacked` | 8 | 8 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/run/outputs/plots/trend` | 11 | 26 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/run/outputs/plots/trend/pub` | 3 | 3 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/run/outputs/plots/trend/pub_latex` | 12 | 12 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/data/phase3_prompt_v4/run/outputs/tables` | 90 | 90 | Prompt-v4 full-corpus recoding data, normalized variants, and temporary run workspaces. |
| `outputs/llm_api_prompt_compare` | 4 | 6 | Prompt comparison outputs summarizing which prompt/workflow performed better. |
| `outputs/llm_api_prompt_compare/master_new` | 1 | 1 | Prompt comparison outputs summarizing which prompt/workflow performed better. |
| `outputs/llm_api_prompt_compare/master_old` | 1 | 1 | Prompt comparison outputs summarizing which prompt/workflow performed better. |
| `outputs/llm_api_prompt_compare_direct` | 4 | 4 | Prompt comparison outputs summarizing which prompt/workflow performed better. |
| `outputs/llm_api_recode` | 3 | 3 | Tracked artifact subdirectory; see file inventory for exact files. |
| `outputs/llm_api_recode_full` | 1 | 1 | Prompt-v4 full-corpus extraction, predictions, copied workbooks, and run metadata. |
| `outputs/llm_api_recode_phase` | 14 | 22 | Phase-sample API recoding outputs for a prompt/workflow variant. |
| `outputs/llm_api_recode_phase/copied_gh_issues_with_openai_labels` | 8 | 8 | Phase-sample API recoding outputs for a prompt/workflow variant. |
| `outputs/llm_api_recode_phase_newflow` | 14 | 22 | Phase-sample API recoding outputs for a prompt/workflow variant. |
| `outputs/llm_api_recode_phase_newflow/copied_gh_issues_with_openai_labels` | 8 | 8 | Phase-sample API recoding outputs for a prompt/workflow variant. |
| `outputs/llm_api_recode_phase_newflow_v2` | 14 | 22 | Phase-sample API recoding outputs for a prompt/workflow variant. |
| `outputs/llm_api_recode_phase_newflow_v2/copied_gh_issues_with_openai_labels` | 8 | 8 | Phase-sample API recoding outputs for a prompt/workflow variant. |
| `outputs/llm_api_recode_phase_newflow_v3` | 14 | 22 | Phase-sample API recoding outputs for a prompt/workflow variant. |
| `outputs/llm_api_recode_phase_newflow_v3/copied_gh_issues_with_openai_labels` | 8 | 8 | Phase-sample API recoding outputs for a prompt/workflow variant. |
| `outputs/llm_api_recode_phase_newflow_v4` | 15 | 23 | Phase-sample API recoding outputs for a prompt/workflow variant. |
| `outputs/llm_api_recode_phase_newflow_v4/copied_gh_issues_with_openai_labels` | 8 | 8 | Phase-sample API recoding outputs for a prompt/workflow variant. |
| `outputs/llm_api_recode_phase_newflow_v4_assoccomp_semantic` | 17 | 25 | Phase-sample API recoding outputs for a prompt/workflow variant. |
| `outputs/llm_api_recode_phase_newflow_v4_assoccomp_semantic/copied_gh_issues_with_openai_labels` | 8 | 8 | Phase-sample API recoding outputs for a prompt/workflow variant. |
| `outputs/llm_api_recode_phase_newflow_v5` | 14 | 22 | Phase-sample API recoding outputs for a prompt/workflow variant. |
| `outputs/llm_api_recode_phase_newflow_v5/copied_gh_issues_with_openai_labels` | 8 | 8 | Phase-sample API recoding outputs for a prompt/workflow variant. |
| `outputs/llm_api_recode_phase_oldflow` | 14 | 22 | Phase-sample API recoding outputs for a prompt/workflow variant. |
| `outputs/llm_api_recode_phase_oldflow/copied_gh_issues_with_openai_labels` | 8 | 8 | Phase-sample API recoding outputs for a prompt/workflow variant. |
| `outputs/tables` | 12 | 12 | Generated CSV/LaTeX tables for the corresponding analysis tree. |
| `scripts` | 13 | 22 | Analysis, plotting, reliability, comparison, and artifact-generation scripts. |
| `scripts/openai_api` | 7 | 7 | OpenAI API recoding, prompt-comparison, and isolated-analysis scripts. |
| `scripts/pubfigs` | 2 | 2 | Publication figure-generation helpers. |

## Complete File Inventory

The complete tracked-file inventory is generated in:

- `ARTIFACT_FILE_INVENTORY.md`

That file groups each tracked artifact file by role (`result tables`, `result figures`, `raw/source data`, `LLM-classified data`, `prompts`, scripts, and documentation), describes the information each file contains, and flags files that have been superseded by newer prompt-v4/core-normalized outputs where that relationship is clear. It excludes Python bytecode caches and `.DS_Store` files because those are local/runtime artifacts rather than research artifacts.

## Data Collection Inputs

`issue_collections/export_issues_discussions.py` collects issues, issue comments, discussions, discussion comments, and repository README content from GitHub. It expects a GitHub access token in the environment variable `github_access_token` when authenticated API access is needed.

Example command for the study window, November 5, 2021 through November 5, 2025:

```bash
export github_access_token="<your GitHub token>"
python issue_collections/export_issues_discussions.py \
  --anchor-date 2025-11-05 \
  --months-back 48
```

The collected issue workbooks used by the LLM/API pipelines are stored under `LLM Prompts/gh-issues/`. The archive `issue_collections/gh_issue_exports.zip` is a convenience snapshot of collected exports.

## LLM Prompt and Human Labeling Inputs

The original phase-based prompt workflow is represented by:

- `LLM Prompts/Specific_Instructions.txt`
- `LLM Prompts/Usability_Coding_Heuristics.txt`

The newer master-prompt API reruns are represented by:

- `LLM Prompts/Master_Classifier_Prompt_v2.txt`
- `LLM Prompts/Master_Classifier_Prompt_v3.txt`
- `LLM Prompts/Master_Classifier_Prompt_v4.txt`
- `LLM Prompts/Master_Classifier_Prompt_v5.txt`

The prompt used for the paper-version API rerun is `LLM Prompts/Master_Classifier_Prompt_v4.txt`. Among the tested master-prompt variants, v4 produced the best validation performance, so the prompt-v4 outputs are the primary OpenAI API rerun results reported for the revised analysis. `Master_Classifier_Prompt_v5.txt` is retained for transparency as a later experimental variant, not as the paper's selected prompt.

Human-labeled calibration and validation workbooks live under `LLM Prompts/human issues/`. The main sample workbook is `LLM Prompts/human issues/issue_sample_180_50.xlsx`.

## Validation Sample and Confidence Intervals

The combined 180-issue validation sample is saved here:

- `outputs/tables/issue_sample_180_50_with_remaining80_analystA.csv`
- `outputs/tables/issue_sample_180_50_with_remaining80_analystA.xlsx`

This file uses `Usability_agreed` consensus labels for the first 100 issues and single-human labels for the additional 80 issues. The final analysis column is `Validation usability label`; its provenance is recorded in `Validation usability label source`.

Current combined validation estimate:

- usability-related issues: `131/180`
- usability rate: `72.78%`
- 95% finite-population corrected CI at the observed rate: `66.43%` to `79.13%`
- maximum-variance 95% margin of error: `+-7.13 percentage points`

The corresponding CI support tables are:

- `outputs/tables/stratified_sampling_ci_from_pipeline_extract.csv`
- `outputs/tables/stratified_sampling_ci_from_updated_validation_sample.csv`
- `outputs/tables/stratified_sampling_ci_summary.csv`
- `outputs/tables/stratified_sampling_strata_table.csv`
- `outputs/tables/stratified_sampling_strata_match_check.csv`

## Full-Corpus Context Table

To describe the 3,900-issue corpus context, run:

```bash
python scripts/summarize_issue_context_by_repo.py
```

Outputs:

- `outputs/tables/issue_context_metrics_full_corpus.csv`
- `outputs/tables/issue_context_distributions_by_repo.csv`
- `outputs/tables/issue_context_distributions_by_repo.tex`

These tables report per-repository median `[IQR]` for comment count, unique commenters, thread participants, and approximate issue-thread word count.

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

Both scripts use shared OpenAI API code in `scripts/openai_api/recode_phase_samples.py`. API calls set `temperature` to `0` and request structured JSON output. Run metadata is saved in `run_meta.json` files inside each output directory.

## Analysis Output Families

| Output family | Path | Data source | Purpose |
|---|---|---|---|
| Paper-version analysis | `outputs/analysis_out_paper_version/` | `outputs/data/phase3_paper_version/` | Original manuscript tables/figures generated from the earlier ChatGPT-UI-coded phase-3 labels. |
| Prompt-v4 analysis | `outputs/analysis_out_prompt_v4/` | `outputs/data/phase3_prompt_v4/recode_full/` | Full-corpus OpenAI API prompt-v4 rerun before core-label normalization. |
| Prompt-v4 core-normalized analysis | `outputs/analysis_out_prompt_v4_core_normalized/` | `outputs/data/phase3_prompt_v4/recode_full_core_normalized/` | Latest preferred prompt-v4 output tree; removes `Core` when a more specific component label is present. |
| Prompt comparison outputs | `outputs/llm_api_prompt_compare*/` | Phase-sample recoding outputs | Agreement comparisons between old phase-based prompting and newer master-prompt workflows. |
| Shared support tables | `outputs/tables/` | Validation sample and full corpus | CI summaries, context tables, strata checks, and manuscript support tables. |

## Important Generated Tables

Main chi-square association tables:

- `outputs/analysis_out_paper_version/tables/rq_chi_square_tool_theme_association_all.csv`
- `outputs/analysis_out_prompt_v4_core_normalized/tables/rq_binary_chi_all_categories_summary.csv`

Poisson trend diagnostics and robustness checks:

- `outputs/analysis_out_prompt_v4_core_normalized/tables/poisson_diagnostics_summary_by_model_set.csv`
- `outputs/analysis_out_prompt_v4_core_normalized/tables/poisson_diagnostics_all_models.csv`
- `outputs/analysis_out_prompt_v4_core_normalized/tables/poisson_overall_usability_trends_by_tool_robust_se.csv`
- `outputs/analysis_out_prompt_v4_core_normalized/tables/negative_binomial_sensitivity_overall_by_tool.csv`

Prompt-v4 versus paper-version comparison reports:

- `outputs/analysis_out_prompt_v4/tables/diff_report_paper_vs_prompt_v4.csv`
- `outputs/analysis_out_prompt_v4/tables/diff_report_paper_vs_prompt_v4_table_guide.txt`

Corpus context and validation support tables:

- `outputs/tables/issue_context_distributions_by_repo.tex`
- `outputs/tables/issue_context_metrics_full_corpus.csv`
- `outputs/tables/stratified_sampling_ci_from_pipeline_extract.csv`

## Regeneration Commands

Create the prompt-v4 core-normalized corpus from the prompt-v4 full-corpus recode:

```bash
python scripts/openai_api/make_core_normalized_corpus.py \
  --input-dir outputs/data/phase3_prompt_v4/recode_full \
  --output-dir outputs/data/phase3_prompt_v4/recode_full_core_normalized
```

Run the isolated prompt-v4 core-normalized analysis:

```bash
bash scripts/openai_api/run_isolated_phase3_analysis.sh \
  outputs/data/phase3_prompt_v4 \
  outputs/data/phase3_prompt_v4/recode_full_core_normalized \
  outputs/analysis_out_prompt_v4_core_normalized
```

Compare paper-version and prompt-v4 outputs:

```bash
python scripts/compare_analysis_outputs.py \
  --old outputs/analysis_out_paper_version \
  --new outputs/analysis_out_prompt_v4_core_normalized \
  --out outputs/analysis_out_prompt_v4_core_normalized/tables
```

## Notes

- The study window is November 5, 2021 through November 5, 2025. Monthly trend outputs use 49 calendar month bins (`2021-11` through `2025-11` inclusive), representing a 48-month elapsed window.
- The latest corpus-level rerun yielded `2,965/3,900 = 76%` usability-related issues. This falls inside the human-validation CI above.
- `.DS_Store`, virtual environments, local scratch outputs, and Python bytecode caches should remain ignored by `.gitignore`.
- Do not publish unredacted raw issue-context files if they contain token-like strings copied from public GitHub issues. Redaction is preferred over hiding those files inside ZIP archives.
