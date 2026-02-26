# IDTools Usability Artifact

This repository contains the analysis pipeline and outputs for a longitudinal study of usability issues in identity-based software signing ecosystems.

For full reproducibility details (inputs, pipeline graph, output manifest, and script-to-result mapping), see `ARTIFACT.md`.

## 1. Quick Start

Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
```

Run the packaged entrypoints:

```bash
idtools_usability run reliability
idtools_usability run phase3-themes
```

Rebuild the artifact outputs:

```bash
bash scripts/regenerate_artifact.sh
```

Output provenance file:

- `outputs/RESULTS_MANIFEST.csv`

## 2. Repository Structure (Stratified)

### 2.1 Top-Level Folders and What They Store

- `data/`: primary analysis inputs used by the pipeline.
  - `data/phase3/`: core coded issue-level dataset(s) for Phase 3+ analyses.
  - `data/agreement/`: coder-agreement/reliability input files.
- `issue_collections/`: raw-data collection assets used to freshly pull GitHub issues/discussions for this study window.
  - Contains the collection script and an archived export bundle.
- `LLM Prompts/`: the prompt/protocol documents used to guide LLM-assisted qualitative coding.
- `scripts/`: executable analysis and figure/table-generation scripts.
- `idtools_usability/`: package/CLI entrypoints that orchestrate major analysis stages.
- `outputs/`: generated artifact outputs.
  - `outputs/tables/`: manuscript and intermediate tables.
  - `outputs/plots/`: manuscript and intermediate figures.
- `outputs copy/`: backup/legacy copy of outputs retained locally.

### 2.2 Meta/Config Files

- `ARTIFACT.md`: detailed reproducibility and regeneration map.
- `readme.md`: top-level orientation (this document).
- `pyproject.toml`, `requirements.txt`: environment and dependency definitions.
- `.gitignore`: ignored local/system/environment files.

## 3. New Folder Documentation

### 3.1 `issue_collections/`

This folder is the study's collection layer: it is used to freshly pull the GitHub issues/discussions that feed the downstream coding and analysis pipeline.
The exporter is designed for reproducible windowed pulls and includes bot-aware handling in comment summaries (non-bot comments are prioritized for `top_3_comments_text` and `all_comments_text`).

#### `issue_collections/export_issues_discussions.py`

Purpose:
- Exports GitHub issues, issue comments, discussions, discussion comments/replies, and README content to one Excel file per repository.
- Adds coding scaffold columns directly in the `issues` sheet for downstream annotation workflows.

Key behavior:
- Pulls issues (`state=all`) and discussions from GitHub REST API.
- Applies time window filtering by `created_at` client-side.
- Supports anchored windows via `--anchor-date` and `--months-back`.
- Collects both text-inferred PR links and issue-timeline PR links.
- Extracts code/log blocks from issue/discussion text.
- Adds README raw text plus an extractive summary.
- Writes one workbook per repo with sheets: `issues`, `comments`, `discussions`, `discussion_comments`, `readme`.

Authentication:
- Reads token from environment variable `github_access_token` (a GitHub personal access token / PAT).
- If unset, script still runs but will be heavily rate-limited.

Example usage:

```bash
python issue_collections/export_issues_discussions.py \
  --repo sigstore/cosign \
  --repo hashicorp/vault \
  --anchor-date 2025-11-05 \
  --months-back 48 \
  --outdir issue_collections/gh_issue_exports
```

Entire selected repo set (script defaults) for the project study window ending November 5, 2025 and looking back 48 months to November 5, 2021:

```bash
export github_access_token="ghp_xxx"
python issue_collections/export_issues_discussions.py \
  --anchor-date 2025-11-05 \
  --months-back 48 \
  --outdir issue_collections/gh_issue_exports
```

```bash
python issue_collections/export_issues_discussions.py \
  --repos-csv data/phase3/tool_repos.csv \
  --since-months 24 \
  --outdir issue_collections/gh_issue_exports
```

#### `issue_collections/gh_issue_exports.zip`

- Archived bundle of exported issue/discussion workbooks.
- Treated as data snapshot, not executable code.

### 3.2 `LLM Prompts/`

These files document the annotation protocol used during coding rounds.

#### `LLM Prompts/Specific Instructions.docx`

Purpose:
- Project-level instruction set for LLM co-annotation.
- Provides study context, RQs/H hypotheses framing, phase workflow, and expected outputs.

What it covers:
- Domain and problem framing for identity-based signing usability.
- Coding workflow phases and handoff expectations.
- Column-level expectations for annotation outputs.

#### `LLM Prompts/Usability Coding Heuristics.docx`

Purpose:
- Operational coding rubric used in annotation/calibration.

What it covers:
- Phase 1: associated component + `code_primary` construction rules.
- Phase 2: usability vs non-usability decision rules (with contrast examples).
- Phase 3: Nielsen mapping guidelines (10 heuristics).
- Phase 4: primary-theme mapping rules.
- Phase 4B: secondary-theme mapping rules.
- Phase 5: affected-component-theme mapping rules.
- Multi-label handling and tool-specific notes.

#### `LLM Prompts/Usability Coding Heuristics_UPDATED.docx`

Purpose:
- Updated version of the heuristics document reflecting calibration revisions.

What changed (high level):
- Maintains the same phase structure as the base heuristics.
- Includes explicit calibration updates at the top (component coding, theme mapping conventions, and agreement-related clarifications).
- Use this as the canonical prompt/rubric version when reproducing LLM-assisted coding steps.

## 4. Reproducibility Notes

- Main analysis outputs are regenerated from `scripts/regenerate_artifact.sh`.
- If you need to regenerate raw GitHub exports, run `issue_collections/export_issues_discussions.py` separately (requires network/API access).
- For script-to-output mappings, use:
  - `ARTIFACT.md`
  - `outputs/RESULTS_MANIFEST.csv`
