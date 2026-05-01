#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/openai_api/run_isolated_phase3_analysis.sh [EXPERIMENT_ROOT] [RECODE_DIR]
#
# Defaults:
#   EXPERIMENT_ROOT = outputs/data/phase3_prompt_v4
#   RECODE_DIR      = ${EXPERIMENT_ROOT}/recode_full
#   ANALYSIS_OUT    = outputs/analysis_out_prompt_v4
#
# This script automates:
# 1) Preparing an isolated run workspace under ${EXPERIMENT_ROOT}/run
# 2) Copying full-corpus labeled xlsx files into run/data/phase3
# 3) Running the analysis scripts so tables/plots are written to run/outputs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EXPERIMENT_ROOT="${1:-outputs/data/phase3_prompt_v4}"
RECODE_DIR="${2:-${EXPERIMENT_ROOT}/recode_full}"
ANALYSIS_OUT="${3:-outputs/analysis_out_prompt_v4}"

ABS_EXPERIMENT_ROOT="${REPO_ROOT}/${EXPERIMENT_ROOT}"
ABS_RECODE_DIR="${REPO_ROOT}/${RECODE_DIR}"
ABS_ANALYSIS_OUT="${REPO_ROOT}/${ANALYSIS_OUT}"
RUN_DIR="${ABS_EXPERIMENT_ROOT}/run"
RUN_DATA_PHASE3_DIR="${RUN_DIR}/data/phase3"
ABS_ANALYSIS_PHASE3_THEME_OUT="${ABS_ANALYSIS_OUT}/phase3_theme_outputs"

SOURCE_XLSX_GLOB="${ABS_RECODE_DIR}/copied_gh_issues_with_openai_labels/*.xlsx"

if ! compgen -G "${SOURCE_XLSX_GLOB}" > /dev/null; then
  echo "ERROR: No labeled xlsx files found at:"
  echo "  ${ABS_RECODE_DIR}/copied_gh_issues_with_openai_labels/"
  echo "Run full-corpus recoding first."
  exit 1
fi

mkdir -p "${RUN_DATA_PHASE3_DIR}"
mkdir -p "${ABS_ANALYSIS_OUT}"
mkdir -p "${ABS_ANALYSIS_PHASE3_THEME_OUT}/tables" "${ABS_ANALYSIS_PHASE3_THEME_OUT}/plots"
rm -f "${RUN_DATA_PHASE3_DIR}"/*.xlsx
cp "${ABS_RECODE_DIR}"/copied_gh_issues_with_openai_labels/*.xlsx "${RUN_DATA_PHASE3_DIR}/"

echo "[1/8] analyze_phase3_themes.py"
( cd "${RUN_DIR}" && python "${REPO_ROOT}/scripts/analyze_phase3_themes.py" )

echo "[2/8] trend_poisson_phase3.py"
( cd "${RUN_DIR}" && python "${REPO_ROOT}/scripts/trend_poisson_phase3.py" )

echo "[3/8] compute_binary_chi_by_theme.py"
( cd "${RUN_DIR}" && python "${REPO_ROOT}/scripts/compute_binary_chi_by_theme.py" )

echo "[4/8] plot_raw_monthly_usability_by_tool.py"
( cd "${RUN_DIR}" && python "${REPO_ROOT}/scripts/plot_raw_monthly_usability_by_tool.py" )

echo "[5/8] build_phase3_theme_codebook_table.py"
( cd "${RUN_DIR}" && python "${REPO_ROOT}/scripts/build_phase3_theme_codebook_table.py" )

echo "[6/8] build_fig_rq3_tables.py"
(
  cd "${RUN_DIR}" && \
  python "${REPO_ROOT}/scripts/openai_api/build_fig_rq3_tables.py" \
    --tables-dir outputs/tables
)

echo "[7/8] make_pub_rq3_figs.py"
( cd "${RUN_DIR}" && python "${REPO_ROOT}/scripts/make_pub_rq3_figs.py" )

echo "[8/8] make_rq3_figs_latex.py"
(
  cd "${RUN_DIR}" && \
  python "${REPO_ROOT}/scripts/pubfigs/make_rq3_figs_latex.py" \
    --tables-dir outputs/tables \
    --out-dir outputs/plots/trend/pub_latex \
    --style "${REPO_ROOT}/scripts/pubfigs/paper.mplstyle" \
    --use-tex off
)

rm -rf "${ABS_ANALYSIS_OUT:?}"/*
cp -R "${RUN_DIR}/outputs/." "${ABS_ANALYSIS_OUT}/"

# Keep a mirrored legacy-style subtree for phase3 theme outputs.
mkdir -p "${ABS_ANALYSIS_PHASE3_THEME_OUT}/tables" "${ABS_ANALYSIS_PHASE3_THEME_OUT}/plots"
if [ -d "${ABS_ANALYSIS_OUT}/tables" ]; then
  rm -rf "${ABS_ANALYSIS_PHASE3_THEME_OUT}/tables"/*
  cp -R "${ABS_ANALYSIS_OUT}/tables/." "${ABS_ANALYSIS_PHASE3_THEME_OUT}/tables/"
fi
if [ -d "${ABS_ANALYSIS_OUT}/plots" ]; then
  rm -rf "${ABS_ANALYSIS_PHASE3_THEME_OUT}/plots"/*
  cp -R "${ABS_ANALYSIS_OUT}/plots/." "${ABS_ANALYSIS_PHASE3_THEME_OUT}/plots/"
fi

echo "Done."
echo "Isolated run workspace outputs:"
echo "  ${RUN_DIR}/outputs"
echo "Published analysis outputs:"
echo "  ${ABS_ANALYSIS_OUT}"
