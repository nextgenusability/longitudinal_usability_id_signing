#!/usr/bin/env bash
set -euo pipefail

echo "[1/8] Reliability analysis"
idtools_usability run reliability

echo "[2/8] Phase 3 descriptive themes"
idtools_usability run phase3-themes

echo "[3/8] Poisson trend analysis"
python scripts/trend_poisson_phase3.py

echo "[4/8] Chi-square association analyses"
python scripts/compute_binary_chi_by_theme.py

echo "[5/8] Raw monthly trend plot (non-Poisson)"
python scripts/plot_raw_monthly_usability_by_tool.py

echo "[6/8] Phase 3 theme codebook table"
python scripts/build_phase3_theme_codebook_table.py

echo "[7/8] Publication-style figures (basic)"
python scripts/make_pub_rq3_figs.py

echo "[8/9] Publication-style figures (LaTeX style)"
python scripts/pubfigs/make_rq3_figs_latex.py --tables-dir outputs/tables --out-dir outputs/plots/trend/pub_latex --use-tex off

echo "[9/9] Results manifest"
python scripts/build_results_manifest.py

echo "Done. See ARTIFACT.md and outputs/RESULTS_MANIFEST.csv."
