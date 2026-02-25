# IDTools Usability

## Research Artifact

For full reproducibility documentation (inputs, pipeline, output manifest, and script-to-result mapping), see:

- `ARTIFACT.md`

## Layout

```
data/
  phase3/
  agreement/
scripts/
  analyze_phase3_themes.py
  analyze_reliability.py
idtools_usability/
  cli.py
outputs/
  tables/
  plots/
requirements.txt
pyproject.toml
readme.md
ARTIFACT.md
```

## CLI Usage

Install as an editable package:

```bash
python -m pip install -e . --no-build-isolation
```

Run reliability analysis:

```bash
idtools_usability run reliability
```

Run phase 3 theme analysis:

```bash
idtools_usability run phase3-themes
```

Direct module invocation also works:

```bash
python -m idtools_usability run reliability
```

Reproduce the full artifact:

```bash
bash scripts/regenerate_artifact.sh
```

Per-file output provenance:

- `outputs/RESULTS_MANIFEST.csv`
