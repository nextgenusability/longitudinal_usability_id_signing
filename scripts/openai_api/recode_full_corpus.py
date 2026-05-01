from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

try:
    from scripts.openai_api.recode_phase_samples import (
        GH_ISSUES_DIR,
        HEURISTICS_PATH,
        MASTER_PROMPT_PATH,
        OPENAI_BASE_URL_DEFAULT,
        OPENAI_MODEL_DEFAULT,
        OpenAIClient,
        SPECIFIC_INSTR_PATH,
        _read_txt,
        annotate_rows,
        build_annotation_system_prompt,
        load_issue_corpus,
        write_updated_repo_copies,
    )
except ModuleNotFoundError:
    from recode_phase_samples import (  # type: ignore
        GH_ISSUES_DIR,
        HEURISTICS_PATH,
        MASTER_PROMPT_PATH,
        OPENAI_BASE_URL_DEFAULT,
        OPENAI_MODEL_DEFAULT,
        OpenAIClient,
        SPECIFIC_INSTR_PATH,
        _read_txt,
        annotate_rows,
        build_annotation_system_prompt,
        load_issue_corpus,
        write_updated_repo_copies,
    )


OUT_DIR = Path("outputs/llm_api_recode_full")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Full-corpus OpenAI recoding for all issues in `LLM Prompts/gh-issues` with "
            "resume/checkpoint support and workbook copy outputs."
        )
    )
    parser.add_argument("--model", default=OPENAI_MODEL_DEFAULT)
    parser.add_argument("--base-url", default=OPENAI_BASE_URL_DEFAULT)
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Output directory for full-corpus recoding artifacts.",
    )
    parser.add_argument(
        "--master-prompt",
        default=str(MASTER_PROMPT_PATH),
        help=(
            "Optional master prompt text file. "
            "If present, used instead of separate specific/heuristics prompts."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Ignore prior cache and recode everything.")
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap for testing.")
    parser.add_argument("--skip-api", action="store_true", help="Only prepare extracted files; no API calls.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_issue_corpus(GH_ISSUES_DIR).copy()
    corpus = corpus[corpus["issue_number"].notna()].copy()
    corpus["sample_id"] = range(1, len(corpus) + 1)
    if args.max_items is not None:
        corpus = corpus.head(args.max_items).copy()

    extract_cols = [
        "sample_id",
        "repo",
        "issue_number",
        "issue_url",
        "title",
        "labels",
        "body_text",
        "top_3_comments_text",
        "all_comments_text",
        "__source_file__",
    ]
    corpus[extract_cols].to_csv(out_dir / "full_corpus_extracted.csv", index=False)

    if args.skip_api:
        print(f"Extracted corpus written to: {out_dir}")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAIClient(api_key=api_key, model=args.model, base_url=args.base_url)
    master_txt = None
    mp = Path(args.master_prompt)
    if args.master_prompt and mp.exists():
        master_txt = _read_txt(mp)
    system_prompt = build_annotation_system_prompt(
        _read_txt(SPECIFIC_INSTR_PATH),
        _read_txt(HEURISTICS_PATH),
        master_txt=master_txt,
    )

    # Core resume file. annotate_rows skips repo/issue_number keys already present.
    cache_path = out_dir / "full_predictions_cache.csv"
    pred = annotate_rows(
        corpus,
        client=client,
        system_prompt=system_prompt,
        cache_path=cache_path,
        overwrite=args.overwrite,
    )

    merged = corpus.merge(
        pred,
        on=["sample_id", "repo", "issue_number", "issue_url"],
        how="left",
    )
    merged.to_csv(out_dir / "full_predictions_and_context.csv", index=False)

    with pd.ExcelWriter(out_dir / "full_predictions_and_context.xlsx", engine="xlsxwriter") as wr:
        merged.to_excel(wr, sheet_name="full_predictions", index=False)

    pred_out = pred[
        [
            "repo",
            "issue_number",
            "llm_associated_component",
            "llm_codes_primary",
            "llm_usability_type",
            "llm_associated_component_theme",
            "llm_l1_theme",
            "llm_l1_theme_secondary",
            "llm_nielsen_theme",
        ]
    ].drop_duplicates(["repo", "issue_number"], keep="first")

    write_updated_repo_copies(
        gh_folder=GH_ISSUES_DIR,
        predictions_df=pred_out,
        out_folder=out_dir / "copied_gh_issues_with_openai_labels",
    )

    run_meta = {
        "model": args.model,
        "base_url": args.base_url,
        "out_dir": str(out_dir),
        "master_prompt": str(mp) if args.master_prompt else "",
        "n_total_issues": int(len(corpus)),
        "n_pred_rows": int(len(pred)),
        "cache_path": str(cache_path),
        "overwrite": bool(args.overwrite),
        "max_items": args.max_items,
        "timestamp_epoch": int(time.time()),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print("Wrote full-corpus outputs to:", out_dir.resolve())
    print("Resume behavior: rerun the same command; existing rows in full_predictions_cache.csv are skipped.")


if __name__ == "__main__":
    main()
