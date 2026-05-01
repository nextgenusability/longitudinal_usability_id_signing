from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


NEW_PROMPT_DEFAULT = Path("LLM Prompts/master_prompt.txt")
# Sentinel path that should not exist; this forces recode_phase_samples.py to
# fall back to Specific_Instructions + Usability_Coding_Heuristics.
LEGACY_BASELINE_SENTINEL = Path("LLM Prompts/__use_specific_plus_heuristics__.txt")
ROOT_OUT_DEFAULT = Path("outputs/llm_api_prompt_compare")


def _run_variant(
    recode_script: Path,
    model: str,
    base_url: str,
    phase12_range: str,
    phase35_range: str,
    master_prompt: Path | None,
    out_dir: Path,
    overwrite: bool,
    skip_api: bool,
) -> None:
    cmd = [
        sys.executable,
        str(recode_script),
        "--model",
        model,
        "--base-url",
        base_url,
        "--phase12-range",
        phase12_range,
        "--phase35-range",
        phase35_range,
        "--out-dir",
        str(out_dir),
    ]
    if master_prompt is not None:
        cmd.extend(["--master-prompt", str(master_prompt)])
    if overwrite:
        cmd.append("--overwrite")
    if skip_api:
        cmd.append("--skip-api")
    subprocess.run(cmd, check=True)


def _load_agreement(path: Path, variant: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "phase12_subset" not in df.columns:
        df["phase12_subset"] = ""
    df["variant"] = variant
    return df


def _write_tex(df: pd.DataFrame, out_path: Path, caption: str, label: str) -> None:
    latex = df.to_latex(index=False, escape=False)
    latex = latex.replace("\\begin{tabular}", "\\begin{table}[t]\n\\centering\n\\scriptsize\n\\begin{tabular}", 1)
    latex = latex.replace("\\end{tabular}", f"\\end{{tabular}}\n\\caption{{{caption}}}\n\\label{{{label}}}", 1)
    latex += "\n\\end{table}\n"
    out_path.write_text(latex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run phase-sample recoding with two prompt variants "
            "(master_prompt vs specific+heuristics baseline) "
            "and generate side-by-side agreement comparison outputs."
        )
    )
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--phase12-range", default="2-161")
    parser.add_argument("--phase35-range", default="102-113,170-181")
    parser.add_argument("--new-prompt", default=str(NEW_PROMPT_DEFAULT))
    parser.add_argument(
        "--old-prompt",
        default=str(LEGACY_BASELINE_SENTINEL),
        help=(
            "Baseline prompt path. By default uses a non-existent sentinel path so the "
            "pipeline falls back to Specific_Instructions + Usability_Coding_Heuristics."
        ),
    )
    parser.add_argument("--root-out", default=str(ROOT_OUT_DEFAULT))
    parser.add_argument("--new-tag", default="master_prompt")
    parser.add_argument("--old-tag", default="specific_plus_heuristics")
    parser.add_argument(
        "--new-agreement-csv",
        default="",
        help="Optional direct path to new-variant agreement_summary.csv.",
    )
    parser.add_argument(
        "--old-agreement-csv",
        default="",
        help="Optional direct path to old-variant agreement_summary.csv.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="Skip recoding runs and only compare existing agreement_summary.csv files.",
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Pass --skip-api to recode script (extract-only).",
    )
    args = parser.parse_args()

    recode_script = Path("scripts/openai_api/recode_phase_samples.py")
    root_out = Path(args.root_out)
    root_out.mkdir(parents=True, exist_ok=True)

    new_prompt = Path(args.new_prompt)
    old_prompt_arg = Path(args.old_prompt)
    old_prompt = old_prompt_arg
    new_out = root_out / args.new_tag
    old_out = root_out / args.old_tag
    new_out.mkdir(parents=True, exist_ok=True)
    old_out.mkdir(parents=True, exist_ok=True)

    use_direct_csv = bool(args.new_agreement_csv) or bool(args.old_agreement_csv)
    if use_direct_csv and (not args.new_agreement_csv or not args.old_agreement_csv):
        raise ValueError("Provide both --new-agreement-csv and --old-agreement-csv, or neither.")

    if not args.skip_runs and not use_direct_csv:
        _run_variant(
            recode_script=recode_script,
            model=args.model,
            base_url=args.base_url,
            phase12_range=args.phase12_range,
            phase35_range=args.phase35_range,
            master_prompt=new_prompt,
            out_dir=new_out,
            overwrite=args.overwrite,
            skip_api=args.skip_api,
        )
        _run_variant(
            recode_script=recode_script,
            model=args.model,
            base_url=args.base_url,
            phase12_range=args.phase12_range,
            phase35_range=args.phase35_range,
            master_prompt=old_prompt,
            out_dir=old_out,
            overwrite=args.overwrite,
            skip_api=args.skip_api,
        )

    if use_direct_csv:
        new_ag_path = Path(args.new_agreement_csv)
        old_ag_path = Path(args.old_agreement_csv)
    else:
        new_ag_path = new_out / "agreement_summary.csv"
        old_ag_path = old_out / "agreement_summary.csv"
    if not new_ag_path.exists() or not old_ag_path.exists():
        raise FileNotFoundError(
            f"Missing agreement summaries. Expected:\n - {new_ag_path}\n - {old_ag_path}"
        )

    new_df = _load_agreement(new_ag_path, args.new_tag)
    old_df = _load_agreement(old_ag_path, args.old_tag)

    long_df = pd.concat([new_df, old_df], ignore_index=True, sort=False)
    long_df.to_csv(root_out / "prompt_agreement_long.csv", index=False)

    keys = ["phase", "phase12_subset", "metric", "n"]
    new_sel = new_df[keys + ["agreement_pct", "error_rate_pct", "cohen_kappa", "gwet_ac1"]].rename(
        columns={
            "agreement_pct": f"agreement_pct_{args.new_tag}",
            "error_rate_pct": f"error_rate_pct_{args.new_tag}",
            "cohen_kappa": f"cohen_kappa_{args.new_tag}",
            "gwet_ac1": f"gwet_ac1_{args.new_tag}",
        }
    )
    old_sel = old_df[keys + ["agreement_pct", "error_rate_pct", "cohen_kappa", "gwet_ac1"]].rename(
        columns={
            "agreement_pct": f"agreement_pct_{args.old_tag}",
            "error_rate_pct": f"error_rate_pct_{args.old_tag}",
            "cohen_kappa": f"cohen_kappa_{args.old_tag}",
            "gwet_ac1": f"gwet_ac1_{args.old_tag}",
        }
    )
    wide = new_sel.merge(old_sel, on=keys, how="outer")
    wide["delta_agreement_pct"] = (
        wide[f"agreement_pct_{args.new_tag}"] - wide[f"agreement_pct_{args.old_tag}"]
    )
    wide["delta_cohen_kappa"] = (
        wide[f"cohen_kappa_{args.new_tag}"] - wide[f"cohen_kappa_{args.old_tag}"]
    )
    wide["delta_gwet_ac1"] = (
        wide[f"gwet_ac1_{args.new_tag}"] - wide[f"gwet_ac1_{args.old_tag}"]
    )
    wide["winner_by_agreement"] = wide.apply(
        lambda r: (
            args.new_tag
            if pd.notna(r[f"agreement_pct_{args.new_tag}"])
            and pd.notna(r[f"agreement_pct_{args.old_tag}"])
            and r[f"agreement_pct_{args.new_tag}"] > r[f"agreement_pct_{args.old_tag}"]
            else (
                args.old_tag
                if pd.notna(r[f"agreement_pct_{args.new_tag}"])
                and pd.notna(r[f"agreement_pct_{args.old_tag}"])
                and r[f"agreement_pct_{args.new_tag}"] < r[f"agreement_pct_{args.old_tag}"]
                else "tie_or_na"
            )
        ),
        axis=1,
    )
    wide.to_csv(root_out / "prompt_agreement_comparison.csv", index=False)

    tex_view = wide.copy()
    tex_view = tex_view.sort_values(["phase", "metric"]).reset_index(drop=True)
    _write_tex(
        tex_view,
        root_out / "prompt_agreement_comparison.tex",
        caption=f"Agreement comparison between prompt variants ({args.new_tag} vs {args.old_tag}).",
        label="tab:prompt-agreement-comparison",
    )

    wins = (
        wide["winner_by_agreement"]
        .value_counts(dropna=False)
        .rename_axis("winner")
        .reset_index(name="count")
    )
    wins.to_csv(root_out / "prompt_agreement_winner_counts.csv", index=False)

    print("Prompt comparison outputs written to:", root_out.resolve())
    print(" -", root_out / "prompt_agreement_long.csv")
    print(" -", root_out / "prompt_agreement_comparison.csv")
    print(" -", root_out / "prompt_agreement_comparison.tex")
    print(" -", root_out / "prompt_agreement_winner_counts.csv")


if __name__ == "__main__":
    main()
