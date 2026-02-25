from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


DATA_DIR = Path("data/phase3")
OUT_DIR = Path("outputs/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COL_TOOL = "repo"
COL_USABILITY = "usability_non-usability_type"
COL_URL = "issue_url"
COL_NUM = "issue_number"
COL_TITLE = "title"
COL_CREATED = "created_at"


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def norm_usability(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"1", "usability", "true", "yes"}:
        return 1
    if s in {"0", "non-usability", "nonusability", "false", "no"}:
        return 0
    try:
        return int(float(s))
    except Exception:
        return np.nan


def split_multi(cell) -> list[str]:
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def split_multi_nielsen(cell) -> list[str]:
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    matches = re.findall(r"(?:^|,\s*)(\d+\.\s.*?)(?=(?:,\s*\d+\.|$))", s)
    if matches:
        return [m.strip() for m in matches if m.strip()]
    return split_multi(s)


def latex_escape(s: str) -> str:
    out = str(s)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out


PRIMARY_THEMES: list[tuple[str, str]] = [
    (
        "Missing feature / enhancement request",
        "User requests functionality that does not currently exist, or enhancement of existing capability.",
    ),
    (
        "Unexpected behavior",
        "Tool behaves incorrectly despite user following expected instructions or documentation.",
    ),
    (
        "Authentication friction",
        "User cannot authenticate/authorize smoothly due to token, flow, timeout, or identity setup problems.",
    ),
    (
        "Configuration friction",
        "User struggles to configure the tool due to complexity, rigid settings, or unclear setup requirements.",
    ),
    (
        "Integration failure/issues",
        "Interoperation with external systems fails or requires additional support/enhancement.",
    ),
    (
        "User confusion / unclear documentation",
        "User is confused, asks usage questions, or documentation is unclear/outdated/insufficient.",
    ),
    (
        "Build/CI/installation/distribution release issues",
        "Problems running/installing/building/distributing the tool in local or CI/CD environments.",
    ),
    (
        "Performance issue",
        "Tool functions but is too slow or resource intensive (or needs performance improvements).",
    ),
    (
        "Security concerns",
        "Reported vulnerability, unsafe default, leakage risk, or request to harden security posture.",
    ),
    (
        "Notification/Logging Issues",
        "Logs, errors, status output, or web/client feedback are unclear, noisy, or not actionable.",
    ),
    (
        "Tedious Workflows",
        "Workflow is overly manual or requires too many steps to achieve a routine task.",
    ),
]

SECONDARY_THEMES: list[tuple[str, str]] = [
    (
        "Operational Friction",
        "Barriers in setup, environment, integration, or workflow execution where users know what to do but the system resists.",
    ),
    (
        "Cognitive Friction",
        "Barriers in understanding or mental model clarity, where users do not know how to proceed.",
    ),
    (
        "Functional Reliability",
        "Friction when expected function is undermined by failures, slowness, or security risk.",
    ),
    (
        "Functional Gap",
        "Tool currently lacks capability needed for the user’s intended task.",
    ),
]

NIELSEN_THEMES: list[tuple[str, str]] = [
    (
        "1. Visibility of system status",
        "System should keep users informed with timely, clear feedback about ongoing operations.",
    ),
    (
        "2. Match between system and the real world",
        "Use familiar language and concepts; avoid internal jargon that conflicts with user expectations.",
    ),
    (
        "3. User control and freedom",
        "Provide clear exits/undo paths so users can recover from unwanted states.",
    ),
    (
        "4. Consistency and standards",
        "Use consistent terminology/behavior and follow ecosystem conventions.",
    ),
    (
        "5. Error prevention",
        "Prevent avoidable mistakes with checks, guardrails, and confirmations before commitment.",
    ),
    (
        "6. Recognition rather than recall",
        "Reduce memory burden by making required actions/options visible in the interface/help.",
    ),
    (
        "7. Flexibility and efficiency of use",
        "Support efficient workflows for experts while remaining usable for less-experienced users.",
    ),
    (
        "8. Aesthetic and minimalist design",
        "Avoid irrelevant/noisy information that hides key outputs or decisions.",
    ),
    (
        "9. Help users recognize, diagnose, and recover from errors",
        "Error messages should be clear, specific, and provide actionable recovery guidance.",
    ),
    (
        "10. Help and documentation",
        "Documentation and help text should be discoverable, accurate, and aligned with current behavior.",
    ),
]

COMPONENT_THEMES: list[tuple[str, str]] = [
    (
        "Authentication/Authorization tools",
        "Identity verification and permission logic, including OIDC flows, token handling, MFA, and RBAC behavior.",
    ),
    (
        "CLI tooling",
        "Command-line UX including command/flag behavior, prompts, and output formatting.",
    ),
    (
        "Signing workflow",
        "End-to-end client-side path used to produce cryptographic signatures for artifacts.",
    ),
    (
        "Verification workflow",
        "End-to-end client-side path used to validate signatures and trust decisions.",
    ),
    (
        "Policy/configuration",
        "Configuration and policy expression/interpretation (e.g., YAML/JSON, OPA/Rego, policy enforcement).",
    ),
    (
        "Build/CI/Installation",
        "Friction in installation/build and automated execution in CI/CD or scripted environments.",
    ),
    (
        "Release pipeline",
        "Maintainer-side release engineering: building, signing, and publishing official binaries/artifacts.",
    ),
    (
        "Notification/Logging",
        "Clarity/actionability of logs, warnings, status output, and debug traces.",
    ),
    (
        "Core",
        "Issue affects the overall tool/core behavior when a specific component boundary is unclear.",
    ),
    (
        "API",
        "Issues in API endpoints, request/response behavior, and API-facing interactions.",
    ),
    (
        "Web Client",
        "Issues in web UI/client/admin dashboard interactions and feedback.",
    ),
    (
        "Key Management Core / Secrets Backend",
        "Key/secret lifecycle and secure storage infrastructure, including KMS/HSM/keychain integrations.",
    ),
]


def build_alias_map(theme_names: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for t in theme_names:
        out[norm_key(t)] = t
    return out


PRIMARY_ALIAS = build_alias_map([x[0] for x in PRIMARY_THEMES])
PRIMARY_ALIAS.update(
    {
        norm_key("User confusion / unclear documentation/documentation improvements"): "User confusion / unclear documentation",
        norm_key("Notification/Logging /Web UI Issues"): "Notification/Logging Issues",
    }
)

SECONDARY_ALIAS = build_alias_map([x[0] for x in SECONDARY_THEMES])

NIELSEN_ALIAS = build_alias_map([x[0] for x in NIELSEN_THEMES])
for t, _ in NIELSEN_THEMES:
    t_no_num = re.sub(r"^\s*\d+\.\s*", "", t)
    NIELSEN_ALIAS[norm_key(t_no_num)] = t
NIELSEN_ALIAS[norm_key("9. Help users recognize and recover from errors")] = (
    "9. Help users recognize, diagnose, and recover from errors"
)

COMPONENT_ALIAS = build_alias_map([x[0] for x in COMPONENT_THEMES])
COMPONENT_ALIAS.update(
    {
        norm_key("Build/CI"): "Build/CI/Installation",
        norm_key("Build/CI/installation"): "Build/CI/Installation",
        norm_key("Key Management Core (secret engine)"): "Key Management Core / Secrets Backend",
    }
)


def choose_secondary_col(df: pd.DataFrame) -> str:
    candidates = [c for c in ["L1_Theme Secondary", "L1_Theme secondary"] if c in df.columns]
    if not candidates:
        raise KeyError("Missing both 'L1_Theme Secondary' and 'L1_Theme secondary'.")
    if len(candidates) == 1:
        return candidates[0]
    counts = {c: int(df[c].notna().sum()) for c in candidates}
    return max(counts, key=counts.get)


def load_all_data() -> pd.DataFrame:
    paths = sorted(DATA_DIR.glob("*.xlsx"))
    if not paths:
        raise FileNotFoundError(f"No xlsx files found under {DATA_DIR}")
    frames = []
    for p in paths:
        d = pd.read_excel(p)
        d["__source_file__"] = p.name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df[COL_USABILITY] = df[COL_USABILITY].apply(norm_usability)
    df[COL_CREATED] = pd.to_datetime(df[COL_CREATED], errors="coerce", utc=True)
    return df


def extract_example_map(
    df_u: pd.DataFrame,
    theme_col: str,
    alias_map: dict[str, str],
    split_fn,
) -> dict[str, dict]:
    keep_cols = [COL_TOOL, COL_NUM, COL_URL, COL_TITLE, COL_CREATED, theme_col]
    tmp = df_u[keep_cols].copy()
    tmp["theme_raw"] = tmp[theme_col].apply(split_fn)
    tmp = tmp.explode("theme_raw")
    tmp = tmp[tmp["theme_raw"].notna() & (tmp["theme_raw"].astype(str).str.strip() != "")]
    tmp["theme_raw"] = tmp["theme_raw"].astype(str).str.strip()
    tmp["theme"] = tmp["theme_raw"].map(lambda x: alias_map.get(norm_key(x)))
    tmp = tmp[tmp["theme"].notna()].copy()
    tmp = tmp[tmp[COL_URL].notna() & (tmp[COL_URL].astype(str).str.startswith("http"))]
    tmp = tmp.sort_values([COL_CREATED, COL_TOOL, COL_NUM], na_position="last")
    first = tmp.groupby("theme", as_index=False).first()
    return {
        str(r["theme"]): {
            "repo": str(r[COL_TOOL]),
            "issue_number": str(r[COL_NUM]),
            "issue_url": str(r[COL_URL]),
            "title": str(r[COL_TITLE]) if pd.notna(r[COL_TITLE]) else "",
        }
        for _, r in first.iterrows()
    }


def format_example_link(example: dict | None) -> str:
    if not example:
        return r"\textit{No example URL in current Phase 3 usability set}"
    repo = latex_escape(example["repo"])
    num = latex_escape(example["issue_number"])
    url = example["issue_url"]
    return rf"\href{{{url}}}{{{repo}\#{num}}}"


def build_rows(
    family_name: str,
    themes: list[tuple[str, str]],
    examples: dict[str, dict],
) -> list[dict]:
    rows = []
    for theme, definition in themes:
        ex = examples.get(theme)
        rows.append(
            {
                "theme_family": family_name,
                "theme": theme,
                "definition": definition,
            }
        )
    return rows


def to_longtable_latex(df: pd.DataFrame) -> str:
    lines = [
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{p{0.17\textwidth} p{0.25\textwidth} p{0.52\textwidth}}",
        r"\caption{Phase 3 theme codebook with heuristic definitions.}\label{tab:phase3-theme-codebook}\\",
        r"\hline",
        r"\textbf{Theme Family} & \textbf{Theme} & \textbf{Definition} \\",
        r"\hline",
        r"\endfirsthead",
        r"\hline",
        r"\textbf{Theme Family} & \textbf{Theme} & \textbf{Definition} \\",
        r"\hline",
        r"\endhead",
        r"\hline",
        r"\endfoot",
        r"\hline",
        r"\endlastfoot",
    ]

    for _, r in df.iterrows():
        family = latex_escape(r["theme_family"])
        theme = latex_escape(r["theme"])
        definition = latex_escape(r["definition"])
        lines.append(f"{family} & {theme} & {definition} \\\\")

    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"


def main():
    df = load_all_data()
    df_u = df[df[COL_USABILITY] == 1].copy()
    secondary_col = choose_secondary_col(df_u)

    primary_examples = extract_example_map(df_u, "L1_Theme", PRIMARY_ALIAS, split_multi)
    secondary_examples = extract_example_map(df_u, secondary_col, SECONDARY_ALIAS, split_multi)
    nielsen_examples = extract_example_map(df_u, "Nielsen_theme", NIELSEN_ALIAS, split_multi_nielsen)
    component_examples = extract_example_map(df_u, "Associated Component Theme", COMPONENT_ALIAS, split_multi)

    rows = []
    rows.extend(build_rows("Primary theme (inductively generated)", PRIMARY_THEMES, primary_examples))
    rows.extend(build_rows("Secondary theme (inductively generated)", SECONDARY_THEMES, secondary_examples))
    rows.extend(build_rows("Nielsen theme (deductive)", NIELSEN_THEMES, nielsen_examples))
    rows.extend(build_rows("Affected component theme", COMPONENT_THEMES, component_examples))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_DIR / "phase3_theme_codebook_with_examples.csv", index=False)

    latex = to_longtable_latex(out_df)
    (OUT_DIR / "phase3_theme_codebook_with_examples.tex").write_text(latex)

    print("Wrote:", OUT_DIR / "phase3_theme_codebook_with_examples.csv")
    print("Wrote:", OUT_DIR / "phase3_theme_codebook_with_examples.tex")


if __name__ == "__main__":
    main()
