from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests


SAMPLE_PATH = Path("LLM Prompts/human issues/issue_sample_180_50.xlsx")
SPECIFIC_INSTR_PATH = Path("LLM Prompts/Specific_Instructions.txt")
HEURISTICS_PATH = Path("LLM Prompts/Usability_Coding_Heuristics.txt")
HUMAN_ISSUES_DIR = Path("LLM Prompts/human issues")
GH_ISSUES_DIR = Path("LLM Prompts/gh-issues")
OUT_DIR = Path("outputs/llm_api_recode")

PHASE12_RANGE_DEFAULT = "2-101"
PHASE35_RANGE_DEFAULT = "102-113,170-181"

OPENAI_BASE_URL_DEFAULT = "https://api.openai.com/v1"
OPENAI_MODEL_DEFAULT = "gpt-4.1"


ASSOCIATED_COMPONENT_THEME_CANONICAL = [
    "Authentication/Authorization tools",
    "CLI tooling",
    "Signing workflow",
    "Verification workflow",
    "Policy/configuration",
    "Build/CI",
    "Release pipeline",
    "Notification/Logging",
    "Core",
    "API",
    "Web Client",
    "Key Management Core / Secrets Backend",
]

L1_THEME_CANONICAL = [
    "Missing feature / enhancement request",
    "Unexpected behavior",
    "Authentication friction",
    "Configuration friction",
    "Integration failure/issues",
    "User confusion / unclear documentation",
    "Build/CI/installation/distribution release issues",
    "Performance issue",
    "Security concerns",
    "Notification/Logging /Web UI Issues",
    "Tedious Workflows",
]

L1_THEME_SECONDARY_CANONICAL = [
    "Operational Friction",
    "Cognitive Friction",
    "Functional Reliability",
    "Functional Gap",
]

NIELSEN_THEME_CANONICAL = [
    "1. Visibility of system status",
    "2. Match between system and the real world",
    "3. User control and freedom",
    "4. Consistency and standards",
    "5. Error prevention",
    "6. Recognition rather than recall",
    "7. Flexibility and efficiency of use",
    "8. Aesthetic and minimalist design",
    "9. Help users recognize, diagnose, and recover from errors",
    "10. Help and documentation",
]

L1_TO_SECONDARY = {
    "Configuration friction": "Operational Friction",
    "Authentication friction": "Operational Friction",
    "Build/CI/installation/distribution release issues": "Operational Friction",
    "Integration failure/issues": "Operational Friction",
    "Tedious Workflows": "Operational Friction",
    "User confusion / unclear documentation": "Cognitive Friction",
    "Notification/Logging /Web UI Issues": "Cognitive Friction",
    "Unexpected behavior": "Functional Reliability",
    "Performance issue": "Functional Reliability",
    "Security concerns": "Functional Reliability",
    "Missing feature / enhancement request": "Functional Gap",
}


def _norm(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def _norm_ws(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip())


def _trunc(s: Any, n: int) -> str:
    s2 = _norm_ws(s)
    return s2 if len(s2) <= n else s2[:n] + " ...[truncated]"


def _parse_list_cell(cell: Any) -> list[str]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    parts = re.split(r"[;,]", s)
    out = [p.strip() for p in parts if p and p.strip()]
    return out


def _as_yes_no(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return "yes"
    return "no"


def _parse_ranges(spec: str, max_value: int | None = None) -> list[int]:
    values: set[int] = set()
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if "-" in tok:
            a, b = tok.split("-", 1)
            start = int(a)
            end = int(b)
            lo, hi = (start, end) if start <= end else (end, start)
            values.update(range(lo, hi + 1))
        else:
            values.add(int(tok))
    out = sorted(values)
    if max_value is not None:
        out = [v for v in out if v <= max_value]
    return out


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_phase_section(txt: str) -> str:
    m = re.search(r"\bPhase 1\b", txt, flags=re.IGNORECASE)
    if not m:
        return txt[:12000]
    section = txt[m.start() :]
    return section[:22000]


def _load_issues_sheet(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="issues")
    # Some files have first row as header and "Unnamed:*" columns.
    if len(df.columns) > 0 and all(str(c).startswith("Unnamed:") for c in df.columns):
        header = df.iloc[0].astype(str).tolist()
        df = df.iloc[1:].copy()
        df.columns = header
    return df


def _canon_colname_map(cols: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in cols:
        out[_norm(c)] = c
    return out


def _first_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    cmap = _canon_colname_map(df.columns)
    for a in aliases:
        key = _norm(a)
        if key in cmap:
            return cmap[key]
    return None


def _ensure_col(df: pd.DataFrame, aliases: list[str], create_name: str) -> str:
    c = _first_col(df, aliases)
    if c is not None:
        return c
    df[create_name] = ""
    return create_name


def _canon_issues_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    repo_col = _first_col(out, ["repo"])
    issue_col = _first_col(out, ["issue_number"])
    title_col = _first_col(out, ["title"])
    body_col = _first_col(out, ["body_text"])
    labels_col = _first_col(out, ["labels"])
    top3_col = _first_col(out, ["top_3_comments_text"])
    allc_col = _first_col(out, ["all_comments_text"])
    issue_url_col = _first_col(out, ["issue_url"])

    if repo_col is None or issue_col is None:
        raise ValueError("issues sheet missing required columns: repo and/or issue_number")

    out = out.rename(
        columns={
            repo_col: "repo",
            issue_col: "issue_number",
            title_col or "title": "title",
            body_col or "body_text": "body_text",
            labels_col or "labels": "labels",
            top3_col or "top_3_comments_text": "top_3_comments_text",
            allc_col or "all_comments_text": "all_comments_text",
            issue_url_col or "issue_url": "issue_url",
        }
    )
    for c in ["title", "body_text", "labels", "top_3_comments_text", "all_comments_text", "issue_url"]:
        if c not in out.columns:
            out[c] = ""
    out["repo"] = out["repo"].astype(str)
    out["issue_number"] = pd.to_numeric(out["issue_number"], errors="coerce").astype("Int64")
    return out


def _load_issue_corpus(folder: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(folder.glob("*.xlsx")):
        if "issue_sample" in p.name:
            continue
        df = _load_issues_sheet(p)
        df = _canon_issues_df(df)
        df["__source_file__"] = p.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No xlsx files found in {folder}")
    all_df = pd.concat(frames, ignore_index=True, sort=False)
    all_df = all_df.dropna(subset=["issue_number"])
    all_df = all_df.sort_values(["repo", "issue_number", "__source_file__"]).drop_duplicates(
        ["repo", "issue_number"], keep="first"
    )
    return all_df


def _load_human_issue_labels(folder: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(folder.glob("*.xlsx")):
        if "issue_sample" in p.name:
            continue
        df = _load_issues_sheet(p)
        if _first_col(df, ["repo"]) is None or _first_col(df, ["issue_number"]) is None:
            continue
        canon = _canon_issues_df(df)
        # Keep all original columns for label extraction.
        for c in df.columns:
            if c not in canon.columns:
                canon[c] = df[c]
        canon["__human_file__"] = p.name
        canon["__phase3_file_priority__"] = 1 if "phase-3" in p.name.lower() else 0
        frames.append(canon)
    if not frames:
        raise FileNotFoundError(f"No usable human issue files in {folder}")
    h = pd.concat(frames, ignore_index=True, sort=False)
    return h


def _pick_human_col(df: pd.DataFrame, aliases: list[str], out_name: str) -> pd.Series:
    col = _first_col(df, aliases)
    if col is None:
        return pd.Series([""] * len(df), index=df.index, name=out_name, dtype="object")
    return df[col].fillna("").astype(str).rename(out_name)


def _build_human_gold_table(human_labels_df: pd.DataFrame) -> pd.DataFrame:
    h = human_labels_df.copy()
    h["issue_number"] = pd.to_numeric(h["issue_number"], errors="coerce").astype("Int64")
    h["repo"] = h["repo"].astype(str)

    h["human_associated_component"] = _pick_human_col(
        h,
        aliases=["Associated Component", "Associated component", "Associated Components"],
        out_name="human_associated_component",
    )
    h["human_codes_primary"] = _pick_human_col(
        h, aliases=["codes_primary", "code_primary", "Codes Primary"], out_name="human_codes_primary"
    )
    h["human_usability_type"] = _pick_human_col(
        h,
        aliases=[
            "usability_non-usability_type",
            "Usability_non-usability_type",
            "Usability_non-usability Type",
            "Usability_Non-Usability Type",
            "usability_non-usability Type",
        ],
        out_name="human_usability_type",
    )
    h["human_associated_component_theme"] = _pick_human_col(
        h,
        aliases=["Associated Component Theme", "Associated Component theme", "Associated component Theme"],
        out_name="human_associated_component_theme",
    )
    h["human_l1_theme"] = _pick_human_col(h, aliases=["L1_Theme"], out_name="human_l1_theme")
    h["human_l1_theme_secondary"] = _pick_human_col(
        h, aliases=["L1_Theme Secondary", "L1_Theme secondary"], out_name="human_l1_theme_secondary"
    )
    h["human_nielsen_theme"] = _pick_human_col(
        h, aliases=["Nielsen_theme", "Nielsen_Theme"], out_name="human_nielsen_theme"
    )

    use_cols = [
        "human_associated_component",
        "human_codes_primary",
        "human_usability_type",
        "human_associated_component_theme",
        "human_l1_theme",
        "human_l1_theme_secondary",
        "human_nielsen_theme",
    ]
    h["__label_fill_score__"] = h[use_cols].apply(lambda r: sum(1 for v in r if _norm_ws(v)), axis=1)

    # Prefer notation phase-3 workbook when duplicates exist, then higher fill score.
    h = h.sort_values(
        ["repo", "issue_number", "__phase3_file_priority__", "__label_fill_score__"],
        ascending=[True, True, False, False],
    ).drop_duplicates(["repo", "issue_number"], keep="first")
    return h[
        ["repo", "issue_number", "__human_file__"] + use_cols
    ].reset_index(drop=True)


def _normalize_usability_binary(v: Any) -> float:
    s = str(v).strip().lower()
    if not s:
        return np.nan
    if s in {"1", "usability", "yes", "true"}:
        return 1.0
    if s in {"0", "non-usability", "nonusability", "non-usability.", "no", "false"}:
        return 0.0
    if "non" in s:
        return 0.0
    if "usability" in s:
        return 1.0
    if re.search(r"\b1\b", s):
        return 1.0
    if re.search(r"\b0\b", s):
        return 0.0
    return np.nan


def _safe_json_loads(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    raise ValueError("Could not parse JSON object from model response.")


@dataclass
class OpenAIClient:
    api_key: str
    model: str
    base_url: str = OPENAI_BASE_URL_DEFAULT
    timeout_s: int = 180
    max_retries: int = 6

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        err = None
        for i in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                if resp.status_code in {429, 500, 502, 503, 504}:
                    wait = min(60, 2 ** i)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return _safe_json_loads(content if isinstance(content, str) else json.dumps(content))
            except Exception as e:
                err = e
                wait = min(60, 2 ** i)
                time.sleep(wait)
        raise RuntimeError(f"OpenAI API call failed after retries: {err}")


def _canonicalize_list(values: Any, canonical_values: list[str]) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        candidates = _parse_list_cell(values)
    elif isinstance(values, list):
        candidates = [str(v).strip() for v in values if str(v).strip()]
    else:
        candidates = [str(values).strip()]

    aliases: dict[str, str] = {}
    for c in canonical_values:
        aliases[_norm(c)] = c
        aliases[_norm(re.sub(r"^\d+\.\s*", "", c))] = c

    # Common aliases from historical labels.
    extra = {
        "documentationhelp": "10. Help and documentation",
        "helpanddocumentation": "10. Help and documentation",
        "notificationloggingissues": "Notification/Logging /Web UI Issues",
        "notificationlogging": "Notification/Logging",
        "buildciinstallation": "Build/CI",
        "buildciinstallationdistributionreleaseissues": "Build/CI/installation/distribution release issues",
        "matchbetweensystemandrealworld": "2. Match between system and the real world",
        "consistencyandstandards": "4. Consistency and standards",
        "flexibilityandefficiencyofuse": "7. Flexibility and efficiency of use",
        "helpusersrecognizediagnoseandrecoverfromerrors": "9. Help users recognize, diagnose, and recover from errors",
        "recognitionratherthanrecall": "6. Recognition rather than recall",
    }
    for k, v in extra.items():
        if v in canonical_values:
            aliases[k] = v

    out = []
    seen = set()
    for x in candidates:
        k = _norm(x)
        if not k:
            continue
        if k in aliases:
            val = aliases[k]
        else:
            # Fallback: keep value if it exactly matches a canonical after whitespace normalization.
            val = None
            for c in canonical_values:
                if _norm_ws(x).lower() == _norm_ws(c).lower():
                    val = c
                    break
            if val is None:
                # unknown label; keep original so user can inspect
                val = x
        if val not in seen:
            out.append(val)
            seen.add(val)
    return out


def _build_annotation_system_prompt(specific_txt: str, heuristics_txt: str) -> str:
    specific_phase = _extract_phase_section(specific_txt)
    heur_phase = _extract_phase_section(heuristics_txt)

    # Keep prompts bounded to avoid huge per-call overhead.
    specific_phase = specific_phase[:10000]
    heur_phase = heur_phase[:12000]

    return f"""
You are a qualitative coding co-annotator for GitHub issues in identity-based software signing tools.
Follow the project instructions and coding heuristics below.

--- Specific Instructions (phase-focused excerpt) ---
{specific_phase}

--- Coding Heuristics (phase-focused excerpt) ---
{heur_phase}

Hard requirements:
1) Return STRICT JSON only (no markdown, no explanation text outside JSON).
2) For multi-label outputs, return JSON arrays.
3) Use canonical labels exactly when possible.
4) If usability_non_usability_type is non-usability, leave phase 3-5 arrays empty unless text gives very strong reason otherwise.
5) Be conservative: do not invent facts beyond the issue text.
""".strip()


def _build_annotation_user_prompt(row: pd.Series) -> str:
    return f"""
Code this single issue across phases 1-5.

Issue metadata:
- sample_id: {row["sample_id"]}
- repo: {row["repo"]}
- issue_number: {row["issue_number"]}
- issue_url: {row.get("issue_url", "")}
- labels: {_trunc(row.get("labels", ""), 1200)}
- title: {_trunc(row.get("title", ""), 2000)}

Issue content:
- body_text: {_trunc(row.get("body_text", ""), 7000)}
- top_3_comments_text: {_trunc(row.get("top_3_comments_text", ""), 5000)}
- all_comments_text: {_trunc(row.get("all_comments_text", ""), 8000)}

Allowed canonical values:
- usability_non_usability_type: ["usability", "non-usability"]
- associated_component_theme: {json.dumps(ASSOCIATED_COMPONENT_THEME_CANONICAL)}
- l1_theme: {json.dumps(L1_THEME_CANONICAL)}
- l1_theme_secondary: {json.dumps(L1_THEME_SECONDARY_CANONICAL)}
- nielsen_theme: {json.dumps(NIELSEN_THEME_CANONICAL)}

Return JSON object with this exact schema:
{{
  "associated_component": "string",
  "codes_primary": "string",
  "usability_non_usability_type": "usability|non-usability",
  "associated_component_theme": ["..."],
  "l1_theme": ["..."],
  "l1_theme_secondary": ["..."],
  "nielsen_theme": ["..."],
  "confidence": 0.0
}}
""".strip()


def _postprocess_prediction(pred: dict[str, Any]) -> dict[str, Any]:
    p = dict(pred)
    usability = str(p.get("usability_non_usability_type", "")).strip().lower()
    if "non" in usability:
        usability = "non-usability"
    else:
        usability = "usability"
    p["usability_non_usability_type"] = usability

    p["associated_component"] = _norm_ws(p.get("associated_component", ""))
    p["codes_primary"] = _norm_ws(p.get("codes_primary", ""))

    p["associated_component_theme"] = _canonicalize_list(
        p.get("associated_component_theme", []), ASSOCIATED_COMPONENT_THEME_CANONICAL
    )
    p["l1_theme"] = _canonicalize_list(p.get("l1_theme", []), L1_THEME_CANONICAL)
    p["l1_theme_secondary"] = _canonicalize_list(
        p.get("l1_theme_secondary", []), L1_THEME_SECONDARY_CANONICAL
    )
    p["nielsen_theme"] = _canonicalize_list(p.get("nielsen_theme", []), NIELSEN_THEME_CANONICAL)

    # If model omitted secondary themes, derive from L1.
    if not p["l1_theme_secondary"] and p["l1_theme"]:
        derived = []
        for t in p["l1_theme"]:
            sec = L1_TO_SECONDARY.get(t)
            if sec and sec not in derived:
                derived.append(sec)
        p["l1_theme_secondary"] = derived

    # Per heuristics, only apply phase 3-5 labels to usability issues.
    if usability == "non-usability":
        p["associated_component_theme"] = []
        p["l1_theme"] = []
        p["l1_theme_secondary"] = []
        p["nielsen_theme"] = []

    try:
        conf = float(p.get("confidence", np.nan))
    except Exception:
        conf = np.nan
    p["confidence"] = conf
    return p


def _annotate_rows(
    df: pd.DataFrame,
    client: OpenAIClient,
    system_prompt: str,
    cache_path: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not overwrite:
        cached = pd.read_csv(cache_path)
    else:
        cached = pd.DataFrame()

    done_keys = set()
    if not cached.empty:
        done_keys = set(zip(cached["repo"].astype(str), pd.to_numeric(cached["issue_number"], errors="coerce")))

    rows = []
    if not cached.empty:
        rows.extend(cached.to_dict("records"))

    for _, r in df.iterrows():
        key = (str(r["repo"]), float(r["issue_number"]))
        if key in done_keys:
            continue

        user_prompt = _build_annotation_user_prompt(r)
        raw = client.chat_json(system_prompt, user_prompt)
        pred = _postprocess_prediction(raw)

        rows.append(
            {
                "sample_id": int(r["sample_id"]),
                "repo": str(r["repo"]),
                "issue_number": int(r["issue_number"]),
                "issue_url": r.get("issue_url", ""),
                "llm_associated_component": pred["associated_component"],
                "llm_codes_primary": pred["codes_primary"],
                "llm_usability_type": pred["usability_non_usability_type"],
                "llm_associated_component_theme": ", ".join(pred["associated_component_theme"]),
                "llm_l1_theme": ", ".join(pred["l1_theme"]),
                "llm_l1_theme_secondary": ", ".join(pred["l1_theme_secondary"]),
                "llm_nielsen_theme": ", ".join(pred["nielsen_theme"]),
                "llm_confidence": pred.get("confidence", np.nan),
                "llm_raw_json": json.dumps(raw, ensure_ascii=False),
            }
        )

        # Save progressively.
        pd.DataFrame(rows).to_csv(cache_path, index=False)
        done_keys.add(key)
        time.sleep(0.2)

    out = pd.DataFrame(rows)
    out["issue_number"] = pd.to_numeric(out["issue_number"], errors="coerce").astype("Int64")
    out["repo"] = out["repo"].astype(str)
    return out.sort_values(["sample_id"]).reset_index(drop=True)


def _judge_phase1_semantic(
    df: pd.DataFrame,
    client: OpenAIClient,
    cache_path: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not overwrite:
        cached = pd.read_csv(cache_path)
    else:
        cached = pd.DataFrame()

    done = set()
    rows = []
    if not cached.empty:
        rows.extend(cached.to_dict("records"))
        done = set(zip(cached["repo"].astype(str), pd.to_numeric(cached["issue_number"], errors="coerce")))

    system_prompt = """
You are an agreement judge. Compare HUMAN vs LLM labels semantically.
Return STRICT JSON only with booleans:
{
  "associated_component_agree": true/false,
  "codes_primary_agree": true/false
}
Guidelines:
- Treat synonyms and naming variants as agreement if they refer to the same functional surface.
- For codes_primary, agree if both capture the same core issue/problem and practical meaning.
- Be strict enough to flag different problem meanings as disagreement.
""".strip()

    for _, r in df.iterrows():
        key = (str(r["repo"]), float(r["issue_number"]))
        if key in done:
            continue
        user_prompt = f"""
Issue: {r["repo"]}#{int(r["issue_number"])}
Human associated component: {r.get("human_associated_component", "")}
LLM associated component: {r.get("llm_associated_component", "")}

Human codes_primary: {r.get("human_codes_primary", "")}
LLM codes_primary: {r.get("llm_codes_primary", "")}
""".strip()
        j = client.chat_json(system_prompt, user_prompt)
        rows.append(
            {
                "sample_id": int(r["sample_id"]),
                "repo": str(r["repo"]),
                "issue_number": int(r["issue_number"]),
                "associated_component_agree": _as_yes_no(j.get("associated_component_agree")),
                "codes_primary_agree": _as_yes_no(j.get("codes_primary_agree")),
                "judge_raw_json": json.dumps(j, ensure_ascii=False),
            }
        )
        pd.DataFrame(rows).to_csv(cache_path, index=False)
        done.add(key)
        time.sleep(0.2)

    out = pd.DataFrame(rows)
    out["issue_number"] = pd.to_numeric(out["issue_number"], errors="coerce").astype("Int64")
    return out.sort_values(["sample_id"]).reset_index(drop=True)


def _judge_phase35_semantic(
    df: pd.DataFrame,
    client: OpenAIClient,
    cache_path: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not overwrite:
        cached = pd.read_csv(cache_path)
    else:
        cached = pd.DataFrame()

    done = set()
    rows = []
    if not cached.empty:
        rows.extend(cached.to_dict("records"))
        done = set(zip(cached["repo"].astype(str), pd.to_numeric(cached["issue_number"], errors="coerce")))

    system_prompt = """
You are an agreement judge for qualitative labels.
Compare HUMAN vs LLM semantic meaning for each field and return STRICT JSON:
{
  "associated_component_theme_agree": true/false,
  "l1_theme_agree": true/false,
  "l1_theme_secondary_agree": true/false,
  "nielsen_theme_agree": true/false
}
Guidelines:
- For multi-label fields, partial overlap on the main meaning can count as agreement.
- Treat wording variants and close synonyms as agreement.
- If human field is empty/missing, return true (not evaluable mismatch).
""".strip()

    for _, r in df.iterrows():
        key = (str(r["repo"]), float(r["issue_number"]))
        if key in done:
            continue
        user_prompt = f"""
Issue: {r["repo"]}#{int(r["issue_number"])}

Human associated component theme: {r.get("human_associated_component_theme", "")}
LLM associated component theme: {r.get("llm_associated_component_theme", "")}

Human L1 theme: {r.get("human_l1_theme", "")}
LLM L1 theme: {r.get("llm_l1_theme", "")}

Human L1 secondary theme: {r.get("human_l1_theme_secondary", "")}
LLM L1 secondary theme: {r.get("llm_l1_theme_secondary", "")}

Human Nielsen theme: {r.get("human_nielsen_theme", "")}
LLM Nielsen theme: {r.get("llm_nielsen_theme", "")}
""".strip()
        j = client.chat_json(system_prompt, user_prompt)
        rows.append(
            {
                "sample_id": int(r["sample_id"]),
                "repo": str(r["repo"]),
                "issue_number": int(r["issue_number"]),
                "associated_component_theme_agree": _as_yes_no(j.get("associated_component_theme_agree")),
                "l1_theme_agree": _as_yes_no(j.get("l1_theme_agree")),
                "l1_theme_secondary_agree": _as_yes_no(j.get("l1_theme_secondary_agree")),
                "nielsen_theme_agree": _as_yes_no(j.get("nielsen_theme_agree")),
                "judge_raw_json": json.dumps(j, ensure_ascii=False),
            }
        )
        pd.DataFrame(rows).to_csv(cache_path, index=False)
        done.add(key)
        time.sleep(0.2)

    out = pd.DataFrame(rows)
    out["issue_number"] = pd.to_numeric(out["issue_number"], errors="coerce").astype("Int64")
    return out.sort_values(["sample_id"]).reset_index(drop=True)


def _cohen_kappa_binary(y1: np.ndarray, y2: np.ndarray) -> float:
    assert len(y1) == len(y2)
    n = len(y1)
    if n == 0:
        return np.nan
    po = float((y1 == y2).mean())
    p1 = float((y1 == 1).mean())
    p2 = float((y2 == 1).mean())
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    den = 1 - pe
    if den == 0:
        return np.nan
    return (po - pe) / den


def _gwet_ac1_binary(y1: np.ndarray, y2: np.ndarray) -> float:
    assert len(y1) == len(y2)
    n = len(y1)
    if n == 0:
        return np.nan
    po = float((y1 == y2).mean())
    p_cat1 = ((y1 == 1).sum() + (y2 == 1).sum()) / (2 * n)
    p_cat0 = 1 - p_cat1
    pe = p_cat1 * (1 - p_cat1) + p_cat0 * (1 - p_cat0)
    den = 1 - pe
    if den == 0:
        return np.nan
    return (po - pe) / den


def _write_updated_repo_copies(
    gh_folder: Path,
    predictions_df: pd.DataFrame,
    out_folder: Path,
) -> None:
    out_folder.mkdir(parents=True, exist_ok=True)
    pred = predictions_df.copy()
    pred["repo"] = pred["repo"].astype(str)
    pred["issue_number"] = pd.to_numeric(pred["issue_number"], errors="coerce").astype("Int64")

    for p in sorted(gh_folder.glob("*.xlsx")):
        xls = pd.ExcelFile(p)
        sheet_frames = {s: xls.parse(s) for s in xls.sheet_names}
        if "issues" not in sheet_frames:
            continue

        issues_raw = sheet_frames["issues"].copy()
        # header-fix case
        fixed_header = False
        if len(issues_raw.columns) > 0 and all(str(c).startswith("Unnamed:") for c in issues_raw.columns):
            hdr = issues_raw.iloc[0].astype(str).tolist()
            issues = issues_raw.iloc[1:].copy()
            issues.columns = hdr
            fixed_header = True
        else:
            issues = issues_raw

        canon = _canon_issues_df(issues)
        m = canon[["repo", "issue_number"]].merge(
            pred,
            on=["repo", "issue_number"],
            how="left",
        )

        col_ac = _ensure_col(issues, ["Associated Component", "Associated component"], "Associated Component")
        col_cp = _ensure_col(issues, ["codes_primary", "code_primary"], "codes_primary")
        col_ut = _ensure_col(
            issues,
            ["usability_non-usability_type", "Usability_non-usability_type", "Usability_Non-Usability Type"],
            "usability_non-usability_type",
        )
        col_act = _ensure_col(
            issues,
            ["Associated Component Theme", "Associated Component theme", "Associated component Theme"],
            "Associated Component Theme",
        )
        col_l1 = _ensure_col(issues, ["L1_Theme"], "L1_Theme")
        col_l1s = _ensure_col(issues, ["L1_Theme Secondary", "L1_Theme secondary"], "L1_Theme Secondary")
        col_n = _ensure_col(issues, ["Nielsen_theme", "Nielsen_Theme"], "Nielsen_theme")

        # Align by positional index of canon->issues.
        for idx in range(len(issues)):
            if idx >= len(m):
                break
            if pd.isna(m.at[idx, "llm_associated_component"]):
                continue
            issues.at[issues.index[idx], col_ac] = m.at[idx, "llm_associated_component"]
            issues.at[issues.index[idx], col_cp] = m.at[idx, "llm_codes_primary"]
            issues.at[issues.index[idx], col_ut] = m.at[idx, "llm_usability_type"]
            issues.at[issues.index[idx], col_act] = m.at[idx, "llm_associated_component_theme"]
            issues.at[issues.index[idx], col_l1] = m.at[idx, "llm_l1_theme"]
            issues.at[issues.index[idx], col_l1s] = m.at[idx, "llm_l1_theme_secondary"]
            issues.at[issues.index[idx], col_n] = m.at[idx, "llm_nielsen_theme"]

        # Rebuild original header-quirk if needed.
        if fixed_header:
            back = pd.concat([pd.DataFrame([issues.columns], columns=issues.columns), issues], ignore_index=True)
            back.columns = issues_raw.columns
            sheet_frames["issues"] = back
        else:
            sheet_frames["issues"] = issues

        out_path = out_folder / p.name
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as wr:
            for sname, sdf in sheet_frames.items():
                sdf.to_excel(wr, sheet_name=sname, index=False)


def _summarize_yes_rate(s: pd.Series) -> tuple[float, float, int]:
    v = s.astype(str).str.lower().str.strip()
    n = int(v.notna().sum())
    if n == 0:
        return np.nan, np.nan, 0
    agree = (v == "yes").mean()
    return float(agree), float(1 - agree), n


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run OpenAI API recoding for phases 1-5 on issue samples, then compute agreements "
            "(phase 1 semantic %, phase 2 Cohen Kappa + Gwet AC1, phase 3-5 semantic %)."
        )
    )
    parser.add_argument("--model", default=OPENAI_MODEL_DEFAULT)
    parser.add_argument("--base-url", default=OPENAI_BASE_URL_DEFAULT)
    parser.add_argument("--phase12-range", default=PHASE12_RANGE_DEFAULT)
    parser.add_argument("--phase35-range", default=PHASE35_RANGE_DEFAULT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-semantic-judge", action="store_true")
    parser.add_argument("--skip-api", action="store_true", help="Only build extracted files; skip API labeling.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sample = pd.read_excel(SAMPLE_PATH)
    sample["sample_id"] = range(1, len(sample) + 1)
    sample["repo"] = sample["repo"].astype(str)
    sample["issue_number"] = pd.to_numeric(sample["issue_number"], errors="coerce").astype("Int64")

    phase12_ids = _parse_ranges(args.phase12_range, max_value=len(sample))
    phase35_ids = _parse_ranges(args.phase35_range, max_value=len(sample))

    phase12_sel = sample[sample["sample_id"].isin(phase12_ids)].copy()
    phase35_sel = sample[sample["sample_id"].isin(phase35_ids)].copy()

    gh = _load_issue_corpus(GH_ISSUES_DIR)
    human_all = _load_human_issue_labels(HUMAN_ISSUES_DIR)
    human_gold = _build_human_gold_table(human_all)

    # Build extracted phase data.
    phase12 = phase12_sel.merge(
        gh[
            [
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
        ],
        on=["repo", "issue_number"],
        how="left",
    ).merge(
        human_gold[
            [
                "repo",
                "issue_number",
                "__human_file__",
                "human_associated_component",
                "human_codes_primary",
                "human_usability_type",
            ]
        ],
        on=["repo", "issue_number"],
        how="left",
    )
    phase12["human_phase2_analyst_a"] = phase12[
        "Usability_Non-Usability Type- Analyst A"
    ].astype(str)
    phase12["human_phase2_analyst_a_bin"] = phase12["human_phase2_analyst_a"].map(_normalize_usability_binary)

    phase35 = phase35_sel.merge(
        gh[
            [
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
        ],
        on=["repo", "issue_number"],
        how="left",
    ).merge(
        human_gold[
            [
                "repo",
                "issue_number",
                "__human_file__",
                "human_associated_component_theme",
                "human_l1_theme",
                "human_l1_theme_secondary",
                "human_nielsen_theme",
            ]
        ],
        on=["repo", "issue_number"],
        how="left",
    )

    phase12.to_csv(OUT_DIR / "phase12_extracted_with_human.csv", index=False)
    phase35.to_csv(OUT_DIR / "phase35_extracted_with_human.csv", index=False)
    with pd.ExcelWriter(OUT_DIR / "phase_samples_extracted_with_human.xlsx", engine="xlsxwriter") as wr:
        phase12.to_excel(wr, sheet_name="phase12_sample", index=False)
        phase35.to_excel(wr, sheet_name="phase35_sample", index=False)

    if args.skip_api:
        print(f"Extracted files written to: {OUT_DIR}")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    specific_txt = _read_txt(SPECIFIC_INSTR_PATH)
    heuristics_txt = _read_txt(HEURISTICS_PATH)
    system_prompt = _build_annotation_system_prompt(specific_txt, heuristics_txt)

    client = OpenAIClient(api_key=api_key, model=args.model, base_url=args.base_url)

    pred12 = _annotate_rows(
        phase12,
        client=client,
        system_prompt=system_prompt,
        cache_path=OUT_DIR / "phase12_llm_predictions_cache.csv",
        overwrite=args.overwrite,
    )
    pred35 = _annotate_rows(
        phase35,
        client=client,
        system_prompt=system_prompt,
        cache_path=OUT_DIR / "phase35_llm_predictions_cache.csv",
        overwrite=args.overwrite,
    )

    phase12_eval = phase12.merge(pred12, on=["sample_id", "repo", "issue_number", "issue_url"], how="left")
    phase35_eval = phase35.merge(pred35, on=["sample_id", "repo", "issue_number", "issue_url"], how="left")

    # Phase 2 metrics.
    phase12_eval["llm_phase2_bin"] = phase12_eval["llm_usability_type"].map(_normalize_usability_binary)
    p2_valid = phase12_eval[
        phase12_eval["human_phase2_analyst_a_bin"].notna() & phase12_eval["llm_phase2_bin"].notna()
    ].copy()
    y_h = p2_valid["human_phase2_analyst_a_bin"].astype(int).to_numpy()
    y_l = p2_valid["llm_phase2_bin"].astype(int).to_numpy()
    p2_po = float((y_h == y_l).mean()) if len(p2_valid) else np.nan
    p2_kappa = _cohen_kappa_binary(y_h, y_l) if len(p2_valid) else np.nan
    p2_ac1 = _gwet_ac1_binary(y_h, y_l) if len(p2_valid) else np.nan
    phase2_metrics = pd.DataFrame(
        [
            {
                "n": int(len(p2_valid)),
                "observed_agreement_pct": round(100 * p2_po, 2) if pd.notna(p2_po) else np.nan,
                "error_rate_pct": round(100 * (1 - p2_po), 2) if pd.notna(p2_po) else np.nan,
                "cohen_kappa": p2_kappa,
                "gwet_ac1": p2_ac1,
            }
        ]
    )
    phase2_metrics.to_csv(OUT_DIR / "phase2_agreement_metrics.csv", index=False)

    # Semantic judges.
    if args.skip_semantic_judge:
        phase1_sem = pd.DataFrame()
        phase35_sem = pd.DataFrame()
    else:
        phase1_sem = _judge_phase1_semantic(
            phase12_eval,
            client=client,
            cache_path=OUT_DIR / "phase1_semantic_judgement_cache.csv",
            overwrite=args.overwrite,
        )
        phase35_sem = _judge_phase35_semantic(
            phase35_eval,
            client=client,
            cache_path=OUT_DIR / "phase35_semantic_judgement_cache.csv",
            overwrite=args.overwrite,
        )

    phase12_eval = phase12_eval.merge(
        phase1_sem[["sample_id", "repo", "issue_number", "associated_component_agree", "codes_primary_agree"]]
        if not phase1_sem.empty
        else pd.DataFrame(columns=["sample_id", "repo", "issue_number", "associated_component_agree", "codes_primary_agree"]),
        on=["sample_id", "repo", "issue_number"],
        how="left",
    )
    phase35_eval = phase35_eval.merge(
        phase35_sem[
            [
                "sample_id",
                "repo",
                "issue_number",
                "associated_component_theme_agree",
                "l1_theme_agree",
                "l1_theme_secondary_agree",
                "nielsen_theme_agree",
            ]
        ]
        if not phase35_sem.empty
        else pd.DataFrame(
            columns=[
                "sample_id",
                "repo",
                "issue_number",
                "associated_component_theme_agree",
                "l1_theme_agree",
                "l1_theme_secondary_agree",
                "nielsen_theme_agree",
            ]
        ),
        on=["sample_id", "repo", "issue_number"],
        how="left",
    )

    # Save detailed outputs.
    phase12_eval.to_csv(OUT_DIR / "phase12_predictions_and_eval.csv", index=False)
    phase35_eval.to_csv(OUT_DIR / "phase35_predictions_and_eval.csv", index=False)
    with pd.ExcelWriter(OUT_DIR / "phase_predictions_and_eval.xlsx", engine="xlsxwriter") as wr:
        phase12_eval.to_excel(wr, sheet_name="phase12", index=False)
        phase35_eval.to_excel(wr, sheet_name="phase35", index=False)
        if not phase1_sem.empty:
            phase1_sem.to_excel(wr, sheet_name="phase1_semantic_judge", index=False)
        if not phase35_sem.empty:
            phase35_sem.to_excel(wr, sheet_name="phase35_semantic_judge", index=False)
        phase2_metrics.to_excel(wr, sheet_name="phase2_metrics", index=False)

    # Summary tables.
    summary_rows: list[dict[str, Any]] = []

    if not phase1_sem.empty:
        comp_agree, comp_err, n_comp = _summarize_yes_rate(phase1_sem["associated_component_agree"])
        code_agree, code_err, n_code = _summarize_yes_rate(phase1_sem["codes_primary_agree"])
        summary_rows += [
            {
                "phase": "Phase 1",
                "metric": "Associated Component semantic agreement",
                "n": n_comp,
                "agreement_pct": round(100 * comp_agree, 2) if pd.notna(comp_agree) else np.nan,
                "error_rate_pct": round(100 * comp_err, 2) if pd.notna(comp_err) else np.nan,
            },
            {
                "phase": "Phase 1",
                "metric": "codes_primary semantic agreement",
                "n": n_code,
                "agreement_pct": round(100 * code_agree, 2) if pd.notna(code_agree) else np.nan,
                "error_rate_pct": round(100 * code_err, 2) if pd.notna(code_err) else np.nan,
            },
        ]

    summary_rows += [
        {
            "phase": "Phase 2",
            "metric": "Observed agreement",
            "n": int(phase2_metrics.at[0, "n"]),
            "agreement_pct": phase2_metrics.at[0, "observed_agreement_pct"],
            "error_rate_pct": phase2_metrics.at[0, "error_rate_pct"],
            "cohen_kappa": phase2_metrics.at[0, "cohen_kappa"],
            "gwet_ac1": phase2_metrics.at[0, "gwet_ac1"],
        }
    ]

    if not phase35_sem.empty:
        for col, label in [
            ("associated_component_theme_agree", "Associated Component Theme semantic agreement"),
            ("l1_theme_agree", "L1_Theme semantic agreement"),
            ("l1_theme_secondary_agree", "L1_Theme Secondary semantic agreement"),
            ("nielsen_theme_agree", "Nielsen_theme semantic agreement"),
        ]:
            a, e, n = _summarize_yes_rate(phase35_sem[col])
            summary_rows.append(
                {
                    "phase": "Phase 3-5",
                    "metric": label,
                    "n": n,
                    "agreement_pct": round(100 * a, 2) if pd.notna(a) else np.nan,
                    "error_rate_pct": round(100 * e, 2) if pd.notna(e) else np.nan,
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "openai_recode_agreement_summary.csv", index=False)

    # Write copied gh issue files with updated columns for the selected samples.
    pred_all = pd.concat(
        [
            phase12_eval[
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
            ],
            phase35_eval[
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
            ],
        ],
        ignore_index=True,
    ).drop_duplicates(["repo", "issue_number"], keep="first")

    _write_updated_repo_copies(
        gh_folder=GH_ISSUES_DIR,
        predictions_df=pred_all,
        out_folder=OUT_DIR / "copied_gh_issues_with_openai_labels",
    )

    run_meta = {
        "model": args.model,
        "base_url": args.base_url,
        "phase12_range": args.phase12_range,
        "phase35_range": args.phase35_range,
        "n_phase12_selected": int(len(phase12)),
        "n_phase35_selected": int(len(phase35)),
        "skip_semantic_judge": args.skip_semantic_judge,
        "timestamp_epoch": int(time.time()),
    }
    (OUT_DIR / "openai_recode_run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print("Wrote outputs to:", OUT_DIR.resolve())
    print("Key files:")
    for p in [
        OUT_DIR / "phase12_extracted_with_human.csv",
        OUT_DIR / "phase35_extracted_with_human.csv",
        OUT_DIR / "phase12_predictions_and_eval.csv",
        OUT_DIR / "phase35_predictions_and_eval.csv",
        OUT_DIR / "phase2_agreement_metrics.csv",
        OUT_DIR / "openai_recode_agreement_summary.csv",
        OUT_DIR / "copied_gh_issues_with_openai_labels",
    ]:
        print(" -", p)


if __name__ == "__main__":
    main()

