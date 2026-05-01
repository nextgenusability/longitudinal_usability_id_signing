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
MASTER_PROMPT_PATH = Path("LLM Prompts/master_prompt.txt")
HUMAN_ISSUES_DIR = Path("LLM Prompts/human issues")
GH_ISSUES_DIR = Path("LLM Prompts/gh-issues")
OUT_DIR = Path("outputs/llm_api_recode_phase")

PHASE12_RANGE_DEFAULT = "2-161"
PHASE35_RANGE_DEFAULT = "102-113,170-181"

OPENAI_BASE_URL_DEFAULT = "https://api.openai.com/v1"
OPENAI_MODEL_DEFAULT = "gpt-4.1"

TOOL_TOKENS = {
    "sigstore",
    "cosign",
    "fulcio",
    "rekor",
    "notary",
    "notation",
    "notaryproject",
    "vault",
    "hashicorp",
    "openpubkey",
    "opk",
    "keyfactor",
    "ejbca",
    "signserver",
}

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
    try:
        if pd.isna(s):
            return ""
    except Exception:
        pass
    if isinstance(s, float) and np.isnan(s):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def _norm_ws(s: Any) -> str:
    if s is None:
        return ""
    try:
        if pd.isna(s):
            return ""
    except Exception:
        pass
    if isinstance(s, float) and np.isnan(s):
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
    return [p.strip() for p in parts if p and p.strip()]


def _parse_ranges(spec: str, max_value: int | None = None) -> list[int]:
    values: set[int] = set()
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if "-" in tok:
            a, b = tok.split("-", 1)
            lo, hi = sorted((int(a), int(b)))
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
    return txt[m.start() : m.start() + 22000]


def _extract_master_actionable(txt: str) -> str:
    # Prefer operational instructions over long project background.
    m = re.search(r"\b3\.\s*EXPECTATION AND DIRECTIONS\b", txt, flags=re.IGNORECASE)
    if m:
        return txt[m.start() : m.start() + 16000]
    return _extract_phase_section(txt)[:16000]


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
                    time.sleep(min(60, 2 ** i))
                    continue
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return _safe_json_loads(content if isinstance(content, str) else json.dumps(content))
            except Exception as e:
                err = e
                time.sleep(min(60, 2 ** i))
        raise RuntimeError(f"OpenAI API call failed after retries: {err}")


def _canon_colname_map(cols: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in cols:
        out[_norm(c)] = c
    return out


def _first_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    cmap = _canon_colname_map(df.columns)
    for a in aliases:
        k = _norm(a)
        if k in cmap:
            return cmap[k]
    return None


def _ensure_col(df: pd.DataFrame, aliases: list[str], create_name: str) -> str:
    c = _first_col(df, aliases)
    if c is not None:
        return c
    df[create_name] = ""
    return create_name


def _load_issues_sheet(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="issues")
    if len(df.columns) > 0 and all(str(c).startswith("Unnamed:") for c in df.columns):
        header = df.iloc[0].astype(str).tolist()
        df = df.iloc[1:].copy()
        df.columns = header
    return df


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
        raise ValueError("issues sheet missing required columns: repo and issue_number")
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


def load_issue_corpus(folder: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(folder.glob("*.xlsx")):
        if "issue_sample" in p.name:
            continue
        df = _canon_issues_df(_load_issues_sheet(p))
        df["__source_file__"] = p.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No xlsx files in {folder}")
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.dropna(subset=["issue_number"])
    out = out.sort_values(["repo", "issue_number", "__source_file__"]).drop_duplicates(
        ["repo", "issue_number"], keep="first"
    )
    return out


def _load_human_issue_labels(folder: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(folder.glob("*.xlsx")):
        if "issue_sample" in p.name:
            continue
        df = _load_issues_sheet(p)
        if _first_col(df, ["repo"]) is None or _first_col(df, ["issue_number"]) is None:
            continue
        canon = _canon_issues_df(df)
        for c in df.columns:
            if c not in canon.columns:
                canon[c] = df[c]
        canon["__human_file__"] = p.name
        canon["__phase3_priority__"] = 1 if "phase-3" in p.name.lower() else 0
        frames.append(canon)
    if not frames:
        raise FileNotFoundError(f"No usable human issue files in {folder}")
    return pd.concat(frames, ignore_index=True, sort=False)


def _pick_human_col(df: pd.DataFrame, aliases: list[str], out_name: str) -> pd.Series:
    alias_keys = {_norm(a) for a in aliases}
    matched = [c for c in df.columns if _norm(c) in alias_keys]
    if not matched:
        return pd.Series([""] * len(df), index=df.index, name=out_name, dtype="object")
    out = pd.Series([""] * len(df), index=df.index, name=out_name, dtype="object")
    for c in matched:
        vals = df[c].fillna("").astype(str).map(_norm_ws)
        fill_mask = out.map(_norm_ws).eq("") & vals.ne("")
        out.loc[fill_mask] = vals.loc[fill_mask]
    return out


def build_human_gold_table(human_labels_df: pd.DataFrame) -> pd.DataFrame:
    h = human_labels_df.copy()
    h["repo"] = h["repo"].astype(str)
    h["issue_number"] = pd.to_numeric(h["issue_number"], errors="coerce").astype("Int64")

    h["human_associated_component"] = _pick_human_col(
        h, ["Associated Component", "Associated component", "Associated Components"], "human_associated_component"
    )
    h["human_codes_primary"] = _pick_human_col(
        h, ["codes_primary", "code_primary", "Codes Primary"], "human_codes_primary"
    )
    h["human_usability_type"] = _pick_human_col(
        h,
        [
            "usability_non-usability_type",
            "Usability_non-usability_type",
            "Usability_non-usability Type",
            "Usability_Non-Usability Type",
            "usability_non-usability Type",
        ],
        "human_usability_type",
    )
    h["human_associated_component_theme"] = _pick_human_col(
        h,
        ["Associated Component Theme", "Associated Component theme", "Associated component Theme"],
        "human_associated_component_theme",
    )
    h["human_l1_theme"] = _pick_human_col(h, ["L1_Theme"], "human_l1_theme")
    h["human_l1_theme_secondary"] = _pick_human_col(
        h, ["L1_Theme Secondary", "L1_Theme secondary"], "human_l1_theme_secondary"
    )
    h["human_nielsen_theme"] = _pick_human_col(h, ["Nielsen_theme", "Nielsen_Theme"], "human_nielsen_theme")

    label_cols = [
        "human_associated_component",
        "human_codes_primary",
        "human_usability_type",
        "human_associated_component_theme",
        "human_l1_theme",
        "human_l1_theme_secondary",
        "human_nielsen_theme",
    ]
    h["__label_fill_score__"] = h[label_cols].apply(lambda r: sum(1 for v in r if _norm_ws(v)), axis=1)
    h = h.sort_values(
        ["repo", "issue_number", "__phase3_priority__", "__label_fill_score__"],
        ascending=[True, True, False, False],
    ).drop_duplicates(["repo", "issue_number"], keep="first")
    return h[["repo", "issue_number", "__human_file__"] + label_cols].reset_index(drop=True)


def normalize_usability_binary(v: Any) -> float:
    s = str(v).strip().lower()
    if not s:
        return np.nan
    if s in {"1", "usability", "yes", "true"}:
        return 1.0
    if s in {"0", "non-usability", "nonusability", "no", "false"}:
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


def _canon_from_allowed(values: Any, allowed: list[str], extra_aliases: dict[str, str] | None = None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, list):
        raw = [str(v).strip() for v in values if str(v).strip()]
    else:
        raw = _parse_list_cell(values)

    aliases = {}
    for a in allowed:
        aliases[_norm(a)] = a
        aliases[_norm(re.sub(r"^\d+\.\s*", "", a))] = a
    if extra_aliases:
        for k, v in extra_aliases.items():
            if v in allowed:
                aliases[_norm(k)] = v

    out: list[str] = []
    seen = set()
    for x in raw:
        k = _norm(x)
        if not k:
            continue
        if k in aliases:
            c = aliases[k]
        else:
            c = x.strip()
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def canonicalize_associated_component(s: Any) -> list[str]:
    toks = _parse_list_cell(s)
    out = []
    seen = set()
    for t in toks:
        x = _norm_ws(t).lower()
        x = re.sub(r"[\(\)\[\]{}]+", " ", x)
        x = x.replace("&", " and ")
        x = re.sub(r"[-_/]+", " ", x)
        x = re.sub(r"\s+", " ", x).strip()
        words = [w for w in x.split() if w not in TOOL_TOKENS]
        x = " ".join(words).strip()
        # light semantic normalization
        x = x.replace("documentation", "doc")
        x = x.replace("docs", "doc")
        x = x.replace("web ui", "web client")
        x = x.replace("webclient", "web client")
        x = x.replace("rest api", "api")
        x = x.replace("command line", "cli")
        x = re.sub(r"\s+", " ", x).strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def associated_component_semantic_concepts(s: Any) -> set[str]:
    """
    Convert free-text associated-component labels into normalized semantic concepts
    so near-equivalent labels (e.g., 'rest api' vs 'api') compare as matches.
    """
    raw = _norm_ws(s).lower()
    comps = canonicalize_associated_component(s)
    concepts: set[str] = set()
    for c in comps:
        t = c.lower()
        matched = False

        if "cli" in t or "command line" in t:
            concepts.add("cli")
            matched = True
        if "api" in t:
            concepts.add("api")
            matched = True
        if any(k in t for k in ["web", "ui", "dashboard", "frontend", "page", "portal"]):
            concepts.add("web_client")
            matched = True
        if any(k in t for k in ["verify", "verification"]):
            concepts.add("verification_workflow")
            matched = True
        if any(k in t for k in ["sign", "signature"]):
            concepts.add("signing_workflow")
            matched = True
        if any(k in t for k in ["auth", "oidc", "rbac", "mfa", "authorization", "identity", "token"]):
            concepts.add("authentication_authorization")
            matched = True
        if any(k in t for k in ["policy", "config", "rego", "opa"]):
            concepts.add("policy_configuration")
            matched = True
        if any(k in t for k in ["build", "ci", "install", "installation", "docker", "pipeline"]):
            concepts.add("build_ci")
            matched = True
        if any(k in t for k in ["release", "distribution", "artifact publish"]):
            concepts.add("release_pipeline")
            matched = True
        if any(k in t for k in ["log", "notif", "debug", "error message"]):
            concepts.add("notification_logging")
            matched = True
        if any(k in t for k in ["key", "kms", "hsm", "secret", "keychain", "certificate authority", "ca"]):
            concepts.add("key_management_or_ca")
            matched = True

        # Fallback to a normalized phrase if no concept rule fired.
        if not matched and t:
            concepts.add(t)

    # If the label is mostly tool-name shorthand (e.g., "ejbca", "hashicorp vault"),
    # assign a coarse concept instead of leaving it empty.
    if not concepts and raw:
        raw_norm = re.sub(r"[-_/]+", " ", raw)
        raw_norm = re.sub(r"\s+", " ", raw_norm).strip()
        if (
            any(k in raw_norm for k in ["ejbca", "fulcio", "certificate authority"])
            or re.search(r"\bca\b", raw_norm) is not None
        ):
            concepts.add("key_management_or_ca")
        elif any(k in raw_norm for k in ["vault", "cosign", "notation", "openpubkey", "rekor", "signserver"]):
            concepts.add("core")

    return concepts


def canonicalize_associated_component_theme(s: Any) -> list[str]:
    extra = {
        "build ci installation": "Build/CI",
        "build ci": "Build/CI",
        "ci cd": "Build/CI",
        "cicd": "Build/CI",
        "installation": "Build/CI",
        "install": "Build/CI",
        "build": "Build/CI",
        "key management core secret engine": "Key Management Core / Secrets Backend",
        "key management core secrets backend": "Key Management Core / Secrets Backend",
        "secret engine": "Key Management Core / Secrets Backend",
        "secrets backend": "Key Management Core / Secrets Backend",
        "key management": "Key Management Core / Secrets Backend",
        "certificate authority": "Key Management Core / Secrets Backend",
        "ca": "Key Management Core / Secrets Backend",
        "pki": "Key Management Core / Secrets Backend",
        "authentication authorization tools": "Authentication/Authorization tools",
        "authentication": "Authentication/Authorization tools",
        "authorization": "Authentication/Authorization tools",
        "auth": "Authentication/Authorization tools",
        "oidc": "Authentication/Authorization tools",
        "authn authz": "Authentication/Authorization tools",
        "authn": "Authentication/Authorization tools",
        "authz": "Authentication/Authorization tools",
        "policy configuration": "Policy/configuration",
        "policy": "Policy/configuration",
        "configuration": "Policy/configuration",
        "config": "Policy/configuration",
        "cli": "CLI tooling",
        "command line": "CLI tooling",
        "signing": "Signing workflow",
        "signing workflow": "Signing workflow",
        "sign workflow": "Signing workflow",
        "verification": "Verification workflow",
        "verify": "Verification workflow",
        "verification workflow": "Verification workflow",
        "release": "Release pipeline",
        "distribution": "Release pipeline",
        "logging": "Notification/Logging",
        "logs": "Notification/Logging",
        "notification": "Notification/Logging",
        "web ui": "Web Client",
        "web page": "Web Client",
        "webpage": "Web Client",
        "dashboard": "Web Client",
        "web": "Web Client",
        "rest api": "API",
        "api": "API",
        "core service": "Core",
        "tool core": "Core",
    }
    return _canon_from_allowed(s, ASSOCIATED_COMPONENT_THEME_CANONICAL, extra_aliases=extra)


def derive_component_theme_from_component(s: Any) -> list[str]:
    comps = canonicalize_associated_component(s)
    out: list[str] = []
    for c in comps:
        t = c.lower()
        if "cli" in t and "CLI tooling" not in out:
            out.append("CLI tooling")
        if ("verify" in t or "verification" in t) and "Verification workflow" not in out:
            out.append("Verification workflow")
        if ("sign" in t or "signature" in t) and "Signing workflow" not in out:
            out.append("Signing workflow")
        if ("auth" in t or "oidc" in t or "rbac" in t or "mfa" in t) and "Authentication/Authorization tools" not in out:
            out.append("Authentication/Authorization tools")
        if ("policy" in t or "config" in t or "rego" in t or "opa" in t) and "Policy/configuration" not in out:
            out.append("Policy/configuration")
        if ("build" in t or "ci" in t or "install" in t or "docker" in t) and "Build/CI" not in out:
            out.append("Build/CI")
        if ("release" in t or "distribution" in t) and "Release pipeline" not in out:
            out.append("Release pipeline")
        if ("log" in t or "notif" in t or "debug" in t) and "Notification/Logging" not in out:
            out.append("Notification/Logging")
        if "api" in t and "API" not in out:
            out.append("API")
        if ("web" in t or "ui" in t or "dashboard" in t) and "Web Client" not in out:
            out.append("Web Client")
        if ("key management" in t or "secret" in t or "kms" in t or "hsm" in t or "keychain" in t) and "Key Management Core / Secrets Backend" not in out:
            out.append("Key Management Core / Secrets Backend")

    if not out and comps:
        out.append("Core")
    return out


def canonicalize_l1_theme(s: Any) -> list[str]:
    extra = {
        "feature request": "Missing feature / enhancement request",
        "missing feature": "Missing feature / enhancement request",
        "enhancement request": "Missing feature / enhancement request",
        "enhancement": "Missing feature / enhancement request",
        "missing capability": "Missing feature / enhancement request",
        "documentation improvements": "User confusion / unclear documentation",
        "unclear documentation": "User confusion / unclear documentation",
        "user confusion": "User confusion / unclear documentation",
        "confusion": "User confusion / unclear documentation",
        "docs": "User confusion / unclear documentation",
        "documentation": "User confusion / unclear documentation",
        "auth": "Authentication friction",
        "authentication": "Authentication friction",
        "authentication issues": "Authentication friction",
        "authorization": "Authentication friction",
        "config": "Configuration friction",
        "configuration": "Configuration friction",
        "configuration issues": "Configuration friction",
        "integration": "Integration failure/issues",
        "integration issues": "Integration failure/issues",
        "build": "Build/CI/installation/distribution release issues",
        "ci": "Build/CI/installation/distribution release issues",
        "ci cd": "Build/CI/installation/distribution release issues",
        "installation": "Build/CI/installation/distribution release issues",
        "release issues": "Build/CI/installation/distribution release issues",
        "perf": "Performance issue",
        "performance": "Performance issue",
        "security": "Security concerns",
        "security issue": "Security concerns",
        "security issues": "Security concerns",
        "notification logging issues": "Notification/Logging /Web UI Issues",
        "notification logging": "Notification/Logging /Web UI Issues",
        "notification": "Notification/Logging /Web UI Issues",
        "logging": "Notification/Logging /Web UI Issues",
        "log": "Notification/Logging /Web UI Issues",
        "web ui issues": "Notification/Logging /Web UI Issues",
        "unexpected": "Unexpected behavior",
        "unexpected behaviour": "Unexpected behavior",
        "unexpected behavior": "Unexpected behavior",
        "workflow friction": "Tedious Workflows",
        "tedious workflow": "Tedious Workflows",
        "tedious workflows": "Tedious Workflows",
    }
    return _canon_from_allowed(s, L1_THEME_CANONICAL, extra_aliases=extra)


def canonicalize_l1_secondary(s: Any, l1_fallback: list[str] | None = None) -> list[str]:
    extra = {
        "operational": "Operational Friction",
        "cognitive": "Cognitive Friction",
        "functional reliability": "Functional Reliability",
        "functional gap": "Functional Gap",
    }
    vals = _canon_from_allowed(s, L1_THEME_SECONDARY_CANONICAL, extra_aliases=extra)
    if vals:
        return vals
    if l1_fallback:
        out = []
        for t in l1_fallback:
            sec = L1_TO_SECONDARY.get(t)
            if sec and sec not in out:
                out.append(sec)
        return out
    return []


def canonicalize_nielsen(s: Any) -> list[str]:
    extra = {
        "1": "1. Visibility of system status",
        "2": "2. Match between system and the real world",
        "3": "3. User control and freedom",
        "4": "4. Consistency and standards",
        "5": "5. Error prevention",
        "6": "6. Recognition rather than recall",
        "7": "7. Flexibility and efficiency of use",
        "8": "8. Aesthetic and minimalist design",
        "9": "9. Help users recognize, diagnose, and recover from errors",
        "10": "10. Help and documentation",
        "help and documentation": "10. Help and documentation",
        "help documentation": "10. Help and documentation",
        "error diagnosis and recovery": "9. Help users recognize, diagnose, and recover from errors",
    }
    raw = _parse_list_cell(s)
    # allow values that are just numerals separated by comma
    out = []
    for r in raw:
        m = re.match(r"^\s*(\d{1,2})\s*$", r)
        if m:
            out.append(m.group(1))
        else:
            out.append(r)
    return _canon_from_allowed(out, NIELSEN_THEME_CANONICAL, extra_aliases=extra)


def build_annotation_system_prompt(
    specific_txt: str,
    heuristics_txt: str,
    master_txt: str | None = None,
) -> str:
    if master_txt and _norm_ws(master_txt):
        master = _extract_master_actionable(master_txt)
        return f"""
You are a qualitative coding co-annotator for GitHub issues in identity-based software signing tools.
Use the operational instructions below as the coding protocol.

--- Master Prompt (actionable excerpt) ---
{master}

Hard requirements:
1) Return STRICT JSON only.
2) Multi-label outputs must be arrays.
3) Use canonical labels when possible.
4) Be conservative and avoid hallucinating facts.
""".strip()

    specific_phase = _extract_phase_section(specific_txt)[:9000]
    heur_phase = _extract_phase_section(heuristics_txt)[:11000]
    return f"""
You are a qualitative coding co-annotator for GitHub issues in identity-based software signing tools.
Follow project instructions and coding heuristics below.

--- Specific Instructions (excerpt) ---
{specific_phase}

--- Coding Heuristics (excerpt) ---
{heur_phase}

Hard requirements:
1) Return STRICT JSON only.
2) Multi-label outputs must be arrays.
3) Use canonical labels when possible.
4) Be conservative and avoid hallucinating facts.
""".strip()


def build_annotation_user_prompt(row: pd.Series) -> str:
    return f"""
Code this single issue for phases 1-5.

Issue metadata:
- sample_id: {row["sample_id"]}
- repo: {row["repo"]}
- issue_number: {row["issue_number"]}
- issue_url: {row.get("issue_url", "")}
- labels: {_trunc(row.get("labels", ""), 1200)}
- title: {_trunc(row.get("title", ""), 1800)}

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

Return JSON with exactly:
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


def postprocess_prediction(pred: dict[str, Any]) -> dict[str, Any]:
    p = dict(pred)
    u = str(p.get("usability_non_usability_type", "")).strip().lower()
    if u in {"0", "non-usability", "nonusability", "non usability", "no", "false"} or "non" in u:
        p["usability_non_usability_type"] = "non-usability"
    elif u in {"1", "usability", "yes", "true"} or "usability" in u:
        p["usability_non_usability_type"] = "usability"
    else:
        p["usability_non_usability_type"] = "usability"
    p["associated_component"] = _norm_ws(p.get("associated_component", ""))
    p["codes_primary"] = _norm_ws(p.get("codes_primary", ""))
    p["associated_component_theme"] = canonicalize_associated_component_theme(p.get("associated_component_theme", []))
    if not p["associated_component_theme"] and p["associated_component"]:
        p["associated_component_theme"] = derive_component_theme_from_component(p["associated_component"])
    p["l1_theme"] = canonicalize_l1_theme(p.get("l1_theme", []))
    p["l1_theme_secondary"] = canonicalize_l1_secondary(
        p.get("l1_theme_secondary", []), l1_fallback=p["l1_theme"]
    )
    p["nielsen_theme"] = canonicalize_nielsen(p.get("nielsen_theme", []))
    if p["usability_non_usability_type"] == "non-usability":
        p["associated_component_theme"] = []
        p["l1_theme"] = []
        p["l1_theme_secondary"] = []
        p["nielsen_theme"] = []
    try:
        p["confidence"] = float(p.get("confidence", np.nan))
    except Exception:
        p["confidence"] = np.nan
    return p


def annotate_rows(
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

    rows = []
    done = set()
    if not cached.empty:
        rows.extend(cached.to_dict("records"))
        done = set(zip(cached["repo"].astype(str), pd.to_numeric(cached["issue_number"], errors="coerce")))

    total = len(df)
    completed = len(done)
    print(f"[annotate] starting with {completed}/{total} already cached")

    for _, r in df.iterrows():
        key = (str(r["repo"]), float(r["issue_number"]))
        if key in done:
            continue
        raw = client.chat_json(system_prompt, build_annotation_user_prompt(r))
        pred = postprocess_prediction(raw)
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
                "llm_confidence": pred["confidence"],
                "llm_raw_json": json.dumps(raw, ensure_ascii=False),
            }
        )
        pd.DataFrame(rows).to_csv(cache_path, index=False)
        done.add(key)
        completed += 1
        if completed % 10 == 0 or completed == total:
            print(f"[annotate] completed {completed}/{total}")
        time.sleep(0.2)

    out = pd.DataFrame(rows)
    out["repo"] = out["repo"].astype(str)
    out["issue_number"] = pd.to_numeric(out["issue_number"], errors="coerce").astype("Int64")
    return out.sort_values("sample_id").reset_index(drop=True)


def judge_code_primary_semantic(
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

    rows = []
    done = set()
    if not cached.empty:
        rows.extend(cached.to_dict("records"))
        done = set(zip(cached["repo"].astype(str), pd.to_numeric(cached["issue_number"], errors="coerce")))

    system_prompt = """
You are judging semantic agreement between two issue summaries.
Return strict JSON:
{
  "code_primary_semantic_agree": true/false
}
Rule:
- true if both summaries convey the same core issue/problem and practical meaning,
  even if wording differs.
- false if they describe different primary problems.
""".strip()

    total = len(df)
    completed = len(done)
    print(f"[judge-code-primary] starting with {completed}/{total} already cached")

    for _, r in df.iterrows():
        key = (str(r["repo"]), float(r["issue_number"]))
        if key in done:
            continue
        human = _norm_ws(r.get("human_codes_primary", ""))
        llm = _norm_ws(r.get("llm_codes_primary", ""))
        if not human:
            agree = "yes"
            raw = {"code_primary_semantic_agree": True, "note": "human missing"}
        else:
            user_prompt = f"""
Issue: {r["repo"]}#{int(r["issue_number"])}
Human codes_primary: {human}
LLM codes_primary: {llm}
""".strip()
            j = client.chat_json(system_prompt, user_prompt)
            agree = "yes" if bool(j.get("code_primary_semantic_agree", False)) else "no"
            raw = j

        rows.append(
            {
                "sample_id": int(r["sample_id"]),
                "repo": str(r["repo"]),
                "issue_number": int(r["issue_number"]),
                "code_primary_semantic_agree": agree,
                "judge_raw_json": json.dumps(raw, ensure_ascii=False),
            }
        )
        pd.DataFrame(rows).to_csv(cache_path, index=False)
        done.add(key)
        completed += 1
        if completed % 10 == 0 or completed == total:
            print(f"[judge-code-primary] completed {completed}/{total}")
        time.sleep(0.2)

    out = pd.DataFrame(rows)
    out["repo"] = out["repo"].astype(str)
    out["issue_number"] = pd.to_numeric(out["issue_number"], errors="coerce").astype("Int64")
    return out.sort_values("sample_id").reset_index(drop=True)


def judge_associated_component_semantic(
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

    rows = []
    done = set()
    if not cached.empty:
        rows.extend(cached.to_dict("records"))
        done = set(zip(cached["repo"].astype(str), pd.to_numeric(cached["issue_number"], errors="coerce")))

    system_prompt = """
You are judging semantic agreement between two software issue component labels.
Return strict JSON:
{
  "associated_component_semantic_agree": true/false
}
Rule:
- true if the two labels refer to the same practical component/surface, even if wording differs
  (e.g., "REST API" vs "API", "web page" vs "web client", tool-prefixed labels with same surface).
- false if they refer to different component surfaces.
""".strip()

    total = len(df)
    completed = len(done)
    print(f"[judge-associated-component] starting with {completed}/{total} already cached")

    for _, r in df.iterrows():
        key = (str(r["repo"]), float(r["issue_number"]))
        if key in done:
            continue
        human = _norm_ws(r.get("human_associated_component", ""))
        llm = _norm_ws(r.get("llm_associated_component", ""))
        if not human:
            agree = "yes"
            raw = {"associated_component_semantic_agree": True, "note": "human missing"}
        else:
            user_prompt = f"""
Issue: {r["repo"]}#{int(r["issue_number"])}
Human associated component: {human}
LLM associated component: {llm}
""".strip()
            j = client.chat_json(system_prompt, user_prompt)
            agree = "yes" if bool(j.get("associated_component_semantic_agree", False)) else "no"
            raw = j

        rows.append(
            {
                "sample_id": int(r["sample_id"]),
                "repo": str(r["repo"]),
                "issue_number": int(r["issue_number"]),
                "associated_component_semantic_agree": agree,
                "judge_raw_json": json.dumps(raw, ensure_ascii=False),
            }
        )
        pd.DataFrame(rows).to_csv(cache_path, index=False)
        done.add(key)
        completed += 1
        if completed % 10 == 0 or completed == total:
            print(f"[judge-associated-component] completed {completed}/{total}")
        time.sleep(0.2)

    out = pd.DataFrame(rows)
    out["repo"] = out["repo"].astype(str)
    out["issue_number"] = pd.to_numeric(out["issue_number"], errors="coerce").astype("Int64")
    return out.sort_values("sample_id").reset_index(drop=True)


def _sets_full_partial(human_sets: list[set[str]], llm_sets: list[set[str]]) -> tuple[np.ndarray, np.ndarray]:
    full = np.array([int(h == l) for h, l in zip(human_sets, llm_sets)], dtype=int)
    partial = np.array([int((h == l) or (len(h.intersection(l)) > 0)) for h, l in zip(human_sets, llm_sets)], dtype=int)
    return full, partial


def _cohen_kappa_binary(y1: np.ndarray, y2: np.ndarray) -> float:
    n = len(y1)
    if n == 0:
        return np.nan
    po = float((y1 == y2).mean())
    p1 = float((y1 == 1).mean())
    p2 = float((y2 == 1).mean())
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    den = 1 - pe
    return np.nan if den == 0 else (po - pe) / den


def _gwet_ac1_binary(y1: np.ndarray, y2: np.ndarray) -> float:
    n = len(y1)
    if n == 0:
        return np.nan
    po = float((y1 == y2).mean())
    p = ((y1 == 1).sum() + (y2 == 1).sum()) / (2 * n)
    q = 1 - p
    pe = p * (1 - p) + q * (1 - q)
    den = 1 - pe
    return np.nan if den == 0 else (po - pe) / den


def _multilabel_kappa_ac1(human_sets: list[set[str]], llm_sets: list[set[str]], universe: list[str]) -> tuple[float, float]:
    if not universe:
        return np.nan, np.nan
    idx = {u: i for i, u in enumerate(universe)}
    h = np.zeros((len(human_sets), len(universe)), dtype=int)
    l = np.zeros((len(llm_sets), len(universe)), dtype=int)
    for i, s in enumerate(human_sets):
        for x in s:
            if x in idx:
                h[i, idx[x]] = 1
    for i, s in enumerate(llm_sets):
        for x in s:
            if x in idx:
                l[i, idx[x]] = 1
    yh = h.reshape(-1)
    yl = l.reshape(-1)
    return _cohen_kappa_binary(yh, yl), _gwet_ac1_binary(yh, yl)


def _binary_precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int = 1,
) -> tuple[float, float, float, float, int, int, int, int]:
    yt = (y_true == pos_label).astype(int)
    yp = (y_pred == pos_label).astype(int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)) if pd.notna(precision) and pd.notna(recall) and (precision + recall) > 0 else np.nan
    return precision, recall, specificity, f1, tp, fp, fn, tn


def _metric_row(
    phase: str,
    metric: str,
    n: int,
    agreement: float,
    error_rate: float,
    kappa: float = np.nan,
    ac1: float = np.nan,
    phase12_subset: str = "",
    precision: float = np.nan,
    recall: float = np.nan,
    specificity: float = np.nan,
    f1: float = np.nan,
    tp: int = 0,
    fp: int = 0,
    fn: int = 0,
    tn: int = 0,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "phase12_subset": phase12_subset,
        "metric": metric,
        "n": int(n),
        "agreement_pct": round(100 * agreement, 2) if pd.notna(agreement) else np.nan,
        "error_rate_pct": round(100 * error_rate, 2) if pd.notna(error_rate) else np.nan,
        "precision_pct": round(100 * precision, 2) if pd.notna(precision) else np.nan,
        "recall_pct": round(100 * recall, 2) if pd.notna(recall) else np.nan,
        "specificity_pct": round(100 * specificity, 2) if pd.notna(specificity) else np.nan,
        "f1_pct": round(100 * f1, 2) if pd.notna(f1) else np.nan,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "cohen_kappa": kappa,
        "gwet_ac1": ac1,
    }


def compute_agreements(
    phase12_eval: pd.DataFrame,
    phase35_eval: pd.DataFrame,
    code_primary_sem_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    p12 = phase12_eval.copy()
    p12["phase12_subset"] = np.where(
        p12["sample_id"].between(2, 101),
        "first_100",
        np.where(p12["sample_id"].between(102, 161), "last_60", "other"),
    )

    # Phase 1: Associated Component semantic agreement, split into first_100 and last_60.
    p1 = p12[p12["human_associated_component"].fillna("").astype(str).str.strip() != ""].copy()
    h_comp = [associated_component_semantic_concepts(x) for x in p1["human_associated_component"]]
    l_comp = [associated_component_semantic_concepts(x) for x in p1["llm_associated_component"]]
    full, partial = _sets_full_partial(h_comp, l_comp)
    p1["full_agree"] = full
    p1["partial_agree"] = partial
    p1["human_associated_component_semantic"] = [", ".join(sorted(x)) for x in h_comp]
    p1["llm_associated_component_semantic"] = [", ".join(sorted(x)) for x in l_comp]
    phase1_detail = p1[
        [
            "sample_id",
            "phase12_subset",
            "repo",
            "issue_number",
            "human_associated_component",
            "llm_associated_component",
            "human_associated_component_semantic",
            "llm_associated_component_semantic",
            "full_agree",
            "partial_agree",
        ]
    ].copy()
    p1a = p1.copy()
    # Semantic agreement for Associated Component uses concept overlap:
    # agree if canonical concept sets overlap (or exactly match).
    p1a["associated_component_semantic_agree_bin"] = p1a["partial_agree"].astype(int)
    p1a["associated_component_semantic_agree"] = np.where(
        p1a["associated_component_semantic_agree_bin"].eq(1), "yes", "no"
    )
    for subset, p1ag in [
        ("all", p1a),
        ("first_100", p1a[p1a["phase12_subset"] == "first_100"]),
        ("last_60", p1a[p1a["phase12_subset"] == "last_60"]),
    ]:
        if p1ag.empty:
            agr = np.nan
            n = 0
        else:
            agr = p1ag["associated_component_semantic_agree_bin"].mean()
            n = len(p1ag)
        rows.append(
            _metric_row(
                "Phase 1",
                "Associated Component (semantic agreement)",
                n,
                agr,
                (1 - agr) if pd.notna(agr) else np.nan,
                phase12_subset=subset,
            )
        )
    phase1_detail = phase1_detail.merge(
        p1a[
            [
                "sample_id",
                "repo",
                "issue_number",
                "associated_component_semantic_agree",
            ]
        ],
        on=["sample_id", "repo", "issue_number"],
        how="left",
    )

    if code_primary_sem_df is not None and not code_primary_sem_df.empty:
        p1s = p1.merge(
            code_primary_sem_df[["sample_id", "repo", "issue_number", "code_primary_semantic_agree"]],
            on=["sample_id", "repo", "issue_number"],
            how="left",
        )
        p1s["code_primary_semantic_agree_bin"] = (
            p1s["code_primary_semantic_agree"].fillna("no").astype(str).str.lower().eq("yes").astype(int)
        )
        for subset, p1sg in [
            ("all", p1s),
            ("first_100", p1s[p1s["phase12_subset"] == "first_100"]),
            ("last_60", p1s[p1s["phase12_subset"] == "last_60"]),
        ]:
            if p1sg.empty:
                agr = np.nan
                n = 0
            else:
                agr = p1sg["code_primary_semantic_agree_bin"].mean()
                n = len(p1sg)
            rows.append(
                _metric_row(
                    "Phase 1",
                    "codes_primary (semantic agreement)",
                    n,
                    agr,
                    (1 - agr) if pd.notna(agr) else np.nan,
                    phase12_subset=subset,
                )
            )
        phase1_detail = phase1_detail.merge(
            p1s[["sample_id", "repo", "issue_number", "code_primary_semantic_agree"]],
            on=["sample_id", "repo", "issue_number"],
            how="left",
        )

    # Phase 2: binary usability with kappa + AC1, split into first_100 and last_60.
    v = p12[p12["human_phase2_analyst_a_bin"].notna() & p12["llm_phase2_bin"].notna()].copy()
    for subset, vg in [
        ("all", v),
        ("first_100", v[v["phase12_subset"] == "first_100"]),
        ("last_60", v[v["phase12_subset"] == "last_60"]),
    ]:
        if vg.empty:
            rows.append(
                _metric_row(
                    "Phase 2",
                    "Usability_Non-Usability Type",
                    0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    phase12_subset=subset,
                    precision=np.nan,
                    recall=np.nan,
                    specificity=np.nan,
                    f1=np.nan,
                    tp=0,
                    fp=0,
                    fn=0,
                    tn=0,
                )
            )
            continue
        y_h = vg["human_phase2_analyst_a_bin"].astype(int).to_numpy()
        y_l = vg["llm_phase2_bin"].astype(int).to_numpy()
        po = float((y_h == y_l).mean())
        precision, recall, specificity, f1, tp, fp, fn, tn = _binary_precision_recall_f1(
            y_h, y_l, pos_label=1
        )
        rows.append(
            _metric_row(
                "Phase 2",
                "Usability_Non-Usability Type",
                len(vg),
                po,
                1 - po if pd.notna(po) else np.nan,
                _cohen_kappa_binary(y_h, y_l),
                _gwet_ac1_binary(y_h, y_l),
                phase12_subset=subset,
                precision=precision,
                recall=recall,
                specificity=specificity,
                f1=f1,
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
            )
        )

    phase2_cols = [
        "sample_id",
        "phase12_subset",
        "repo",
        "issue_number",
        "human_phase2_analyst_a",
        "llm_usability_type",
        "human_phase2_analyst_a_bin",
        "llm_phase2_bin",
    ]
    if "human_phase2_label_source" in v.columns:
        phase2_cols.append("human_phase2_label_source")
    phase2_detail = v[phase2_cols].copy()
    phase2_detail["agree"] = (phase2_detail["human_phase2_analyst_a_bin"] == phase2_detail["llm_phase2_bin"]).astype(int)

    # Phase 3: multi-label columns (full + partial + kappa/ac1)
    phase3_detail_frames = []
    cfg = [
        ("Associated Component Theme", "human_associated_component_theme", "llm_associated_component_theme", canonicalize_associated_component_theme),
        ("L1_Theme", "human_l1_theme", "llm_l1_theme", canonicalize_l1_theme),
        ("L1_Theme Secondary", "human_l1_theme_secondary", "llm_l1_theme_secondary", canonicalize_l1_secondary),
        ("Nielsen_theme", "human_nielsen_theme", "llm_nielsen_theme", canonicalize_nielsen),
    ]

    if "human_phase3_usability_seed" in phase35_eval.columns:
        phase35_u = phase35_eval[phase35_eval["human_phase3_usability_seed"] == True].copy()  # noqa: E712
    else:
        phase35_u = phase35_eval[
            phase35_eval["llm_usability_type"].fillna("").astype(str).str.lower().eq("usability")
        ].copy()

    for name, hcol, lcol, canon_fn in cfg:
        if canon_fn is canonicalize_l1_secondary:
            # Secondary labels are often omitted in human files but derivable from human L1 labels.
            p3 = phase35_u[
                (phase35_u[hcol].fillna("").astype(str).str.strip() != "")
                | (phase35_u["human_l1_theme"].fillna("").astype(str).str.strip() != "")
            ].copy()
        else:
            p3 = phase35_u[phase35_u[hcol].fillna("").astype(str).str.strip() != ""].copy()
        if p3.empty:
            continue
        if canon_fn is canonicalize_l1_secondary:
            hs = []
            ls = []
            for _, rr in p3.iterrows():
                h_l1 = canonicalize_l1_theme(rr.get("human_l1_theme", ""))
                l_l1 = canonicalize_l1_theme(rr.get("llm_l1_theme", ""))
                hs.append(set(canonicalize_l1_secondary(rr.get(hcol, ""), l1_fallback=h_l1)))
                ls.append(set(canonicalize_l1_secondary(rr.get(lcol, ""), l1_fallback=l_l1)))
        else:
            hs = [set(canon_fn(x)) for x in p3[hcol]]
            ls = [set(canon_fn(x)) for x in p3[lcol]]

        full3, partial3 = _sets_full_partial(hs, ls)
        universe = sorted(set().union(*hs).union(*ls))
        kappa3, ac13 = _multilabel_kappa_ac1(hs, ls, universe)
        n3 = len(p3)

        rows.append(
            _metric_row(
                "Phase 3",
                f"{name} (full)",
                n3,
                full3.mean(),
                1 - full3.mean(),
                kappa3,
                ac13,
                phase12_subset="phase3_sample",
            )
        )
        rows.append(
            _metric_row(
                "Phase 3",
                f"{name} (partial overlap)",
                n3,
                partial3.mean(),
                1 - partial3.mean(),
                kappa3,
                ac13,
                phase12_subset="phase3_sample",
            )
        )

        tmp = p3[["sample_id", "repo", "issue_number", hcol, lcol]].copy()
        tmp["metric"] = name
        tmp["phase12_subset"] = "phase3_sample"
        tmp["full_agree"] = full3
        tmp["partial_agree"] = partial3
        phase3_detail_frames.append(tmp)

    phase3_detail = (
        pd.concat(phase3_detail_frames, ignore_index=True)
        if phase3_detail_frames
        else pd.DataFrame(columns=["sample_id", "repo", "issue_number", "metric", "phase12_subset", "full_agree", "partial_agree"])
    )
    summary = pd.DataFrame(rows)
    return summary, phase1_detail, pd.concat([phase2_detail, phase3_detail], ignore_index=True, sort=False)


def _canonicalize_and_join_for_write(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    out["llm_associated_component_theme"] = out["llm_associated_component_theme"].fillna("").map(
        lambda s: ", ".join(canonicalize_associated_component_theme(s))
    )
    out["llm_l1_theme"] = out["llm_l1_theme"].fillna("").map(lambda s: ", ".join(canonicalize_l1_theme(s)))
    out["llm_l1_theme_secondary"] = out.apply(
        lambda r: ", ".join(canonicalize_l1_secondary(r.get("llm_l1_theme_secondary", ""), l1_fallback=canonicalize_l1_theme(r.get("llm_l1_theme", "")))),
        axis=1,
    )
    out["llm_nielsen_theme"] = out["llm_nielsen_theme"].fillna("").map(lambda s: ", ".join(canonicalize_nielsen(s)))
    return out


def write_updated_repo_copies(gh_folder: Path, predictions_df: pd.DataFrame, out_folder: Path) -> None:
    out_folder.mkdir(parents=True, exist_ok=True)
    pred = _canonicalize_and_join_for_write(predictions_df.copy())
    pred["repo"] = pred["repo"].astype(str)
    pred["issue_number"] = pd.to_numeric(pred["issue_number"], errors="coerce").astype("Int64")

    for p in sorted(gh_folder.glob("*.xlsx")):
        xls = pd.ExcelFile(p)
        sheet_frames = {s: xls.parse(s) for s in xls.sheet_names}
        if "issues" not in sheet_frames:
            continue

        issues_raw = sheet_frames["issues"].copy()
        fixed_header = False
        if len(issues_raw.columns) > 0 and all(str(c).startswith("Unnamed:") for c in issues_raw.columns):
            hdr = issues_raw.iloc[0].astype(str).tolist()
            issues = issues_raw.iloc[1:].copy()
            issues.columns = hdr
            fixed_header = True
        else:
            issues = issues_raw

        canon = _canon_issues_df(issues)
        m = canon[["repo", "issue_number"]].merge(pred, on=["repo", "issue_number"], how="left")

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

        # Some source sheets infer these columns as float64 when mostly empty.
        # Force object dtype so string labels can be assigned safely.
        for c in [col_ac, col_cp, col_ut, col_act, col_l1, col_l1s, col_n]:
            issues[c] = issues[c].astype("object")

        for i in range(min(len(issues), len(m))):
            if pd.isna(m.at[i, "llm_associated_component"]):
                continue
            ridx = issues.index[i]
            issues.at[ridx, col_ac] = m.at[i, "llm_associated_component"]
            issues.at[ridx, col_cp] = m.at[i, "llm_codes_primary"]
            issues.at[ridx, col_ut] = m.at[i, "llm_usability_type"]
            issues.at[ridx, col_act] = m.at[i, "llm_associated_component_theme"]
            issues.at[ridx, col_l1] = m.at[i, "llm_l1_theme"]
            issues.at[ridx, col_l1s] = m.at[i, "llm_l1_theme_secondary"]
            issues.at[ridx, col_n] = m.at[i, "llm_nielsen_theme"]

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run OpenAI API recoding for phase samples (phase 1/2 and phase 3 sets), "
            "then compute full + partial agreement and phase 2/3 kappa + Gwet AC1."
        )
    )
    parser.add_argument("--model", default=OPENAI_MODEL_DEFAULT)
    parser.add_argument("--base-url", default=OPENAI_BASE_URL_DEFAULT)
    parser.add_argument("--phase12-range", default=PHASE12_RANGE_DEFAULT)
    parser.add_argument("--phase35-range", default=PHASE35_RANGE_DEFAULT)
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Output directory for caches, predictions, and agreement files.",
    )
    parser.add_argument(
        "--master-prompt",
        default=str(MASTER_PROMPT_PATH),
        help="Optional master prompt text file. If present, used instead of separate specific/heuristics prompts.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-api", action="store_true", help="Only prepare extracted files; do not call API.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_excel(SAMPLE_PATH)
    sample["sample_id"] = range(1, len(sample) + 1)
    sample["repo"] = sample["repo"].astype(str)
    sample["issue_number"] = pd.to_numeric(sample["issue_number"], errors="coerce").astype("Int64")

    phase12_ids = _parse_ranges(args.phase12_range, max_value=len(sample))
    phase35_ids = _parse_ranges(args.phase35_range, max_value=len(sample))
    phase12_sel = sample[sample["sample_id"].isin(phase12_ids)].copy()
    phase35_sel = sample[sample["sample_id"].isin(phase35_ids)].copy()

    gh = load_issue_corpus(GH_ISSUES_DIR)
    human_all = _load_human_issue_labels(HUMAN_ISSUES_DIR)
    human_gold = build_human_gold_table(human_all)

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
    if "Usability_Non-Usability Type- Analyst A" in phase12.columns:
        phase12["human_phase2_analyst_a_raw"] = phase12["Usability_Non-Usability Type- Analyst A"].astype(str)
    else:
        phase12["human_phase2_analyst_a_raw"] = ""
    phase12["human_phase2_analyst_a_bin_raw"] = phase12["human_phase2_analyst_a_raw"].map(normalize_usability_binary)
    phase12["human_usability_type_bin"] = phase12["human_usability_type"].map(normalize_usability_binary)
    phase12["human_phase2_analyst_a_bin"] = np.where(
        phase12["human_phase2_analyst_a_bin_raw"].notna(),
        phase12["human_phase2_analyst_a_bin_raw"],
        phase12["human_usability_type_bin"],
    )
    phase12["human_phase2_label_source"] = np.where(
        phase12["human_phase2_analyst_a_bin_raw"].notna(),
        "sample_analyst_a",
        np.where(phase12["human_usability_type_bin"].notna(), "human_issue_file", "missing"),
    )
    phase12["human_phase2_analyst_a"] = np.where(
        phase12["human_phase2_label_source"] == "sample_analyst_a",
        phase12["human_phase2_analyst_a_raw"].map(_norm_ws),
        np.where(
            phase12["human_phase2_label_source"] == "human_issue_file",
            phase12["human_usability_type"].fillna("").astype(str).map(_norm_ws),
            "",
        ),
    )

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
                "human_usability_type",
                "human_associated_component",
                "human_associated_component_theme",
                "human_l1_theme",
                "human_l1_theme_secondary",
                "human_nielsen_theme",
            ]
        ],
        on=["repo", "issue_number"],
        how="left",
    )

    # Human-seeded usability filter for Phase 3 agreement.
    # Precedence:
    # 1) explicit analyst-A usability label in sample sheet (if present),
    # 2) explicit usability label from human issue files,
    # 3) inferred usability if human phase-3 thematic labels exist.
    if "Usability_Non-Usability Type- Analyst A" in phase35.columns:
        phase35["human_phase3_analyst_a"] = phase35["Usability_Non-Usability Type- Analyst A"].astype(str)
    else:
        phase35["human_phase3_analyst_a"] = ""
    phase35["human_phase3_analyst_a_bin"] = phase35["human_phase3_analyst_a"].map(normalize_usability_binary)
    phase35["human_usability_type_bin"] = phase35["human_usability_type"].map(normalize_usability_binary)
    phase35["human_phase3_theme_seed"] = phase35[
        ["human_l1_theme", "human_l1_theme_secondary", "human_nielsen_theme"]
    ].fillna("").astype(str).apply(
        lambda r: any(_norm_ws(v) != "" for v in r),
        axis=1,
    )
    phase35["human_phase3_usability_seed"] = np.where(
        phase35["human_phase3_analyst_a_bin"].notna(),
        phase35["human_phase3_analyst_a_bin"].eq(1.0),
        np.where(
            phase35["human_usability_type_bin"].notna(),
            phase35["human_usability_type_bin"].eq(1.0),
            phase35["human_phase3_theme_seed"],
        ),
    ).astype(bool)

    phase12.to_csv(out_dir / "phase12_extracted_with_human.csv", index=False)
    phase35.to_csv(out_dir / "phase35_extracted_with_human.csv", index=False)
    with pd.ExcelWriter(out_dir / "phase_samples_extracted_with_human.xlsx", engine="xlsxwriter") as wr:
        phase12.to_excel(wr, sheet_name="phase12_sample", index=False)
        phase35.to_excel(wr, sheet_name="phase35_sample", index=False)

    if args.skip_api:
        print(f"Extracted files written to: {out_dir}")
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

    pred12 = annotate_rows(
        phase12,
        client=client,
        system_prompt=system_prompt,
        cache_path=out_dir / "phase12_llm_predictions_cache.csv",
        overwrite=args.overwrite,
    )
    pred35 = annotate_rows(
        phase35,
        client=client,
        system_prompt=system_prompt,
        cache_path=out_dir / "phase35_llm_predictions_cache.csv",
        overwrite=args.overwrite,
    )

    phase12_eval = phase12.merge(pred12, on=["sample_id", "repo", "issue_number"], how="left")
    phase35_eval = phase35.merge(pred35, on=["sample_id", "repo", "issue_number"], how="left")
    phase12_eval["llm_phase2_bin"] = phase12_eval["llm_usability_type"].map(normalize_usability_binary)

    # Backfill Associated Component Theme from Associated Component when missing (human + llm).
    def _fill_h_act(row: pd.Series) -> str:
        cur = _norm_ws(row.get("human_associated_component_theme", ""))
        if cur:
            return ", ".join(canonicalize_associated_component_theme(cur))
        return ", ".join(derive_component_theme_from_component(row.get("human_associated_component", "")))

    phase35_eval["human_associated_component_theme"] = phase35_eval.apply(_fill_h_act, axis=1)
    phase35_eval["llm_associated_component_theme"] = phase35_eval.apply(
        lambda r: (
            ", ".join(canonicalize_associated_component_theme(r.get("llm_associated_component_theme", "")))
            if _norm_ws(r.get("llm_associated_component_theme", ""))
            else ", ".join(derive_component_theme_from_component(r.get("llm_associated_component", "")))
        ),
        axis=1,
    )

    code_primary_sem_df = judge_code_primary_semantic(
        phase12_eval,
        client=client,
        cache_path=out_dir / "phase1_code_primary_semantic_cache.csv",
        overwrite=args.overwrite,
    )
    summary, phase1_detail, phase23_detail = compute_agreements(
        phase12_eval,
        phase35_eval,
        code_primary_sem_df=code_primary_sem_df,
    )

    phase12_eval.to_csv(out_dir / "phase12_predictions_and_eval.csv", index=False)
    phase35_eval.to_csv(out_dir / "phase35_predictions_and_eval.csv", index=False)
    summary.to_csv(out_dir / "agreement_summary.csv", index=False)
    phase1_detail.to_csv(out_dir / "phase1_agreement_detail.csv", index=False)
    phase23_detail.to_csv(out_dir / "phase2_phase3_agreement_detail.csv", index=False)
    code_primary_sem_df.to_csv(out_dir / "phase1_code_primary_semantic_detail.csv", index=False)

    with pd.ExcelWriter(out_dir / "phase_predictions_and_eval.xlsx", engine="xlsxwriter") as wr:
        phase12_eval.to_excel(wr, sheet_name="phase12", index=False)
        phase35_eval.to_excel(wr, sheet_name="phase35", index=False)
        summary.to_excel(wr, sheet_name="agreement_summary", index=False)
        phase1_detail.to_excel(wr, sheet_name="phase1_detail", index=False)
        phase23_detail.to_excel(wr, sheet_name="phase2_3_detail", index=False)
        code_primary_sem_df.to_excel(wr, sheet_name="phase1_code_primary_sem", index=False)

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

    write_updated_repo_copies(
        gh_folder=GH_ISSUES_DIR,
        predictions_df=pred_all,
        out_folder=out_dir / "copied_gh_issues_with_openai_labels",
    )

    run_meta = {
        "model": args.model,
        "base_url": args.base_url,
        "phase12_range": args.phase12_range,
        "phase35_range": args.phase35_range,
        "n_phase12_selected": int(len(phase12)),
        "n_phase35_selected": int(len(phase35)),
        "n_phase35_human_usability_seed_true": int(phase35["human_phase3_usability_seed"].sum()),
        "timestamp_epoch": int(time.time()),
    }
    run_meta["out_dir"] = str(out_dir)
    run_meta["master_prompt"] = str(mp) if args.master_prompt else ""
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print("Wrote outputs to:", out_dir.resolve())
    print("Key files:")
    for p in [
        out_dir / "agreement_summary.csv",
        out_dir / "phase12_predictions_and_eval.csv",
        out_dir / "phase35_predictions_and_eval.csv",
        out_dir / "copied_gh_issues_with_openai_labels",
    ]:
        print(" -", p)


if __name__ == "__main__":
    main()
