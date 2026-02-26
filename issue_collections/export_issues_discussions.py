#!/usr/bin/env python3
"""
Export GitHub Issues (open+closed) + Discussions (if enabled) + full threads,
and README (raw + summary) to one Excel workbook per repo.

Sheets:
- issues                : issues metadata + content + coding scaffold (item_type='issue')
- comments              : all issue comments
- discussions           : discussions metadata + content (item_type='discussion')
- discussion_comments   : all discussion comments (and replies flattened)
- readme                : README url, summary, raw text

Filters:
- Server uses 'since' (updated_at) where available; client filters by created_at
- --since-months N (default 24)
"""

import argparse
import base64
import os
import re
import time
import textwrap
import requests
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any
from typing import Optional, List
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

API = "https://api.github.com"
TOKEN = os.getenv("github_access_token")
SESSION = requests.Session()
SESSION.headers.update({
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
})
if TOKEN:
    SESSION.headers["Authorization"] = f"Bearer {TOKEN}"

BOT_PATTERNS = [
    r".*\[bot\]$",
    r"dependabot.*",
    r"renovate.*",
    r"github-actions",
]

# ------------------------- Helpers -------------------------

def is_bot(login: Optional[str], author_type: Optional[str]) -> bool:
    if not login:
        return False
    if author_type and author_type.lower() == "bot":
        return True
    ll = login.lower()
    return any(re.fullmatch(pat, ll) for pat in [re.compile(p) for p in BOT_PATTERNS])

def rate_limit_sleep(resp: requests.Response):
    if resp.status_code == 403:
        reset = resp.headers.get("X-RateLimit-Reset")
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            wait = int(retry_after)
        elif reset:
            wait = max(0, int(reset) - int(time.time())) + 5
        else:
            wait = 15
        tqdm.write(f"[rate-limit] sleeping {wait}s…")
        time.sleep(wait)

def paged_get(url, params=None):
    """
    Generator over paginated endpoints.
    Treat 404/410 as 'endpoint not available' and stop cleanly.
    """
    params = dict(params or {})
    params.setdefault("per_page", 100)
    while url:
        resp = SESSION.get(url, params=params, timeout=60)
        if resp.status_code == 403:
            rate_limit_sleep(resp)
            resp = SESSION.get(url, params=params, timeout=60)
        if resp.status_code in (404, 410):
            return
        resp.raise_for_status()
        data = resp.json()
        yield data
        link = resp.headers.get("Link", "")
        next_url = None
        if link:
            for part in [p.strip() for p in link.split(",")]:
                if 'rel="next"' in part:
                    next_url = part[part.find("<")+1:part.find(">")]
                    break
        url = next_url
        params = None

# def within_created_window(created_at: str, since_dt: datetime) -> bool:
#     try:
#         dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
#         return dt >= since_dt
#     except Exception:
#         return True
def within_created_window(created_at: str, start_dt, end_dt) -> bool:
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        return (dt >= start_dt) and (dt <= end_dt)
    except Exception:
        return False


def normalize_list(items):
    return ", ".join(sorted(set(items))) if items else ""

def extract_code_blocks(text: str) -> list:
    if not text:
        return []
    blocks = re.findall(r"```(?:[^\n]*\n)?(.*?)```", text, flags=re.DOTALL)
    indented = re.findall(r"(?:^|\n)(?: {4}|\t)(.+)", text)
    return [b.strip() for b in blocks + indented]

def extract_pr_links(text: str) -> list:
    if not text:
        return []
    pattern = r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/pulls?/\d+"
    return re.findall(pattern, text)

# put near concat_top3_nonbot
def concat_all_nonbot(comments, max_chars=20000):
    """Join all non-bot comments as 'author: text' lines. Truncate to max_chars."""
    parts = []
    for c in comments:
        # normalize author
        author = None
        if isinstance(c.get("user"), dict) and c["user"].get("login"):
            author = c["user"]["login"]
            a_type = c["user"].get("type")
        elif isinstance(c.get("author"), dict) and c["author"].get("login"):
            author = c["author"]["login"]
            a_type = (c.get("author") or {}).get("type")
        else:
            author = c.get("author_login")
            a_type = None

        if is_bot(author, a_type):
            continue

        body = (c.get("body") or c.get("body_text") or "").strip()
        if not body:
            continue
        body = re.sub(r"\s+", " ", body)
        parts.append(f"{author}: {body}")

    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + " ...[truncated]"
    return text


def concat_top3_nonbot(comments):
    txts = []
    for c in comments:
        # Normalize author fields from issues or discussions
        author = None
        if isinstance(c.get("user"), dict) and c["user"].get("login"):
            author = c["user"]["login"]
        elif isinstance(c.get("author"), dict) and c["author"].get("login"):
            author = c["author"]["login"]
        elif c.get("author_login"):
            author = c["author_login"]

        a_type = None
        if isinstance(c.get("user"), dict):
            a_type = c["user"].get("type")

        if is_bot(author, a_type):
            continue

        body = (c.get("body") or c.get("body_text") or "").strip()
        body = re.sub(r"\s+", " ", body)
        if body and author:
            txts.append(f"{author}: {body}")
        if len(txts) == 3:
            break
    return " | ".join(txts)

# -------------------- Issues + comments --------------------

def get_issues(owner, repo, since_iso):
    url = f"{API}/repos/{owner}/{repo}/issues"
    params = {"state": "all", "since": since_iso, "direction": "desc", "per_page": 100}
    for page in paged_get(url, params=params):
        for it in page:
            if "pull_request" in it:
                continue
            yield it

def get_issue_comments(owner, repo, number):
    url = f"{API}/repos/{owner}/{repo}/issues/{number}/comments"
    for page in paged_get(url):
        for c in page:
            yield c

# -------------------- Discussions + comments --------------------

def get_discussions(owner, repo, since_iso):
    url = f"{API}/repos/{owner}/{repo}/discussions"
    params = {"since": since_iso, "direction": "desc", "per_page": 100}
    for page in paged_get(url, params=params):
        for d in page:
            yield d

def get_discussion_comments(owner, repo, discussion_number):
    url = f"{API}/repos/{owner}/{repo}/discussions/{discussion_number}/comments"
    for page in paged_get(url):
        for c in page:
            yield c
            # Replies nested; flatten
            replies_url = c.get("replies_url")
            if replies_url:
                for rp in paged_get(replies_url):
                    for r in rp:
                        child = dict(r)
                        child["_parent_comment_id"] = c.get("id")
                        yield child

# -------------------- Timeline-based PR linking --------------------

def get_issue_timeline(owner: str, repo: str, issue_number: int):
    """
    Walk the issue timeline and return raw events (handles pagination + 404/410).
    """
    url = f"{API}/repos/{owner}/{repo}/issues/{issue_number}/timeline"
    for page in paged_get(url, params={"per_page": 100}):
        for ev in page:
            yield ev

def extract_linked_prs_from_timeline(owner: str, repo: str, issue_number: int) -> Tuple[List[str], Optional[str]]:
    """
    Returns (all_pr_urls, closing_pr_url). Uses:
      - event 'cross-referenced' with 'source' pointing to a PR
      - event 'connected' where 'source' is PR
      - for 'cross-referenced', if 'will_close_target' is true -> closing_pr_url
    Falls back gracefully if timeline not available.
    """
    pr_urls: List[str] = []
    closing: Optional[str] = None

    for ev in get_issue_timeline(owner, repo, issue_number) or []:
        etype = ev.get("event", "")
        # Many cross-referenced events include:
        #   "source": {"type": "issue", "issue": {...}} or {"type":"pull_request","issue":{...}}
        # In practice, "source.issue.pull_request" exists if it's a PR.
        source = ev.get("source") or {}
        source_issue = source.get("issue") or {}
        source_is_pr = bool(source_issue.get("pull_request"))
        source_url = source_issue.get("html_url")

        if etype in ("cross-referenced", "connected") and source_is_pr and source_url:
            pr_urls.append(source_url)
            # Some payloads carry a flag 'will_close_target' (or 'will_close' historically)
            will_close = ev.get("will_close_target")
            if will_close is None:
                # some older payloads used 'will_close'
                will_close = ev.get("will_close")
            if will_close and not closing:
                closing = source_url

        # Some 'closed' events reference 'commit_id' not PR; we ignore for PR linking.

    pr_urls = sorted(set(pr_urls))
    return pr_urls, closing

# -------------------- README fetch + summarize --------------------

def fetch_readme(owner, repo):
    url = f"{API}/repos/{owner}/{repo}/readme"
    resp = SESSION.get(url, timeout=60)
    if resp.status_code == 403:
        rate_limit_sleep(resp)
        resp = SESSION.get(url, timeout=60)
    if resp.status_code == 404:
        return None, ""
    resp.raise_for_status()
    j = resp.json()
    download_url = j.get("html_url")
    content = j.get("content")
    encoding = j.get("encoding")
    if content and encoding == "base64":
        try:
            raw = base64.b64decode(content).decode("utf-8", errors="replace")
        except Exception:
            raw = ""
    else:
        raw = ""
    return download_url, raw

def summarize_text(text: str, max_sentences: int = 8) -> str:
    if not text:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', textwrap.dedent(text).strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return ""
    stop = set("""
        a an the and or but if then else when while for on in to from of with by as at
        is are was were be been being have has had do does did can could should would
        this that these those it its into about over under up down not no yes you your
    """.split())
    def tok(s): return [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", s) if len(t) > 2 and t.lower() not in stop]
    tf, sent_tokens = {}, []
    for s in sentences:
        toks = tok(s); sent_tokens.append(toks)
        for t in toks: tf[t] = tf.get(t, 0) + 1
    scores = [(i, sum(tf.get(t,0) for t in toks)/(len(toks) or 1)) for i, toks in enumerate(sent_tokens)]
    k = min(max_sentences, max(1, int(len(sentences)*0.25)))
    top_idx = {i for i,_ in sorted(scores, key=lambda x: x[1], reverse=True)[:k]}
    summary = " ".join(sentences[i] for i in range(len(sentences)) if i in top_idx)
    return summary[:2000]

# -------------------- Export per repo --------------------

def export_repo(owner_repo: str, since_months: int, outdir: str, max_issues: Optional[int],
                anchor_date_str: Optional[str] = None, months_back_override: Optional[int] = None):

    owner, repo = owner_repo.split("/", 1)
    os.makedirs(outdir, exist_ok=True)

    # since_dt = datetime.now(timezone.utc) - relativedelta(months=since_months)
    # since_iso = since_dt.isoformat()

    # tqdm.write(f"\n=== {owner_repo} | window since {since_dt.date()} ===")
    if anchor_date_str:
        from datetime import datetime, timezone
        from dateutil.relativedelta import relativedelta
        try:
            end_dt = datetime.strptime(anchor_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = end_dt + relativedelta(days=1, seconds=-1)  # include full anchor day
        except ValueError:
            raise ValueError("--anchor-date must be YYYY-MM-DD (e.g., 2025-11-05)")
    else:
        end_dt = datetime.now(timezone.utc)

    months_back = months_back_override if (months_back_override is not None) else since_months
    start_dt = end_dt - relativedelta(months=months_back)

    since_iso = start_dt.isoformat()  # API lower bound (updated_at); we still post-filter by created_at

    tqdm.write(f"\n=== {owner_repo} | window {start_dt.date()} → {end_dt.date()} ===")

    

    # -------- Issues
    raw_issues = list(get_issues(owner, repo, since_iso))
    # issues = [i for i in raw_issues if within_created_window(i.get("created_at",""), since_dt)]
    issues = [i for i in raw_issues if within_created_window(i.get("created_at",""), start_dt, end_dt)]

    if max_issues:
        issues = issues[:max_issues]

    issues_rows, comments_rows = [], []
    for it in tqdm(issues, desc=f"{owner_repo}: issues", unit="issue"):
        num = it["number"]
        url = it.get("html_url")
        title = it.get("title") or ""
        body = it.get("body") or ""
        author = (it.get("user") or {}).get("login")
        author_type = (it.get("user") or {}).get("type")
        created_at = it.get("created_at")
        closed_at = it.get("closed_at")
        labels = [l.get("name","") for l in it.get("labels", []) if isinstance(l, dict)]
        assignees = [(a or {}).get("login","") for a in (it.get("assignees") or [])]

        all_comments = list(get_issue_comments(owner, repo, num))
        for c in all_comments:
            comments_rows.append({
                "repo": owner_repo,
                "issue_number": num,
                "comment_id": c.get("id"),
                "comment_url": c.get("html_url"),
                "created_at": c.get("created_at"),
                "author": (c.get("user") or {}).get("login"),
                "author_type": (c.get("user") or {}).get("type"),
                "body_text": c.get("body") or "",
            })

        top3 = concat_top3_nonbot(all_comments)
        all_txt = concat_all_nonbot(all_comments)  # NEW
        code_blocks = extract_code_blocks(body) + sum((extract_code_blocks(c.get("body") or "") for c in all_comments), [])
        # Heuristic PR links from text (keep as backup)
        pr_links_text = set(extract_pr_links(body))
        for c in all_comments:
            pr_links_text.update(extract_pr_links(c.get("body") or ""))

        # Timeline-based PRs (authoritative)
        pr_urls_tl, closing_pr = extract_linked_prs_from_timeline(owner, repo, num)

        issues_rows.append({
            "item_type": "issue",
            "repo": owner_repo,
            "issue_number": num,
            "issue_url": url,
            "created_at": created_at,
            "closed_at": closed_at,
            "author": author,
            "labels": normalize_list(labels),
            "assignees": normalize_list(assignees),
            "title": title,
            "body_text": body,
            "top_3_comments_text": top3,
            "all_comments_text": all_txt,          # NEW
            # keep both:
            "linked_prs": normalize_list(list(pr_links_text)),           # from text (legacy)
            "linked_prs_timeline": normalize_list(pr_urls_tl),           # from timeline (authoritative)
            "closing_pr_url": closing_pr or "",
            "error_msgs/logs": "\n\n---\n\n".join(code_blocks[:10]),
            # coding scaffold placeholders
            "codes_primary": "",
            "codes_secondary": "",
            "agreement": "",
            "disagreement_notes": "",
            "theme_secondary": "",
            "sscfm_point": "",
            "challenge_type": "",
            "importance_view": "",
            "memo_secondary": "",
        })

    # -------- Discussions
    discussions_rows, discussion_comments_rows = [], []
    raw_discussions = list(get_discussions(owner, repo, since_iso)) or []
    # discussions = [d for d in raw_discussions if within_created_window(d.get("created_at",""), since_dt)]
    discussions = [d for d in raw_discussions if within_created_window(d.get("created_at",""), start_dt, end_dt)]

    for d in tqdm(discussions, desc=f"{owner_repo}: discussions", unit="disc"):
        dnum = d.get("number")
        url = d.get("html_url")
        title = d.get("title") or ""
        body = d.get("body") or ""
        author = (d.get("user") or {}).get("login") or (d.get("author") or {}).get("login")
        created_at = d.get("created_at")
        state = d.get("state") or ""
        category = (d.get("category") or {}).get("slug") or (d.get("category") or {}).get("name") or ""

        all_dcom = list(get_discussion_comments(owner, repo, dnum)) or []
        for c in all_dcom:
            au = c.get("user") or c.get("author") or {}
            author_login = (au.get("login") if isinstance(au, dict) else au) or c.get("author_login")
            discussion_comments_rows.append({
                "repo": owner_repo,
                "discussion_number": dnum,
                "comment_id": c.get("id"),
                "parent_comment_id": c.get("_parent_comment_id") if "_parent_comment_id" in c else "",
                "comment_url": c.get("html_url"),
                "created_at": c.get("created_at"),
                "author": author_login,
                "body_text": c.get("body") or "",
            })

        top3 = concat_top3_nonbot(all_dcom)
        code_blocks = extract_code_blocks(body) + sum((extract_code_blocks(c.get("body") or "") for c in all_dcom), [])
        pr_links = set(extract_pr_links(body))
        for c in all_dcom:
            pr_links.update(extract_pr_links(c.get("body") or ""))

        discussions_rows.append({
            "item_type": "discussion",
            "repo": owner_repo,
            "discussion_number": dnum,
            "discussion_url": url,
            "created_at": created_at,
            "state": state,
            "category": category,
            "author": author,
            "title": title,
            "body_text": body,
            "top_3_comments_text": top3,
            "linked_prs": normalize_list(list(pr_links)),
            "error_msgs/logs": "\n\n---\n\n".join(code_blocks[:10]),
        })

    # -------- README
    readme_url, readme_raw = fetch_readme(owner, repo)
    readme_summary = summarize_text(readme_raw, max_sentences=8)
    readme_df = pd.DataFrame([{
        "repo": owner_repo,
        "readme_url": readme_url or "",
        "readme_summary": readme_summary,
        "readme_raw": readme_raw,
    }])

    # -------- DataFrames with predefined columns
    issues_cols = ["item_type","repo","issue_number","issue_url","created_at","closed_at",
                   "author","labels","assignees","title","body_text","top_3_comments_text","all_comments_text",
                   "linked_prs","linked_prs_timeline","closing_pr_url","error_msgs/logs",
                   "codes_primary","codes_secondary","agreement","disagreement_notes",
                   "theme_secondary","sscfm_point","challenge_type","importance_view","memo_secondary"]
    comments_cols = ["repo","issue_number","comment_id","comment_url","created_at","author",
                     "author_type","body_text"]
    discussions_cols = ["item_type","repo","discussion_number","discussion_url","created_at",
                        "state","category","author","title","body_text","top_3_comments_text",
                        "linked_prs","error_msgs/logs"]
    discussion_comments_cols = ["repo","discussion_number","comment_id","parent_comment_id",
                                "comment_url","created_at","author","body_text"]

    issues_df = pd.DataFrame(issues_rows, columns=issues_cols)
    comments_df = pd.DataFrame(comments_rows, columns=comments_cols)
    discussions_df = pd.DataFrame(discussions_rows, columns=discussions_cols)
    discussion_comments_df = pd.DataFrame(discussion_comments_rows, columns=discussion_comments_cols)

    if not issues_df.empty:
        issues_df = issues_df.sort_values(["created_at","issue_number"], ascending=[False, False])
    if not comments_df.empty:
        comments_df = comments_df.sort_values(["issue_number","created_at"], ascending=[True, True])
    if not discussions_df.empty:
        discussions_df = discussions_df.sort_values(["created_at","discussion_number"], ascending=[False, False])
    if not discussion_comments_df.empty:
        discussion_comments_df = discussion_comments_df.sort_values(["discussion_number","created_at"], ascending=[True, True])

    # -------- Write Excel
    safe = owner_repo.replace("/", "__")
    path = os.path.join(outdir, f"{safe}.xlsx")
    os.makedirs(outdir, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as wr:
        wr.book.strings_to_urls = False  # disable auto URL conversion for older combos
        issues_df.to_excel(wr, index=False, sheet_name="issues")
        comments_df.to_excel(wr, index=False, sheet_name="comments")
        discussions_df.to_excel(wr, index=False, sheet_name="discussions")
        discussion_comments_df.to_excel(wr, index=False, sheet_name="discussion_comments")
        readme_df.to_excel(wr, index=False, sheet_name="readme")

        ws_i = wr.sheets["issues"]; ws_c = wr.sheets["comments"]
        ws_d = wr.sheets["discussions"]; ws_dc = wr.sheets["discussion_comments"]
        ws_r = wr.sheets["readme"]
        for ws in (ws_i, ws_c, ws_d, ws_dc, ws_r):
            ws.freeze_panes(1, 1)

        # Column widths
        # ws_i.set_column("A:A", 12); ws_i.set_column("B:B", 22); ws_i.set_column("C:C", 12)
        # ws_i.set_column("D:D", 45); ws_i.set_column("E:F", 20); ws_i.set_column("G:I", 22)
        # ws_i.set_column("J:J", 60); ws_i.set_column("K:K", 90); ws_i.set_column("L:M", 90)
        # ws_i.set_column("N:N", 45)  # closing_pr_url
        # ws_i.set_column("O:O", 70)  # error logs
        # ws_i.set_column("P:U", 22)  # coding scaffold
        # Column widths (issues)
        ws_i.set_column("A:A", 12)   # item_type
        ws_i.set_column("B:B", 22)   # repo
        ws_i.set_column("C:C", 12)   # issue_number
        ws_i.set_column("D:D", 45)   # issue_url
        ws_i.set_column("E:F", 20)   # created_at, closed_at
        ws_i.set_column("G:I", 22)   # author, labels, assignees
        ws_i.set_column("J:J", 60)   # title
        ws_i.set_column("K:K", 90)   # body_text
        ws_i.set_column("L:L", 90)   # top_3_comments_text
        ws_i.set_column("M:M", 120)  # all_comments_text  (NEW, usually longer)
        ws_i.set_column("N:O", 90)   # linked_prs, linked_prs_timeline
        ws_i.set_column("P:P", 45)   # closing_pr_url
        ws_i.set_column("Q:Q", 70)   # error logs
        ws_i.set_column("R:Z", 22)   # coding scaffold: codes_* ... memo_secondary


        ws_c.set_column("A:C", 18); ws_c.set_column("D:D", 45); ws_c.set_column("E:F", 22)
        ws_c.set_column("G:G", 18); ws_c.set_column("H:H", 100)

        ws_d.set_column("A:A", 12); ws_d.set_column("B:B", 22); ws_d.set_column("C:C", 16)
        ws_d.set_column("D:D", 45); ws_d.set_column("E:E", 20); ws_d.set_column("F:G", 18)
        ws_d.set_column("H:H", 22); ws_d.set_column("I:I", 60); ws_d.set_column("J:J", 90)
        ws_d.set_column("K:K", 90); ws_d.set_column("L:L", 70)

        ws_dc.set_column("A:C", 18); ws_dc.set_column("D:D", 18); ws_dc.set_column("E:E", 45)
        ws_dc.set_column("F:G", 22); ws_dc.set_column("H:H", 100)

        ws_r.set_column("A:B", 45); ws_r.set_column("C:C", 100); ws_r.set_column("D:D", 120)

    print(f"✓ {owner_repo}: {len(issues_df)} issues ({(issues_df['closing_pr_url']!='').sum()} with closing PR), "
          f"{len(comments_df)} issue-comments, {len(discussions_df)} discussions, {len(discussion_comments_df)} discussion-comments → {path}")

# -------------------- IO + CLI --------------------

def read_repos_from_csv(path: str) -> List[str]:
    df = pd.read_csv(path)
    if "full_name" not in df.columns:
        raise ValueError(f"{path} must have a 'full_name' column (e.g., sigstore/cosign).")
    return [r for r in df["full_name"].dropna().astype(str).tolist() if "/" in r]

def read_repos_from_xlsx(path: str) -> List[str]:
    df = pd.read_excel(path)
    if "full_name" not in df.columns:
        raise ValueError(f"{path} must have a 'full_name' column (e.g., sigstore/cosign).")
    return [r for r in df["full_name"].dropna().astype(str).tolist() if "/" in r]

def parse_args():
    p = argparse.ArgumentParser(description="Export GitHub Issues + Discussions + README + linked PRs to XLSX per repo")
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--repos-csv", help="CSV with 'full_name' column")
    src.add_argument("--repos-xlsx", help="XLSX with 'full_name' column")
    p.add_argument("--repo", action="append", help="owner/name (repeatable)")
    p.add_argument("--since-months", type=int, default=24, help="months back (default: 24)")
    p.add_argument("--max-issues", type=int, default=None, help="limit per repo (debug)")
    p.add_argument("--outdir", default="gh_issue_exports", help="output directory")
    p.add_argument("--anchor-date", type=str, default=None,
               help="End of window in YYYY-MM-DD (UTC). If omitted, uses now.")
    p.add_argument("--months-back", type=int, default=None,
               help="Months to look back from anchor-date. If omitted, uses --since-months.")

    return p.parse_args()

def main():
    args = parse_args()

    repos: List[str] = []
    if args.repos_csv:
        repos = read_repos_from_csv(args.repos_csv)
    elif args.repos_xlsx:
        repos = read_repos_from_xlsx(args.repos_xlsx)
    if args.repo:
        repos.extend(args.repo)

    if not repos:
        repos = [
            "notaryproject/notation",
            "sigstore/cosign",
            "sigstore/rekor",
            "sigstore/fulcio",
            "Keyfactor/ejbca-ce",
            "Keyfactor/signserver-ce",
            "openpubkey/openpubkey",
            "hashicorp/vault",
        ]

    seen = set()
    repos = [r for r in repos if not (r in seen or seen.add(r))]

    if not TOKEN:
        print("WARNING: github_access_token not set—expect heavy rate limiting.")

    for r in repos:
        try:
            # export_repo(r, args.since_months, args.outdir, args.max_issues)
            export_repo(r, args.since_months, args.outdir, args.max_issues,
            anchor_date_str=args.anchor_date, months_back_override=args.months_back)
        except requests.HTTPError as e:
            print(f"[HTTPError] {r}: {e}")
        except Exception as e:
            print(f"[ERROR] {r}: {e}")

if __name__ == "__main__":
    main()
