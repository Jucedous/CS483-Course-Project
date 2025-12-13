'''
 - python tag_events_llm.py --input_dir ./events --output_dir ./tagged --bad_dir ./needs_review --model gpt-4o-mini --max_concurrency 8 --registry_path ./tag_registry.json --api_base http://localhost:8000/v1 --api_key sk-xxx
'''

import os
import re
import json
import asyncio
import argparse
import shutil
from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass

from tqdm.asyncio import tqdm as tqdm_async
from openai import AsyncOpenAI


def normalize_tag(tag: Any) -> str:
    """
    Normalize tags to short snake_case:
    - lowercased
    - spaces, slashes, hyphens -> underscores
    - remove non [a-z0-9_]
    - collapse multiple underscores and trim
    """
    if not isinstance(tag, str):
        return ""
    s = tag.lower()
    s = re.sub(r"[ \t\-\/]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s


class TagRegistry:
    """
    Open tag registry for message_tags and relation_tags.
    Loads an existing registry file if present; merges and saves new tags.
    """
    def __init__(self):
        self.message_tags: Set[str] = set()
        self.relation_tags: Set[str] = set()

    def load(self, path: Optional[str]):
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for t in data.get("message_tags", []):
                nt = normalize_tag(t)
                if nt:
                    self.message_tags.add(nt)
            for t in data.get("relation_tags", []):
                nt = normalize_tag(t)
                if nt:
                    self.relation_tags.add(nt)
        except Exception:
            # Silently ignore loading errors to avoid interrupting runs
            pass

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "message_tags": sorted(self.message_tags),
                    "relation_tags": sorted(self.relation_tags),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def add_message_tags(self, tags: List[str]):
        for t in tags:
            nt = normalize_tag(t)
            if nt:
                self.message_tags.add(nt)

    def add_relation_tags(self, tags: List[str]):
        for t in tags:
            nt = normalize_tag(t)
            if nt:
                self.relation_tags.add(nt)


# =========================
# LLM prompts (English)
# =========================

SYSTEM_PROMPT = (
    "You are an expert LKML discussion annotation assistant. "
    "Given an LKML event containing multiple messages, annotate each valid message with message_tags (open set, short snake_case), "
    "and extract inter-message relations with relation_tags (open set, short snake_case). "
    "Return strict JSON only. "
    "If a message has both subject and content as null or empty, skip it (no tags). "
    "Use source/target message indices for relations (indices follow the input order, starting at 0). "
    "Examples of concise reusable tags: patch_submission, review_comment, tested_by, requests_changes, confirms_on_platform, replies_to, references_patch, provides_test_for. "
    "However, the tag set is open and you may add new tags if helpful."
)

def build_user_prompt(event: Dict[str, Any]) -> str:
    """
    Build a concise user prompt with only necessary fields.
    """
    event_id = event.get("event_id", "")
    root_url = event.get("root_url", "")
    msgs = event.get("messages", [])

    lines = []
    lines.append(f"Event ID: {event_id}")
    lines.append(f"root_url: {root_url}")
    lines.append("Data: messages is an array in original order, index starts at 0. Each item provides:")
    lines.append("  {index, url, subject, content}")
    lines.append("Rules:")
    lines.append("1) Only output message_tags (open set) and relations.relation_tags (open set).")
    lines.append("2) Skip messages where subject AND content are both empty or null.")
    lines.append("3) Relations use fields: source_index, target_index, relation_tags (multiple allowed).")
    lines.append("4) Output strict JSON with this schema:")
    lines.append("""
{
  "event_id": "<same as input>",
  "skipped_indices": [<int>...],
  "messages": [
    {
      "index": <int>,
      "message_tags": ["..."]
    }
  ],
  "relations": [
    {
      "source_index": <int>,
      "target_index": <int>,
      "relation_tags": ["..."]
    }
  ]
}
""".strip())
    lines.append("Messages:")
    for i, m in enumerate(msgs):
        subject = m.get("subject", None)
        content = m.get("content", None)
        url = m.get("url", "")
        subj_show = subject if isinstance(subject, str) else "null"
        cont_show = content if isinstance(content, str) else "null"
        lines.append(f"index={i} | url={url}\nsubject={subj_show}\ncontent={cont_show}\n---")
    return "\n".join(lines)


# =========================
# IO helpers
# =========================

def safe_load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_dump_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def is_message_skippable(m: Dict[str, Any]) -> bool:
    subj = m.get("subject", None)
    cont = m.get("content", None)
    subj_empty = (subj is None) or (isinstance(subj, str) and subj.strip() == "")
    cont_empty = (cont is None) or (isinstance(cont, str) and cont.strip() == "")
    return subj_empty and cont_empty


# =========================
# LLM output validation/cleaning
# =========================

def _try_extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort fallback: extract the largest {...} block and parse as JSON.
    Useful for non-JSON-mode backends that still output JSON-like content.
    """
    if not isinstance(text, str):
        return None
    # Find first '{' and last '}' to get a likely JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        blob = text[start:end+1]
        try:
            return json.loads(blob)
        except Exception:
            return None
    return None

def validate_and_clean_llm_output(raw: Dict[str, Any], n_messages: int) -> Dict[str, Any]:
    """
    - Validate top-level structure
    - Normalize tags to snake_case
    - Validate index bounds
    - Keep only message_tags and relations (open sets)
    """
    if not isinstance(raw, dict):
        raise ValueError("LLM output is not a JSON object.")

    out = {
        "event_id": raw.get("event_id", None),
        "skipped_indices": [],
        "messages": [],
        "relations": []
    }

    # skipped_indices
    skipped = raw.get("skipped_indices", [])
    if isinstance(skipped, list):
        out["skipped_indices"] = [int(i) for i in skipped if isinstance(i, int)]

    # messages
    msgs = raw.get("messages", [])
    if not isinstance(msgs, list):
        raise ValueError("LLM output missing 'messages' list.")
    for item in msgs:
        if not isinstance(item, dict):
            continue
        idx = item.get("index", None)
        if not isinstance(idx, int) or idx < 0 or idx >= n_messages:
            continue
        raw_tags = item.get("message_tags", []) or []
        if not isinstance(raw_tags, list):
            raw_tags = []
        norm_tags = []
        for t in raw_tags:
            nt = normalize_tag(t)
            if nt:
                norm_tags.append(nt)
        norm_tags = sorted(set(norm_tags))
        out["messages"].append({
            "index": idx,
            "message_tags": norm_tags
        })

    # relations
    rels = raw.get("relations", [])
    if isinstance(rels, list):
        for r in rels:
            if not isinstance(r, dict):
                continue
            si = r.get("source_index", None)
            ti = r.get("target_index", None)
            if not isinstance(si, int) or not isinstance(ti, int):
                continue
            if si < 0 or si >= n_messages or ti < 0 or ti >= n_messages:
                continue
            raw_rtags = r.get("relation_tags", []) or []
            if not isinstance(raw_rtags, list):
                raw_rtags = []
            norm_rtags = []
            for t in raw_rtags:
                nt = normalize_tag(t)
                if nt:
                    norm_rtags.append(nt)
            norm_rtags = sorted(set(norm_rtags))
            if not norm_rtags:
                continue
            out["relations"].append({
                "source_index": si,
                "target_index": ti,
                "relation_tags": norm_rtags
            })
    return out


# =========================
# Per-event processing
# =========================

@dataclass
class EventResult:
    path: str
    ok: bool
    error: str = ""
    discovered_message_tags: Set[str] = None
    discovered_relation_tags: Set[str] = None


async def llm_call(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    use_json_mode: bool,
    timeout: int
) -> Dict[str, Any]:
    """
    Make one chat.completions call and parse JSON result.
    If JSON mode is off or backend ignores it, try a best-effort JSON extraction fallback.
    """
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    # await with timeout
    resp = await asyncio.wait_for(client.chat.completions.create(**kwargs), timeout=timeout)
    content = resp.choices[0].message.content

    # First try strict JSON parsing
    try:
        return json.loads(content)
    except Exception:
        pass

    # Fallback: attempt to extract a JSON block
    maybe = _try_extract_json_block(content)
    if maybe is not None:
        return maybe

    # Give up
    raise ValueError("Failed to parse LLM output as JSON.")


async def process_one_event(
    path: str,
    out_dir: str,
    bad_dir: str,
    client: AsyncOpenAI,
    model: str,
    retries: int = 1,
    timeout: int = 120,
    use_json_mode: bool = True
) -> EventResult:
    # 1) Load JSON
    try:
        event = safe_load_json(path)
    except Exception as e:
        os.makedirs(bad_dir, exist_ok=True)
        shutil.move(path, os.path.join(bad_dir, os.path.basename(path)))
        return EventResult(path=path, ok=False, error=f"JSON parse failed: {e}")

    if not isinstance(event, dict) or "messages" not in event or not isinstance(event["messages"], list):
        os.makedirs(bad_dir, exist_ok=True)
        shutil.move(path, os.path.join(bad_dir, os.path.basename(path)))
        return EventResult(path=path, ok=False, error="JSON structure missing 'messages' list")

    messages = event["messages"]
    n_messages = len(messages)

    # 2) Build prompt and call LLM
    prompt = build_user_prompt(event)

    last_err = ""
    llm_json = None
    attempt = 0

    while attempt <= retries:
        attempt += 1
        try:
            llm_raw = await llm_call(
                client=client,
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                use_json_mode=use_json_mode,
                timeout=timeout
            )
            llm_json = validate_and_clean_llm_output(llm_raw, n_messages)
            break
        except Exception as e:
            last_err = f"LLM call/parse failed: {e}"

    if llm_json is None:
        # Move to bad_dir for manual inspection
        os.makedirs(bad_dir, exist_ok=True)
        shutil.move(path, os.path.join(bad_dir, os.path.basename(path)))
        return EventResult(path=path, ok=False, error=last_err)

    # 3) Merge tags into event (only message_tags; keep skipped info)
    idx_to_tags = {m["index"]: m for m in llm_json["messages"]}
    discovered_message_tags: Set[str] = set()
    discovered_relation_tags: Set[str] = set()

    for i, m in enumerate(messages):
        if is_message_skippable(m):
            m["tags"] = {
                "skipped": True,
                "message_tags": []
            }
            continue
        tags = idx_to_tags.get(i, {"message_tags": []})
        mtags = tags.get("message_tags", [])
        m["tags"] = {
            "skipped": False,
            "message_tags": mtags
        }
        for t in mtags:
            discovered_message_tags.add(t)

    # Event-level relations
    event["relations"] = llm_json.get("relations", [])
    for r in event["relations"]:
        for t in r.get("relation_tags", []):
            discovered_relation_tags.add(t)

    event["tagged"] = True

    # 4) Write output
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(path))
    safe_dump_json(event, out_path)

    return EventResult(
        path=path,
        ok=True,
        discovered_message_tags=discovered_message_tags,
        discovered_relation_tags=discovered_relation_tags
    )


# =========================
# Main (parallel + progress)
# =========================

async def main_async(args):
    # Resolve API key
    api_key = args.api_key or os.getenv(args.api_key_env) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: No API key provided. Set --api_key or environment variable.")
    # Optional custom headers for non-OpenAI endpoints
    default_headers = None
    if args.api_headers:
        try:
            default_headers = json.loads(args.api_headers)
            if not isinstance(default_headers, dict):
                default_headers = None
        except Exception:
            default_headers = None

    # Build client (supports custom base_url and headers for OpenAI-compatible APIs)
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.api_base or os.getenv("OPENAI_BASE_URL") or None,
        default_headers=default_headers
    )

    # Open registry
    registry = TagRegistry()
    registry.load(args.registry_path)

    # Collect files
    files = []
    for name in os.listdir(args.input_dir):
        if name.lower().endswith(".json"):
            files.append(os.path.join(args.input_dir, name))
    files.sort()
    if not files:
        print("No JSON files found in input_dir.")
        return

    sem = asyncio.Semaphore(args.max_concurrency)
    results: List[EventResult] = []

    async def run_one(p):
        async with sem:
            return await process_one_event(
                path=p,
                out_dir=args.output_dir,
                bad_dir=args.bad_dir,
                client=client,
                model=args.model,
                retries=args.retries,
                timeout=args.timeout,
                use_json_mode=not args.no_json_mode
            )

    tasks = [run_one(p) for p in files]

    # Progress bar over async tasks
    async for task in tqdm_async.as_completed(tasks, total=len(tasks), desc="Tagging progress"):
        res = await task
        results.append(res)
        if res.ok:
            if res.discovered_message_tags:
                registry.add_message_tags(list(res.discovered_message_tags))
            if res.discovered_relation_tags:
                registry.add_relation_tags(list(res.discovered_relation_tags))

    # Summary
    ok = sum(1 for r in results if r.ok)
    bad = len(results) - ok
    print(f"Done. Succeeded: {ok}, Moved to needs review: {bad}")

    # Save registry
    registry.save(args.registry_path)

    # Error log
    errlog = [{"file": r.path, "error": r.error} for r in results if not r.ok]
    if errlog:
        os.makedirs(args.bad_dir, exist_ok=True)
        safe_dump_json(errlog, os.path.join(args.bad_dir, "errors.json"))


def main():
    parser = argparse.ArgumentParser(description="LKML event open-tag message and relation labeling (LLM parallel)")
    parser.add_argument("--input_dir", required=True, help="Folder of input event JSON files")
    parser.add_argument("--output_dir", required=True, help="Folder for labeled JSON files")
    parser.add_argument("--bad_dir", required=True, help="Folder for malformed/needs manual review files")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name (default gpt-4o-mini)")
    parser.add_argument("--max_concurrency", type=int, default=5, help="Parallel events (default 5)")
    parser.add_argument("--retries", type=int, default=1, help="Retries per event on LLM failure (default 1)")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout seconds per event call (default 120)")

    # Open tag registry path (accumulates across runs)
    parser.add_argument("--registry_path", default="./tag_registry.json", help="Path to open tag registry JSON (default ./tag_registry.json)")

    # API compatibility options
    parser.add_argument("--api_base", default=None, help="Custom API base URL for OpenAI-compatible servers (e.g., http://host:port/v1)")
    parser.add_argument("--api_key", default=None, help="API key value (overrides environment)")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY", help="Name of env var that holds the API key (default OPENAI_API_KEY)")
    parser.add_argument("--api_headers", default=None, help='Extra HTTP headers as JSON string (e.g., \'{"Authorization":"Bearer ...","X-API-KEY":"..."}\')')
    parser.add_argument("--no_json_mode", action="store_true", help="Disable response_format=json mode for non-compliant backends")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
