#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import sys
from pathlib import Path
import threading
import time
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tqdm import tqdm

from .step1_eval import (
    effective_step1_chunk_size,
    image_chunk_index_ranges,
    run_step1_window_ocr_with_fallback,
)
from .step2_eval import run_step2_claim_evidence_mapping
from .step3_eval import normalize_step3_judgement, run_step3_global_rubric_judgement

from reports.html_statistic import (
    calculate_batch_statistics,
    missing_judge_cache_payload,
    save_batch_html_report,
    save_per_task_type_reports,
)

DEFAULT_QWEN_PORT = 8005
DEFAULT_QWEN_API_BASE = f"http://127.0.0.1:{DEFAULT_QWEN_PORT}/v1"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"

_RETRYABLE_CHAT_ERRORS: tuple[type[Exception], ...] = (
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    RuntimeError,  
    json.JSONDecodeError,  # malformed JSON from the judge model
)

_LLM_CALL_COUNTER = 0
_LLM_CALL_LOCK = threading.Lock()


def _next_llm_call_id() -> int:
    global _LLM_CALL_COUNTER
    with _LLM_CALL_LOCK:
        _LLM_CALL_COUNTER += 1
        return _LLM_CALL_COUNTER


def _clip_debug_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 24] + "\n...(truncated)..."


def api_base_for_port(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1"


def resolve_api_base(*, port: int, api_base: str | None) -> str:
    if api_base:
        return api_base
    env_base = os.environ.get("QWEN_API_BASE")
    if env_base:
        return env_base
    return api_base_for_port(port)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS_JSON = _PROJECT_ROOT / "task_gen" / "outputs" / "final" / "possible_translate_ko_metadata.json"

# --- Rubric profiles: primary profile per episode (by task_type / id). ---
_RUBRIC_DIMENSIONS: dict[str, list[tuple[str, str]]] = {
    "common": [
        (
            "intent_grounding",
            "Intent Grounding — Did the agent correctly interpret the user's request and core goal?",
        ),
        (
            "evidence_grounding",
            "Evidence Grounding — (1) Was the selected or recommended target, such as a product, recipe, article, or item, "
            "actually visible in the merged evidence through screen or text evidence? If the target is insufficiently verified "
            "or not evidenced, this dimension fails. "
            "(2) Was the reason for selecting that target grounded in site-observed facts, such as price, specifications, filters, "
            "availability, description, ingredients, quantities, rankings, reviews, or visible attributes? "
            "If the reason relies only on general knowledge, assumptions, or plausible inference, this dimension fails. "
            "(Hard check: concrete values such as Korean-won prices, bold product names, quantities, ingredients, and named items "
            "must match the Step 1 evidence strings in screenshot_ocr or visual_elements.)",
        ),
        (
            "response_completion",
            "Response Completion — Did the final answer fully complete the user's request?",
        ),
    ],
    "semantic_grounding": [
        (
            "ui_label_mapping",
            "UI Label Mapping — Did the agent correctly map the user's expression to the intended UI meaning, "
            "such as labels, menus, filters, tags, sorting options, or site-specific categories? "
            "(Refer to task_spec.user_expression and expected_grounding.)",
        ),
    ],
    "contextual_inference": [
        (
            "context_to_requirement",
            "Context-to-Requirement Extraction — Did the agent correctly extract the actionable requirements "
            "needed to complete the task from the user's situational context?",
        ),
        (
            "implicit_need_validity",
            "Implicit Need Validity — Were the inferred implicit requirements reasonable and necessary given the user's situation?",
        ),
        (
            "priority_tradeoff",
            "Priority & Tradeoff Handling — When not all conditions could be perfectly satisfied, did the agent prioritize "
            "the core requirements and make a reasonable substitution or tradeoff?",
        ),
    ],
    "visual_grounding": [
        (
            "visual_intent_grounding",
            "Visual Intent Grounding — Did the agent interpret the user's references to location, icons, badges, colors, "
            "images, or other visual cues as concrete visual conditions to look for?",
        ),
        (
            "visual_target_disambiguation",
            "Visual Target Disambiguation — Did the agent select the correct target associated with the visual cue described by the user?",
        ),
    ]
}


def _combined_rubric_dimension_tuples(profile: str) -> list[tuple[str, str]]:
    """Always include the three common dimensions; append profile-specific rows.

    If a task profile reuses the same dimension id as common, merge the descriptions into a single row for that id.
    """
    common = _RUBRIC_DIMENSIONS["common"]
    common_ids = {did for did, _ in common}
    if profile == "common":
        return list(common)
    spec = _RUBRIC_DIMENSIONS.get(profile) or []
    out: list[tuple[str, str]] = [(did, desc) for did, desc in common]
    by_id: dict[str, int] = {did: i for i, (did, _) in enumerate(out)}
    for did, desc in spec:
        if did in common_ids:
            i = by_id[did]
            prev_d, prev_desc = out[i]
            out[i] = (prev_d, prev_desc + " — 추가 관점: " + desc)
        else:
            out.append((did, desc))
            by_id[did] = len(out) - 1
    return out


def _eval_checkpoints_from_task_spec(task_spec: dict[str, Any]) -> list[dict[str, str]]:
    """Task metadata: human ``evidence_checks`` strings → ``ec_0``… ids; plus ``checkpoints`` / ``eval_checkpoints``."""
    if not isinstance(task_spec, dict):
        return []
    out: list[dict[str, str]] = []

    ev = task_spec.get("evidence_checks")
    if isinstance(ev, list):
        for i, x in enumerate(ev):
            if isinstance(x, str) and x.strip():
                out.append({"id": f"ec_{i}", "description": x.strip()})

    raw = task_spec.get("checkpoints") or task_spec.get("eval_checkpoints")
    if not isinstance(raw, list):
        return out
    for x in raw:
        if isinstance(x, str) and x.strip():
            out.append({"id": x.strip(), "description": ""})
        elif isinstance(x, dict) and x.get("id"):
            out.append(
                {
                    "id": str(x["id"]).strip(),
                    "description": str(x.get("description") or ""),
                }
            )
    return out


def _rubric_dimension_ids(profile: str) -> list[str]:
    return [did for did, _ in _combined_rubric_dimension_tuples(profile)]


def _rubric_dimensions_structured(profile: str) -> list[dict[str, str]]:
    return [
        {"id": did, "description": desc} for did, desc in _combined_rubric_dimension_tuples(profile)
    ]


def _format_rubric_dimension_lines(profile: str) -> str:
    dims = _combined_rubric_dimension_tuples(profile)
    if profile != "common":
        header = (
            "Evaluate **every** dimension below. The first three (`intent_grounding` … "
            "`response_completion`) are **공통** and apply to all tasks; remaining rows "
            f"are specific to profile `{profile}`.\n\n"
        )
    else:
        header = ""
    body = "\n".join(
        f"{i}. `{did}` — {desc} (**pass** or **fail** only; output boolean in `dimension_scores`)"
        for i, (did, desc) in enumerate(dims, 1)
    )
    return header + body


def _rubric_profile_for_task(task_type: str | None, task_id: str | None) -> str:
    """Pick primary rubric bundle for task-specific rows. Common three dimensions are always merged in Step 3."""
    tt_compact = (task_type or "").lower().replace(" ", "").replace("-", "").replace("_", "")
    tid = ((task_id or "").lower().replace("-", "_"))
    # Dead-end: task id *_deadend_* / *_deadend or task_type like "Dead-end Detection"
    if "deadend" in tid or "deadend" in tt_compact:
        return "dead_end"
    if "visual" in tt_compact:
        return "visual_grounding"
    if "contextual" in tt_compact:
        return "contextual_inference"
    # Semantic Grounding; Graceful Fallback overlaps with UI-mapping/fallback quality, so it reuses the same rubric
    if "semantic" in tt_compact or "graceful" in tt_compact:
        return "semantic_grounding"
    return "common"


def _task_spec_from_metadata_row(task: dict[str, Any] | None) -> dict[str, Any]:
    if not task:
        return {}
    keys = ("user_intent", "evidence_checks", "task_type", "id", "start_url")
    return {k: task[k] for k in keys if k in task}


_STEP1_DROP_TOP_KEYS = frozenset(
    {
        "final_answer",
        "eval_checkpoints",
        "actions",
        "visited_urls",
        "rubric_dimensions",
        "rubric_dimensions_structured",
        "rubric_dimension_ids",
        "constraints",
    }
)
_STEP1_DROP_TASK_SPEC_KEYS = frozenset(
    {
        "evidence_checks",
        "checkpoints",
        "eval_checkpoints",
    }
)


def _build_step1_common_payload(
    common_payload: dict[str, Any],
    task_spec: dict[str, Any],
    *,
    shot_rows_used: list[dict[str, Any]],
) -> dict[str, Any]:
    """Step 1 input: no final_answer, no human checkpoints (Step 2 maps those)."""
    step1 = {k: v for k, v in common_payload.items() if k not in _STEP1_DROP_TOP_KEYS}
    step1["task_spec"] = {
        k: v for k, v in task_spec.items() if k not in _STEP1_DROP_TASK_SPEC_KEYS
    }
    step1["note"] = (
        "OCR-focused screen extraction only. For each screenshot, extract readable visible text, "
        "UI labels, buttons, filters, tabs, card/list text, prices, ratings, counts, and other OCR-like text "
        "into screenshot_ocr. Also emit non-text visual cues (icons, badges, images, colors, layout) "
        "into visual_elements. "
        "Do not judge task success or evidence_checks. "
        "Thought in actions is for locating frames, not admissible as site content."
    )
    step1["screenshot_window_steps"] = shot_rows_used
    return step1


_STEP2_TASK_SPEC_KEYS = frozenset({"user_intent", "task_type"})


def _build_step2_payload(
    common_payload: dict[str, Any],
    task_spec: dict[str, Any],
) -> dict[str, Any]:
    """Step 2 input: claims + checkpoints only (no actions, rubric, or visited URLs)."""
    return {
        "instruction": common_payload.get("instruction", ""),
        "final_answer": common_payload.get("final_answer", ""),
        "eval_checkpoints": common_payload.get("eval_checkpoints") or [],
        "task_spec": {
            k: v for k, v in task_spec.items() if k in _STEP2_TASK_SPEC_KEYS
        },
    }


_STEP3_TASK_SPEC_KEYS = frozenset(
    {
        "user_intent",
        "task_type",
        "id",
        "user_expression",
        "expected_grounding",
    }
)


def _build_step3_payload(
    common_payload: dict[str, Any],
    task_spec: dict[str, Any],
) -> dict[str, Any]:
    """Step 3 input: audit + rubric context (no trajectory actions or visited URLs)."""
    return {
        "instruction": common_payload.get("instruction", ""),
        "start_url": common_payload.get("start_url", ""),
        "final_answer": common_payload.get("final_answer", ""),
        "eval_checkpoints": common_payload.get("eval_checkpoints") or [],
        "task_spec": {
            k: v for k, v in task_spec.items() if k in _STEP3_TASK_SPEC_KEYS
        },
        "rubric_profile": common_payload.get("rubric_profile"),
        "rubric_dimension_ids": common_payload.get("rubric_dimension_ids"),
        "rubric_dimensions": common_payload.get("rubric_dimensions"),
    }


def _strip_code_fence(text: str) -> str:
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                out.append(item["text"])
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    out.append(text)
        return "\n".join(x for x in out if x)
    return ""


def _strip_json_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ] (common LLM JSON mistake)."""
    prev = None
    cur = text
    while prev != cur:
        prev = cur
        cur = re.sub(r",(\s*[}\]])", r"\1", cur)
    return cur


def _json_load_robust(text: str) -> dict[str, Any]:
    t = _strip_code_fence(text)
    candidates: list[str] = [t]
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        inner = t[start : end + 1]
        if inner != t:
            candidates.append(inner)
    last_err: json.JSONDecodeError | None = None
    for cand in candidates:
        for attempt in (cand, _strip_json_trailing_commas(cand)):
            try:
                parsed = json.loads(attempt)
            except json.JSONDecodeError as e:
                last_err = e
                continue
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("Judge response JSON must be an object.")
    if last_err is not None:
        raise last_err
    raise ValueError("Judge response contains no JSON object.")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_instruction(run_dir: Path, instruction: str | None) -> str:
    if instruction:
        return instruction
    user_message_dir = run_dir / "user_message"
    if user_message_dir.exists():
        first = user_message_dir / "user_message_001.txt"
        if first.exists():
            return first.read_text(encoding="utf-8").strip()
    raise ValueError("instruction is required (not found in user_message_001.txt)")


def _read_final_answer(
    run_dir: Path, final_answer: str | None, final_answer_file: str | None
) -> str:
    if final_answer is not None:
        return final_answer.strip()
    if final_answer_file:
        return Path(final_answer_file).read_text(encoding="utf-8").strip()
    for name in ("final_answer.txt", "answer.txt", "assistant_final_answer.txt"):
        p = run_dir / name
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return ""


def _sorted_steps(trajectory: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return sorted(trajectory.items(), key=lambda kv: int(kv[0]))


def _extract_final_answer_from_trajectory(trajectory: dict[str, Any]) -> str:
    answer_candidates: list[str] = []
    for _, step in _sorted_steps(trajectory):
        action = (step or {}).get("action", {}) or {}
        a_out = action.get("action_output", {}) or {}
        action_name = a_out.get("action_name")
        msg = None
        action_args = a_out.get("action")
        if isinstance(action_args, dict):
            msg = action_args.get("msg")
        if action_name == "send_msg_to_user" and isinstance(msg, str) and msg.strip():
            answer_candidates.append(msg.strip())
    if not answer_candidates:
        return ""
    filtered = [x for x in answer_candidates if x != "[EXIT]"]
    if not filtered:
        return ""
    latest = filtered[-1]
    if latest.startswith("[ANSWER]"):
        return latest[len("[ANSWER]") :].strip()
    return latest


def _step_thought_from_step(step: dict[str, Any]) -> str:
    """Prefer action.action_output.thought, else other_obs.thought."""
    action = step.get("action") if isinstance(step.get("action"), dict) else {}
    ao = action.get("action_output") if isinstance(action.get("action_output"), dict) else {}
    t = ao.get("thought")
    if isinstance(t, str) and t.strip():
        return t.strip()
    obs = step.get("other_obs") if isinstance(step.get("other_obs"), dict) else {}
    t2 = obs.get("thought")
    if isinstance(t2, str) and t2.strip():
        return t2.strip()
    return ""


_MAX_THOUGHT_JSON_CHARS = 8000


def _truncate_text(s: str, max_chars: int = _MAX_THOUGHT_JSON_CHARS) -> str:
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 24] + "\n...(truncated)"


def _build_actions_and_urls(trajectory: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    actions: list[dict[str, Any]] = []
    urls: list[str] = []
    for k, step in _sorted_steps(trajectory):
        action = (step or {}).get("action", {}) or {}
        a_out = action.get("action_output", {}) or {}
        url = a_out.get("url") or action.get("url")
        if isinstance(url, str) and url:
            urls.append(url)
        thought = _truncate_text(_step_thought_from_step(step or {}))
        row: dict[str, Any] = {
            "step": int(k),
            "action_str": action.get("action_str"),
            "action_name": a_out.get("action_name"),
            "action_args": a_out.get("action"),
            "url": url,
        }
        if thought:
            row["thought"] = thought
        actions.append(row)
    dedup_urls = list(dict.fromkeys(urls))
    return actions, dedup_urls


def _list_trajectory_screenshots_with_rows(
    run_dir: Path, trajectory: dict[str, Any]
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Screenshots in trajectory order, each with step id and agent thought (truncated)."""
    paths: list[Path] = []
    rows: list[dict[str, Any]] = []
    for k, step in _sorted_steps(trajectory):
        shot = (step or {}).get("screenshot")
        if not isinstance(shot, str) or not shot:
            continue
        img = run_dir / "images" / shot
        if not img.exists():
            continue
        paths.append(img)
        th = _step_thought_from_step(step or {})
        rows.append(
            {
                "step": int(k),
                "screenshot": shot,
                "thought": _truncate_text(th),
            }
        )
    return paths, rows


def _sample_screenshots_pair(
    paths: list[Path], rows: list[dict[str, Any]], max_images: int
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Same head+tail subsampling as before; keeps paths and per-shot rows aligned."""
    if max_images <= 0 or len(paths) <= max_images:
        return list(paths), list(rows)
    head = max(1, max_images // 3)
    tail = max_images - head
    idx = sorted(set(list(range(head)) + list(range(len(paths) - tail, len(paths)))))
    return [paths[i] for i in idx], [rows[i] for i in idx]


def _image_to_data_url(path: Path) -> str:
    b = path.read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _make_chat_json(
    client: OpenAI,
    *,
    llm_semaphore: threading.Semaphore | None = None,
    max_tokens: int | None = None,
    max_retries: int = 2,
    retry_backoff: float = 2.0,
    print_llm_raw: bool = False,
    print_llm_raw_max_chars: int = 0,
):
    """Wrap ``_chat_json`` with an in-flight cap, an output-token cap, and transient-error retries.

    - ``llm_semaphore`` bounds concurrent vLLM HTTP calls so the server-side queue cannot blow up
      (an unbounded queue makes the per-request timeout count queue-wait time → cascade timeouts).
    - ``max_tokens`` caps generation length so one runaway response cannot hog the GPU batch.
    - ``max_retries`` retries timeouts / connection / 5xx / empty / malformed JSON with
      exponential backoff; the semaphore slot is released between attempts.
    """

    def chat_json(**kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("max_tokens", max_tokens)
        if print_llm_raw:
            kwargs.setdefault("print_llm_raw", True)
            kwargs.setdefault("print_llm_raw_max_chars", print_llm_raw_max_chars)
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                if llm_semaphore is not None:
                    with llm_semaphore:
                        return _chat_json(client=client, **kwargs)
                return _chat_json(client=client, **kwargs)
            except _RETRYABLE_CHAT_ERRORS as e:
                last_exc = e
                if attempt >= max_retries:
                    break
                delay = retry_backoff * (2 ** attempt)
                print(
                    f"[WARN chat_json retry {attempt + 1}/{max_retries}] "
                    f"{type(e).__name__}: {e}; backoff {delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    return chat_json


def _chat_json(
    *,
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_text: str,
    image_paths: list[Path],
    request_timeout: float,
    max_tokens: int | None = None,
    print_llm_raw: bool = False,
    print_llm_raw_max_chars: int = 0,
) -> dict[str, Any]:
    # Qwen3-VL on vLLM returns degenerate empty JSON for text-only Step 2/3 when
    # user content is sent as a multimodal part list without images; use a plain string.
    if image_paths:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for idx, p in enumerate(image_paths, start=1):
            content.append({"type": "text", "text": f"Screenshot {idx}: {p.name}"})
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(p)}})
        content.append({"type": "text", "text": "Return ONLY valid JSON. No markdown."})
        user_content: str | list[dict[str, Any]] = content
    else:
        user_content = user_text + "\n\nReturn ONLY valid JSON. No markdown."

    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "timeout": request_timeout,
    }
    if max_tokens and max_tokens > 0:
        create_kwargs["max_tokens"] = max_tokens

    call_id: int | None = None
    if print_llm_raw:
        call_id = _next_llm_call_id()
        api_base = str(getattr(client, "base_url", "") or "")
        img_names = [p.name for p in image_paths]
        print(
            f"\n{'=' * 60}\n[LLM REQ #{call_id}] → {api_base}\n"
            f"  model={model}\n"
            f"  images={len(image_paths)} {img_names[:8]}{'...' if len(img_names) > 8 else ''}\n"
            f"  system_chars={len(system_prompt)} user_chars={len(user_text)} "
            f"max_tokens={max_tokens}\n"
            f"[LLM REQ #{call_id}] user_text:\n"
            f"{_clip_debug_text(user_text, print_llm_raw_max_chars)}\n"
            f"[LLM REQ #{call_id}] sending...",
            flush=True,
        )

    response = client.chat.completions.create(**create_kwargs)
    choice = response.choices[0]
    raw = _extract_text_content(choice.message.content)

    if print_llm_raw and call_id is not None:
        finish = getattr(choice, "finish_reason", None)
        print(
            f"[LLM RAW #{call_id}] finish_reason={finish} chars={len(raw or '')}\n"
            f"{_clip_debug_text(raw or '', print_llm_raw_max_chars)}\n"
            f"{'=' * 60}\n",
            flush=True,
        )

    if not raw:
        raise RuntimeError("Judge model returned empty response.")
    if getattr(choice, "finish_reason", None) == "length":
        raise ValueError(
            f"Judge output truncated at output_tokens={max_tokens} (finish_reason=length); "
            "raise --max-output-tokens or lower --max-images for this episode."
        )
    return _json_load_robust(raw)


def _dimension_pass_numeric(v: Any) -> float | None:
    """Map a stored dimension value to 1.0 (pass) / 0.0 (fail); None if absent or unusable.

    Supports binary booleans, 0–1 floats, and legacy 0–10 rubric scores (>=7 pass).
    """
    if v is True:
        return 1.0
    if v is False:
        return 0.0
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return None
    x = float(v)
    if 0.0 <= x <= 1.0:
        return 1.0 if x >= 0.5 else 0.0
    return 1.0 if x >= 7.0 else 0.0


def _compute_metrics(out: dict[str, Any], *, dim_ids: list[str] | None = None) -> None:
    """EGSR / ETSR from answer_success, metadata, and core-claim grounding (RSS is diagnostic only)."""
    _ = dim_ids
    ans = bool(out.get("answer_success"))
    meta_ok = bool(out.get("metadata_requirement_verified"))
    claims_ok = bool(out.get("key_claims_all_supported"))
    out["egsr"] = ans and meta_ok
    out["etsr"] = ans and meta_ok and claims_ok


def _sr_from_trajectory(trajectory: dict[str, Any] | None) -> dict[str, Any]:
    """SR: longest prefix of ordered steps with no ``error`` (consecutive step-level success)."""
    if not trajectory or not isinstance(trajectory, dict):
        return {"total_steps": 0, "sr_prefix_consecutive_ok_steps": 0, "sr_rate": None}
    try:
        steps = sorted(trajectory.items(), key=lambda kv: int(kv[0]))
    except (TypeError, ValueError):
        return {"total_steps": 0, "sr_prefix_consecutive_ok_steps": 0, "sr_rate": None}
    total = len(steps)
    prefix = 0
    for _, step in steps:
        st = step if isinstance(step, dict) else {}
        err = st.get("error")
        bad = err is not None and not (isinstance(err, str) and not err.strip())
        if bad:
            break
        prefix += 1
    sr_rate = (prefix / total) if total else None
    return {
        "total_steps": total,
        "sr_prefix_consecutive_ok_steps": prefix,
        "sr_rate": round(float(sr_rate), 4) if sr_rate is not None else None,
    }

def _claim_supported_bool(c: dict[str, Any]) -> bool:
    """Final claim grounding: Step 3 uses ``supported``; accept legacy ``satisfied`` fallback."""
    if "supported" in c:
        return bool(c.get("supported"))
    if "satisfied" in c:
        return bool(c.get("satisfied"))
    return False


def _is_core_claim_row(c: dict[str, Any]) -> bool:
    """Core (Type B / reason) claims drive ``key_claims_all_supported`` and EGSR grounding."""
    if "is_core_claim" in c:
        return bool(c.get("is_core_claim"))
    ct = str(c.get("claim_type") or "").strip().lower()
    if ct == "reason":
        return True
    if ct == "existence":
        return False
    return True  # legacy rows without type fields


def _reconcile_step2_judgement(
    j: dict[str, Any],
    *,
    dim_ids: list[str],
    step1_merged: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill RSS from ``dimension_scores``; recompute EGSR; align label (keys match Step 3 JSON)."""
    _ = step1_merged
    out = dict(j)
    for k in ("ccr", "ccr_numerator", "ccr_denominator"):
        out.pop(k, None)
    out.pop("path_success", None)

    claims = out.get("final_answer_claims") or []
    if isinstance(claims, list) and claims:
        claim_rows = [c for c in claims if isinstance(c, dict)]
        core_rows = [c for c in claim_rows if _is_core_claim_row(c)]
        if core_rows:
            out["key_claims_all_supported"] = all(_claim_supported_bool(c) for c in core_rows)
        else:
            out["key_claims_all_supported"] = True

    ds = out.get("dimension_scores")
    rss_01: int | None = None
    if isinstance(ds, dict):
        if dim_ids:
            passes = [_dimension_pass_numeric(ds.get(did)) for did in dim_ids]
            if any(p is None for p in passes):
                rss_01 = 0
            else:
                rss_01 = 1 if all(p >= 1.0 for p in passes) else 0
        else:
            vals = [p for p in (_dimension_pass_numeric(v) for v in ds.values()) if p is not None]
            if vals:
                rss_01 = 1 if all(p >= 1.0 for p in vals) else 0
    if rss_01 is not None:
        out["rss"] = rss_01
    _compute_metrics(out, dim_ids=dim_ids)
    _reconcile_label_with_metrics(out)
    return out

_STEP3_JUDGEMENT_KEYS = (
    "step3_final_judgement",
    "step2_global_judgement",
    "pass3_judgement",
    "pass2_judgement",
)


def _step1_merged_from_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("step1_window_evidence", "pass2_evidence_mapping", "step2_evidence_mapping"):
        block = data.get(key)
        if isinstance(block, dict):
            return block
    return None


def _step3_judgement_from_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    for key in _STEP3_JUDGEMENT_KEYS:
        block = data.get(key)
        if isinstance(block, dict):
            return block
    return None


def _dim_ids_for_cached_payload(
    data: dict[str, Any],
    *,
    task_row: dict[str, Any] | None,
    run_dir: Path | None,
) -> list[str]:
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    profile = meta.get("rubric_profile")
    if not profile and task_row:
        profile = _rubric_profile_for_task(
            str(task_row.get("task_type") or ""),
            str(task_row.get("id") or (run_dir.name if run_dir else "")),
        )
    if not profile and run_dir is not None:
        metadata_path = run_dir / "metadata.json"
        if metadata_path.is_file():
            md = _read_json(metadata_path)
            if isinstance(md, dict):
                profile = _rubric_profile_for_task(
                    str(md.get("task_type") or ""),
                    str(md.get("id") or run_dir.name),
                )
    if profile:
        return _rubric_dimension_ids(str(profile))
    j = _step3_judgement_from_payload(data)
    if isinstance(j, dict) and isinstance(j.get("dimension_scores"), dict):
        return list(j["dimension_scores"].keys())
    return []


def reconcile_cached_eval_payload(
    data: dict[str, Any],
    *,
    task_row: dict[str, Any] | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Re-run deterministic reconcile (claim-derived key_claims + EGSR/labels) on cached final judgement."""
    if not isinstance(data, dict) or data.get("error"):
        return data
    step1 = _step1_merged_from_payload(data)
    judgement = _step3_judgement_from_payload(data)
    if not step1 or not judgement:
        return data

    dim_ids = _dim_ids_for_cached_payload(data, task_row=task_row, run_dir=run_dir)
    j2 = normalize_step3_judgement(judgement, rubric_dimension_ids=dim_ids or None)
    j2 = _reconcile_step2_judgement(j2, dim_ids=dim_ids, step1_merged=step1)

    out = dict(data)
    for key in _STEP3_JUDGEMENT_KEYS:
        if key in out:
            out[key] = j2
    if "step3_final_judgement" not in out:
        out["step3_final_judgement"] = j2

    meta = out.get("meta")
    rubric_profile = meta.get("rubric_profile") if isinstance(meta, dict) else None
    trajectory = None
    if run_dir is not None:
        traj_path = run_dir / "trajectory.json"
        if traj_path.is_file():
            trajectory = _read_json(traj_path)
    dm = _derived_task_metrics(
        j2,
        trajectory=trajectory if isinstance(trajectory, dict) else None,
        rubric_profile=str(rubric_profile) if rubric_profile else None,
    )
    out["derived_metrics"] = dm
    if isinstance(meta, dict):
        meta = dict(meta)
        meta["derived_metrics"] = dm
        traj_steps = dm.get("trajectory_step_count")
        if isinstance(traj_steps, int) and traj_steps > 0:
            meta["trajectory_step_count"] = traj_steps
        out["meta"] = meta
    return out


_LABEL_RANK = {"FAIL": 0, "UNGROUNDED_PASS": 1, "CLAIM_UNGROUNDED_PASS": 2, "ETSR_PASS": 3}


def _expected_label_from_metrics(out: dict[str, Any]) -> str:
    ans = bool(out.get("answer_success"))
    egsr = bool(out.get("egsr"))
    etsr = bool(out.get("etsr"))
    if ans and etsr:
        return "ETSR_PASS"
    if ans and egsr:
        return "CLAIM_UNGROUNDED_PASS"
    if ans:
        return "UNGROUNDED_PASS"
    return "FAIL"


def _reconcile_label_with_metrics(out: dict[str, Any]) -> None:
    """Demote pass labels that exceed recomputed ``etsr`` / ``egsr`` / ``answer_success``.

    Does not auto-promote weaker labels (e.g. ``UNGROUNDED_PASS`` → ``ETSR_PASS``).
    """
    lab = str(out.get("label", "") or "").strip()
    if lab == "CRITICAL_FAIL":
        return
    expected = _expected_label_from_metrics(out)
    if lab == expected:
        return
    if _LABEL_RANK.get(lab, -1) <= _LABEL_RANK.get(expected, 0):
        return
    out["label"] = expected
    note = (
        f"[reconciled_label] Model had label={lab!r} but recomputed "
        f"answer_success={bool(out.get('answer_success'))} egsr={bool(out.get('egsr'))} "
        f"etsr={bool(out.get('etsr'))}; set label to {expected!r}."
    )
    rat = out.get("rationale", "")
    if isinstance(rat, str):
        out["rationale"] = (rat.rstrip() + "\n\n" + note) if rat.strip() else note
    elif rat:
        out["rationale"] = str(rat) + "\n\n" + note
    else:
        out["rationale"] = note


def _derived_task_metrics(
    step2_j: dict[str, Any],
    *,
    trajectory: dict[str, Any] | None,
    rubric_profile: str | None,
) -> dict[str, Any]:
    claims = step2_j.get("final_answer_claims") or []
    supported = unsupported = core_supported = core_unsupported = 0
    if isinstance(claims, list):
        for c in claims:
            if not isinstance(c, dict):
                continue
            if _claim_supported_bool(c):
                supported += 1
            else:
                unsupported += 1
            if _is_core_claim_row(c):
                if _claim_supported_bool(c):
                    core_supported += 1
                else:
                    core_unsupported += 1
    sr_block = _sr_from_trajectory(trajectory)
    traj_steps = int(sr_block["total_steps"] or 0)
    ans_ok = bool(step2_j.get("answer_success"))
    return {
        "tsr": ans_ok,
        "egsr": step2_j.get("egsr"),
        "etsr": step2_j.get("etsr"),
        "rss": step2_j.get("rss"),
        "sr_prefix_consecutive_ok_steps": sr_block.get("sr_prefix_consecutive_ok_steps"),
        "sr_rate": sr_block.get("sr_rate"),
        "total_steps": traj_steps,
        "ass_step_count_if_success": traj_steps if ans_ok else None,
        "ddr_applicable": step2_j.get("ddr_applicable"),
        "ddr_pass": step2_j.get("ddr_pass"),
        "rubric_profile": rubric_profile,
        "final_answer_claims_supported": supported,
        "final_answer_claims_unsupported": unsupported,
        "core_claims_supported": core_supported,
        "core_claims_unsupported": core_unsupported,
        "trajectory_step_count": traj_steps,
    }


def run_three_step_eval(
    *,
    run_dir: Path,
    instruction: str | None,
    start_url: str | None,
    constraints: str | None,
    final_answer: str | None,
    final_answer_file: str | None,
    max_images: int,
    api_base: str,
    model: str,
    api_key: str,
    request_timeout: float,
    task_row: dict[str, Any] | None = None,
    image_chunk_size: int = 0,
    step1_chunk_workers: int = 1,
    llm_max_inflight: int = 0,
    llm_semaphore: threading.Semaphore | None = None,
    step1_retry_on_empty: bool = False,
    step2_retry_on_empty: bool = False,
    max_output_tokens: int = 0,
    llm_retries: int = 2,
    print_llm_raw: bool = False,
    print_llm_raw_max_chars: int = 0,
) -> dict[str, Any]:
    """Step 1 (literal screen extraction) → Step 2 (claims + evidence_checks ↔ evidence, satisfied) → Step 3 (audit + rubric).

    Eval metrics (episode / batch):
    - **RSS** — diagnostic **0 or 1**: **1** iff every rubric dimension passes (not used for EGSR/ETSR gating).
    - **EGSR** — ``answer_success`` and ``metadata_requirement_verified``.
    - **ETSR** — ``answer_success``, ``metadata_requirement_verified``, and ``key_claims_all_supported``.
    - **TSR** — task success; same boolean as ``answer_success`` (``derived_metrics.tsr``).
    - **SR** — consecutive successful steps from the start of ``trajectory.json`` until first non-empty ``error``.
    - **ASS** — batch mean of step counts for successful tasks (see batch ``meta.ass_mean_successful_steps``).
    - **DDR** — for ``dead_end`` profile episodes, ``ddr_pass`` / ``ddr_applicable`` (batch DDR rate in ``meta``).

    Image pipeline: ``max_images > 0`` caps screenshots (head+tail sample); ``max_images <= 0`` uses all on-disk shots.
    ``image_chunk_size``: Step 1 shots per API call (0 = all at once). Steps 2–3 are text-only (no screenshots).
    ``step1_chunk_workers``: parallel Step 1 chunk API calls per episode (2–3 typical).
    ``llm_semaphore``: caps concurrent LLM HTTP calls across batch workers (match vLLM ``max-num-seqs``).
    """
    trajectory_path = run_dir / "trajectory.json"
    if not trajectory_path.exists():
        raise FileNotFoundError(f"trajectory.json not found: {trajectory_path}")

    trajectory = _read_json(trajectory_path)
    metadata = _read_json(run_dir / "metadata.json") if (run_dir / "metadata.json").exists() else {}
    resolved_instruction = _read_instruction(run_dir, instruction)
    resolved_start_url = start_url or metadata.get("initial_url") or ""
    resolved_final_answer = _read_final_answer(run_dir, final_answer, final_answer_file)
    if not resolved_final_answer:
        resolved_final_answer = _extract_final_answer_from_trajectory(trajectory)

    actions, visited_urls = _build_actions_and_urls(trajectory)
    all_shots, shot_rows_full = _list_trajectory_screenshots_with_rows(run_dir, trajectory)
    images, shot_rows_used = _sample_screenshots_pair(all_shots, shot_rows_full, max_images)

    _s1_eff = effective_step1_chunk_size(image_chunk_size, len(images))
    _s1_chunks = (
        1
        if len(images) <= _s1_eff
        else len(image_chunk_index_ranges(len(images), _s1_eff))
    )
    print(
        f"[DEBUG] episode={run_dir.name} api={api_base} "
        f"images={len(images)} step1_chunks≈{_s1_chunks} (chunk_size={image_chunk_size})",
        flush=True,
    )

    task_spec = _task_spec_from_metadata_row(task_row)
    if isinstance(metadata, dict):
        for key in (
            "user_intent",
            "user_expression",
            "expected_grounding",
            "required_actions",
            "fallback_expected",
            "fallback_reason",
            "acceptable_fallbacks",
            "task_type",
            "id",
            "prompt",
            "checkpoints",
            "eval_checkpoints",
            "evidence_checks",
        ):
            if key not in task_spec and key in metadata:
                task_spec[key] = metadata[key]
    rubric_profile = _rubric_profile_for_task(
        str(task_spec.get("task_type") or metadata.get("task_type") or ""),
        str(task_spec.get("id") or metadata.get("id") or run_dir.name),
    )

    eval_ck = _eval_checkpoints_from_task_spec(task_spec)
    dim_ids = _rubric_dimension_ids(rubric_profile)

    common_payload: dict[str, Any] = {
        "instruction": resolved_instruction,
        "start_url": resolved_start_url,
        "constraints": constraints or "",
        "visited_urls": visited_urls,
        "actions": actions,
        "final_answer": resolved_final_answer,
        "task_spec": task_spec,
        "rubric_profile": rubric_profile,
        "rubric_dimensions": _format_rubric_dimension_lines(rubric_profile),
        "rubric_dimension_ids": dim_ids,
        "rubric_dimensions_structured": _rubric_dimensions_structured(rubric_profile),
        "eval_checkpoints": eval_ck,
        "note": (
            "Step 1: extract screenshot_ocr + visual_elements (no final_answer, no eval_checkpoints). "
            "Step 2: map final_answer + eval_checkpoints to evidence with satisfied. "
            "Step 3: audit supported + rubric."
        ),
    }

    step1_common = _build_step1_common_payload(
        common_payload,
        task_spec,
        shot_rows_used=shot_rows_used,
    )

    client = OpenAI(base_url=api_base, api_key=api_key, timeout=request_timeout)

    chat_json = _make_chat_json(
        client,
        llm_semaphore=llm_semaphore,
        max_tokens=max_output_tokens if max_output_tokens > 0 else None,
        max_retries=max(0, llm_retries),
        print_llm_raw=print_llm_raw,
        print_llm_raw_max_chars=print_llm_raw_max_chars,
    )

    print(
        f"[DEBUG] Step1 OCR start → {api_base} "
        f"(chunk_workers={max(1, step1_chunk_workers)})",
        flush=True,
    )
    step1_merged, step1_raw_chunks, step1_run_meta = run_step1_window_ocr_with_fallback(
        chat_json=chat_json,
        model=model,
        common_payload=step1_common,
        image_paths=images,
        shot_rows=shot_rows_used,
        request_timeout=request_timeout,
        chunk_size=image_chunk_size,
        chunk_workers=step1_chunk_workers,
        retry_on_empty=step1_retry_on_empty,
    )
    step1_ocr = (
        step1_merged.get("screenshot_ocr")
        if isinstance(step1_merged.get("screenshot_ocr"), list)
        else []
    )
    step1_visual = (
        step1_merged.get("visual_elements")
        if isinstance(step1_merged.get("visual_elements"), list)
        else []
    )
    step1_extraction_empty = bool(images) and len(step1_ocr) == 0

    print("[DEBUG Step1 api_base]", api_base, flush=True)
    print("[DEBUG Step1 model]", model, flush=True)
    print("[DEBUG Step1 images_sent]", len(images), flush=True)
    print("[DEBUG Step1 ocr_row_count]", len(step1_ocr), flush=True)
    print("[DEBUG Step1 visual_element_count]", len(step1_visual), flush=True)
    for i, row in enumerate(step1_ocr[:3]):
        if not isinstance(row, dict):
            continue
        print(
            f"[DEBUG Step1 ocr sample {i + 1}]",
            row.get("screenshot_filename") or row.get("screenshot"),
            repr((row.get("ocr_text") or "")[:400]),
            flush=True,
        )
    for i, row in enumerate(step1_visual[:3]):
        if not isinstance(row, dict):
            continue
        print(
            f"[DEBUG Step1 visual sample {i + 1}]",
            row.get("element_type"),
            row.get("location"),
            repr((row.get("description") or "")[:200]),
            flush=True,
        )

    step2_payload = _build_step2_payload(common_payload, task_spec)
    print("[DEBUG Step2 start]", flush=True)
    print("[DEBUG Step2 final_answer]", repr(step2_payload.get("final_answer")), flush=True)
    print("[DEBUG Step2 eval_checkpoints]", step2_payload.get("eval_checkpoints"), flush=True)

    step2_claim_ev = run_step2_claim_evidence_mapping(
        chat_json=chat_json,
        model=model,
        common_payload=step2_payload,
        step1_merged_evidence=step1_merged,
        request_timeout=request_timeout,
    )
    step3_payload = _build_step3_payload(common_payload, task_spec)
    step3_j = run_step3_global_rubric_judgement(
        chat_json=chat_json,
        model=model,
        common_payload=step3_payload,
        step1_merged_ocr=step1_merged,
        step2_claim_evidence=step2_claim_ev,
        request_timeout=request_timeout,
    )
    step3_j = _reconcile_step2_judgement(
        step3_j,
        dim_ids=dim_ids,
        step1_merged=step1_merged,
    )

    trajectory_step_count = len(trajectory) if isinstance(trajectory, dict) else 0
    derived_metrics = _derived_task_metrics(
        step3_j,
        trajectory=trajectory if isinstance(trajectory, dict) else None,
        rubric_profile=rubric_profile,
    )

    meta_images: dict[str, Any] = {
        "run_dir": str(run_dir),
        "api_base": api_base,
        "model": model,
        "image_count_step1_total": len(images),
        "screenshots_on_disk": len(all_shots),
        "max_images_cap": max_images if max_images > 0 else None,
        "images_mode": "all" if max_images <= 0 else "subsampled",
        "image_chunk_size": image_chunk_size if image_chunk_size > 0 else None,
        "step1_requested_chunk_size": step1_run_meta.get("step1_requested_chunk_size"),
        "step1_effective_chunk_size": step1_run_meta.get("step1_effective_chunk_size"),
        "step1_chunk_workers": max(1, step1_chunk_workers),
        "llm_max_inflight": llm_max_inflight if llm_max_inflight > 0 else None,
        "step1_retry_on_empty": step1_retry_on_empty,
        "step1_empty_retry": step1_run_meta.get("step1_empty_retry"),
        "step1_extraction_empty": step1_extraction_empty,
        "step1_screenshot_ocr_count": len(step1_ocr),
        "step1_visual_elements_count": len(step1_visual),
        "step2_reviews_synthesized": bool(
            isinstance(step2_claim_ev, dict) and step2_claim_ev.get("step2_reviews_synthesized")
        ),
        "step1_used_chunking": bool(step1_raw_chunks and len(step1_raw_chunks) > 1),
        "step1_llm_calls": (
            (len(step1_raw_chunks) if step1_raw_chunks else 1)
            * int(step1_run_meta.get("step1_llm_passes") or 1)
        ),
        "step3_global_rubric_calls": 1,
        "rubric_profile": rubric_profile,
        "pipeline": "step1_ocr_visual_extract_step2_map_satisfied_step3_audit_rubric",
        "trajectory_step_count": trajectory_step_count,
        "eval_metrics": {
            "rss": "Diagnostic rollup 0/1: 1 iff every rubric dimension passes (not used for EGSR/ETSR gating).",
            "egsr": "answer_success + metadata_requirement_verified.",
            "etsr": "answer_success + metadata_requirement_verified + key_claims_all_supported.",
            "tsr": "Task success = answer_success (Step 3 `step3_final_judgement`).",
            "sr": "Longest prefix of trajectory steps with empty/null error; sr_rate = prefix / total_steps.",
            "ass": "Batch-only: mean total_steps over episodes with answer_success (meta.ass_mean_successful_steps).",
            "ddr": "Dead-end episodes only: ddr_pass when impossibility handled honestly (ddr_applicable from rubric_profile dead_end).",
        },
        "derived_metrics": derived_metrics,
    }
    eff_chunk = step1_run_meta.get("step1_effective_chunk_size")
    if step1_raw_chunks and isinstance(eff_chunk, int) and eff_chunk > 0 and len(images) > eff_chunk:
        rngs = image_chunk_index_ranges(len(images), eff_chunk)
        meta_images["step1_chunk_sizes"] = [e - s for s, e in rngs]

    return {
        "meta": meta_images,
        "step1_window_evidence": step1_merged,
        "step2_claim_evidence": step2_claim_ev,
        "step3_final_judgement": step3_j,
        "derived_metrics": derived_metrics,
    }


def _load_tasks_by_id(tasks_json_path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(tasks_json_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        tasks = raw
    elif isinstance(raw, dict) and isinstance(raw.get("tasks"), list):
        tasks = raw["tasks"]
    else:
        raise ValueError(
            "tasks JSON must be a list of tasks or an object with a 'tasks' array "
            f"(e.g. *_metadata.json); got {type(raw).__name__}"
        )
    out: dict[str, dict[str, Any]] = {}
    for item in tasks:
        if not isinstance(item, dict):
            continue
        tid = item.get("id")
        if isinstance(tid, str) and tid:
            out[tid] = item
    return out


def _list_episode_run_dirs(results_root: Path) -> list[Path]:
    """Episode dirs under results_root: skip dotdirs and folders without trajectory.json."""
    out: list[Path] = []
    for p in sorted(results_root.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        if not (p / "trajectory.json").is_file():
            continue
        out.append(p)
    return out


def run_batch_from_results_root(
    *,
    results_root: Path,
    tasks_json: Path,
    constraints: str | None,
    max_images: int,
    image_chunk_size: int,
    step1_chunk_workers: int = 1,
    llm_max_inflight: int = 0,
    step1_retry_on_empty: bool = False,
    step2_retry_on_empty: bool = False,
    max_output_tokens: int = 0,
    llm_retries: int = 2,
    print_llm_raw: bool = False,
    print_llm_raw_max_chars: int = 0,
    api_base: str,
    model: str,
    api_key: str,
    episode_id: str | None,
    cache_filename: str,
    force: bool,
    workers: int,
    report_filename: str,
    request_timeout: float,
    show_progress: bool = True,
) -> dict[str, Any]:
    tasks_by_id = _load_tasks_by_id(tasks_json)
    run_dirs = _list_episode_run_dirs(results_root)
    if episode_id:
        run_dirs = [p for p in run_dirs if p.name == episode_id]
    total = len(run_dirs)
    max_workers = max(1, workers)
    use_tqdm = show_progress and total > 0 and sys.stderr.isatty()
    llm_semaphore: threading.Semaphore | None = None
    if llm_max_inflight > 0:
        llm_semaphore = threading.Semaphore(llm_max_inflight)
    print(
        f"🚀 Starting Step 1–3 eval batch: total={total}, workers={max_workers}, "
        f"force={force}, llm_max_inflight={llm_max_inflight or 'unlimited'}, "
        f"step1_chunk_workers={max(1, step1_chunk_workers)}",
    )

    def _judge_one(ep_dir: Path) -> tuple[str, dict[str, Any], bool]:
        trajectory_path = ep_dir / "trajectory.json"
        if not trajectory_path.exists():
            return ep_dir.name, {"error": "trajectory.json not found"}, False
        cache_path = ep_dir / cache_filename
        if cache_path.exists() and not force:
            try:
                cached = _read_json(cache_path)
                cached = reconcile_cached_eval_payload(
                    cached,
                    task_row=tasks_by_id.get(ep_dir.name),
                    run_dir=ep_dir,
                )
                meta = cached.get("meta")
                if isinstance(meta, dict):
                    meta["cache_hit"] = True
                else:
                    cached["meta"] = {"cache_hit": True}
                return ep_dir.name, cached, True
            except Exception as e:  # noqa: BLE001
                return ep_dir.name, {"error": f"failed to read cache: {e}"}, False
        task = tasks_by_id.get(ep_dir.name)
        metadata_path = ep_dir / "metadata.json"
        metadata = _read_json(metadata_path) if metadata_path.exists() else {}
        instruction = (task or {}).get("prompt") or metadata.get("goal")
        start_url = (task or {}).get("start_url")
        if not instruction:
            return ep_dir.name, {"error": "No prompt found in tasks_json or metadata.goal"}, False
        print(f"[DEBUG] batch judging start: {ep_dir.name}", flush=True)
        try:
            judged = run_three_step_eval(
                run_dir=ep_dir,
                instruction=instruction,
                start_url=start_url,
                constraints=constraints,
                final_answer=None,
                final_answer_file=None,
                max_images=max_images,
                api_base=api_base,
                model=model,
                api_key=api_key,
                request_timeout=request_timeout,
                task_row=task,
                image_chunk_size=image_chunk_size,
                step1_chunk_workers=step1_chunk_workers,
                llm_max_inflight=llm_max_inflight,
                llm_semaphore=llm_semaphore,
                step1_retry_on_empty=step1_retry_on_empty,
                step2_retry_on_empty=step2_retry_on_empty,
                max_output_tokens=max_output_tokens,
                llm_retries=llm_retries,
                print_llm_raw=print_llm_raw,
                print_llm_raw_max_chars=print_llm_raw_max_chars,
            )
            _write_json(cache_path, judged)
            return ep_dir.name, judged, False
        except Exception as e:  # noqa: BLE001
            return ep_dir.name, {"error": str(e)}, False

    outputs: dict[str, Any] = {}
    cache_hits = 0
    completed = 0
    errors = 0
    lock = threading.Lock()

    def _log(msg: str) -> None:
        if use_tqdm:
            tqdm.write(msg)
        else:
            print(msg)

    def _consume(rid: str, data: dict[str, Any], hit: bool, pbar: tqdm | None = None) -> None:
        nonlocal completed, cache_hits, errors
        with lock:
            completed += 1
            outputs[rid] = data
            if hit:
                cache_hits += 1
            is_error = isinstance(data, dict) and "error" in data
            if is_error:
                errors += 1
            ok = completed - errors
            status = "cache" if hit else ("error" if is_error else "judged")
            if pbar is not None:
                pbar.set_postfix(ok=ok, err=errors, cache=cache_hits, refresh=False)
                pbar.update(1)
            if is_error:
                _log(
                    f"❌ [{completed}/{total}] {rid}\n"
                    f"   error: {data.get('error')}"
                )
            else:
                _log(
                    f"✅ [{completed}/{total}] {rid} ({ok} ok) "
                    f"[mode={status}, cache_hits={cache_hits}]"
                )

    if max_workers == 1:
        with tqdm(total=total, desc="Episodes", unit="ep", disable=not use_tqdm) as pbar:
            for run_dir in run_dirs:
                rid, data, hit = _judge_one(run_dir)
                _consume(rid, data, hit, pbar if use_tqdm else None)
    else:
        with tqdm(total=total, desc="Episodes", unit="ep", disable=not use_tqdm) as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(_judge_one, run_dir): run_dir.name for run_dir in run_dirs}
                for fut in as_completed(future_map):
                    rid, data, hit = fut.result()
                    _consume(rid, data, hit, pbar if use_tqdm else None)

    # Metadata tasks never reached in batch (no trajectory) or skipped dirs → explicit errors in report.
    for tid in tasks_by_id:
        if tid in outputs:
            continue
        err = missing_judge_cache_payload(results_root / tid, cache_filename)
        if err:
            outputs[tid] = err

    per_type_reports = save_per_task_type_reports(
        results_root=results_root,
        outputs=outputs,
        tasks_by_id=tasks_by_id,
        base_report_filename=report_filename,
        cache_filename=cache_filename,
    )
    nav_links = [(ttype, p.name, count) for ttype, p, count in per_type_reports]

    report_path = save_batch_html_report(
        results_root=results_root,
        outputs=outputs,
        report_filename=report_filename,
        cache_filename=cache_filename,
        tasks_by_id=tasks_by_id,
        nav_links=nav_links,
    )
    print(f"🏁 Finished Step 1–3 eval batch: attempted={completed}/{total}, success={completed - errors}, errors={errors}, cache_hits={cache_hits}")   
    print(f"📄 Saved report: {report_path}")
    for ttype, p, count in per_type_reports:
        print(f"📄 Saved per-task_type report ({ttype}, n={count}): {p}")

    batch_stats = calculate_batch_statistics(outputs, tasks_by_id)

    return {
        "meta": {
            "mode": "batch_results_root",
            "results_root": str(results_root),
            "tasks_json": str(tasks_json),
            "count_total_dirs": len(run_dirs),
            "count_evaluated": len(outputs),
            "cache_filename": cache_filename,
            "cache_hits": cache_hits,
            "force": force,
            "workers": max_workers,
            "errors": errors,
            "report_path": str(report_path),
            "tsr_rate": batch_stats["tsr_rate"],
            "egsr_rate": batch_stats["egsr_rate"],
            "etsr_rate": batch_stats["etsr_rate"],
            "ddr_rate_dead_end_tasks": batch_stats["ddr_rate"],
            "ddr_applicable_count": batch_stats["ddr_applicable_count"],
            "ass_mean_successful_steps": batch_stats["ass_mean_successful_steps"],
            "avg_steps_successful_trajectories": batch_stats["avg_steps_successful_trajectories"],
            "count_answer_success_for_avg_steps": batch_stats["count_answer_success_for_avg_steps"],
            "per_task_type_reports": [
                {"task_type": ttype, "path": str(p), "count": count}
                for ttype, p, count in per_type_reports
            ],
        },
        "episodes": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
    description="Step 1 OCR evidence + Step 2 claim mapping + Step 3 final rubric judge"
    )
    parser.add_argument(
        "--tasks-json",
        type=Path,
        default=DEFAULT_TASKS_JSON,
        help="Task list JSON: array of {id, prompt, start_url, ...} or metadata bundle {tasks, meta} "
        f"(default: {DEFAULT_TASKS_JSON})",
    )
    parser.add_argument("--results-root", default=None, help="Root dir containing many episode dirs")
    parser.add_argument("--episode-id", default=None, help="Evaluate only this episode id (batch mode)")
    parser.add_argument("--run-dir", required=False, help="Run directory with trajectory.json/images")
    parser.add_argument("--instruction", default=None, help="Override instruction")
    parser.add_argument("--start-url", default=None, help="Override start_url")
    parser.add_argument("--constraints", default=None, help="Constraint text")
    parser.add_argument("--final-answer", default=None, help="Final answer string override")
    parser.add_argument("--final-answer-file", default=None, help="Path to final answer file")
    parser.add_argument(
        "--max-images",
        type=int,
        default=16,
        help="Screenshots fed to Step 1/Step 2: positive=cap (default 16, samples first ~1/3 + tail), <=0=all images in the trajectory",
    )
    parser.add_argument(
        "--image-chunk-size",
        type=int,
        default=8,
        help="Screenshots per Step 1 API call (default 8). 0=all at once. Steps 2/3 are text-only (no images).",
    )
    parser.add_argument(
        "--step1-chunk-workers",
        type=int,
        default=2,
        help="Parallel Step 1 chunks within an episode (default 2). 1=sequential.",
    )
    parser.add_argument(
        "--llm-max-inflight",
        type=int,
        default=12,
        help="Max concurrent LLM HTTP requests (default 12; keep <= vLLM max-num-seqs 16). 0=unlimited.",
    )
    parser.add_argument(
        "--step1-retry-on-empty",
        action="store_true",
        help=(
            "If screenshots were sent in Step 1 but screenshot_ocr is empty, "
            "re-run extraction once with the same chunk size (default: off). visual_elements is not part of the retry condition."
        ),
    )
    parser.add_argument(
        "--step2-retry-on-empty",
        action="store_true",
        help=(
            "If Step 2 returns empty arrays for both final_answer_claims and evidence_check_reviews, "
            "re-run the mapping API once (default: off)."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("QWEN_PORT", str(DEFAULT_QWEN_PORT))),
        help=f"vLLM OpenAI API port (default: {DEFAULT_QWEN_PORT}; e.g. --port 8004)",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help=(
            f"Override API base URL (default: {DEFAULT_QWEN_API_BASE}; env: QWEN_API_BASE)"
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN_MODEL", DEFAULT_QWEN_MODEL),
        help=f"Qwen judge model on local vLLM (default: {DEFAULT_QWEN_MODEL})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("QWEN_API_KEY", "dummy"),
        help="API key for local vLLM (default: QWEN_API_KEY or dummy)",
    )
    parser.add_argument("--cache-filename", default="two_pass_judge.json", help="Per-episode cache filename")
    parser.add_argument("--force", action="store_true", help="Ignore existing cache and re-evaluate")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for batch mode")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar (batch mode; also off when stderr is not a TTY)",
    )
    parser.add_argument("--report-filename", default="!__two_pass_verdicts.html", help="Batch HTML report filename")
    parser.add_argument("--request-timeout", type=float, default=180.0, help="Per-request timeout seconds")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=16000,
        help="Cap on LLM response max_tokens (default 12000). Prevents Step 3 runaway (max-token overflow) and timeouts. 0=unlimited.",
    )
    parser.add_argument(
        "--llm-retries",
        type=int,
        default=2,
        help="Retry count on LLM call failure (timeout/connection/5xx/empty response/JSON parse failure) (default 2).",
    )
    parser.add_argument(
        "--print-llm-raw",
        action="store_true",
        help="Print each LLM API request/response to stdout (for debugging).",
    )
    parser.add_argument(
        "--print-llm-raw-max-chars",
        type=int,
        default=0,
        help="Char cap for user_text/response output under --print-llm-raw. 0=print everything.",
    )
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()
    api_base = resolve_api_base(port=args.port, api_base=args.api_base)
    api_key = args.api_key

    result: dict[str, Any]
    if args.tasks_json and args.results_root:
        result = run_batch_from_results_root(
            results_root=Path(args.results_root),
            tasks_json=Path(args.tasks_json),
            constraints=args.constraints,
            max_images=args.max_images,
            image_chunk_size=args.image_chunk_size,
            step1_chunk_workers=args.step1_chunk_workers,
            llm_max_inflight=args.llm_max_inflight,
            step1_retry_on_empty=args.step1_retry_on_empty,
            step2_retry_on_empty=args.step2_retry_on_empty,
            max_output_tokens=args.max_output_tokens,
            llm_retries=args.llm_retries,
            print_llm_raw=args.print_llm_raw,
            print_llm_raw_max_chars=args.print_llm_raw_max_chars,
            api_base=api_base,
            model=args.model,
            api_key=api_key,
            episode_id=args.episode_id,
            cache_filename=args.cache_filename,
            force=args.force,
            workers=args.workers,
            report_filename=args.report_filename,
            request_timeout=args.request_timeout,
            show_progress=not args.no_progress,
        )
    else:
        if not args.run_dir:
            raise ValueError("--run-dir is required in single-run mode.")
        run_dir = Path(args.run_dir)
        cache_path = run_dir / args.cache_filename
        if cache_path.exists() and not args.force:
            result = _read_json(cache_path)
            task_row_single = None
            tpath = args.tasks_json
            if tpath.is_file():
                task_row_single = _load_tasks_by_id(tpath).get(run_dir.name)
            result = reconcile_cached_eval_payload(
                result,
                task_row=task_row_single,
                run_dir=run_dir,
            )
            meta = result.get("meta")
            if isinstance(meta, dict):
                meta["cache_hit"] = True
            else:
                result["meta"] = {"cache_hit": True}
        else:
            task_row_single: dict[str, Any] | None = None
            tpath = args.tasks_json
            if tpath.is_file():
                task_row_single = _load_tasks_by_id(tpath).get(run_dir.name)
            result = run_three_step_eval(
                run_dir=run_dir,
                instruction=args.instruction,
                start_url=args.start_url,
                constraints=args.constraints,
                final_answer=args.final_answer,
                final_answer_file=args.final_answer_file,
                max_images=args.max_images,
                api_base=api_base,
                model=args.model,
                api_key=api_key,
                request_timeout=args.request_timeout,
                task_row=task_row_single,
                image_chunk_size=args.image_chunk_size,
                step1_chunk_workers=args.step1_chunk_workers,
                llm_max_inflight=args.llm_max_inflight,
                llm_semaphore=(
                    threading.Semaphore(args.llm_max_inflight)
                    if args.llm_max_inflight > 0
                    else None
                ),
                step1_retry_on_empty=args.step1_retry_on_empty,
                step2_retry_on_empty=args.step2_retry_on_empty,
                max_output_tokens=args.max_output_tokens,
                llm_retries=args.llm_retries,
                print_llm_raw=args.print_llm_raw,
                print_llm_raw_max_chars=args.print_llm_raw_max_chars,
            )
            _write_json(cache_path, result)

    dumped = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(dumped, encoding="utf-8")
    print(dumped)


if __name__ == "__main__":
    main()
