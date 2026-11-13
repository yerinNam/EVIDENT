from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


def effective_step1_chunk_size(requested_chunk_size: int, n_images: int) -> int:
    """Window size for one Step 1 call: ``--image-chunk-size`` (0 = all images at once)."""
    if n_images <= 0:
        return 0
    if requested_chunk_size <= 0:
        size = n_images
    else:
        size = requested_chunk_size
    return max(1, min(size, n_images))

STEP1_WINDOW_OCR_PROMPT = """You are a **screenshot OCR** pass for a GUI/web agent trajectory.

## Your role

For each attached screenshot, extract visible on-screen text as faithfully as possible.

Your priority is OCR-like text extraction, not image captioning.

Do not write long image descriptions.
Do not infer hidden meanings.
Do not judge task success.
Do not compare to final_answer.
Do not decide whether evidence supports the answer.
Agent thought may help locate frames, but is not admissible as page evidence.

## Screenshot OCR

For **every attached screenshot**, emit one `screenshot_ocr` row.

Each row should contain:
- readable visible text from the screenshot
- UI labels, menu labels, tabs, buttons, filters, selected options
- card/list item text: titles, subtitles, prices, ratings, counts, badges, dates, rankings
- search query text if visible
- URL bar text only if visible in the screenshot
- uncertainty notes for unreadable or partially readable text

Preserve the original language and numbers.
Preserve important line breaks when helpful.
Do not summarize when exact text is readable.
If text is partially unreadable, write the readable part and mark unreadable spans as `[unclear]`.

Include important visual-only cues inside `ocr_text` when they matter and have no readable label, prefixed with `VISUAL:`:
- visible badge/icon without readable text
- product photo contents (box, accessory, logo, color, object)
- selected/disabled/highlighted state, ranking position, or spatial relation

## Non-empty extraction

If any screenshot shows a loaded page, emit at least one `screenshot_ocr` row per screenshot.

`screenshot_ocr: []` is valid only when every image is blank, loading-only, error-only, or login/captcha wall.

When OCR text appears slightly corrupted but the visible screenshot strongly indicates a common Korean product term, preserve the most visually plausible text and add the uncertainty note.

Examples:
- "빗방사" may be "빛반사" if the visible product context is raincoat/safety/reflection.
- Do not silently invent; write the corrected text in ocr_text and mention the ambiguity in unreadable_or_uncertain.

## Visual Element Extraction (for Visual Grounding)

For each screenshot, also emit a `visual_elements` list for non-text visual cues
that could serve as selection targets.

Each entry:
- `element_type`: "icon" | "badge" | "image" | "color_cue" | "layout_position"
- `description`: what it looks like (e.g. "CARE 인증 아이콘, 방패 모양")  
- `location`: spatial anchor (e.g. "3번째 상품 카드 우측 상단", "검색창 좌측")
- `associated_text`: nearby readable text if any
- `screenshot_index_in_request`: number

## Output
Return ONLY valid JSON:

{
  "window_index": number | null,
  "screenshot_ocr": [
    {
      "screenshot_index_in_request": number,
      "trajectory_step": number | null,
      "screenshot_filename": string | null,
      "ocr_text": string,
      "unreadable_or_uncertain": string
    }
  ],
    "visual_elements": [
    {
        "screenshot_index_in_request": number,
        "element_type": "icon" | "badge" | "image" | "color_cue" | "layout_position",
        "description": string,
        "location": string,
        "associated_text": string | null
    }
    ],
  "window_notes": string
}

Rules:
- `ocr_text` should be OCR-like text, not a prose description; include all readable UI text in one field.
- Do not output checkpoint judgments.
- Do not output task success.
- Do not use agent thought as evidence.
"""

def merge_step1_window_outputs(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge Step 1 per-window JSON: screenshot OCR + visual elements."""
    if not chunks:
        return {
            "screenshot_ocr": [],
            "visual_elements": [],
            "window_notes": "",
            "chunk_merge": True,
            "chunk_count": 0,
        }

    all_ocr: list[dict[str, Any]] = []
    all_visual: list[dict[str, Any]] = []
    notes_parts: list[str] = []

    for wi, ch in enumerate(chunks):
        raw_ocr = ch.get("screenshot_ocr") or []
        if isinstance(raw_ocr, list):
            for d in raw_ocr:
                if not isinstance(d, dict):
                    continue
                row = dict(d)
                row["merge_source_chunk"] = wi + 1
                all_ocr.append(row)

        raw_visual = ch.get("visual_elements") or []
        if isinstance(raw_visual, list):
            for v in raw_visual:
                if not isinstance(v, dict):
                    continue
                row = dict(v)
                row["merge_source_chunk"] = wi + 1
                all_visual.append(row)

        wn = ch.get("window_notes")
        if isinstance(wn, str) and wn.strip():
            notes_parts.append(f"[window {wi + 1}/{len(chunks)}] {wn.strip()}")

    return {
        "screenshot_ocr": all_ocr,
        "visual_elements": all_visual,
        "window_notes": "\n".join(notes_parts),
        "chunk_merge": True,
        "chunk_count": len(chunks),
    }


def run_step1_window_ocr(
    *,
    chat_json,
    model: str,
    common_payload: dict[str, Any],
    image_paths: list,
    request_timeout: float,
) -> dict[str, Any]:
    user_text = (
        "Step 1. For each attached screenshot: extract OCR-like visible text in ocr_text"
        " and visual elements (icons, badges, images) in visual_elements.\n"
        "No task judgment, no final_answer comparison, no checklist pass/fail.\n\n"
        + json.dumps({"input": common_payload}, ensure_ascii=False, indent=2)
    )
    out = chat_json(
        model=model,
        system_prompt=STEP1_WINDOW_OCR_PROMPT,
        user_text=user_text,
        image_paths=image_paths,
        request_timeout=request_timeout,
    )
    if isinstance(out, dict):
        out.setdefault("screenshot_ocr", [])
        out.setdefault("visual_elements", [])
        if out.get("window_index") is None:
            out["window_index"] = 1
    return out


def image_chunk_index_ranges(n_total: int, window: int) -> list[tuple[int, int]]:
    """0-based half-open [start, end) indices; non-overlapping windows, stride = window."""
    if n_total <= 0 or window <= 0:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < n_total:
        end = min(start + window, n_total)
        ranges.append((start, end))
        if end >= n_total:
            break
        start += window
    return ranges


def _run_step1_single_chunk(
    *,
    ci: int,
    s: int,
    e: int,
    ranges_len: int,
    chat_json,
    model: str,
    common_payload: dict[str, Any],
    paths: list,
    rows: list[dict[str, Any]],
    eff: int,
    chunk_size: int,
    request_timeout: float,
) -> dict[str, Any]:
    chunk_paths = paths[s:e]
    chunk_rows = rows[s:e]
    print(
        f"[DEBUG Step1 chunk {ci + 1}/{ranges_len}] "
        f"images {s + 1}-{e} ({len(chunk_paths)} files) → API…",
        flush=True,
    )
    payload = {
        **common_payload,
        "screenshot_window_steps": chunk_rows,
        "screenshot_subset": {
            "chunk_index": ci + 1,
            "chunk_count": ranges_len,
            "window_size": eff,
            "stride": eff,
            "requested_chunk_size": chunk_size,
            "effective_chunk_size": eff,
            "global_screenshot_index_1based": [s + 1, e],
            "images_in_this_request": len(chunk_paths),
            "total_screenshots_available": len(paths),
            "filenames": [p.name for p in chunk_paths],
            "note": (
                "Extract every visible UI element in this chunk only. "
                "Do not judge task requirements."
            ),
        },
    }
    out = run_step1_window_ocr(
        chat_json=chat_json,
        model=model,
        common_payload=payload,
        image_paths=chunk_paths,
        request_timeout=request_timeout,
    )
    if isinstance(out, dict):
        out["window_index"] = ci + 1
    n_ocr = len(out.get("screenshot_ocr") or []) if isinstance(out, dict) else 0
    print(f"[DEBUG Step1 chunk {ci + 1}/{ranges_len}] done, ocr_rows={n_ocr}", flush=True)
    return out


def run_step1_window_ocr_chunked(
    *,
    chat_json,
    model: str,
    common_payload: dict[str, Any],
    image_paths: list,
    shot_rows: list[dict[str, Any]] | None,
    request_timeout: float,
    chunk_size: int,
    chunk_workers: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run Step 1 once per screenshot window (``chunk_size`` images per call)."""
    paths = list(image_paths)
    rows = list(shot_rows) if shot_rows is not None else []
    if len(rows) != len(paths):
        rows = [{"step": None, "screenshot": p.name, "thought": ""} for p in paths]

    eff = effective_step1_chunk_size(chunk_size, len(paths))
    if len(paths) <= eff:
        print(f"[DEBUG Step1 chunk 1/1] images 1-{len(paths)} → API…", flush=True)
        single = run_step1_window_ocr(
            chat_json=chat_json,
            model=model,
            common_payload=common_payload,
            image_paths=paths,
            request_timeout=request_timeout,
        )
        n_ocr = len(single.get("screenshot_ocr") or []) if isinstance(single, dict) else 0
        print(f"[DEBUG Step1 chunk 1/1] done, ocr_rows={n_ocr}", flush=True)
        return single, [single]

    ranges = image_chunk_index_ranges(len(paths), eff)
    parallel = max(1, chunk_workers)
    raw_chunks: list[dict[str, Any] | None] = [None] * len(ranges)

    if parallel <= 1 or len(ranges) <= 1:
        for ci, (s, e) in enumerate(ranges):
            raw_chunks[ci] = _run_step1_single_chunk(
                ci=ci,
                s=s,
                e=e,
                ranges_len=len(ranges),
                chat_json=chat_json,
                model=model,
                common_payload=common_payload,
                paths=paths,
                rows=rows,
                eff=eff,
                chunk_size=chunk_size,
                request_timeout=request_timeout,
            )
    else:
        with ThreadPoolExecutor(max_workers=min(parallel, len(ranges))) as executor:
            futures = {
                executor.submit(
                    _run_step1_single_chunk,
                    ci=ci,
                    s=s,
                    e=e,
                    ranges_len=len(ranges),
                    chat_json=chat_json,
                    model=model,
                    common_payload=common_payload,
                    paths=paths,
                    rows=rows,
                    eff=eff,
                    chunk_size=chunk_size,
                    request_timeout=request_timeout,
                ): ci
                for ci, (s, e) in enumerate(ranges)
            }
            for fut in as_completed(futures):
                ci = futures[fut]
                raw_chunks[ci] = fut.result()

    ordered = [c for c in raw_chunks if isinstance(c, dict)]
    merged = merge_step1_window_outputs(ordered)
    return merged, ordered


def _step1_merged_extraction_empty(merged: dict[str, Any]) -> bool:
    """True when screenshot_ocr is empty after merge."""
    if not isinstance(merged, dict):
        return True
    ocr = merged.get("screenshot_ocr")
    return not isinstance(ocr, list) or len(ocr) == 0


def run_step1_window_ocr_with_fallback(
    *,
    chat_json,
    model: str,
    common_payload: dict[str, Any],
    image_paths: list,
    shot_rows: list[dict[str, Any]] | None,
    request_timeout: float,
    chunk_size: int,
    chunk_workers: int = 1,
    retry_on_empty: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Run Step 1 (chunked per ``image_chunk_size``). Optional full re-run if extraction is empty."""
    n = len(image_paths)
    eff = effective_step1_chunk_size(chunk_size, n)
    merged, raw_chunks = run_step1_window_ocr_chunked(
        chat_json=chat_json,
        model=model,
        common_payload=common_payload,
        image_paths=image_paths,
        shot_rows=shot_rows,
        request_timeout=request_timeout,
        chunk_size=chunk_size,
        chunk_workers=chunk_workers,
    )
    meta: dict[str, Any] = {
        "step1_requested_chunk_size": chunk_size if chunk_size > 0 else None,
        "step1_effective_chunk_size": eff if n else None,
        "step1_chunk_workers": max(1, chunk_workers),
        "step1_llm_passes": 1,
    }

    if retry_on_empty and n > 0 and _step1_merged_extraction_empty(merged):
        print(
            "[WARN Step1 empty extraction] "
            f"screenshots={n}, chunk_size={chunk_size}; retrying once (same chunk size)"
        )
        merged_retry, raw_chunks_retry = run_step1_window_ocr_chunked(
            chat_json=chat_json,
            model=model,
            common_payload=common_payload,
            image_paths=image_paths,
            shot_rows=shot_rows,
            request_timeout=request_timeout,
            chunk_size=chunk_size,
            chunk_workers=chunk_workers,
        )
        meta["step1_llm_passes"] = 2
        if not _step1_merged_extraction_empty(merged_retry):
            merged, raw_chunks = merged_retry, raw_chunks_retry
            meta["step1_empty_retry"] = "used_retry"
        else:
            meta["step1_empty_retry"] = "retry_still_empty"
    elif retry_on_empty and n > 0:
        meta["step1_empty_retry"] = "skipped_not_empty"
    elif retry_on_empty:
        meta["step1_empty_retry"] = "skipped_no_images"

    return merged, raw_chunks, meta
