from __future__ import annotations

import json
from typing import Any
import re

STEP2_API_SYSTEM_PROMPT = """
You are Step 2 evidence mapping for GUI/web trajectory evaluation.

`evidence_catalog` is built from Step 1 `screenshot_ocr` and `visual_elements` rows.
OCR entry ids look like `screen_ocr_1`, `screen_ocr_2`, ...; visual entry ids look like `screen_visual_1`, `screen_visual_2`, ...
Pick `supporting_evidence_ids` from those ids and quote the matching snippet in `evidence_quote`.

Return ONLY valid JSON with exactly these top-level keys:
- final_answer_claims
- evidence_check_reviews

Critical rule:
Evidence fields may be empty, but required rows must never be omitted.

If input.final_answer is non-empty:
- final_answer_claims MUST contain at least one row.
- Even unsupported claims must be included.
- For unsupported claims, use:
  supporting_evidence_ids=[]
  evidence_quote=""
  satisfied=false

If input.eval_checkpoints has N items:
- evidence_check_reviews MUST contain exactly N rows.
- Preserve checkpoint order.
- Copy checkpoint id into checkpoint_id.
- Even unsupported checkpoints must be included.
- For unsupported checkpoints, use:
  supporting_evidence_ids=[]
  evidence_quote=""
  satisfied=false
  notes explaining what is missing.

Use `satisfied`, not `supported` or `verified`.

A visible title/search result supports only existence.
Recommendation reasons, suitability, ease, popularity, taste, healthiness, ingredients, quantities, or methods require explicit visible evidence.

Required output shape:
{
  "final_answer_claims": [
    {
      "claim": "string",
      "claim_type": "existence",
      "is_core_claim": false,
      "supporting_evidence_ids": [],
      "evidence_quote": "",
      "satisfied": false
    }
  ],
  "evidence_check_reviews": [
    {
      "checkpoint_id": "ec_0",
      "description": "string",
      "supporting_evidence_ids": [],
      "evidence_quote": "",
      "satisfied": false,
      "notes": "No direct visible evidence."
    }
  ]
}
"""


def _normalize_step2_claim_row(row: dict[str, Any]) -> dict[str, Any]:
    r = dict(row)
    if not r.get("claim"):
        for key in ("claim_text", "text", "statement"):
            if isinstance(r.get(key), str) and r[key].strip():
                r["claim"] = r[key].strip()
                break
    if not r.get("supporting_evidence_ids") and isinstance(r.get("evidence_ids"), list):
        r["supporting_evidence_ids"] = [str(x) for x in r["evidence_ids"] if str(x).strip()]
    claim_type = str(r.get("claim_type") or "").strip().lower()
    if claim_type not in ("existence", "reason"):
        claim_type = "reason" if r.get("is_core_claim") else "existence"
    r["claim_type"] = claim_type
    if "is_core_claim" not in r:
        r["is_core_claim"] = claim_type == "reason"
    else:
        r["is_core_claim"] = bool(r["is_core_claim"])
    r.setdefault("supporting_evidence_ids", [])
    r.setdefault("evidence_quote", "")
    if "satisfied" not in r and "supported" in r:
        r["satisfied"] = r["supported"]
    r.setdefault("satisfied", False)
    r.pop("supported", None)
    for k in ("claim_id", "claim_text", "text", "statement", "evidence", "evidence_ids"):
        r.pop(k, None)
    return r


def _normalize_step2_review_row(row: dict[str, Any]) -> dict[str, Any]:
    r = dict(row)
    if not r.get("description"):
        for key in ("requirement", "checkpoint_description", "text"):
            if isinstance(r.get(key), str) and r[key].strip():
                r["description"] = r[key].strip()
                break
    if not r.get("supporting_evidence_ids") and isinstance(r.get("evidence_ids"), list):
        r["supporting_evidence_ids"] = [str(x) for x in r["evidence_ids"] if str(x).strip()]
    r.setdefault("supporting_evidence_ids", [])
    r.setdefault("evidence_quote", "")
    if "satisfied" not in r and r.get("verified") is not None:
        r["satisfied"] = r["verified"]
    r.setdefault("satisfied", False)
    r.setdefault("notes", "")
    r.pop("supported", None)
    r.pop("verified", None)
    for k in ("evidence", "evidence_ids", "review_text", "requirement", "checkpoint_description", "text"):
        r.pop(k, None)
    return r


def normalize_step2_output(data: dict[str, Any]) -> dict[str, Any]:
    """Step 2 schema: `satisfied` only on claims/checkpoints."""
    if not isinstance(data, dict):
        return {}

    out = dict(data)

    # Accept common alias keys from models
    if "final_answer_claims" not in out:
        for key in ("claims", "claim_reviews", "claim_evidence", "final_claims"):
            if isinstance(out.get(key), list):
                out["final_answer_claims"] = out[key]
                break

    if "evidence_check_reviews" not in out:
        for key in ("checkpoint_reviews", "checkpoints", "evidence_reviews", "metadata_requirement_checks"):
            if isinstance(out.get(key), list):
                out["evidence_check_reviews"] = out[key]
                break

    claims = out.get("final_answer_claims")
    if isinstance(claims, list):
        fixed_claims: list[dict[str, Any]] = []
        for row in claims:
            if not isinstance(row, dict):
                continue
            fixed_claims.append(_normalize_step2_claim_row(row))

        out["final_answer_claims"] = fixed_claims
    else:
        out["final_answer_claims"] = []

    reviews = out.get("evidence_check_reviews")
    if isinstance(reviews, list):
        fixed_reviews: list[dict[str, Any]] = []
        for row in reviews:
            if not isinstance(row, dict):
                continue
            fixed_reviews.append(_normalize_step2_review_row(row))

        out["evidence_check_reviews"] = fixed_reviews
    else:
        out["evidence_check_reviews"] = []

    return out

def build_evidence_catalog_from_merged(step1_merged: dict[str, Any]) -> list[dict[str, str]]:
    """Build Step 2 evidence catalog from Step 1 screenshot OCR and visual_elements."""
    evidence_catalog: list[dict[str, str]] = []

    ocr_rows = step1_merged.get("screenshot_ocr") if isinstance(step1_merged, dict) else []
    if isinstance(ocr_rows, list):
        for i, row in enumerate(ocr_rows):
            if not isinstance(row, dict):
                continue

            idx = row.get("screenshot_index_in_request")
            step = row.get("trajectory_step")
            filename = row.get("screenshot_filename")

            parts: list[str] = []

            ocr_text = row.get("ocr_text")
            if isinstance(ocr_text, str) and ocr_text.strip():
                parts.append(f"ocr_text: {ocr_text.strip()}")

            uncertain = row.get("unreadable_or_uncertain")
            if isinstance(uncertain, str) and uncertain.strip():
                parts.append(f"uncertain: {uncertain.strip()}")

            visible_support = " | ".join(parts).strip()
            if visible_support:
                evidence_catalog.append(
                    {
                        "evidence_id": f"screen_ocr_{i + 1}",
                        "visible_support": (
                            f"SCREEN_OCR step={step}, screenshot={filename}, "
                            f"index={idx}: {visible_support}"
                        ),
                        "source_kind": "screenshot_text",
                    }
                )

    visual_rows = step1_merged.get("visual_elements") if isinstance(step1_merged, dict) else []
    if isinstance(visual_rows, list):
        for i, row in enumerate(visual_rows):
            if not isinstance(row, dict):
                continue

            idx = row.get("screenshot_index_in_request")
            element_type = row.get("element_type")
            description = row.get("description")
            location = row.get("location")
            associated = row.get("associated_text")

            parts: list[str] = []
            if isinstance(element_type, str) and element_type.strip():
                parts.append(f"type: {element_type.strip()}")
            if isinstance(description, str) and description.strip():
                parts.append(f"description: {description.strip()}")
            if isinstance(location, str) and location.strip():
                parts.append(f"location: {location.strip()}")
            if isinstance(associated, str) and associated.strip():
                parts.append(f"associated_text: {associated.strip()}")

            visible_support = " | ".join(parts).strip()
            if visible_support:
                evidence_catalog.append(
                    {
                        "evidence_id": f"screen_visual_{i + 1}",
                        "visible_support": (
                            f"VISUAL_ELEMENT index={idx}: {visible_support}"
                        ),
                        "source_kind": "visual_element",
                    }
                )

    return evidence_catalog


def collect_step2_referenced_evidence_ids_ordered(
    step2_claim_evidence: dict[str, Any],
) -> list[str]:
    """Step 2 cited ids in claim/review order, first occurrence kept."""
    ordered: list[str] = []
    seen: set[str] = set()
    if not isinstance(step2_claim_evidence, dict):
        return ordered
    for key in ("final_answer_claims", "evidence_check_reviews"):
        rows = step2_claim_evidence.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            refs = row.get("supporting_evidence_ids")
            if not isinstance(refs, list):
                continue
            for ref in refs:
                s = str(ref).strip()
                if s and s not in seen:
                    seen.add(s)
                    ordered.append(s)
    return ordered


def build_step3_evidence_catalog(
    step1_merged: dict[str, Any],
    step2_claim_evidence: dict[str, Any],
) -> list[dict[str, str]]:
    """Step 3 catalog: only Step 1 rows whose ids appear in Step 2 supporting_evidence_ids."""
    full = build_evidence_catalog_from_merged(step1_merged)
    ordered_ids = collect_step2_referenced_evidence_ids_ordered(step2_claim_evidence)
    if not ordered_ids:
        return []
    by_id: dict[str, dict[str, str]] = {}
    for entry in full:
        if not isinstance(entry, dict):
            continue
        eid = str(entry.get("evidence_id") or "").strip()
        if eid:
            by_id[eid] = entry
    return [by_id[eid] for eid in ordered_ids if eid in by_id]


def split_final_answer_to_min_claims(final_answer: str) -> list[dict[str, Any]]:
    text = str(final_answer or "").strip()
    if not text:
        return []

    # Avoid over-splitting; produce minimal claims at line/sentence granularity
    candidates: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        line = line.strip("-*• \t")
        if not line:
            continue
        candidates.append(line)

    if len(candidates) <= 1:
        # Few lines: fall back to sentence-level splitting
        parts = re.split(r"(?<=[.!?。！？])\s+|(?<=[다요죠함음됨니다])\.\s*", text)
        candidates = [p.strip("-*• \t") for p in parts if p.strip()]

    rows: list[dict[str, Any]] = []
    for i, claim in enumerate(candidates[:10]):
        rows.append(
            {
                "claim": claim,
                "claim_type": "existence" if i == 0 else "reason",
                "is_core_claim": False if i == 0 else True,
                "supporting_evidence_ids": [],
                "evidence_quote": "",
                "satisfied": False,
            }
        )

    if not rows:
        rows.append(
            {
                "claim": text,
                "claim_type": "reason",
                "is_core_claim": True,
                "supporting_evidence_ids": [],
                "evidence_quote": "",
                "satisfied": False,
            }
        )

    return rows


    
def fill_step2_minimum_from_checkpoints(
    out: dict[str, Any],
    eval_checkpoints: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Ensure exactly one evidence_check_reviews row per checkpoint id."""
    if not eval_checkpoints:
        return out, False

    fixed = dict(out) if isinstance(out, dict) else {}
    reviews = fixed.get("evidence_check_reviews")
    if not isinstance(reviews, list):
        reviews = []

    valid_reviews: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in reviews:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("checkpoint_id") or "").strip()
        if not cid:
            continue
        if cid in seen:
            continue
        seen.add(cid)
        valid_reviews.append(row)

    synthesized = False

    for ck in eval_checkpoints:
        if not isinstance(ck, dict):
            continue

        cid = str(ck.get("id") or "").strip()
        if not cid:
            continue

        if cid in seen:
            continue

        valid_reviews.append(
            {
                "checkpoint_id": cid,
                "description": str(ck.get("description") or ""),
                "supporting_evidence_ids": [],
                "evidence_quote": "",
                "satisfied": False,
                "notes": "[synthesized] Step 2 omitted this checkpoint review; row was created as unsupported fallback.",
            }
        )
        seen.add(cid)
        synthesized = True

    fixed["evidence_check_reviews"] = valid_reviews

    if not isinstance(fixed.get("final_answer_claims"), list):
        fixed["final_answer_claims"] = []

    return fixed, synthesized


def _step2_both_lists_empty(out: dict[str, Any]) -> bool:
    """True when normalized output is only empty claims and empty reviews."""
    claims = out.get("final_answer_claims")
    reviews = out.get("evidence_check_reviews")
    claims_empty = not isinstance(claims, list) or len(claims) == 0
    reviews_empty = not isinstance(reviews, list) or len(reviews) == 0
    return claims_empty and reviews_empty


def _step2_input_needs_rows(common_payload: dict[str, Any]) -> bool:
    final_answer_text = str(common_payload.get("final_answer") or "").strip()
    eval_ck = common_payload.get("eval_checkpoints") or []
    has_ck = isinstance(eval_ck, list) and bool(eval_ck)
    return bool(final_answer_text) or has_ck


def _fetch_step2_mapping(
    *,
    chat_json,
    model: str,
    common_payload: dict[str, Any],
    evidence_catalog: list[dict[str, str]],
    request_timeout: float,
    attempt: int,
) -> dict[str, Any]:
    user_text = (
        "Create the required Step 2 JSON.\n"
        "When final_answer has text, include at least one claim row.\n"
        "When eval_checkpoints has items, include one review row per checkpoint id.\n\n"
        + json.dumps(
            {
                "input": common_payload,
                "evidence_catalog": evidence_catalog,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    raw = chat_json(
        model=model,
        system_prompt=STEP2_API_SYSTEM_PROMPT,
        user_text=user_text,
        image_paths=[],
        request_timeout=request_timeout,
    )
    print(f"[DEBUG Step2 raw response attempt={attempt}] {json.dumps(raw, ensure_ascii=False)}")
    return normalize_step2_output(raw)


def run_step2_claim_evidence_mapping(
    *,
    chat_json,
    model: str,
    common_payload: dict[str, Any],
    step1_merged_evidence: dict[str, Any],
    request_timeout: float,
    retry_on_empty: bool = False,
) -> dict[str, Any]:
    evidence_catalog = build_evidence_catalog_from_merged(step1_merged_evidence)

    out = _fetch_step2_mapping(
        chat_json=chat_json,
        model=model,
        common_payload=common_payload,
        evidence_catalog=evidence_catalog,
        request_timeout=request_timeout,
        attempt=1,
    )
    llm_calls = 1

    if (
        retry_on_empty
        and _step2_input_needs_rows(common_payload)
        and _step2_both_lists_empty(out)
    ):
        print("[WARN Step2 empty both lists] retrying mapping once")
        out_retry = _fetch_step2_mapping(
            chat_json=chat_json,
            model=model,
            common_payload=common_payload,
            evidence_catalog=evidence_catalog,
            request_timeout=request_timeout,
            attempt=2,
        )
        llm_calls = 2
        if not _step2_both_lists_empty(out_retry):
            out = out_retry
            out["step2_empty_retry"] = "used_retry"
        else:
            out["step2_empty_retry"] = "retry_still_empty"
    elif retry_on_empty:
        out["step2_empty_retry"] = "skipped_not_both_empty"

    final_answer_text = str(common_payload.get("final_answer") or "").strip()
    eval_ck = common_payload.get("eval_checkpoints") or []

    bad_empty_claims = bool(final_answer_text) and not out.get("final_answer_claims")
    bad_empty_reviews = isinstance(eval_ck, list) and bool(eval_ck) and not out.get("evidence_check_reviews")

    if bad_empty_claims or bad_empty_reviews:
        print(
            "[WARN Step2 invalid empty output] "
            f"bad_empty_claims={bad_empty_claims}, "
            f"bad_empty_reviews={bad_empty_reviews}, "
            f"final_answer_len={len(final_answer_text)}, "
            f"eval_checkpoints={len(eval_ck) if isinstance(eval_ck, list) else 0}, "
            f"evidence_catalog={len(evidence_catalog)}"
        )

    # fallback: force-generate claims if the list is empty
    if final_answer_text and not out.get("final_answer_claims"):
        out["final_answer_claims"] = split_final_answer_to_min_claims(final_answer_text)
        out["step2_claims_synthesized"] = True

    eval_ck = common_payload.get("eval_checkpoints")
    if isinstance(eval_ck, list):
        out, synthesized = fill_step2_minimum_from_checkpoints(out, eval_ck)
        if synthesized:
            out["step2_reviews_synthesized"] = True

    out["step2_llm_calls"] = llm_calls

    print(
        "[DEBUG Step2 final normalized output] "
        f"claims={len(out.get('final_answer_claims') or [])}, "
        f"reviews={len(out.get('evidence_check_reviews') or [])}, "
        f"claims_synth={out.get('step2_claims_synthesized')}, "
        f"reviews_synth={out.get('step2_reviews_synthesized')}, "
        f"empty_retry={out.get('step2_empty_retry')}, "
        f"llm_calls={llm_calls}"
    )

    return out

