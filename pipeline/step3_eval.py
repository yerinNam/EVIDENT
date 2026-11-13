from __future__ import annotations

import json
from typing import Any

from .step2_eval import build_step3_evidence_catalog  # unified pipeline uses same catalog builder

STEP3_GLOBAL_RUBRIC_PROMPT = """You are the final audit judge for a GUI/web agent trajectory.

Your role is to audit four separate aspects:

1. TSR / outcome:
   Whether the final answer successfully completes the user's task.

2. EGSR / process grounding:
   Whether the trajectory observed the task-required evidence checks.

3. ETSR / answer grounding:
   Whether the final answer's core claims are explicitly supported by visible evidence.

4. Rubric diagnostics:
   Which qualitative rubric dimensions pass or fail.

Use ONLY:
- input.instruction
- input.final_answer
- input.task_spec
- input.eval_checkpoints
- input.rubric_dimension_ids
- input.rubric_dimensions
- evidence_catalog (only Step 1 rows referenced by Step 2 supporting_evidence_ids)
- step2_claim_evidence

Do NOT use:
- Step 1 OCR or visual entries not listed in evidence_catalog
- agent thoughts
- actions as evidence
- task metadata as proof of site content
- general knowledge as proof of site content
- facts not present in evidence_catalog

Step 2 results are provisional hints only.
You may agree with or override Step 2.

# Core distinction

Separate these judgements:

1. answer_success
Whether the final answer usefully completes the user request.
This is TSR / outcome success.
Do not require evidence grounding for answer_success.

2. metadata_requirement_verified
Whether every required evidence check in input.eval_checkpoints is verified by evidence_catalog.
This is EGSR / process evidence success.
It checks whether the agent actually observed the required evidence during the trajectory.

3. key_claims_all_supported
Whether every core claim in the final answer is explicitly supported by evidence_catalog.
This is ETSR / final answer grounding quality.
It checks whether the answer itself is traceable to the observed evidence.

Label tiers (computed in Stage 5; do not mix them up):

- ETSR_PASS — tsr=true, egsr=true, etsr=true: answer succeeds, trajectory verified required evidence checks, and every core final-answer claim is grounded.
- CLAIM_UNGROUNDED_PASS — tsr=true, egsr=true, etsr=false: answer succeeds and evidence checks passed, but some final-answer claims (e.g. unsupported recommendation reasons) are not grounded.
- UNGROUNDED_PASS — tsr=true, egsr=false: answer succeeds on outcome, but the trajectory did not verify required evidence checks.
- FAIL — tsr=false: answer did not complete the task.

Example (claim-level ungrounding with good process):
- answer_success=true, metadata_requirement_verified=true, key_claims_all_supported=false
- egsr=true, etsr=false
- label="CLAIM_UNGROUNDED_PASS"

# Stage 1. Claim audit

Audit the substantive claims in final_answer.

Use Step 2 claims if available, but add missing substantive claims if necessary.

Each claim must be one of:

1. existence
The claim says that a named entity, option, page, category, result, value, or visible property exists.
Named item and visible value claims are existence claims.

2. reason
The claim gives a rationale, interpretation, suitability judgement, comparison, recommendation reason, inferred attribute, effect, or quality judgement.
Rationale, suitability, inferred attribute, effect, or quality claims are reason claims.

For each claim, output:
- claim
- claim_type: "existence" or "reason"
- is_core_claim
- supported
- supporting_evidence_ids
- evidence_quote

Claim support rules:
- supported=true only if every important fact in the claim is explicitly supported by evidence_catalog.
- Partial support is not enough.
- If a claim adds details beyond the cited evidence, supported=false.
- A visible name, title, category, brand, image, or listing can support existence.
- A visible name, title, category, brand, image, or listing does NOT by itself support hidden qualities, effects, suitability, or recommendation reasons.
- Reason claims require explicit evidence for the stated rationale.
- Plausible inference is not evidence.
- General knowledge is not evidence.

# Explicit evidence rule for reason claims

Reason claims are supported ONLY when the exact rationale is explicitly visible in evidence_catalog.

Do NOT mark a reason claim as supported using:
- implication
- common sense
- cultural norms
- general domain knowledge
- typical product/recipe/menu knowledge
- likely user preference
- inferred suitability
- inferred functionality
- inferred quality
- inferred taste
- inferred ease
- inferred popularity beyond visible metrics

If your support explanation uses words like:
"암시", "시사", "일반적으로", "보통", "likely", "suggests", "implies", "seems", "can be inferred"
then the claim is NOT explicitly supported and must be supported=false.

For recipe, food, product, travel, shopping, and recommendation tasks:
A visible title, category, result count, view count, rating, image, or listing can support only:
- the item exists
- the item appears in search results
- the visible title/category/value/rating/view count exists

It cannot support:
- ease of cooking or use
- beginner friendliness
- suitability for parents/children/guests
- taste, healthiness, spiciness, freshness, lightness
- visual impressiveness
- ingredients
- cooking method
- exact preparation steps
- menu pairing rationale
- recommendation reason
unless those facts are explicitly visible in evidence_catalog.

If a final answer recommends an item that is not explicitly visible in evidence_catalog, that recommendation claim must be supported=false.

# Stage 2. Evidence-check audit

Audit every item in input.eval_checkpoints.

For each checkpoint, output:
- requirement
- verified
- supporting_evidence_ids
- supporting_checkpoint_ids
- notes

verified=true only if the requirement is clearly satisfied by evidence_catalog.
You may combine multiple evidence rows.
If a checkpoint has multiple required conditions, all must be supported.

metadata_requirement_verified=true iff all mandatory evidence checks are verified.

This field is process-oriented:
- It evaluates whether the trajectory observed the required evidence.
- It does NOT evaluate whether the final answer's claims are all grounded.
- Final answer grounding is handled separately by key_claims_all_supported and ETSR.

# Stage 3. Answer success / TSR

answer_success=true if the final answer completes the user's task in a useful way.

Do not fail answer_success only because some claims are not grounded.
Fail answer_success when the answer is wrong, incomplete, off-task, or does not satisfy the requested output.

tsr=true iff answer_success=true.

# Stage 4. Rubric diagnostics

For every id in input.rubric_dimension_ids, output one boolean in dimension_scores.

Rubric dimensions are diagnostic.
They should explain what kind of capability passed or failed.
Do not use rubric dimensions to compute TSR, EGSR, or ETSR.

Important distinction:
- metadata_requirement_verified evaluates process evidence checks.
- key_claims_all_supported evaluates final answer claim grounding.
- dimension_scores provide additional diagnostic breakdown.

# Stage 5. Final consistency calculation

Compute these fields exactly:

key_claims_all_supported =
true iff every final_answer_claims row with is_core_claim=true has supported=true

tsr =
true iff:
- answer_success=true

egsr =
true iff:
- answer_success=true
- metadata_requirement_verified=true

etsr =
true iff:
- answer_success=true
- metadata_requirement_verified=true
- key_claims_all_supported=true

rss =
1 iff every dimension in input.rubric_dimension_ids has dimension_scores[id]=true
otherwise 0

rss is diagnostic only.
Do not use rss to compute tsr, egsr, or etsr.

label =
- "ETSR_PASS" iff etsr=true
  (answer_success=true, metadata_requirement_verified=true, key_claims_all_supported=true)
- "CLAIM_UNGROUNDED_PASS" iff answer_success=true,
  metadata_requirement_verified=true,
  and key_claims_all_supported=false
- "UNGROUNDED_PASS" iff answer_success=true and metadata_requirement_verified=false
- "FAIL" iff answer_success=false
- "CRITICAL_FAIL" only for severe unrelated, fabricated, unsafe, or impossible-to-recover answers

dimension_summary:
- satisfied: ids with dimension_scores[id]=true
- not_satisfied: ids with dimension_scores[id]=false
- not_applicable: ids that are missing or not applicable

passed_criteria must equal dimension_summary.satisfied.
failed_criteria must equal dimension_summary.not_satisfied.

# Forbidden inconsistent outputs

Never output:
- tsr=true and answer_success=false
- egsr=true and answer_success=false
- egsr=true and metadata_requirement_verified=false
- etsr=true and egsr=false
- etsr=true and key_claims_all_supported=false
- answer_success=true and metadata_requirement_verified=true and egsr=false
- answer_success=true and metadata_requirement_verified=true and key_claims_all_supported=true and etsr=false
- etsr=true and label!="ETSR_PASS"
- etsr=false and label="ETSR_PASS"
- egsr=true and etsr=false and label!="CLAIM_UNGROUNDED_PASS"
- egsr=false and label="CLAIM_UNGROUNDED_PASS"
- answer_success=true and egsr=false and label!="UNGROUNDED_PASS"
- answer_success=false and label in ("ETSR_PASS", "CLAIM_UNGROUNDED_PASS", "UNGROUNDED_PASS")
- dimension_summary that disagrees with dimension_scores
- passed_criteria or failed_criteria that disagree with dimension_summary

Return ONLY valid JSON:

{
  "answer_success": boolean,
  "tsr": boolean,
  "metadata_requirement_verified": boolean,
  "metadata_requirement_checks": [
    {
      "requirement": string,
      "verified": boolean,
      "supporting_evidence_ids": [string],
      "supporting_checkpoint_ids": [string],
      "notes": string
    }
  ],
  "egsr": boolean,
  "etsr": boolean,
  "label": "ETSR_PASS" | "CLAIM_UNGROUNDED_PASS" | "UNGROUNDED_PASS" | "FAIL" | "CRITICAL_FAIL",
  "rss": 0 | 1,
  "egsr_notes": string,
  "etsr_notes": string,
  "key_claims_all_supported": boolean,
  "dimension_scores": { "<dimension_id>": boolean },
  "ddr_applicable": boolean,
  "ddr_pass": boolean | null,
  "ddr_notes": string,
  "final_answer_claims": [
    {
      "claim": string,
      "claim_type": "existence" | "reason",
      "is_core_claim": boolean,
      "supported": boolean,
      "supporting_evidence_ids": [string],
      "evidence_quote": string
    }
  ],
  "rubric_profile": string,
  "dimension_summary": {
    "satisfied": [string],
    "not_satisfied": [string],
    "not_applicable": [string]
  },
  "checkpoint_success": {
    "site_used": boolean,
    "metadata_requirement_verified": boolean,
    "rubric_evidence_coherent": boolean,
    "final_answer_complete": boolean
  },
  "passed_criteria": [string],
  "failed_criteria": [string],
  "rationale": string
}
"""



def _coerce_bool(v: Any) -> bool | None:
    if v is True or v is False:
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "1", "pass"):
            return True
        if s in ("false", "no", "0", "fail"):
            return False
    return None


def _dimension_pass_numeric(v: Any) -> float | None:
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


_INVALID_DIMENSION_SUMMARY_TOKENS = frozenset(
    {"true", "false", "pass", "fail", "yes", "no", "1", "0"}
)


def _is_rubric_dimension_id_string(x: Any) -> bool:
    if not isinstance(x, str):
        return False
    s = x.strip()
    if not s or s.lower() in _INVALID_DIMENSION_SUMMARY_TOKENS:
        return False
    return True


def _collect_dimension_ids_from_summary_value(vals: Any) -> list[str]:
    """Parse list or dict-shaped summary fragments; drop booleans and true/false strings."""
    if isinstance(vals, dict):
        out: list[str] = []
        for k, v in vals.items():
            if not _is_rubric_dimension_id_string(k):
                continue
            kid = k.strip()
            if isinstance(v, bool):
                continue
            pv = _dimension_pass_numeric(v)
            if pv is None:
                out.append(kid)
            elif pv >= 1.0:
                out.append(kid)
        return out
    if not isinstance(vals, list):
        return []
    out_list: list[str] = []
    for item in vals:
        if isinstance(item, bool):
            continue
        if _is_rubric_dimension_id_string(item):
            out_list.append(item.strip())
    return out_list


def _build_dimension_summary_from_scores(
    dimension_scores: dict[str, Any],
    *,
    dim_ids: list[str] | None = None,
) -> dict[str, list[str]]:
    satisfied: list[str] = []
    not_satisfied: list[str] = []
    not_applicable: list[str] = []
    ids = list(dim_ids) if dim_ids else list(dimension_scores.keys())
    for did in ids:
        if not isinstance(did, str) or not did.strip():
            continue
        key = did.strip()
        if key not in dimension_scores:
            not_applicable.append(key)
            continue
        pv = _dimension_pass_numeric(dimension_scores.get(key))
        if pv is None:
            not_applicable.append(key)
        elif pv >= 1.0:
            satisfied.append(key)
        else:
            not_satisfied.append(key)
    return {
        "satisfied": satisfied,
        "not_satisfied": not_satisfied,
        "not_applicable": not_applicable,
    }


def _dimension_summary_needs_rebuild(
    summary: Any,
    dimension_scores: dict[str, Any],
) -> bool:
    if not isinstance(dimension_scores, dict) or not dimension_scores:
        return False
    if not isinstance(summary, dict):
        return True
    saw_valid_id = False
    for bucket in ("satisfied", "not_satisfied", "not_applicable"):
        vals = summary.get(bucket)
        if vals is None:
            continue
        if not isinstance(vals, (list, dict)):
            return True
        for item in vals if isinstance(vals, list) else vals.keys():
            if isinstance(item, bool):
                return True
            if isinstance(item, str) and item.strip().lower() in _INVALID_DIMENSION_SUMMARY_TOKENS:
                return True
            if isinstance(item, str) and _is_rubric_dimension_id_string(item):
                saw_valid_id = True
        if isinstance(vals, dict):
            for k in vals:
                if isinstance(k, bool) or (
                    isinstance(k, str) and k.strip().lower() in _INVALID_DIMENSION_SUMMARY_TOKENS
                ):
                    return True
                if _is_rubric_dimension_id_string(k):
                    saw_valid_id = True
    if not saw_valid_id:
        return True
    return False


def normalize_dimension_summary(
    judgement: dict[str, Any],
    *,
    rubric_dimension_ids: list[str] | None = None,
) -> dict[str, list[str]]:
    """Ensure dimension_summary lists contain rubric id strings, rebuilt from dimension_scores if needed."""
    ds = judgement.get("dimension_scores")
    dimension_scores = ds if isinstance(ds, dict) else {}
    dim_ids = rubric_dimension_ids
    if not dim_ids and dimension_scores:
        dim_ids = list(dimension_scores.keys())

    summary = judgement.get("dimension_summary")
    if _dimension_summary_needs_rebuild(summary, dimension_scores):
        return _build_dimension_summary_from_scores(dimension_scores, dim_ids=dim_ids)

    assert isinstance(summary, dict)
    return {
        "satisfied": _collect_dimension_ids_from_summary_value(summary.get("satisfied")),
        "not_satisfied": _collect_dimension_ids_from_summary_value(summary.get("not_satisfied")),
        "not_applicable": _collect_dimension_ids_from_summary_value(summary.get("not_applicable")),
    }


def normalize_step3_judgement(
    data: dict[str, Any],
    *,
    rubric_dimension_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Final schema: claims use ``supported`` only; drop stray ``satisfied`` on claim rows."""
    if not isinstance(data, dict):
        return data
    out = dict(data)

    claims = out.get("final_answer_claims")
    if isinstance(claims, list):
        fixed: list[dict[str, Any]] = []
        for row in claims:
            if not isinstance(row, dict):
                continue
            r = dict(row)
            if "supported" not in r:
                sat = _coerce_bool(r.get("satisfied"))
                if sat is not None:
                    r["supported"] = sat
            r.pop("satisfied", None)
            fixed.append(r)
        out["final_answer_claims"] = fixed

    checks = out.get("metadata_requirement_checks")
    if isinstance(checks, list):
        fixed_checks: list[dict[str, Any]] = []
        for row in checks:
            if not isinstance(row, dict):
                continue
            r = dict(row)
            if "verified" not in r:
                sat = _coerce_bool(r.get("satisfied"))
                if sat is not None:
                    r["verified"] = sat
            r.pop("satisfied", None)
            fixed_checks.append(r)
        out["metadata_requirement_checks"] = fixed_checks

    dim_ids = rubric_dimension_ids
    if not dim_ids:
        inp = out.get("rubric_dimension_ids")
        if isinstance(inp, list):
            dim_ids = [str(x) for x in inp if isinstance(x, str) and x.strip()]
    out["dimension_summary"] = normalize_dimension_summary(
        out, rubric_dimension_ids=dim_ids
    )

    return out


def step2_claim_evidence_for_step3(step2_claim_evidence: dict[str, Any]) -> dict[str, Any]:
    """Strip Step 2 pipeline metadata before sending to the Step 3 judge."""
    if not isinstance(step2_claim_evidence, dict):
        return {"final_answer_claims": [], "evidence_check_reviews": []}
    return {
        "final_answer_claims": step2_claim_evidence.get("final_answer_claims") or [],
        "evidence_check_reviews": step2_claim_evidence.get("evidence_check_reviews") or [],
    }


def run_step3_global_rubric_judgement(
    *,
    chat_json,
    model: str,
    common_payload: dict[str, Any],
    step1_merged_ocr: dict[str, Any],
    step2_claim_evidence: dict[str, Any],
    request_timeout: float,
) -> dict[str, Any]:
    evidence_catalog = build_step3_evidence_catalog(step1_merged_ocr, step2_claim_evidence)
    step2_for_step3 = step2_claim_evidence_for_step3(step2_claim_evidence)

    user_text = (
        "Step 3. Independently audit Step 2. evidence_catalog contains only Step 2-cited "
        "screenshot OCR/visual rows (not the full Step 1 dump). "
        "Step 2 satisfied values are weak hints only; overturn them when evidence says otherwise. "
        "Output final_answer_claims with supported only; metadata_requirement_checks with verified; "
        "dimension_summary = arrays of rubric dimension id strings, not booleans. "
        "Then compute rubric, RSS (diagnostic), EGSR, ETSR, and label.\n\n"
        + json.dumps(
            {
                "input": common_payload,
                "evidence_catalog": evidence_catalog,
                "step2_claim_evidence": step2_for_step3,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    raw = chat_json(
        model=model,
        system_prompt=STEP3_GLOBAL_RUBRIC_PROMPT,
        user_text=user_text,
        image_paths=[],
        request_timeout=request_timeout,
    )
    dim_ids = common_payload.get("rubric_dimension_ids")
    rubric_dim_ids = (
        [str(x) for x in dim_ids if isinstance(x, str) and x.strip()]
        if isinstance(dim_ids, list)
        else None
    )
    return normalize_step3_judgement(raw, rubric_dimension_ids=rubric_dim_ids)
