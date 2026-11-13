"""Rubric dimension / checkpoint statistics for HTML reports (from two_pass_judge caches)."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .html_statistic import (
    filter_outputs_by_tasks,
    iter_statistics_slices,
    step3_judgement_from_payload,
)

# Keep in sync with main_eval._RUBRIC_DIMENSIONS (avoid importing main_eval / openai).
_RUBRIC_PROFILE_SPEC: dict[str, list[str]] = {
    "common": [
        "intent_grounding",
        "evidence_grounding",
        "response_completion",
    ],
    "semantic_grounding": ["ui_label_mapping"],
    "contextual_inference": [
        "context_to_requirement",
        "implicit_need_validity",
        "priority_tradeoff",
    ],
    "visual_grounding": [
        "visual_intent_grounding",
        "visual_target_disambiguation",
    ],
    "dead_end": ["impossibility_boundary"],
}


def rubric_profile_for_task(task_type: str | None, task_id: str | None) -> str:
    tt_compact = (task_type or "").lower().replace(" ", "").replace("-", "").replace("_", "")
    tid = (task_id or "").lower().replace("-", "_")
    if "deadend" in tid or "deadend" in tt_compact:
        return "dead_end"
    if "visual" in tt_compact:
        return "visual_grounding"
    if "contextual" in tt_compact:
        return "contextual_inference"
    if "semantic" in tt_compact or "graceful" in tt_compact:
        return "semantic_grounding"
    return "common"


def rubric_dimension_ids(profile: str) -> list[str]:
    common = _RUBRIC_PROFILE_SPEC["common"]
    if profile == "common":
        return list(common)
    spec = _RUBRIC_PROFILE_SPEC.get(profile) or []
    out = list(common)
    for did in spec:
        if did not in out:
            out.append(did)
    return out

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS_JSON = _PROJECT_ROOT / "task_gen" / "outputs" / "300" / "150_ko_metadata.json"
DEFAULT_RESULTS_ROOT = _PROJECT_ROOT / "agent_results" / "results_cua" / "final_gemini3" / "korean"
RUBRIC_HTML_FILENAME = "!__rubric.html"
RUBRIC_JSON_FILENAME = "!__rubric_stats.json"
RUBRIC_SUMMARY_JSON_NAME = "rubric_stats.json"

_CHECKPOINT_KEYS = (
    "site_used",
    "metadata_requirement_verified",
    "rubric_evidence_coherent",
    "final_answer_complete",
)

_DIM_STATUS_PASS = "pass"
_DIM_STATUS_FAIL = "fail"
_DIM_STATUS_NA = "na"


def dimension_status_from_judgement(judgement: dict[str, Any]) -> dict[str, str]:
    """Map rubric dimension id -> pass | fail | na."""
    summary = judgement.get("dimension_summary")
    if isinstance(summary, dict):
        out: dict[str, str] = {}
        for dim in summary.get("satisfied") or []:
            if isinstance(dim, str) and dim.strip():
                out[dim.strip()] = _DIM_STATUS_PASS
        for dim in summary.get("not_satisfied") or []:
            if isinstance(dim, str) and dim.strip():
                out[dim.strip()] = _DIM_STATUS_FAIL
        for dim in summary.get("not_applicable") or []:
            if isinstance(dim, str) and dim.strip():
                out[dim.strip()] = _DIM_STATUS_NA
        if out:
            return out

    passed = judgement.get("passed_criteria")
    failed = judgement.get("failed_criteria")
    if isinstance(passed, list) or isinstance(failed, list):
        out = {}
        for dim in passed or []:
            if isinstance(dim, str) and dim.strip():
                out[dim.strip()] = _DIM_STATUS_PASS
        for dim in failed or []:
            if isinstance(dim, str) and dim.strip():
                out[dim.strip()] = _DIM_STATUS_FAIL
        if out:
            return out

    scores = judgement.get("dimension_scores")
    if isinstance(scores, dict):
        from .html_statistic import dimension_pass_numeric

        out = {}
        for dim, raw in scores.items():
            if not isinstance(dim, str) or not dim.strip():
                continue
            pv = dimension_pass_numeric(raw)
            if pv is None:
                continue
            out[dim.strip()] = _DIM_STATUS_PASS if pv >= 1.0 else _DIM_STATUS_FAIL
        if out:
            return out

    return {}


def rubric_profile_from_payload(
    data: dict[str, Any],
    *,
    task_type: str | None = None,
    episode_id: str | None = None,
) -> str:
    judgement = step3_judgement_from_payload(data) or {}
    for block in (judgement, data.get("meta") or {}, data.get("derived_metrics") or {}):
        if isinstance(block, dict):
            profile = block.get("rubric_profile")
            if isinstance(profile, str) and profile.strip():
                return profile.strip()
    return rubric_profile_for_task(task_type, episode_id)


def extract_episode_rubric_row(
    episode_id: str,
    data: dict[str, Any],
    *,
    task_type: str,
) -> dict[str, Any] | None:
    if not isinstance(data, dict) or "error" in data:
        return None
    judgement = step3_judgement_from_payload(data)
    if not isinstance(judgement, dict) or not judgement:
        return None

    dim_status = dimension_status_from_judgement(judgement)
    profile = rubric_profile_from_payload(
        data, task_type=task_type, episode_id=episode_id
    )
    expected_dims = set(rubric_dimension_ids(profile))

    cp_raw = judgement.get("checkpoint_success")
    checkpoints: dict[str, bool | None] = {}
    if isinstance(cp_raw, dict):
        for key in _CHECKPOINT_KEYS:
            v = cp_raw.get(key)
            checkpoints[key] = v if isinstance(v, bool) else None

    applicable = [d for d, s in dim_status.items() if s != _DIM_STATUS_NA]
    all_pass = bool(applicable) and all(dim_status[d] == _DIM_STATUS_PASS for d in applicable)

    return {
        "episode_id": episode_id,
        "task_type": task_type,
        "rubric_profile": profile,
        "label": str(judgement.get("label") or "UNKNOWN"),
        "answer_success": bool(judgement.get("answer_success", False)),
        "egsr": bool(judgement.get("egsr")),
        "etsr": bool(judgement.get("etsr")),
        "rss": judgement.get("rss"),
        "dimension_status": dim_status,
        "expected_dimensions": sorted(expected_dims),
        "all_dimensions_pass": all_pass,
        "checkpoints": checkpoints,
    }


def _aggregate_dimension_counts(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """dim_id -> {pass, fail, na, applicable}."""
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"pass": 0, "fail": 0, "na": 0, "applicable": 0}
    )
    for row in rows:
        for dim, status in (row.get("dimension_status") or {}).items():
            bucket = counts[dim]
            if status == _DIM_STATUS_PASS:
                bucket["pass"] += 1
                bucket["applicable"] += 1
            elif status == _DIM_STATUS_FAIL:
                bucket["fail"] += 1
                bucket["applicable"] += 1
            elif status == _DIM_STATUS_NA:
                bucket["na"] += 1
    return dict(counts)


def _aggregate_checkpoint_counts(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for key in _CHECKPOINT_KEYS:
        pass_n = fail_n = missing = 0
        for row in rows:
            v = (row.get("checkpoints") or {}).get(key)
            if v is True:
                pass_n += 1
            elif v is False:
                fail_n += 1
            else:
                missing += 1
        out[key] = {
            "pass": pass_n,
            "fail": fail_n,
            "missing": missing,
            "judged": pass_n + fail_n,
        }
    return out


def calculate_rubric_slice_statistics(
    outputs: dict[str, Any],
    tasks_by_id: dict[str, dict[str, Any]] | None,
    *,
    slice_label: str,
    meta_ids: list[str],
) -> dict[str, Any]:
    tasks_by_id = tasks_by_id or {}
    rows: list[dict[str, Any]] = []
    judge_errors = 0
    label_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()

    for eid in meta_ids:
        data = outputs.get(eid)
        task_type = str((tasks_by_id.get(eid) or {}).get("task_type") or "Unknown")
        if data is None or (isinstance(data, dict) and "error" in data):
            judge_errors += 1
            continue
        row = extract_episode_rubric_row(eid, data, task_type=task_type)
        if row is None:
            judge_errors += 1
            continue
        rows.append(row)
        label_counts[row["label"]] += 1
        profile_counts[row["rubric_profile"]] += 1

    judged = len(rows)
    meta_total = len(meta_ids)
    dim_counts = _aggregate_dimension_counts(rows)
    checkpoint_counts = _aggregate_checkpoint_counts(rows)

    answer_ok = sum(1 for r in rows if r["answer_success"])
    egsr_ok = sum(1 for r in rows if r["egsr"])
    etsr_ok = sum(1 for r in rows if r["etsr"])
    rss_ok = sum(1 for r in rows if r.get("all_dimensions_pass"))
    ungrounded_pass = sum(
        1 for r in rows if r["answer_success"] and not r["etsr"]
    )

    dim_rates: dict[str, dict[str, Any]] = {}
    for dim, c in sorted(dim_counts.items()):
        app = c["applicable"]
        dim_rates[dim] = {
            **c,
            "pass_rate": (c["pass"] / app * 100.0) if app else None,
            "fail_rate": (c["fail"] / app * 100.0) if app else None,
        }

    cp_rates: dict[str, dict[str, Any]] = {}
    for key, c in checkpoint_counts.items():
        judged_cp = c["judged"]
        cp_rates[key] = {
            **c,
            "pass_rate": (c["pass"] / judged_cp * 100.0) if judged_cp else None,
        }

    return {
        "slice_label": slice_label,
        "meta_total": meta_total,
        "judged": judged,
        "judge_errors": judge_errors,
        "answer_success": answer_ok,
        "egsr_pass": egsr_ok,
        "etsr_pass": etsr_ok,
        "rss_all_dims_pass": rss_ok,
        "ungrounded_pass": ungrounded_pass,
        "answer_success_rate": (answer_ok / meta_total * 100.0) if meta_total else None,
        "egsr_rate": (egsr_ok / meta_total * 100.0) if meta_total else None,
        "etsr_rate": (etsr_ok / meta_total * 100.0) if meta_total else None,
        "rss_rate": (rss_ok / judged * 100.0) if judged else None,
        "ungrounded_pass_rate": (ungrounded_pass / meta_total * 100.0) if meta_total else None,
        "label_counts": dict(label_counts),
        "profile_counts": dict(profile_counts),
        "dimension_rates": dim_rates,
        "checkpoint_rates": cp_rates,
        "rows": rows,
    }


def calculate_rubric_statistics(
    outputs: dict[str, Any],
    tasks_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Overall + per task_type rubric aggregates."""
    outputs = filter_outputs_by_tasks(outputs, tasks_by_id)
    slices: dict[str, Any] = {}
    for label, meta_ids in iter_statistics_slices(outputs, tasks_by_id):
        slices[label] = calculate_rubric_slice_statistics(
            outputs, tasks_by_id, slice_label=label, meta_ids=meta_ids
        )
    return {"slices": slices}


def compact_rubric_report(report: dict[str, Any]) -> dict[str, Any]:
    """Drop per-episode rows; keep slice aggregates only."""
    out = {k: v for k, v in report.items() if k != "slices"}
    slices_out: dict[str, Any] = {}
    for label, slice_stats in (report.get("slices") or {}).items():
        if isinstance(slice_stats, dict):
            slices_out[label] = {k: v for k, v in slice_stats.items() if k != "rows"}
    out["slices"] = slices_out
    return out


def compute_rubric_report(
    *,
    results_root: Path,
    tasks_json: Path,
    cache_filename: str = "two_pass_judge.json",
    reconcile: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Load judge caches; return (report, outputs, tasks_by_id)."""
    from html_from_cache import load_outputs_from_cache
    from main_eval import _load_tasks_by_id

    tasks_by_id = _load_tasks_by_id(tasks_json)
    meta_ids = list(tasks_by_id.keys())
    outputs, load_counts = load_outputs_from_cache(
        results_root=results_root,
        tasks_by_id=tasks_by_id,
        cache_filename=cache_filename,
        episode_ids=meta_ids,
        reconcile=reconcile,
        include_missing=False,
    )
    stats = calculate_rubric_statistics(outputs, tasks_by_id)
    overall = stats["slices"].get("Overall") or {}
    report = {
        "results_root": str(results_root.resolve()),
        "tasks_json": str(tasks_json.resolve()),
        "cache_filename": cache_filename,
        "meta_total": int(overall.get("meta_total") or len(tasks_by_id)),
        "judged": int(overall.get("judged") or 0),
        "judge_errors": int(overall.get("judge_errors") or 0),
        "tsr_ok": int(overall.get("answer_success") or 0),
        "egsr_ok": int(overall.get("egsr_pass") or 0),
        "etsr_ok": int(overall.get("etsr_pass") or 0),
        "tsr_rate": overall.get("answer_success_rate"),
        "egsr_rate": overall.get("egsr_rate"),
        "etsr_rate": overall.get("etsr_rate"),
        "rss_all_dims_pass": int(overall.get("rss_all_dims_pass") or 0),
        "ungrounded_pass": int(overall.get("ungrounded_pass") or 0),
        "load_counts": load_counts,
        "slices": stats["slices"],
    }
    return report, outputs, tasks_by_id


def _pct(v: float | None, *, digits: int = 1) -> str:
    return f"{v:.{digits}f}%" if v is not None else "—"


def _render_dimension_table(dim_rates: dict[str, dict[str, Any]]) -> str:
    if not dim_rates:
        return '<p class="muted">No dimension_summary data in this slice.</p>'
    body_rows: list[str] = []
    for dim, d in dim_rates.items():
        app = d.get("applicable", 0)
        pass_n = d.get("pass", 0)
        fail_n = d.get("fail", 0)
        na_n = d.get("na", 0)
        pass_rate = d.get("pass_rate")
        fail_rate = d.get("fail_rate")
        pass_cls = "pf-pass" if pass_rate is not None and pass_rate >= 50 else "pf-fail"
        body_rows.append(
            "<tr>"
            f"<td><code>{html.escape(dim)}</code></td>"
            f"<td>{app}</td>"
            f'<td class="{pass_cls}">{pass_n}</td>'
            f'<td class="pf-fail">{fail_n}</td>'
            f"<td>{na_n}</td>"
            f"<td>{_pct(pass_rate)}</td>"
            f"<td>{_pct(fail_rate)}</td>"
            "</tr>"
        )
    return (
        '<table class="rubric-dim-table">'
        "<thead><tr>"
        "<th>Dimension</th><th>applicable</th><th>pass</th><th>fail</th>"
        "<th>n/a</th><th>pass %</th><th>fail %</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )


def _render_checkpoint_table(cp_rates: dict[str, dict[str, Any]]) -> str:
    if not cp_rates:
        return ""
    body_rows: list[str] = []
    for key, d in cp_rates.items():
        judged = d.get("judged", 0)
        body_rows.append(
            "<tr>"
            f"<td><code>{html.escape(key)}</code></td>"
            f"<td>{judged}</td>"
            f"<td>{d.get('pass', 0)}</td>"
            f"<td>{d.get('fail', 0)}</td>"
            f"<td>{d.get('missing', 0)}</td>"
            f"<td>{_pct(d.get('pass_rate'))}</td>"
            "</tr>"
        )
    return (
        '<table class="rubric-cp-table">'
        "<thead><tr>"
        "<th>Checkpoint</th><th>judged</th><th>pass</th><th>fail</th>"
        "<th>missing</th><th>pass %</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )


def _render_slice_block(slice_stats: dict[str, Any]) -> str:
    label = html.escape(str(slice_stats["slice_label"]))
    meta_total = slice_stats["meta_total"]
    judged = slice_stats["judged"]
    judge_errors = slice_stats["judge_errors"]

    headline = (
        f"<b>{label}</b> — n(meta)={meta_total}, judged={judged}, "
        f"judge_error={judge_errors} | "
        f"TSR {_pct(slice_stats.get('answer_success_rate'))} | "
        f"EGSR {_pct(slice_stats.get('egsr_rate'))} | "
        f"ETSR {_pct(slice_stats.get('etsr_rate'))} | "
        f"all-dims pass {_pct(slice_stats.get('rss_rate'))} (of judged) | "
        f"ungrounded pass {_pct(slice_stats.get('ungrounded_pass_rate'))}"
    )

    profiles = slice_stats.get("profile_counts") or {}
    profile_bits = ", ".join(
        f"{html.escape(k)}={v}" for k, v in sorted(profiles.items())
    )
    labels = slice_stats.get("label_counts") or {}
    label_bits = ", ".join(
        f"{html.escape(k)}={v}" for k, v in sorted(labels.items())
    )

    open_attr = " open" if slice_stats.get("slice_label") == "Overall" else ""
    return (
        f'<details class="rubric-slice-block"{open_attr}>'
        f"<summary>{headline}</summary>"
        '<div class="rubric-slice-inner">'
        f'<p class="muted" style="margin:4px 0 8px;font-size:12px;">'
        f"Profiles: {profile_bits or '—'} · Labels: {label_bits or '—'}</p>"
        "<h5 style=\"margin:12px 0 6px;font-size:0.85rem;\">Rubric dimensions</h5>"
        f"{_render_dimension_table(slice_stats.get('dimension_rates') or {})}"
        "<h5 style=\"margin:16px 0 6px;font-size:0.85rem;\">Checkpoints</h5>"
        f"{_render_checkpoint_table(slice_stats.get('checkpoint_rates') or {})}"
        "</div></details>"
    )


def render_rubric_statistics_html(
    outputs: dict[str, Any],
    tasks_by_id: dict[str, dict[str, Any]] | None = None,
) -> str:
    """HTML block: per task_type rubric dimension pass rates + checkpoints."""
    stats = calculate_rubric_statistics(outputs, tasks_by_id)
    blocks = [
        _render_slice_block(s)
        for s in stats["slices"].values()
    ]
    return (
        '<div class="rubric-stats-block">'
        '<h4 style="margin:0 0 8px;font-size:0.95rem;">Rubric dimensions by task type</h4>'
        '<p class="muted" style="margin:0 0 10px;font-size:12px;">'
        "Pass/fail % uses <code>dimension_summary</code> per episode "
        "(denominator = applicable = pass + fail). "
        "Rates vs meta n use tasks-json slice size (unjudged episodes count as judge_error only "
        "in headline TSR/EGSR/ETSR; dimension table uses judged episodes only).</p>"
        f"{''.join(blocks)}"
        "</div>"
    )


def save_rubric_statistics_json(
    path: Path,
    outputs: dict[str, Any],
    tasks_by_id: dict[str, dict[str, Any]] | None = None,
    *,
    report: dict[str, Any] | None = None,
) -> Path:
    if report is None:
        stats = calculate_rubric_statistics(outputs, tasks_by_id)
        report = {"slices": stats["slices"]}
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(compact_rubric_report(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


_RUBRIC_PAGE_STYLES = """
    :root { --bg:#1a1d23; --card:#252830; --text:#e8eaed; --muted:#9aa0a6; --accent:#8ab4f8; --border:#3c4043; }
    body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); margin:0; padding:24px; max-width:1200px; margin-inline:auto; line-height:1.45; }
    h1 { font-size:1.35rem; margin:0 0 12px; }
    .meta { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:12px 16px; font-size:13px; margin-bottom:20px; }
    .meta code { color:var(--accent); font-size:12px; word-break:break-all; }
    .muted { color:var(--muted); font-size:12px; }
    .pf-pass { color:#81c995; font-weight:600; }
    .pf-fail { color:#f28b82; font-weight:600; }
    table { border-collapse:collapse; width:100%; font-size:12px; margin-top:4px; }
    th,td { border:1px solid var(--border); padding:6px 8px; text-align:left; vertical-align:top; }
    th { background:#2d333b; color:var(--muted); }
    .rubric-stats-block { background:#1e2028; border:1px solid var(--border); border-radius:10px; padding:12px 16px; margin:0 0 16px 0; }
    .rubric-slice-block { margin:10px 0 0 0; border:1px solid var(--border); border-radius:8px; padding:8px 12px; background:#252830; }
    .rubric-slice-block > summary { cursor:pointer; color:var(--accent); font-size:13px; list-style:none; }
    .rubric-slice-block > summary::-webkit-details-marker { display:none; }
    .rubric-slice-inner { margin-top:10px; }
"""


def render_rubric_page_html(
    outputs: dict[str, Any],
    tasks_by_id: dict[str, dict[str, Any]] | None = None,
    *,
    title: str | None = None,
    report: dict[str, Any] | None = None,
) -> str:
    """Full standalone HTML page for rubric dimension statistics."""
    report = report or {}
    label = title or Path(str(report.get("results_root", ""))).name
    title_esc = html.escape(label)
    results_esc = html.escape(str(report.get("results_root", "")))
    tasks_esc = html.escape(str(report.get("tasks_json", "")))
    cache_esc = html.escape(str(report.get("cache_filename", "two_pass_judge.json")))

    load = report.get("load_counts") if isinstance(report.get("load_counts"), dict) else {}
    load_s = html.escape(
        f"loaded={load.get('loaded', 0)}, missing={load.get('missing', 0)}, "
        f"invalid={load.get('invalid', 0)}"
    )
    meta_total = int(report.get("meta_total") or 0)
    judged = int(report.get("judged") or 0)

    body_inner = render_rubric_statistics_html(outputs, tasks_by_id)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Rubric stats — {title_esc}</title>
  <style>{_RUBRIC_PAGE_STYLES}</style>
</head>
<body>
  <h1>Rubric statistics — {title_esc}</h1>
  <div class="meta">
    <p><b>results_root:</b> <code>{results_esc}</code></p>
    <p><b>tasks_json:</b> <code>{tasks_esc}</code></p>
    <p><b>cache:</b> <code>{cache_esc}</code> · <b>judged:</b> {judged}/{meta_total} · {load_s}</p>
    <p class="muted">Dimension pass/fail from <code>dimension_summary</code> in judge JSON. Denominator for pass % = applicable (pass + fail) per dimension.</p>
  </div>
  {body_inner}
</body>
</html>"""


def save_rubric_statistics_html(
    path: Path,
    outputs: dict[str, Any],
    tasks_by_id: dict[str, dict[str, Any]] | None = None,
    *,
    title: str | None = None,
    report: dict[str, Any] | None = None,
) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_rubric_page_html(
            outputs, tasks_by_id, title=title, report=report
        ),
        encoding="utf-8",
    )
    return path


def build_rubric_index_html(output_base: Path) -> Path:
    """Scan output_base for rubric_stats.json and write index.html."""
    output_base = output_base.resolve()
    rows: list[str] = []
    for json_path in sorted(output_base.rglob(RUBRIC_SUMMARY_JSON_NAME)):
        if json_path.parent == output_base:
            continue
        rel = json_path.parent.relative_to(output_base)
        try:
            report = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        html_path = json_path.parent / RUBRIC_HTML_FILENAME
        if not html_path.is_file():
            html_path = json_path.with_suffix(".html")
        rel_html = html_path.relative_to(output_base)
        label = str(rel).replace("/", " / ")
        overall = (report.get("slices") or {}).get("Overall") or report
        meta_total = int(overall.get("meta_total") or report.get("meta_total") or 0)
        judged = int(overall.get("judged") or report.get("judged") or 0)

        def pct(v: Any) -> str:
            if v is None:
                return "—"
            try:
                return f"{float(v):.1f}%"
            except (TypeError, ValueError):
                return "—"

        rows.append(
            "<tr>"
            f"<td><b>{html.escape(label)}</b></td>"
            f"<td>{meta_total}</td>"
            f"<td>{judged}</td>"
            f"<td>{pct(overall.get('answer_success_rate') or report.get('tsr_rate'))}</td>"
            f"<td>{pct(overall.get('egsr_rate') or report.get('egsr_rate'))}</td>"
            f"<td>{pct(overall.get('etsr_rate') or report.get('etsr_rate'))}</td>"
            f'<td><a href="{html.escape(str(rel_html))}">!__rubric.html</a> · '
            f'<a href="{html.escape(str(json_path.relative_to(output_base)))}">json</a></td>'
            "</tr>"
        )

    index_path = output_base / "index.html"
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Rubric summaries — all runs</title>
  <style>
    :root {{ --bg:#1a1d23; --card:#252830; --text:#e8eaed; --muted:#9aa0a6; --accent:#8ab4f8; --border:#3c4043; }}
    body {{ font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); margin:0; padding:24px; max-width:1200px; margin-inline:auto; }}
    h1 {{ font-size:1.4rem; }}
    table {{ border-collapse:collapse; width:100%; font-size:12px; margin-top:16px; }}
    th,td {{ border:1px solid var(--border); padding:6px 8px; }}
    th {{ background:#2d333b; color:var(--muted); }}
    a {{ color:var(--accent); }}
    .muted {{ color:var(--muted); font-size:13px; }}
  </style>
</head>
<body>
  <h1>Rubric summaries (all model runs)</h1>
  <p class="muted">Generated under <code>{html.escape(str(output_base))}</code>.</p>
  <table>
    <thead><tr>
      <th>Run</th><th>n (meta)</th><th>judged</th><th>TSR %</th><th>EGSR %</th><th>ETSR %</th><th>Links</th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>"""
    index_path.write_text(body, encoding="utf-8")
    return index_path


def print_rubric_summary(
    outputs: dict[str, Any],
    tasks_by_id: dict[str, dict[str, Any]] | None = None,
) -> None:
    stats = calculate_rubric_statistics(outputs, tasks_by_id)
    for label, s in stats["slices"].items():
        print(f"\n📐 Rubric [{label}] judged={s['judged']}/{s['meta_total']}")
        for dim, d in sorted((s.get("dimension_rates") or {}).items()):
            app = d.get("applicable", 0)
            if not app:
                continue
            print(
                f"   {dim}: pass {d['pass']}/{app} ({_pct(d.get('pass_rate'))})"
            )


def _load_tasks_by_id(tasks_json: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(tasks_json.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "tasks" in data:
        rows = data["tasks"]
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("id"):
            out[str(row["id"])] = row
    return out


def _discover_episode_ids(results_root: Path, cache_filename: str) -> list[str]:
    ids: list[str] = []
    if not results_root.is_dir():
        return ids
    for p in sorted(results_root.iterdir()):
        if p.is_dir() and not p.name.startswith(".") and (p / cache_filename).is_file():
            ids.append(p.name)
    return ids


def _load_caches(
    results_root: Path,
    cache_filename: str,
    episode_ids: list[str],
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for eid in episode_ids:
        path = results_root / eid / cache_filename
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict):
            outputs[eid] = data
    return outputs


def run_rubric_report(
    *,
    results_root: Path,
    tasks_json: Path,
    cache_filename: str = "two_pass_judge.json",
    reconcile: bool = True,
    json_out: Path | None = None,
    html_out: Path | None = None,
    title: str | None = None,
    outputs: dict[str, Any] | None = None,
    tasks_by_id: dict[str, dict[str, Any]] | None = None,
    load_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Compute report and optionally write JSON + HTML."""
    if outputs is None or tasks_by_id is None:
        report, outputs, tasks_by_id = compute_rubric_report(
            results_root=results_root,
            tasks_json=tasks_json,
            cache_filename=cache_filename,
            reconcile=reconcile,
        )
    else:
        stats = calculate_rubric_statistics(outputs, tasks_by_id)
        overall = stats["slices"].get("Overall") or {}
        report = {
            "results_root": str(results_root.resolve()),
            "tasks_json": str(tasks_json.resolve()),
            "cache_filename": cache_filename,
            "meta_total": int(overall.get("meta_total") or len(tasks_by_id)),
            "judged": int(overall.get("judged") or 0),
            "judge_errors": int(overall.get("judge_errors") or 0),
            "tsr_ok": int(overall.get("answer_success") or 0),
            "egsr_ok": int(overall.get("egsr_pass") or 0),
            "etsr_ok": int(overall.get("etsr_pass") or 0),
            "tsr_rate": overall.get("answer_success_rate"),
            "egsr_rate": overall.get("egsr_rate"),
            "etsr_rate": overall.get("etsr_rate"),
            "rss_all_dims_pass": int(overall.get("rss_all_dims_pass") or 0),
            "ungrounded_pass": int(overall.get("ungrounded_pass") or 0),
            "load_counts": load_counts or {},
            "slices": stats["slices"],
        }
    print_rubric_summary(outputs, tasks_by_id)

    if json_out is not None:
        save_rubric_statistics_json(json_out, outputs, tasks_by_id, report=report)
        print(f"Wrote {json_out.resolve()}")
    if html_out is not None:
        save_rubric_statistics_html(
            html_out, outputs, tasks_by_id, title=title, report=report
        )
        print(f"Wrote {html_out.resolve()}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rubric dimension statistics from cached two_pass_judge.json files."
    )
    parser.add_argument("--tasks-json", type=Path, default=DEFAULT_TASKS_JSON)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--cache-filename", default="two_pass_judge.json")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help=f"Write compact stats JSON (default: results-root/{RUBRIC_JSON_FILENAME})",
    )
    parser.add_argument(
        "--html-out",
        type=Path,
        default=None,
        help=f"Write standalone HTML (default: results-root/{RUBRIC_HTML_FILENAME})",
    )
    parser.add_argument("--no-json", action="store_true", help="Skip JSON output")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML output")
    parser.add_argument("--no-reconcile", action="store_true")
    parser.add_argument(
        "--title",
        default=None,
        help="HTML page title (default: results-root folder name)",
    )
    parser.add_argument(
        "--build-index",
        type=Path,
        metavar="OUTPUT_BASE",
        help="Only build index.html under OUTPUT_BASE from existing rubric_stats.json files",
    )
    args = parser.parse_args()

    if args.build_index is not None:
        path = build_rubric_index_html(args.build_index.resolve())
        print(path)
        return

    results_root = args.results_root.resolve()
    tasks_json = args.tasks_json.resolve()
    if not results_root.is_dir():
        raise SystemExit(f"results-root not found: {results_root}")
    if not tasks_json.is_file():
        raise SystemExit(f"tasks-json not found: {tasks_json}")

    json_out = None if args.no_json else (args.json_out or results_root / RUBRIC_JSON_FILENAME)
    html_out = None if args.no_html else (args.html_out or results_root / RUBRIC_HTML_FILENAME)
    title = args.title or results_root.name

    run_rubric_report(
        results_root=results_root,
        tasks_json=tasks_json,
        cache_filename=args.cache_filename,
        reconcile=not args.no_reconcile,
        json_out=json_out,
        html_out=html_out,
        title=title,
    )


if __name__ == "__main__":
    main()
