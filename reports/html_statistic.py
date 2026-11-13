"""HTML verdict reports and batch/HTML summary statistics for task_eval."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

_STEP3_JUDGEMENT_KEYS = (
    "step3_final_judgement",
    "step2_global_judgement",
    "pass3_judgement",
    "pass2_judgement",
)


def step3_judgement_from_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    for key in _STEP3_JUDGEMENT_KEYS:
        block = data.get(key)
        if isinstance(block, dict):
            return block
    return None


def trajectory_step_count_from_payload(data: dict[str, Any]) -> int | None:
    """Total trajectory steps from judge payload (meta or derived_metrics)."""
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    dm = data.get("derived_metrics") if isinstance(data.get("derived_metrics"), dict) else {}
    derived_in_meta = meta.get("derived_metrics") if isinstance(meta.get("derived_metrics"), dict) else {}
    for block in (meta, dm, derived_in_meta):
        for key in ("trajectory_step_count", "total_steps"):
            sc = block.get(key)
            if isinstance(sc, int) and sc > 0:
                return sc
    return None


def _mean_positive_ints(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def dimension_pass_numeric(v: Any) -> float | None:
    """Map a stored dimension value to 1.0 (pass) / 0.0 (fail); None if absent."""
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


def _sorted_steps(trajectory: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return sorted(trajectory.items(), key=lambda kv: int(kv[0]))


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


def episode_html_context(run_dir: Path) -> dict[str, Any]:
    instruction = ""
    um = run_dir / "user_message" / "user_message_001.txt"
    if um.exists():
        instruction = um.read_text(encoding="utf-8").strip()
    meta_path = run_dir / "metadata.json"
    if not instruction and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                instruction = str(meta.get("goal") or meta.get("prompt") or "").strip()
        except (json.JSONDecodeError, OSError):
            pass

    answer = ""
    step_count: int | None = None
    last_shot_rel: str | None = None
    traj_path = run_dir / "trajectory.json"
    if traj_path.exists():
        try:
            traj = json.loads(traj_path.read_text(encoding="utf-8"))
            if isinstance(traj, dict):
                step_count = len(traj)
                answer = _read_final_answer(run_dir, None, None) or _extract_final_answer_from_trajectory(
                    traj
                )
                last_name: str | None = None
                for _, step in _sorted_steps(traj):
                    shot = (step or {}).get("screenshot")
                    if isinstance(shot, str) and shot.strip():
                        last_name = shot.strip()
                if last_name:
                    last_shot_rel = f"{run_dir.name}/images/{last_name}"
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return {
        "instruction": instruction,
        "answer": answer,
        "step_count": step_count,
        "last_screenshot_rel": last_shot_rel,
    }


def slugify_task_type(s: str) -> str:
    safe = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in s.strip())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "unknown"


def filter_outputs_by_tasks(
    outputs: dict[str, Any],
    tasks_by_id: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """Restrict outputs to episode ids present in tasks-json metadata."""
    if not tasks_by_id:
        return outputs
    allowed = set(tasks_by_id.keys())
    return {eid: data for eid, data in outputs.items() if eid in allowed}


COVERAGE_JUDGED_OK = "judged_ok"
COVERAGE_JUDGE_ERROR = "judge_error"
COVERAGE_NO_TRAJECTORY = "no_trajectory"
COVERAGE_MISSING_DIR = "missing_dir"


def _is_judge_error_payload(data: dict[str, Any]) -> bool:
    err_msg = str(data.get("error", ""))
    return (
        data.get("coverage") == COVERAGE_JUDGE_ERROR
        or err_msg.startswith("JUDGE_ERROR:")
        or "step3_final_judgement missing" in err_msg
    )


def _is_cua_error_payload(data: dict[str, Any]) -> bool:
    err_msg = str(data.get("error", ""))
    return data.get("coverage") == COVERAGE_NO_TRAJECTORY or err_msg.startswith("CUA_ERROR:")


def _fail_metrics() -> dict[str, bool]:
    """TSR / EGSR / ETSR all fail (used for missing judge, CUA errors, etc.)."""
    return {"answer_success": False, "egsr": False, "etsr": False}


def _judgement_metrics_from_output(data: dict[str, Any]) -> dict[str, bool] | None:
    """Return pass/fail flags when a valid step-3 judgement exists; None if not judged OK."""
    if not isinstance(data, dict) or "error" in data:
        return None
    judgement = step3_judgement_from_payload(data) or {}
    if not isinstance(judgement, dict) or not judgement:
        return None
    return {
        "answer_success": bool(judgement.get("answer_success", False)),
        "egsr": bool(judgement.get("egsr")),
        "etsr": bool(judgement.get("etsr")),
    }


def _metrics_for_output(data: Any) -> dict[str, bool]:
    """Scoring flags for one episode; errors count as TSR/EGSR/ETSR fail."""
    if not isinstance(data, dict) or "error" in data:
        return _fail_metrics()
    metrics = _judgement_metrics_from_output(data)
    return metrics if metrics is not None else _fail_metrics()


def missing_judge_cache_payload(run_dir: Path, cache_filename: str) -> dict[str, Any]:
    """Error dict when per-episode judge cache is absent (for HTML report / batch outputs)."""
    if not run_dir.is_dir():
        return {
            "error": "JUDGE_ERROR: episode directory not found",
            "coverage": COVERAGE_MISSING_DIR,
        }
    if (run_dir / cache_filename).is_file():
        return {}
    has_traj = (run_dir / "trajectory.json").is_file()
    has_cua_errors = (run_dir / "errors").is_dir()
    if has_traj:
        msg = f"JUDGE_ERROR: {cache_filename} not found (trajectory exists; judge not run or failed)"
        if has_cua_errors:
            msg += "; CUA errors/ also present"
        return {"error": msg, "coverage": COVERAGE_JUDGE_ERROR}
    if has_cua_errors:
        return {
            "error": "CUA_ERROR: trajectory.json missing; CUA errors/ present",
            "coverage": COVERAGE_NO_TRAJECTORY,
        }
    return {
        "error": "CUA_ERROR: trajectory.json not found",
        "coverage": COVERAGE_NO_TRAJECTORY,
    }


def rss_display_pf(v: Any) -> str:
    if v is None or v == "" or v == "-":
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    return "pass" if x >= 1.0 else "fail"


def egsr_display_pf(v: Any) -> str:
    if v is True:
        return "pass"
    if v is False:
        return "fail"
    return "—"


def bool_pf(v: Any) -> str:
    if v is True:
        return "pass"
    if v is False:
        return "fail"
    return "—"


def format_dimension_scores_pf_html(ds: Any) -> str:
    if not isinstance(ds, dict) or not ds:
        return "—"
    parts: list[str] = []
    for k in sorted(ds.keys()):
        pv = dimension_pass_numeric(ds.get(k))
        if pv is None:
            pf = "?"
        elif pv >= 1.0:
            pf = '<span class="pf-pass">pass</span>'
        else:
            pf = '<span class="pf-fail">fail</span>'
        parts.append(f'<span class="dim-pf"><code>{html.escape(str(k))}</code>:{pf}</span>')
    return '<span class="dim-wrap">' + " ".join(parts) + "</span>"


def infer_episode_coverage(
    run_dir: Path,
    data: dict[str, Any] | None,
    *,
    cache_filename: str,
) -> str:
    """Classify episode: missing_dir | no_trajectory | judge_error | judged_ok."""
    if not run_dir.is_dir():
        return COVERAGE_MISSING_DIR
    if not (run_dir / "trajectory.json").is_file():
        return COVERAGE_NO_TRAJECTORY

    if not isinstance(data, dict):
        return COVERAGE_JUDGE_ERROR
    cov = data.get("coverage")
    if cov in (
        COVERAGE_JUDGED_OK,
        COVERAGE_JUDGE_ERROR,
        COVERAGE_NO_TRAJECTORY,
        COVERAGE_MISSING_DIR,
    ):
        return str(cov)

    if "error" in data:
        return COVERAGE_JUDGE_ERROR
    if step3_judgement_from_payload(data):
        return COVERAGE_JUDGED_OK

    cache_path = run_dir / cache_filename
    if not cache_path.is_file():
        return COVERAGE_JUDGE_ERROR
    return COVERAGE_JUDGE_ERROR


def calculate_coverage_statistics(
    *,
    episode_ids: list[str],
    results_root: Path,
    outputs: dict[str, Any],
    cache_filename: str,
) -> dict[str, Any]:
    """Coverage counts against metadata episode list (e.g. 150 tasks)."""
    counts = {
        "meta_total": len(episode_ids),
        "with_trajectory": 0,
        COVERAGE_JUDGED_OK: 0,
        COVERAGE_JUDGE_ERROR: 0,
        COVERAGE_NO_TRAJECTORY: 0,
        COVERAGE_MISSING_DIR: 0,
    }
    for eid in episode_ids:
        run_dir = results_root / eid
        data = outputs.get(eid)
        if isinstance(data, dict):
            cov = infer_episode_coverage(run_dir, data, cache_filename=cache_filename)
        else:
            cov = infer_episode_coverage(run_dir, None, cache_filename=cache_filename)
        counts[cov] += 1
        if cov != COVERAGE_MISSING_DIR and cov != COVERAGE_NO_TRAJECTORY:
            counts["with_trajectory"] += 1
    return counts


def render_coverage_summary_html(
    coverage: dict[str, Any],
    *,
    cache_filename: str,
) -> str:
    meta = coverage["meta_total"]
    judged = coverage[COVERAGE_JUDGED_OK]
    judge_err = coverage[COVERAGE_JUDGE_ERROR]
    with_traj = coverage["with_trajectory"]
    no_traj = coverage[COVERAGE_NO_TRAJECTORY]
    missing = coverage[COVERAGE_MISSING_DIR]

    return (
        '<div class="coverage-block">'
        "<h4 style=\"margin:0 0 8px;font-size:0.95rem;\">Episode coverage (metadata scope)</h4>"
        f'<p class="muted" style="margin:0 0 8px;font-size:12px;">'
        f"Judge cache: <code>{html.escape(cache_filename)}</code> · "
        f"meta total = tasks-json episode count</p>"
        "<table class=\"slice-stats-table coverage-table\">"
        "<thead><tr><th>What</th><th>Count</th><th>Meaning</th></tr></thead><tbody>"
        f"<tr><td><b>Meta (total)</b></td><td>{meta}</td>"
        "<td>All tasks in metadata JSON (denominator for coverage)</td></tr>"
        f"<tr><td>With trajectory</td><td>{with_traj}</td>"
        "<td>Folder exists and <code>trajectory.json</code> present</td></tr>"
        f"<tr><td><b>Judged (two-pass OK)</b></td><td>{judged}</td>"
        f"<td>Valid <code>{html.escape(cache_filename)}</code> with step-3 judgement</td></tr>"
        f"<tr><td><b>judge_error</b></td><td>{judge_err}</td>"
        "<td>Trajectory present but judge missing, invalid, or failed</td></tr>"
        f"<tr><td>No trajectory</td><td>{no_traj}</td>"
        "<td>Episode folder exists without <code>trajectory.json</code></td></tr>"
        f"<tr><td>Missing folder</td><td>{missing}</td>"
        "<td>No episode directory under results-root</td></tr>"
        "</tbody></table></div>"
    )


def _error_row(
    episode_id: str,
    task_type: str,
    *,
    error: str,
    cache_hit: bool = False,
    status: str = "FAIL",
) -> dict[str, Any]:
    fail_pf = "fail"
    return {
        "episode_id": episode_id,
        "task_type": task_type,
        "status": status,
        "label": "-",
        "score": "-",
        "rss": "-",
        "rss_pf": fail_pf,
        "egsr": False,
        "egsr_pf": fail_pf,
        "etsr": False,
        "etsr_pf": fail_pf,
        "dims_html": "—",
        "answer_success": False,
        "answer_success_pf": fail_pf,
        "cache_hit": cache_hit,
        "error": error,
        "rationale": "",
        "evidence_audit_fail": False,
    }


def calculate_html_report_statistics(
    outputs: dict[str, Any],
    tasks_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate per-episode rows and summary counters for HTML report rendering."""
    tasks_by_id = tasks_by_id or {}
    outputs = filter_outputs_by_tasks(outputs, tasks_by_id)
    total = len(outputs)
    errors = 0
    judge_errors = 0
    cache_hits = 0
    answer_success = 0
    egsr_pass = 0
    etsr_pass = 0
    label_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []

    for episode_id, data in sorted(outputs.items()):
        task_type = str((tasks_by_id.get(episode_id) or {}).get("task_type") or "Unknown")
        if not isinstance(data, dict):
            errors += 1
            continue
        if "error" in data:
            is_judge_err = _is_judge_error_payload(data)
            if is_judge_err:
                judge_errors += 1
            else:
                errors += 1
            rows.append(
                _error_row(
                    episode_id,
                    task_type,
                    error=str(data.get("error", "")),
                    cache_hit=bool(
                        isinstance(data.get("meta"), dict) and data["meta"].get("cache_hit")
                    ),
                )
            )
            continue

        meta = data.get("meta", {})
        if isinstance(meta, dict) and meta.get("cache_hit"):
            cache_hits += 1

        judgement = step3_judgement_from_payload(data) or {}
        if not isinstance(judgement, dict) or not judgement:
            judge_errors += 1
            rows.append(
                _error_row(
                    episode_id,
                    task_type,
                    error="step3_final_judgement missing or invalid",
                    cache_hit=bool(isinstance(meta, dict) and meta.get("cache_hit")),
                )
            )
            continue

        label = str(judgement.get("label", "UNKNOWN"))
        label_counts[label] = label_counts.get(label, 0) + 1
        ans_ok = bool(judgement.get("answer_success", False))
        if ans_ok:
            answer_success += 1
        if judgement.get("egsr"):
            egsr_pass += 1
        if judgement.get("etsr"):
            etsr_pass += 1

        rationale_raw = judgement.get("rationale", "")
        rationale_str = (
            rationale_raw
            if isinstance(rationale_raw, str)
            else json.dumps(rationale_raw, ensure_ascii=False)
        )

        dm = meta.get("derived_metrics") if isinstance(meta, dict) else {}
        if not isinstance(dm, dict):
            dm = {}
        rss_out = judgement.get("rss", dm.get("rss"))
        egsr_v = judgement.get("egsr")
        etsr_v = judgement.get("etsr")
        ds_raw = judgement.get("dimension_scores")
        dims_html = format_dimension_scores_pf_html(ds_raw)

        rows.append(
            {
                "episode_id": episode_id,
                "task_type": task_type,
                "status": "OK",
                "label": label,
                "score": judgement.get("score", judgement.get("rss", "-")),
                "rss": rss_out if rss_out is not None else "-",
                "rss_pf": rss_display_pf(rss_out),
                "egsr": egsr_v,
                "egsr_pf": egsr_display_pf(egsr_v),
                "etsr": etsr_v,
                "etsr_pf": egsr_display_pf(etsr_v),
                "dims_html": dims_html,
                "answer_success": ans_ok,
                "answer_success_pf": bool_pf(ans_ok),
                "cache_hit": bool(isinstance(meta, dict) and meta.get("cache_hit")),
                "error": "",
                "rationale": rationale_str,
                "evidence_audit_fail": False,
            }
        )

    judged_ok = total
    result: dict[str, Any] = {
        "total": total,
        "errors": errors,
        "judge_errors": judge_errors,
        "judged_ok": judged_ok,
        "cache_hits": cache_hits,
        "attempted": judged_ok,
        "answer_success": answer_success,
        "egsr_pass": egsr_pass,
        "etsr_pass": etsr_pass,
        "answer_success_rate": (answer_success / judged_ok * 100.0) if judged_ok else 0.0,
        "egsr_rate": (egsr_pass / judged_ok * 100.0) if judged_ok else 0.0,
        "etsr_rate": (etsr_pass / judged_ok * 100.0) if judged_ok else 0.0,
        "label_counts": label_counts,
        "rows": rows,
    }
    # Align headline rates with "Metrics by task type" (meta denominator; errors = fail).
    if tasks_by_id:
        meta = calculate_slice_statistics(outputs, tasks_by_id, slice_label="Overall")
        result["meta_total"] = meta["meta_total"]
        result["attempted"] = meta["meta_total"]
        result["answer_success"] = meta["answer_success"]
        result["egsr_pass"] = meta["egsr_pass"]
        result["etsr_pass"] = meta["etsr_pass"]
        result["answer_success_rate"] = meta["answer_success_rate"] or 0.0
        result["egsr_rate"] = meta["egsr_rate"] or 0.0
        result["etsr_rate"] = meta["etsr_rate"] or 0.0
    return result


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TSR_OUTPUTS_ROOT = _PROJECT_ROOT / "task_eval_tsr" / "outputs"

_TASK_TYPE_SLICE_ORDER = (
    "Overall",
    "Visual Grounding",
    "Contextual Inference",
    "Semantic Grounding",
)


def resolve_tsr_outputs_root(results_root: Path) -> Path | None:
    """Map agent_results results_cua/... to task_eval_tsr/outputs/..."""
    rr = results_root.resolve()
    parts = rr.parts
    for i, part in enumerate(parts):
        if part == "results_cua" and i + 1 < len(parts):
            candidate = _TSR_OUTPUTS_ROOT.joinpath(*parts[i + 1 :])
            if candidate.is_dir():
                return candidate
    return None


def load_rssr_pass_by_episode(tsr_outputs_root: Path) -> dict[str, bool]:
    """Episode id -> tsr_pass from task_eval_tsr batch_summary or per-episode caches."""
    summary_path = tsr_outputs_root / "batch_summary.json"
    if summary_path.is_file():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
        episodes = data.get("episodes") if isinstance(data, dict) else {}
        if isinstance(episodes, dict):
            out: dict[str, bool] = {}
            for eid, ep in episodes.items():
                if isinstance(ep, dict) and "tsr_pass" in ep:
                    out[str(eid)] = bool(ep["tsr_pass"])
            if out:
                return out

    out = {}
    for p in sorted(tsr_outputs_root.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        cache = p / "tsr_pass_judge.json"
        if not cache.is_file():
            continue
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and "tsr_pass" in payload:
            out[p.name] = bool(payload["tsr_pass"])
    return out


def calculate_rssr_slice(
    episode_ids: list[str],
    rssr_by_episode: dict[str, bool],
) -> dict[str, Any]:
    """RSSR = tsr_pass rate from task_eval_tsr for episodes in this slice."""
    matched = [eid for eid in episode_ids if eid in rssr_by_episode]
    passed = sum(1 for eid in matched if rssr_by_episode[eid])
    n = len(matched)
    return {
        "rssr_n": n,
        "rssr_pass": passed,
        "rssr_rate": (passed / n * 100.0) if n else None,
    }


def iter_statistics_slices(
    outputs: dict[str, Any],
    tasks_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[tuple[str, list[str]]]:
    """Return (slice_label, metadata episode ids) scoped to tasks-json."""
    tasks_by_id = tasks_by_id or {}
    if not tasks_by_id:
        return [("Overall", list(filter_outputs_by_tasks(outputs, tasks_by_id).keys()))]

    by_type_meta: dict[str, list[str]] = {}
    for eid, row in tasks_by_id.items():
        ttype = str((row or {}).get("task_type") or "Unknown")
        by_type_meta.setdefault(ttype, []).append(eid)

    slices: list[tuple[str, list[str]]] = [("Overall", list(tasks_by_id.keys()))]
    for ttype in _TASK_TYPE_SLICE_ORDER[1:]:
        meta_ids = by_type_meta.pop(ttype, None)
        if meta_ids:
            slices.append((ttype, meta_ids))
    for ttype in sorted(by_type_meta.keys()):
        slices.append((ttype, by_type_meta[ttype]))
    return slices


def _meta_episode_ids_for_slice(
    label: str,
    tasks_by_id: dict[str, dict[str, Any]],
    output_keys: list[str] | set[str],
) -> list[str]:
    if not tasks_by_id:
        return list(output_keys)
    if label == "Overall":
        return list(tasks_by_id.keys())
    return [
        eid
        for eid, row in tasks_by_id.items()
        if str((row or {}).get("task_type") or "Unknown") == label
    ]


def calculate_slice_statistics(
    outputs: dict[str, Any],
    tasks_by_id: dict[str, dict[str, Any]] | None = None,
    *,
    slice_label: str = "Overall",
    meta_ids: list[str] | None = None,
    rssr_by_episode: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Metrics for one metadata slice; rates use meta_total as denominator."""
    tasks_by_id = tasks_by_id or {}
    outputs = filter_outputs_by_tasks(outputs, tasks_by_id)
    meta_ids = meta_ids or _meta_episode_ids_for_slice(slice_label, tasks_by_id, outputs.keys())
    meta_total = len(meta_ids)

    judged_ok = judge_errors = errors = 0
    answer_success = tsr_ok = egsr_pass = etsr_pass = 0
    step_ok_counts: list[int] = []

    for eid in meta_ids:
        data = outputs.get(eid)
        if data is None:
            judge_errors += 1
        elif not isinstance(data, dict):
            errors += 1
        elif "error" in data:
            if _is_judge_error_payload(data):
                judge_errors += 1
            elif _is_cua_error_payload(data):
                errors += 1
            else:
                errors += 1
        elif _judgement_metrics_from_output(data) is None:
            judge_errors += 1

        metrics = _metrics_for_output(data)
        judged_ok += 1
        if metrics["answer_success"]:
            answer_success += 1
            tsr_ok += 1
            sc = trajectory_step_count_from_payload(data)
            if sc is not None:
                step_ok_counts.append(sc)
        if metrics["egsr"]:
            egsr_pass += 1
        if metrics["etsr"]:
            etsr_pass += 1

    def _rate(n: int, d: int) -> float | None:
        return (n / d * 100.0) if d else None

    rssr = calculate_rssr_slice(meta_ids, rssr_by_episode or {})
    return {
        "label": slice_label,
        "total": meta_total,
        "attempted": judged_ok,
        "judged_ok": judged_ok,
        "errors": errors,
        "judge_errors": judge_errors,
        "meta_total": meta_total,
        "answer_success": answer_success,
        "answer_success_rate": _rate(answer_success, meta_total),
        "egsr_pass": egsr_pass,
        "egsr_rate": _rate(egsr_pass, meta_total),
        "etsr_pass": etsr_pass,
        "etsr_rate": _rate(etsr_pass, meta_total),
        "ssr_ok": tsr_ok,
        "ssr_rate": _rate(tsr_ok, meta_total),
        "ass_mean_successful_steps": _mean_positive_ints(step_ok_counts),
        "count_answer_success_for_avg_steps": len(step_ok_counts),
        **rssr,
    }


def render_slice_statistics_html(
    outputs: dict[str, Any],
    tasks_by_id: dict[str, dict[str, Any]] | None = None,
    *,
    results_root: Path | None = None,
    rssr_by_episode: dict[str, bool] | None = None,
) -> str:
    """HTML table: Overall + per task_type SSR/RSSR/EGSR/ETSR/answer rates."""
    if rssr_by_episode is None and results_root is not None:
        tsr_root = resolve_tsr_outputs_root(results_root)
        rssr_by_episode = load_rssr_pass_by_episode(tsr_root) if tsr_root else {}

    rssr_by_episode = rssr_by_episode or {}
    scoped_outputs = filter_outputs_by_tasks(outputs, tasks_by_id)
    rows_html: list[str] = []
    for label, meta_ids in iter_statistics_slices(scoped_outputs, tasks_by_id):
        s = calculate_slice_statistics(
            scoped_outputs,
            tasks_by_id,
            slice_label=label,
            meta_ids=meta_ids,
            rssr_by_episode=rssr_by_episode,
        )
        ssr = s["ssr_rate"]
        rssr = s["rssr_rate"]
        egsr = s["egsr_rate"]
        etsr = s["etsr_rate"]
        ans = s["answer_success_rate"]
        ass = s["ass_mean_successful_steps"]
        ass_n = s["count_answer_success_for_avg_steps"]

        def pct(v: float | None) -> str:
            return f"{v:.2f}%" if v is not None else "—"

        if ass is not None and ass_n:
            ass_cell = f"{ass:.1f} (n={ass_n})"
        else:
            ass_cell = "—"

        if s["rssr_n"]:
            rssr_cell = f"{s['rssr_pass']}/{s['rssr_n']} ({pct(rssr)})"
        else:
            rssr_cell = "—"

        rows_html.append(
            "<tr>"
            f"<td><b>{html.escape(label)}</b></td>"
            f"<td>{s['meta_total']}</td>"
            f"<td>{s['attempted']}</td>"
            f"<td>{s['judge_errors']}</td>"
            f"<td>{s['answer_success']}/{s['meta_total']} ({pct(ans)})</td>"
            f"<td>{s['ssr_ok']}/{s['meta_total']} ({pct(ssr)})</td>"
            f"<td>{rssr_cell}</td>"
            f"<td>{s['egsr_pass']}/{s['meta_total']} ({pct(egsr)})</td>"
            f"<td>{s['etsr_pass']}/{s['meta_total']} ({pct(etsr)})</td>"
            f"<td>{ass_cell}</td>"
            f"<td>{s['errors']}</td>"
            "</tr>"
        )

    tsr_note = ""
    if results_root is not None:
        tsr_root = resolve_tsr_outputs_root(results_root)
        if tsr_root:
            tsr_note = (
                f'<p class="muted" style="margin:0 0 8px;font-size:12px;">'
                f"RSSR from <code>{html.escape(str(tsr_root))}</code> "
                f"(tsr_pass_judge.json)</p>"
            )

    return (
        '<div class="slice-stats-block">'
        "<h4 style=\"margin:0 0 8px;font-size:0.95rem;\">Metrics by task type</h4>"
        f"{tsr_note}"
        '<p class="muted" style="margin:0 0 8px;font-size:12px;">'
        "Slices and <b>n (meta)</b> come from tasks-json. "
        "TSR / EGSR / ETSR rates use meta total as denominator. "
        "Episodes with <b>judge_error</b> or <b>Errors</b> (CUA) count as TSR/EGSR/ETSR fail. "
        "ASS = mean trajectory step count among TSR successes with a known step count.</p>"
        "<table class=\"slice-stats-table\">"
        "<thead><tr>"
        "<th>Slice</th><th>n (meta)</th><th>judged</th><th>judge_error</th>"
        "<th>Answer success</th><th>TSR</th><th>RSSR</th><th>EGSR</th><th>ETSR</th>"
        "<th>ASS (avg steps, TSR ok)</th><th>Errors</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table></div>"
    )


def calculate_batch_statistics(
    outputs: dict[str, Any],
    tasks_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Batch-level TSR/EGSR/ETSR/DDR/ASS rates from evaluated episode outputs."""
    outputs = filter_outputs_by_tasks(outputs, tasks_by_id)
    step_ok_counts: list[int] = []
    egsr_ok = etsr_ok = tsr_ok = 0
    ddr_passed = 0
    ddr_applicable_count = 0
    for data in outputs.values():
        if not isinstance(data, dict):
            continue
        metrics = _metrics_for_output(data)
        m = data.get("meta") or {}
        dm = data.get("derived_metrics") if isinstance(data.get("derived_metrics"), dict) else {}
        if metrics["answer_success"]:
            tsr_ok += 1
            sc = trajectory_step_count_from_payload(data)
            if sc is not None:
                step_ok_counts.append(sc)
        if metrics["egsr"]:
            egsr_ok += 1
        if metrics["etsr"]:
            etsr_ok += 1
        if dm.get("ddr_applicable"):
            ddr_applicable_count += 1
            if dm.get("ddr_pass"):
                ddr_passed += 1
    attempted_ok = len(outputs)
    avg_steps_success = _mean_positive_ints(step_ok_counts)

    if tasks_by_id:
        meta = calculate_slice_statistics(outputs, tasks_by_id, slice_label="Overall")
        meta_total = int(meta["meta_total"])
        tsr_ok = int(meta["answer_success"])
        egsr_ok = int(meta["egsr_pass"])
        etsr_ok = int(meta["etsr_pass"])
        return {
            "attempted_ok": meta_total,
            "meta_total": meta_total,
            "tsr_ok": tsr_ok,
            "egsr_ok": egsr_ok,
            "etsr_ok": etsr_ok,
            "tsr_rate": (tsr_ok / meta_total) if meta_total else None,
            "egsr_rate": (egsr_ok / meta_total) if meta_total else None,
            "etsr_rate": (etsr_ok / meta_total) if meta_total else None,
            "ddr_passed": ddr_passed,
            "ddr_applicable_count": ddr_applicable_count,
            "ddr_rate": (ddr_passed / ddr_applicable_count) if ddr_applicable_count else None,
            "ass_mean_successful_steps": avg_steps_success,
            "avg_steps_successful_trajectories": avg_steps_success,
            "count_answer_success_for_avg_steps": len(step_ok_counts),
        }

    return {
        "attempted_ok": attempted_ok,
        "tsr_ok": tsr_ok,
        "egsr_ok": egsr_ok,
        "etsr_ok": etsr_ok,
        "tsr_rate": (tsr_ok / attempted_ok) if attempted_ok else None,
        "egsr_rate": (egsr_ok / attempted_ok) if attempted_ok else None,
        "etsr_rate": (etsr_ok / attempted_ok) if attempted_ok else None,
        "ddr_passed": ddr_passed,
        "ddr_applicable_count": ddr_applicable_count,
        "ddr_rate": (ddr_passed / ddr_applicable_count) if ddr_applicable_count else None,
        "ass_mean_successful_steps": avg_steps_success,
        "avg_steps_successful_trajectories": avg_steps_success,
        "count_answer_success_for_avg_steps": len(step_ok_counts),
    }


def _rel_href(rel: str) -> str:
    return "/".join(quote(part, safe="") for part in rel.split("/"))


def _data_href(results_root: Path, html_out_root: Path, rel: str) -> str:
    """Href from an HTML file under html_out_root to data under results_root."""
    if html_out_root.resolve() == results_root.resolve():
        return _rel_href(rel)
    target = (results_root / rel).resolve()
    href = os.path.relpath(target, html_out_root.resolve())
    return _rel_href(href.replace("\\", "/"))


def _render_nav_html(nav_links: list[tuple[str, str, int]] | None) -> str:
    if not nav_links:
        return ""
    nav_items = "".join(
        f'<li><a href="{html.escape(href)}">{html.escape(label)}</a> '
        f'<span class="muted">({count})</span></li>'
        for label, href, count in nav_links
    )
    return (
        '<div class="nav-block">'
        '<b>Per-task_type reports:</b>'
        f'<ul class="nav-list">{nav_items}</ul>'
        "</div>"
    )


def _render_verdict_sections(
    *,
    results_root: Path,
    html_out_root: Path,
    rows: list[dict[str, Any]],
    cache_filename: str,
) -> tuple[list[str], list[str]]:
    table_rows: list[str] = []
    verdict_cards: list[str] = []
    for row in rows:
        eid_raw = str(row["episode_id"])
        eid_esc = html.escape(eid_raw)
        task_type_esc = html.escape(str(row.get("task_type") or "Unknown"))
        trajectory_link = f"{_data_href(results_root, html_out_root, eid_raw)}/trajectory.html"
        cache_link = (
            f"{_data_href(results_root, html_out_root, eid_raw)}/{quote(cache_filename, safe='')}"
        )
        run_dir = results_root / eid_raw
        ctx = episode_html_context(run_dir)
        instr_esc = html.escape(ctx["instruction"] or "—")
        answer_esc = html.escape(ctx["answer"] or "—")
        steps_esc = html.escape(str(ctx["step_count"]) if ctx["step_count"] is not None else "—")
        rationale_esc = html.escape(str(row.get("rationale") or ""))
        shot_rel = ctx["last_screenshot_rel"]
        if shot_rel and (results_root / shot_rel).is_file():
            shot_html = (
                f'<img src="{_data_href(results_root, html_out_root, shot_rel)}" '
                f'alt="last screenshot" loading="lazy" />'
            )
        elif shot_rel:
            shot_html = (
                f'<div class="no-shot">Missing image on disk<br/>'
                f"<code>{html.escape(shot_rel)}</code></div>"
            )
        else:
            shot_html = '<div class="no-shot">No screenshot in trajectory</div>'

        extra_summary: list[str] = []
        rpf = row.get("rss_pf")
        if isinstance(rpf, str) and rpf != "—":
            extra_summary.append(f"rss={rpf}")
        epf = row.get("egsr_pf")
        if isinstance(epf, str) and epf != "—":
            extra_summary.append(f"egsr={epf}")
        tpf = row.get("etsr_pf")
        if isinstance(tpf, str) and tpf != "—":
            extra_summary.append(f"etsr={tpf}")
        if row.get("evidence_audit_fail"):
            extra_summary.append("string_audit_fail")
        mid = " · ".join(extra_summary)
        summary_judgement_esc = html.escape(
            f"{row['label']} (score {row['score']})"
            + (f" · {mid}" if mid else "")
            + f" · ans={row.get('answer_success_pf', '—')}"
        )

        header_html = (
            f"<h3>{eid_esc} "
            f'<span class="task-type-badge">{task_type_esc}</span></h3>'
        )

        if row.get("error"):
            err_esc = html.escape(str(row.get("error", "")))
            fail_summary = html.escape(
                f"FAIL · etsr=fail · egsr=fail · ans=fail — {err_esc}"
            )
            verdict_cards.append(
                f'<section class="verdict-card-eval">'
                f"{header_html}"
                f'<p class="task-line">{instr_esc}</p>'
                f'<p class="status-err"><b>{fail_summary}</b></p>'
                f'<div class="verdict-body"><div class="shot-col">{shot_html}</div>'
                f'<div class="text-col">'
                f"<h6>Answer</h6><p>{answer_esc}</p>"
                f"<h6>Final steps</h6><p>{steps_esc}</p>"
                f"</div></div>"
                f'<p class="footer-links"><a href="{trajectory_link}">Trajectory</a> · '
                f'<a href="{cache_link}">judge_json</a></p>'
                f"</section>"
            )
        else:
            body_rationale = (
                f'<div class="rationale">{rationale_esc}</div>'
                if rationale_esc
                else '<div class="rationale muted">(no rationale in judge output)</div>'
            )
            verdict_cards.append(
                f'<section class="verdict-card-eval">'
                f"{header_html}"
                f'<p class="task-line">{instr_esc}</p>'
                f'<div class="verdict-body">'
                f'<div class="shot-col">{shot_html}</div>'
                f'<div class="text-col">'
                f"<h6>Answer</h6><p>{answer_esc}</p>"
                f"<h6>Final steps</h6><p>{steps_esc}</p>"
                f'<h6>Rubric dimensions</h6><p class="dim-line">{row.get("dims_html", "—")}</p>'
                f"<h6>Judgement</h6>"
                f'<details class="judgement"><summary>{summary_judgement_esc}</summary>'
                f"{body_rationale}"
                f"</details>"
                f"</div></div>"
                f'<p class="footer-links"><a href="{trajectory_link}">Trajectory</a> · '
                f'<a href="{cache_link}">judge_json</a></p>'
                f"</section>"
            )

        table_rows.append(
            "<tr>"
            f"<td>{eid_esc}</td>"
            f"<td>{task_type_esc}</td>"
            f"<td>{html.escape(str(row['status']))}</td>"
            f"<td>{html.escape(str(row['label']))}</td>"
            f"<td>{html.escape(str(row['score']))}</td>"
            f"<td>{html.escape(str(row.get('rss_pf', '—')))}</td>"
            f"<td>{html.escape(str(row.get('egsr_pf', '—')))}</td>"
            f"<td>{html.escape(str(row.get('etsr_pf', '—')))}</td>"
            f"<td>{html.escape(str(row.get('answer_success_pf', '—')))}</td>"
            f"<td>{html.escape(str(row['cache_hit']))}</td>"
            f"<td>{html.escape(str(row['error']))}</td>"
            f'<td class="dims-cell">{row.get("dims_html", "—")}</td>'
            f'<td><details class="table-rationale"><summary>rationale</summary>'
            f'<div class="rationale">{rationale_esc or "—"}</div></details></td>'
            f'<td><a href="{trajectory_link}">trajectory</a> | <a href="{cache_link}">judge_json</a></td>'
            "</tr>"
        )
    return table_rows, verdict_cards


def save_batch_html_report(
    results_root: Path,
    outputs: dict[str, Any],
    report_filename: str,
    *,
    cache_filename: str = "three_pass_judge.json",
    tasks_by_id: dict[str, dict[str, Any]] | None = None,
    page_title: str = "Three-step rubric verdicts",
    nav_links: list[tuple[str, str, int]] | None = None,
    stats_outputs: dict[str, Any] | None = None,
    html_out_root: Path | None = None,
) -> Path:
    """Write batch HTML. *outputs* drives episode rows/cards; *stats_outputs* drives coverage + slice metrics (defaults to *outputs*)."""
    html_out_root = results_root if html_out_root is None else html_out_root
    if tasks_by_id:
        outputs = filter_outputs_by_tasks(outputs, tasks_by_id)
    stats_outputs = outputs if stats_outputs is None else stats_outputs
    if tasks_by_id:
        stats_outputs = filter_outputs_by_tasks(stats_outputs, tasks_by_id)
    stats = calculate_html_report_statistics(outputs, tasks_by_id)
    episode_ids = list((tasks_by_id or {}).keys()) or list(stats_outputs.keys())
    coverage = calculate_coverage_statistics(
        episode_ids=episode_ids,
        results_root=results_root,
        outputs=stats_outputs,
        cache_filename=cache_filename,
    )
    coverage_html = render_coverage_summary_html(coverage, cache_filename=cache_filename)
    total = stats["total"]
    errors = stats["errors"]
    judge_errors = stats["judge_errors"]
    cache_hits = stats["cache_hits"]
    attempted = stats["attempted"]
    answer_success_rate = stats["answer_success_rate"]
    egsr_rate = stats["egsr_rate"]
    etsr_rate = stats["etsr_rate"]
    batch_stats = calculate_batch_statistics(outputs, tasks_by_id)
    ass_mean = batch_stats.get("ass_mean_successful_steps")
    ass_n = batch_stats.get("count_answer_success_for_avg_steps", 0)
    if ass_mean is not None and ass_n:
        ass_summary = f"{ass_mean:.1f} (n={ass_n} TSR successes with step count)"
    else:
        ass_summary = "—"
    label_counts = stats["label_counts"]
    rows = stats["rows"]

    label_items = "".join(
        f"<li><b>{html.escape(k)}</b>: {v}</li>"
        for k, v in sorted(label_counts.items(), key=lambda x: x[0])
    )
    nav_html = _render_nav_html(nav_links)
    from .html_rubric import render_rubric_statistics_html

    rubric_html = render_rubric_statistics_html(stats_outputs, tasks_by_id)
    table_rows, verdict_cards = _render_verdict_sections(
        results_root=results_root,
        html_out_root=html_out_root,
        rows=rows,
        cache_filename=cache_filename,
    )

    page_title_esc = html.escape(page_title)
    body = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>{page_title_esc}</title>
    <style>
      :root {{ --bg:#1a1d23; --card:#252830; --text:#e8eaed; --muted:#9aa0a6; --accent:#8ab4f8; --border:#3c4043; --err:#f28b82; }}
      body {{ font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); margin:0; padding:24px; max-width:1100px; margin-left:auto; margin-right:auto; line-height:1.45; }}
      h1 {{ font-size:1.4rem; font-weight:600; margin:0 0 8px 0; }}
      h2 {{ font-size:1.15rem; margin:28px 0 14px 0; font-weight:600; }}
      .summary-wrap {{ background:#1e3a5f; border:1px solid var(--border); border-radius:10px; padding:12px 16px; margin-bottom:8px; }}
      .summary-wrap > summary {{ cursor:pointer; font-weight:600; color:var(--accent); list-style:none; }}
      .summary-wrap > summary::-webkit-details-marker {{ display:none; }}
      .summary-inner {{ margin-top:12px; color:var(--text); }}
      .summary-inner ul {{ margin:8px 0; padding-left:1.2rem; }}
      table {{ border-collapse:collapse; width:100%; margin-top:12px; font-size:12px; }}
      th,td {{ border:1px solid var(--border); padding:6px 8px; vertical-align:top; }}
      th {{ background:#2d333b; color:var(--muted); }}
      .verdict-card-eval {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:16px 18px; margin-bottom:22px; }}
      .verdict-card-eval h3 {{ margin:0 0 6px 0; font-size:1.15rem; font-weight:600; display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
      .task-type-badge {{ font-size:0.72rem; font-weight:500; padding:2px 8px; border-radius:999px; background:#2d333b; color:var(--accent); border:1px solid var(--border); letter-spacing:0.02em; }}
      .task-line {{ font-style:italic; color:var(--muted); margin:0 0 14px 0; }}
      .verdict-body {{ display:flex; gap:20px; flex-wrap:wrap; align-items:flex-start; }}
      .shot-col {{ flex:1 1 300px; max-width:460px; min-width:220px; }}
      .shot-col img {{ width:100%; height:auto; border-radius:8px; border:1px solid var(--border); display:block; }}
      .text-col {{ flex:1 1 260px; min-width:220px; }}
      .text-col h6 {{ margin:14px 0 6px 0; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--muted); font-weight:600; }}
      .text-col h6:first-child {{ margin-top:0; }}
      .text-col p {{ margin:0; }}
      .no-shot {{ background:#1e2028; border:1px dashed var(--border); border-radius:8px; padding:36px 14px; text-align:center; color:var(--muted); font-size:13px; }}
      .no-shot code {{ font-size:11px; word-break:break-all; }}
      details.judgement summary {{ cursor:pointer; color:var(--accent); font-weight:500; }}
      .rationale {{ white-space:pre-wrap; margin-top:10px; color:var(--text); }}
      .muted {{ color:var(--muted); font-style:italic; }}
      .status-err {{ color:var(--err); margin:0 0 12px 0; }}
      .footer-links {{ margin:14px 0 0 0; font-size:14px; }}
      .footer-links a {{ color:var(--accent); text-decoration:none; }}
      .footer-links a:hover {{ text-decoration:underline; }}
      .dim-line {{ font-size:12px; line-height:1.5; margin:0; }}
      .dim-wrap {{ display:block; }}
      .dim-pf code {{ font-size:11px; color:var(--muted); }}
      .pf-pass {{ color:#81c995; font-weight:600; }}
      .pf-fail {{ color:#f28b82; font-weight:600; }}
      .dims-cell {{ max-width:420px; font-size:11px; line-height:1.45; vertical-align:top; }}
      details.table-rationale .rationale {{ max-height:200px; overflow:auto; font-size:11px; margin-top:6px; }}
      .nav-block {{ background:#1e2028; border:1px solid var(--border); border-radius:10px; padding:10px 14px; margin:8px 0 16px 0; font-size:13px; }}
      .nav-list {{ list-style:none; padding:0; margin:6px 0 0 0; display:flex; flex-wrap:wrap; gap:10px 18px; }}
      .nav-list a {{ color:var(--accent); text-decoration:none; }}
      .nav-list a:hover {{ text-decoration:underline; }}
      .slice-stats-block {{ background:#1e2028; border:1px solid var(--border); border-radius:10px; padding:12px 16px; margin:0 0 16px 0; }}
      .slice-stats-table {{ font-size:13px; margin-top:0; }}
      .slice-stats-table th {{ white-space:nowrap; }}
      .rubric-stats-block {{ background:#1e2028; border:1px solid var(--border); border-radius:10px; padding:12px 16px; margin:0 0 16px 0; }}
      .rubric-slice-block {{ margin:10px 0 0 0; border:1px solid var(--border); border-radius:8px; padding:8px 12px; background:#252830; }}
      .rubric-slice-block > summary {{ cursor:pointer; color:var(--accent); font-size:13px; list-style:none; }}
      .rubric-slice-block > summary::-webkit-details-marker {{ display:none; }}
      .rubric-slice-inner {{ margin-top:10px; }}
      .rubric-dim-table, .rubric-cp-table {{ font-size:12px; margin-top:4px; }}
    </style>
  </head>
  <body>
    <h1>{page_title_esc}</h1>
    {coverage_html}
    {render_slice_statistics_html(stats_outputs, tasks_by_id, results_root=results_root)}
    {rubric_html}
    {nav_html}
    <details class="summary-wrap">
      <summary>Summary stats &amp; table (click to expand)</summary>
      <div class="summary-inner">
        <p><b>Total:</b> {total} | <b>n (meta):</b> {attempted} | <b>judge_error:</b> {judge_errors} | <b>Errors:</b> {errors} | <b>Cache hits:</b> {cache_hits}</p>
        <p><b>TSR rate:</b> {answer_success_rate:.2f}% | <b>EGSR rate:</b> {egsr_rate:.2f}% | <b>ETSR rate:</b> {etsr_rate:.2f}% | <b>ASS (avg steps, TSR ok):</b> {html.escape(ass_summary)}</p>
        <p class="muted" style="margin:6px 0 0;font-size:11px;">Rates above use meta total (same as slice table); error episodes count as TSR/EGSR/ETSR fail.</p>
        <h4 style="margin:12px 0 6px;font-size:0.9rem;">Label counts</h4>
        <ul>{label_items}</ul>
        <table><thead><tr><th>episode_id</th><th>task_type</th><th>status</th><th>label</th><th>score</th><th>rss</th><th>egsr</th><th>etsr</th><th>answer</th><th>cache_hit</th><th>error</th><th>rubric_dims</th><th>rationale</th><th>links</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
      </div>
    </details>
    <h2>Verdicts</h2>
    {''.join(verdict_cards)}
  </body>
</html>"""

    html_out_root.mkdir(parents=True, exist_ok=True)
    report_path = html_out_root / report_filename
    report_path.write_text(body, encoding="utf-8")
    return report_path


def save_per_task_type_reports(
    *,
    results_root: Path,
    outputs: dict[str, Any],
    tasks_by_id: dict[str, dict[str, Any]],
    base_report_filename: str,
    cache_filename: str,
    html_out_root: Path | None = None,
) -> list[tuple[str, Path, int]]:
    html_out_root = results_root if html_out_root is None else html_out_root
    scoped_outputs = filter_outputs_by_tasks(outputs, tasks_by_id)
    type_to_meta_ids: dict[str, list[str]] = {}
    for eid, row in tasks_by_id.items():
        ttype = str((row or {}).get("task_type") or "Unknown")
        type_to_meta_ids.setdefault(ttype, []).append(eid)

    base = Path(base_report_filename)
    stem = base.stem
    suffix = base.suffix or ".html"

    base_nav = [
        (ttype, f"{stem}__{slugify_task_type(ttype)}{suffix}", len(meta_ids))
        for ttype, meta_ids in sorted(type_to_meta_ids.items())
    ]

    written: list[tuple[str, Path, int]] = []
    for ttype in sorted(type_to_meta_ids.keys()):
        meta_ids = type_to_meta_ids[ttype]
        sub_outputs = {eid: scoped_outputs[eid] for eid in meta_ids if eid in scoped_outputs}
        slug = slugify_task_type(ttype)
        fname = f"{stem}__{slug}{suffix}"
        type_nav = [(f"← All types", stem + suffix, len(tasks_by_id))] + [
            (label, href, count) for label, href, count in base_nav if label != ttype
        ]
        path = save_batch_html_report(
            results_root=results_root,
            outputs=sub_outputs,
            report_filename=fname,
            cache_filename=cache_filename,
            tasks_by_id=tasks_by_id,
            page_title=f"Two-step rubric verdicts — {ttype}",
            nav_links=type_nav,
            stats_outputs=scoped_outputs,
            html_out_root=html_out_root,
        )
        written.append((ttype, path, len(meta_ids)))
    return written
