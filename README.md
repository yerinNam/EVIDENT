# EVIDENT — 3-step trajectory rubric judge

Pipeline that evaluates GUI/web agent **trajectories** with a vision-language judge.  
Default entry point: **`main_eval.py`** (local vLLM + **Qwen3-VL**).

## Pipeline (3 steps)

| Step | Module | Input | Role |
|------|--------|-------|------|
| **1** | `pipeline/step1_eval.py` | Screenshot chunks | Extract **observed evidence only** (`screenshot_ocr`, `visual_elements`, …) |
| **2** | `pipeline/step2_eval.py` | Step 1 + `final_answer` | Map and review **claims** in the final answer vs evidence (text only, no images) |
| **3** | `pipeline/step3_eval.py` | Steps 1–2 + task metadata | **Global rubric** judgement (`answer_success`, per-dimension pass/fail, …) |

Steps 2 and 3 are text-only API calls. On vLLM, VL models can return empty JSON for multimodal payloads with no images; `main_eval.py` sends user content as a **plain string** when there are no images.

## Layout

```
EVIDENT/
  main_eval.py          # Qwen / local vLLM entry point
  run.sh                # example batch run
  pipeline/            # evaluation stages and prompts
    __init__.py
    step1_eval.py
    step2_eval.py
    step3_eval.py
  reports/             # HTML generation and statistics
    __init__.py
    html_statistic.py   # batch / task-type reports and aggregates
    html_rubric.py      # rubric dimension / checkpoint reports
  submit/              # bundled task metadata
    150_ko_metadata.json
    150_en_metadata.json
  README.md
```

Report modules live in `reports/` so they do not shadow Python's standard-library
`html` module. Generated HTML and judge caches are written to the results directory.
Import evaluation stages as `pipeline.step1_eval` (and similarly for Steps 2–3),
and report utilities as `reports.html_statistic` or `reports.html_rubric`.

## Prerequisites

- Python 3.10+ (`openai`, `tqdm` — project venv recommended)
- **Qwen judge (`main_eval.py`)**: OpenAI-compatible **vLLM** server must be running  
  - Default API: `http://127.0.0.1:8005/v1`  
  - Default model: `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8`

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install openai tqdm
```

Environment variables (Qwen):

| Variable | Description |
|----------|-------------|
| `QWEN_PORT` | vLLM port (default `8005`) |
| `QWEN_API_BASE` | Override API base URL |
| `QWEN_MODEL` | Override model ID |
| `QWEN_API_KEY` | For local vLLM (defaults to `dummy` if unset) |

## Input (per episode / `--run-dir`)

Each episode directory is expected to contain:

| Path | Required | Description |
|------|----------|-------------|
| `trajectory.json` | yes | Steps, screenshots, actions |
| `images/*.png` (or jpg) | yes | Screenshots referenced by the trajectory |
| `metadata.json` | optional | e.g. `start_url` |
| `user_message/user_message_001.txt` | optional | Auxiliary instruction text |

**`final_answer`** resolution order:

1. `--final-answer`
2. `--final-answer-file`
3. `final_answer.txt` / `answer.txt` / `assistant_final_answer.txt` in the run dir
4. Last `send_msg_to_user` message in the trajectory (`[ANSWER]` prefix stripped)

**Batch** (`--tasks-json` + `--results-root`): match each task `id` to `results_root/<id>/`.  
JSON format: array of `{id, prompt, start_url, …}`, or a bundle `{ "tasks": [...], "meta": ... }`.

Default tasks JSON (in code): `../task_gen/outputs/final/possible_translate_ko_metadata.json`

That default is relative to the repository location, not the current directory.
To use the bundled metadata, pass `--tasks-json ./submit/150_ko_metadata.json`
(Korean) or `--tasks-json ./submit/150_en_metadata.json` (English).

## Usage

Run Python commands from the `EVIDENT` directory.

```bash
cd /path/to/EVIDENT
source .venv/bin/activate   # optional
```

### Single episode

```bash
python3 main_eval.py \
  --run-dir /path/to/results/<episode_id> \
  --tasks-json /path/to/tasks.json \
  --final-answer-file /path/to/final_answer.txt \
  --constraints "optional constraint text" \
  --out ./output/<episode_id>.json
```

If a cache file exists in the run dir, it is loaded without calling the judge again (use `--force` to ignore).

### Batch

```bash
python3 main_eval.py \
  --tasks-json /path/to/tasks.json \
  --results-root /path/to/agent_results \
  --workers 8 \
  --llm-max-inflight 12 \
  --max-images 0 \
  --image-chunk-size 8 \
  --port 8005
```

Single episode filter:

```bash
python3 main_eval.py \
  --tasks-json /path/to/tasks.json \
  --results-root /path/to/agent_results \
  --episode-id <episode_id>
```

`bash run.sh` uses `./submit/150_ko_metadata.json` and `./results`, relative to the
repository directory. It switches to that directory automatically and activates
`.venv` if present; otherwise it uses the current environment's `python3`.
Adjust the results path and judge settings in the script for your environment.

## Cache & reports

- Per-episode cache: `<run_dir>/<cache-filename>` (default `two_pass_judge.json`)
- On rerun, cache hit skips judge LLM calls (`meta.cache_hit`)
- Force re-evaluation: `--force`
- After batch: HTML at `results_root/<report-filename>` (default `!__two_pass_verdicts.html`)
- Per–task-type HTML may also be written under `results_root` via `reports/html_statistic.py`

## Important CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--max-images` | `16` | Cap screenshots for Steps 1–2; `≤0` = all images in the trajectory |
| `--image-chunk-size` | `8` | Screenshots per Step 1 call (`0` = all at once) |
| `--step1-chunk-workers` | `2` | Parallel Step 1 chunks within an episode |
| `--llm-max-inflight` | `12` | Max concurrent LLM HTTP requests (keep ≤ vLLM `max-num-seqs`) |
| `--max-output-tokens` | `12000` | Response token cap (`0` = unlimited) |
| `--request-timeout` | `180` | Per-request timeout (seconds) |
| `--llm-retries` | `2` | Retries on timeout / 5xx / empty response / JSON parse failure |
| `--step1-retry-on-empty` | off | Re-run Step 1 once if OCR is empty |
| `--step2-retry-on-empty` | off | Re-run Step 2 once if claims and reviews are both empty |
| `--print-llm-raw` | off | Print request/response for debugging |

## Output JSON (single episode)

```json
{
  "meta": { "...": "model, paths, counts, cache_hit, metric notes ..." },
  "step1_window_evidence": { "screenshot_ocr": [], "visual_elements": [] },
  "step2_claim_evidence": { "final_answer_claims": [], "evidence_check_reviews": [] },
  "step3_final_judgement": { "answer_success": true, "dimensions": {}, "...": "..." },
  "derived_metrics": { "tsr": true, "egsr": true, "etsr": false, "sr_rate": 0.85 }
}
```

**Batch** output adds a top-level `meta` (aggregates, `tsr_rate`, `egsr_rate`, …) and an `episodes` array.

### Metrics (summary)

- **TSR**: `answer_success` (Step 3)
- **EGSR**: `answer_success` ∧ `metadata_requirement_verified`
- **ETSR**: EGSR ∧ `key_claims_all_supported`
- **SR**: Fraction of trajectory steps in the longest error-free prefix
- **DDR / ASS**: Dead-end and successful-trajectory step stats in batch `meta`

Rubrics merge a **common three dimensions** with task-type profiles (`semantic_grounding`, `contextual_inference`, `visual_grounding`, …) for Step 3 scoring (`_RUBRIC_DIMENSIONS` in `main_eval.py`).

## Notes

- `main_eval.py` is the judge entry point included in this checkout.
- Run the rubric module as `python3 -m reports.html_rubric` from the repository directory. Its standalone cache-loading path still depends on `html_from_cache`, which is absent from this checkout; this pre-existing limitation does not affect batch HTML rendering through `main_eval.py`.
- Set API keys **only via environment variables** — do not put them in the README, logs, or commits.
- Outputs and HTML may include trajectory text; check for sensitive URLs, accounts, or PII before sharing externally.
