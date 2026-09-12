"""Raise n on instruct-only leakage + ignore-bet. Thinking off. No steering."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.prompts import bet_prompt, ignore_bet_prompt
from src.run import RESULTS, ROOT, bias_from_rows, log, run_condition, save_json

MODEL_ID = "Qwen/Qwen3.5-4B"
THRESHOLD = 14898555.0
MAX_NEW = 512
# Same independent streams as the original n=4 cells. Do not reuse steer seeds.
SEED = {
    "above_good": 200,
    "below_good": 300,
    "ignore_bet_above": 4100,
    "ignore_bet_below": 4200,
}
OLD_PATH = {
    "above_good": RESULTS / "above_good.json",
    "below_good": RESULTS / "below_good.json",
    "ignore_bet_above": RESULTS / "ignore_bet_above.json",
    "ignore_bet_below": RESULTS / "ignore_bet_below.json",
}
OUT = RESULTS / "instruct_n_boost"
LOAD_ABORT_S = 10 * 60
# 3× if time allows; stop after 2× if the remaining wall budget is tight.
N_MIN = 8
N_TARGET = 12
WALL_BUDGET_S = 75 * 60


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _merge_existing(condition: str) -> list[dict]:
    """Keep the original n=4 draws; also pick up any prior boost rows."""
    old = _load_rows(OLD_PATH[condition])
    boosted = _load_rows(OUT / f"{condition}.json")
    by_i: dict[int, dict] = {}
    for row in old + boosted:
        by_i[int(row["i"])] = row
    return [by_i[i] for i in sorted(by_i)]


def _write_state(rows: dict[str, list[dict]], extra: dict | None = None) -> None:
    for cond, rs in rows.items():
        save_json(OUT / f"{cond}.json", rs)
    above = rows["above_good"]
    below = rows["below_good"]
    ign_a = rows["ignore_bet_above"]
    ign_b = rows["ignore_bet_below"]
    baseline = bias_from_rows(above, below, THRESHOLD) if above and below else None
    ignore = (
        bias_from_rows(ign_a, ign_b, THRESHOLD) if ign_a and ign_b else None
    )
    summary = {
        "model": MODEL_ID,
        "thinking": False,
        "threshold": THRESHOLD,
        "max_new_tokens": MAX_NEW,
        "seeds": SEED,
        "protocol": "independent seeds; continue original 200/300/4100/4200 streams",
        "n_requested_per_cell": {k: len(v) for k, v in rows.items()},
        "baseline": baseline,
        "ignore_bet": ignore,
        "baseline_above_estimates": [r["estimate"] for r in above],
        "baseline_below_estimates": [r["estimate"] for r in below],
        "ignore_bet_above_estimates": [r["estimate"] for r in ign_a],
        "ignore_bet_below_estimates": [r["estimate"] for r in ign_b],
        "parse_fail": {
            "baseline_above": sum(r["estimate"] is None for r in above),
            "baseline_below": sum(r["estimate"] is None for r in below),
            "ignore_bet_above": sum(r["estimate"] is None for r in ign_a),
            "ignore_bet_below": sum(r["estimate"] is None for r in ign_b),
        },
    }
    if extra:
        summary.update(extra)
    save_json(OUT / "summary.json", summary)
    if baseline is not None:
        save_json(OUT / "baseline_bias.json", baseline)
    if ignore is not None:
        save_json(OUT / "ignore_bet_bias.json", ignore)


def _extend(
    model,
    tokenizer,
    device,
    user_text: str,
    condition: str,
    existing: list[dict],
    n_target: int,
    t0: float,
) -> list[dict]:
    have = len(existing)
    if have >= n_target:
        log(f"{condition}: already n={have}, skip")
        return existing
    need = n_target - have
    elapsed = time.time() - t0
    log(
        f"{condition}: +{need} from seed {SEED[condition] + have} "
        f"(have {have}, want {n_target}, elapsed {elapsed/60:.1f}m)"
    )
    extra = run_condition(
        model,
        tokenizer,
        device,
        user_text,
        condition,
        n=need,
        max_new_tokens=MAX_NEW,
        seed0=SEED[condition] + have,
        enable_thinking=False,
    )
    for row in extra:
        row["i"] = have + int(row["i"])
    return existing + extra


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-target", type=int, default=N_TARGET)
    p.add_argument("--n-min", type=int, default=N_MIN)
    p.add_argument("--wall-budget-s", type=float, default=WALL_BUDGET_S)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = {cond: _merge_existing(cond) for cond in SEED}
    _write_state(rows, extra={"status": "pre-load", "t0": t0})
    for cond, rs in rows.items():
        log(f"have {cond} n={len(rs)}")

    log(f"LOADING {MODEL_ID}  abort_if>{LOAD_ABORT_S}s")
    from src.model_io import load_model

    model, tokenizer, device = load_model(MODEL_ID)
    load_s = time.time() - t0
    log(f"LOADED device={device} load_s={load_s:.1f} thinking=False")
    if load_s > LOAD_ABORT_S:
        log("load exceeded 10 min; writing existing-n summary and exiting")
        _write_state(rows, extra={"status": "load_timeout", "load_s": load_s})
        sys.exit(2)

    specs = [
        ("above_good", bet_prompt(THRESHOLD, True)),
        ("below_good", bet_prompt(THRESHOLD, False)),
        ("ignore_bet_above", ignore_bet_prompt(THRESHOLD, True)),
        ("ignore_bet_below", ignore_bet_prompt(THRESHOLD, False)),
    ]

    # Reach 2× first, then 3× if wall budget remains.
    for n_goal, label in ((args.n_min, "2x"), (args.n_target, "3x")):
        if n_goal > args.n_target:
            continue
        remaining = args.wall_budget_s - (time.time() - t0)
        if label == "3x" and remaining < 12 * 60:
            log(f"skip {label}: remaining {remaining/60:.1f}m < 12m")
            break
        log(f"=== {label} target n={n_goal} remaining={remaining/60:.1f}m ===")
        for cond, user in specs:
            remaining = args.wall_budget_s - (time.time() - t0)
            if remaining < 90 and len(rows[cond]) >= args.n_min:
                log(f"stop {cond}: remaining {remaining:.0f}s and already 2x")
                continue
            rows[cond] = _extend(
                model, tokenizer, device, user, cond, rows[cond], n_goal, t0
            )
            _write_state(
                rows,
                extra={
                    "status": f"running_{label}",
                    "elapsed_s": time.time() - t0,
                    "load_s": load_s,
                },
            )

    _write_state(
        rows,
        extra={
            "status": "done",
            "elapsed_s": time.time() - t0,
            "load_s": load_s,
            "n_min": args.n_min,
            "n_target": args.n_target,
        },
    )
    summary = json.loads((OUT / "summary.json").read_text())
    log(
        "done baseline={b} ignore={i} parse_fail={p}".format(
            b=summary.get("baseline"),
            i=summary.get("ignore_bet"),
            p=summary.get("parse_fail"),
        )
    )


if __name__ == "__main__":
    main()
