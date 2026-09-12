"""3-hour pipeline: replicate giraffe Donation Bet, fit a direction, ablate vs random."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from src.device import pick_device
from src.directions import difference_in_means, random_unit
from src.model_io import apply_chat, generate, last_token_residual, load_model
from src.prompts import bet_prompt, no_bet_prompt
from src.score import balanced_bias, extract_estimate, on_good_side

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_all(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if torch.is_tensor(obj):
        return obj.tolist()
    raise TypeError(type(obj))


def run_condition(
    model,
    tokenizer,
    device,
    user_text: str,
    condition: str,
    n: int,
    max_new_tokens: int,
    seed0: int,
    enable_thinking: bool = False,
) -> list[dict]:
    prompt = apply_chat(tokenizer, user_text, enable_thinking=enable_thinking)
    rows = []
    for i in range(n):
        seed_all(seed0 + i, device)
        text = generate(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=max_new_tokens,
        )
        est = extract_estimate(text)
        rows.append(
            {
                "i": i,
                "condition": condition,
                "estimate": est,
                "text": text,
            }
        )
        closed = "</think>" in text
        log(
            f"  {condition} {i+1}/{n} estimate={est} "
            f"chars={len(text)} think_closed={closed}"
        )
    return rows


def bias_from_rows(above: list[dict], below: list[dict], threshold: float):
    a_flags = [
        on_good_side(r["estimate"], threshold, True)
        for r in above
        if r["estimate"] is not None
    ]
    b_flags = [
        on_good_side(r["estimate"], threshold, False)
        for r in below
        if r["estimate"] is not None
    ]
    dropped = (len(above) - len(a_flags)) + (len(below) - len(b_flags))
    result = balanced_bias(a_flags, b_flags)
    out = result.as_dict()
    out["dropped"] = dropped
    out["threshold"] = threshold
    return out


def plot_bias(series: dict[str, float], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib missing; skip plot")
        return
    labels = list(series.keys())
    vals = [series[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(labels, vals, color="#3b6ea5")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Donation Bet bias")
    ax.set_title("Giraffe value leakage under residual interventions")
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layer", type=int, default=23)
    p.add_argument("--n-nobet", type=int, default=4)
    p.add_argument("--n-bet", type=int, default=4)
    p.add_argument("--n-steer", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--thinking", action="store_true", help="Keep Qwen thinking mode on")
    p.add_argument("--stage", default="all", choices=["all", "smoke", "replicate", "steer"])
    args = p.parse_args()
    thinking = args.thinking
    global RESULTS
    RESULTS = ROOT / ("results_thinking" if thinking else "results")

    RESULTS.mkdir(exist_ok=True)
    meta_path = RESULTS / "meta.json"
    save_json(meta_path, vars(args) | {"device_probe": str(pick_device())})

    log(f"loading {args.model}")
    model, tokenizer, device = load_model(args.model)
    n_layers = len(
        __import__("src.model_io", fromlist=["_text_layer_list"])._text_layer_list(model)
    )
    if args.layer >= n_layers:
        raise SystemExit(f"layer {args.layer} >= {n_layers}")
    log(f"device={device} layers={n_layers} intervene_layer={args.layer} thinking={thinking}")

    if args.stage == "smoke":
        log("smoke: 2 no-bet giraffe samples")
        smoke = run_condition(
            model,
            tokenizer,
            device,
            no_bet_prompt(),
            "smoke_nobet",
            n=2,
            max_new_tokens=min(512, args.max_new_tokens),
            seed0=0,
            enable_thinking=thinking,
        )
        save_json(RESULTS / "smoke.json", smoke)
        if all(r["estimate"] is None for r in smoke):
            raise SystemExit("smoke test: could not parse any estimate; stop")
        return

    if args.stage == "steer":
        threshold = json.loads((RESULTS / "threshold.json").read_text())["threshold"]
        baseline_bias = json.loads((RESULTS / "baseline_bias.json").read_text())
        bundle = torch.load(RESULTS / "direction.pt", map_location="cpu", weights_only=True)
        v = bundle["v"]
        log(
            f"steer-only: threshold={threshold:g} "
            f"baseline_bias={baseline_bias['bias']} layer={args.layer} c={args.strength}"
        )
    else:
        log(f"no-bet n={args.n_nobet}")
        nobet = run_condition(
            model,
            tokenizer,
            device,
            no_bet_prompt(),
            "no_bet",
            n=args.n_nobet,
            max_new_tokens=args.max_new_tokens,
            seed0=100,
            enable_thinking=thinking,
        )
        save_json(RESULTS / "no_bet.json", nobet)
        estimates = [r["estimate"] for r in nobet if r["estimate"] is not None]
        if len(estimates) < 3:
            raise SystemExit(f"too few no-bet parses: {estimates}")
        threshold = float(statistics.median(estimates))
        log(f"threshold (no-bet median) = {threshold:g}  n_parsed={len(estimates)}")
        save_json(
            RESULTS / "threshold.json",
            {"threshold": threshold, "no_bet_estimates": estimates},
        )

        log(f"above-good n={args.n_bet}")
        above = run_condition(
            model,
            tokenizer,
            device,
            bet_prompt(threshold, above_good=True),
            "above_good",
            n=args.n_bet,
            max_new_tokens=args.max_new_tokens,
            seed0=200,
            enable_thinking=thinking,
        )
        log(f"below-good n={args.n_bet}")
        below = run_condition(
            model,
            tokenizer,
            device,
            bet_prompt(threshold, above_good=False),
            "below_good",
            n=args.n_bet,
            max_new_tokens=args.max_new_tokens,
            seed0=300,
            enable_thinking=thinking,
        )
        save_json(RESULTS / "above_good.json", above)
        save_json(RESULTS / "below_good.json", below)
        baseline_bias = bias_from_rows(above, below, threshold)
        log(f"baseline bias={baseline_bias['bias']:.3f} ± {baseline_bias['se']:.3f}")
        save_json(RESULTS / "baseline_bias.json", baseline_bias)

        log("fitting last-token difference-in-means (above vs below prompt)")
        above_prompt = apply_chat(
            tokenizer, bet_prompt(threshold, True), enable_thinking=thinking
        )
        below_prompt = apply_chat(
            tokenizer, bet_prompt(threshold, False), enable_thinking=thinking
        )
        h_above = last_token_residual(
            model, tokenizer, above_prompt, args.layer, device
        )
        h_below = last_token_residual(
            model, tokenizer, below_prompt, args.layer, device
        )
        v = difference_in_means(h_above, h_below)
        torch.save(
            {
                "v": v,
                "h_above": h_above,
                "h_below": h_below,
                "layer": args.layer,
                "model": args.model,
            },
            RESULTS / "direction.pt",
        )
        log(f"direction dim={tuple(v.shape)}")

        if args.stage == "replicate":
            return

    rand = random_unit(v.numel(), seed=0)
    # Distinct RNG streams per intervention. The first run reused 400/500 for
    # both ablate and random, so those two conditions were not a control.
    steer_specs = [
        ("ablate_v", "project_out", v, args.strength, 1400, 1500),
        ("random", "project_out", rand, args.strength, 2400, 2500),
    ]

    steered_bias = {"baseline": baseline_bias["bias"]}
    for name, mode, vec, c, seed_above, seed_below in steer_specs:
        log(f"steer {name} mode={mode} c={c} n={args.n_steer}")
        above_s = []
        below_s = []
        for cond, user, store, seed0 in (
            ("above_good", bet_prompt(threshold, True), above_s, seed_above),
            ("below_good", bet_prompt(threshold, False), below_s, seed_below),
        ):
            prompt = apply_chat(tokenizer, user, enable_thinking=thinking)
            for i in range(args.n_steer):
                seed_all(seed0 + i, device)
                text = generate(
                    model,
                    tokenizer,
                    prompt,
                    device,
                    max_new_tokens=args.max_new_tokens,
                    layer=args.layer,
                    direction=vec,
                    steer_mode=mode,
                    strength=c,
                )
                est = extract_estimate(text)
                store.append({"i": i, "condition": cond, "estimate": est, "text": text})
                log(f"  {name} {cond} {i+1}/{args.n_steer} estimate={est}")
        save_json(RESULTS / f"{name}_above.json", above_s)
        save_json(RESULTS / f"{name}_below.json", below_s)
        try:
            b = bias_from_rows(above_s, below_s, threshold)
        except ValueError as exc:
            log(f"  {name} could not score: {exc}")
            b = {"bias": None, "error": str(exc)}
        save_json(RESULTS / f"{name}_bias.json", b)
        steered_bias[name] = b.get("bias")
        log(f"  {name} bias={b.get('bias')}")

    save_json(RESULTS / "summary.json", {"threshold": threshold, "biases": steered_bias})
    numeric = {k: v_ for k, v_ in steered_bias.items() if v_ is not None}
    if numeric:
        plot_bias(numeric, RESULTS / "bias_bars.png")
    log("done")
    log(json.dumps({"threshold": threshold, "biases": steered_bias}, indent=2))


if __name__ == "__main__":
    main()
