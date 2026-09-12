"""Filler-token (dot) accuracy on short grade-school arithmetic. Thinking off."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

from src.model_io import apply_chat, load_model, residual_tensor, _text_layer_list
from src.package4b import try_load_jlens, unembed_topk
from src.run import ROOT, log, save_json

MODEL_ID = "Qwen/Qwen3.5-4B"
LAYER = 23
OUT = ROOT / "results" / "filler"
ANSWER_REQUEST = "Write only the final numeric answer, with no explanation."
# If 256 OOMs, main() retries this shorter ladder.
DOT_LADDER = (0, 16, 64, 256)
DOT_LADDER_FALLBACK = (0, 32, 128)
MAX_NEW = 32
N_PROBLEMS = 20
BUMP_PP = 5.0

PROBLEMS: list[dict] = [
    {"id": 1, "question": "What is 15 + 27?", "answer": "42"},
    {"id": 2, "question": "What is 84 - 19?", "answer": "65"},
    {"id": 3, "question": "What is 8 times 7?", "answer": "56"},
    {"id": 4, "question": "What is 144 divided by 12?", "answer": "12"},
    {"id": 5, "question": "What is 9 + 16?", "answer": "25"},
    {"id": 6, "question": "What is 100 - 37?", "answer": "63"},
    {"id": 7, "question": "What is 6 times 9?", "answer": "54"},
    {"id": 8, "question": "What is 81 divided by 9?", "answer": "9"},
    {"id": 9, "question": "A store had 40 apples and sold 13. How many apples are left?", "answer": "27"},
    {"id": 10, "question": "There are 5 boxes with 8 pencils each. How many pencils in total?", "answer": "40"},
    {"id": 11, "question": "What is 23 + 48?", "answer": "71"},
    {"id": 12, "question": "What is 90 - 45?", "answer": "45"},
    {"id": 13, "question": "What is 11 times 11?", "answer": "121"},
    {"id": 14, "question": "What is 56 divided by 7?", "answer": "8"},
    {"id": 15, "question": "Maya has 18 stickers and gets 14 more. How many stickers does she have?", "answer": "32"},
    {"id": 16, "question": "A bus has 36 seats and 9 are empty. How many seats are filled?", "answer": "27"},
    {"id": 17, "question": "What is 7 + 8 + 9?", "answer": "24"},
    {"id": 18, "question": "What is 50 + 25 - 10?", "answer": "65"},
    {"id": 19, "question": "What is 12 times 4?", "answer": "48"},
    {"id": 20, "question": "What is 200 divided by 25?", "answer": "8"},
]


def user_text(question: str, n_dots: int) -> str:
    if n_dots <= 0:
        return f"{question} {ANSWER_REQUEST}"
    return f"{question} {'.' * n_dots} {ANSWER_REQUEST}"


def extract_number(text: str) -> str | None:
    cleaned = text
    for tag in ("<think>", "</think>"):
        cleaned = cleaned.replace(tag, " ")
    m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    return m.group(0) if m else None


def numbers_equal(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return float(pred) == float(gold)
    except ValueError:
        return pred.strip() == gold.strip()


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt: str, device, max_new_tokens: int) -> tuple[str, int]:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    n_new = int(out.shape[1] - prompt_len)
    text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=False)
    for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        text = text.replace(tok, "")
    return text.strip(), n_new


@torch.no_grad()
def residuals_all_positions(model, tokenizer, prompt: str, layer: int, device):
    layers = _text_layer_list(model)
    captured: dict[str, torch.Tensor] = {}

    def hook(_mod, _inp, output):
        hidden = residual_tensor(output)
        captured["h"] = hidden[0].detach().float().cpu()
        return output

    handle = layers[layer].register_forward_hook(hook)
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model(**inputs, use_cache=False)
        ids = inputs["input_ids"][0].detach().cpu()
    finally:
        handle.remove()
    if "h" not in captured:
        raise RuntimeError(f"No residual captured at layer {layer}")
    return captured["h"], ids


def _dot_token_ids(tokenizer) -> set[int]:
    ids = set()
    for s in (".", " ."):
        enc = tokenizer.encode(s, add_special_tokens=False)
        ids.update(int(x) for x in enc)
    return ids


def run_accuracy(model, tokenizer, device, n_dots: int, max_new_tokens: int) -> dict:
    rows = []
    n_ok = 0
    token_counts = []
    for item in PROBLEMS[:N_PROBLEMS]:
        user = user_text(item["question"], n_dots)
        prompt = apply_chat(tokenizer, user, enable_thinking=False)
        t0 = time.time()
        text, n_new = greedy_generate(model, tokenizer, prompt, device, max_new_tokens)
        pred = extract_number(text)
        ok = numbers_equal(pred, item["answer"])
        n_ok += int(ok)
        token_counts.append(n_new)
        rows.append(
            {
                "id": item["id"],
                "n_dots": n_dots,
                "question": item["question"],
                "gold": item["answer"],
                "pred": pred,
                "correct": ok,
                "n_new_tokens": n_new,
                "seconds": round(time.time() - t0, 3),
                "text": text,
            }
        )
        log(
            f"  dots={n_dots} id={item['id']:02d} gold={item['answer']} "
            f"pred={pred} ok={ok} new_tok={n_new}"
        )
    acc = n_ok / len(rows)
    return {
        "n_dots": n_dots,
        "n": len(rows),
        "n_correct": n_ok,
        "accuracy": acc,
        "mean_new_tokens": float(sum(token_counts) / len(token_counts)),
        "rows": rows,
    }


def maybe_jlens_dot_positions(model, tokenizer, device, n_dots: int) -> dict:
    """If there was an accuracy bump, compare mean-dot vs last-question J-Lens."""
    from src.directions import cosine

    J_l, jmeta = try_load_jlens(LAYER)
    bundle = torch.load(ROOT / "results" / "direction.pt", map_location="cpu", weights_only=True)
    v_leak = bundle["v"].float()
    dot_ids = _dot_token_ids(tokenizer)
    layers_to_read = [LAYER, max(0, LAYER - 8)]
    out = {"jlens": jmeta, "layers": layers_to_read, "items": []}
    for item in PROBLEMS[: min(5, N_PROBLEMS)]:
        user = user_text(item["question"], n_dots)
        prompt = apply_chat(tokenizer, user, enable_thinking=False)
        item_rec = {"id": item["id"], "layers": {}}
        for layer in layers_to_read:
            h, ids = residuals_all_positions(model, tokenizer, prompt, layer, device)
            toks = [tokenizer.decode([int(i)]) for i in ids.tolist()]
            # Dots live in the user span; take '.' tokens that are not sentence-final
            # of the question if possible. Use all '.' after the question text starts.
            q_prompt = apply_chat(
                tokenizer, user_text(item["question"], 0), enable_thinking=False
            )
            q_ids = tokenizer(q_prompt, return_tensors="pt")["input_ids"][0]
            # Last question token ≈ last token of the 0-dot prompt that is still
            # in the question (before answer request). Simpler: last token of
            # tokenized question inside the filled prompt.
            q_only = tokenizer.encode(item["question"], add_special_tokens=False)
            last_q = None
            for i in range(len(ids) - len(q_only) + 1):
                if ids[i : i + len(q_only)].tolist() == q_only:
                    last_q = i + len(q_only) - 1
                    break
            if last_q is None:
                last_q = max(0, len(q_ids) - 8)
            dot_pos = [
                i
                for i, tid in enumerate(ids.tolist())
                if tid in dot_ids and last_q is not None and i > last_q
            ]
            h_q = h[last_q]
            if dot_pos:
                h_dots = h[dot_pos].mean(0)
            else:
                h_dots = h[-1]
            rec = {
                "last_question_token": toks[last_q] if last_q < len(toks) else None,
                "n_dot_positions": len(dot_pos),
                "cosine_mean_dots_vs_last_q": cosine(h_dots, h_q),
                "cosine_mean_dots_vs_v_leak": cosine(h_dots, v_leak),
                "cosine_last_q_vs_v_leak": cosine(h_q, v_leak),
            }
            if J_l is not None:
                rec["jlens_mean_dots"] = unembed_topk(
                    model, tokenizer, h_dots, 10, device, pre_map=J_l, rms_norm=True
                )
                rec["jlens_last_q"] = unembed_topk(
                    model, tokenizer, h_q, 10, device, pre_map=J_l, rms_norm=True
                )
                rec["jlens_v_leak"] = unembed_topk(
                    model, tokenizer, v_leak, 10, device, pre_map=J_l, rms_norm=True
                )
            item_rec["layers"][str(layer)] = rec
        out["items"].append(item_rec)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    log(f"loading {args.model}")
    model, tokenizer, device = load_model(args.model)
    log(f"device={device} thinking=False greedy n={N_PROBLEMS}")

    ladders = [DOT_LADDER, DOT_LADDER_FALLBACK]
    used_ladder = None
    by_dots: dict[int, dict] = {}
    oom_error = None
    for ladder in ladders:
        try:
            by_dots = {}
            for n_dots in ladder:
                log(f"condition dots={n_dots}")
                by_dots[n_dots] = run_accuracy(
                    model, tokenizer, device, n_dots, args.max_new_tokens
                )
            used_ladder = list(ladder)
            oom_error = None
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() and "OOM" not in str(exc):
                raise
            oom_error = str(exc)
            log(f"OOM on ladder={ladder}: {exc}; trying fallback")
            if device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()

    if used_ladder is None:
        raise SystemExit(f"all ladders OOM: {oom_error}")

    accs = {n: by_dots[n]["accuracy"] for n in used_ladder}
    acc0 = accs[0]
    deltas_pp = {n: (accs[n] - acc0) * 100.0 for n in used_ladder}
    bump_ns = [n for n in used_ladder if n > 0 and deltas_pp[n] > BUMP_PP]
    bump = bool(bump_ns)

    summary = {
        "model": args.model,
        "thinking": False,
        "decoding": "greedy",
        "n_problems": N_PROBLEMS,
        "dot_ladder": used_ladder,
        "answer_request": ANSWER_REQUEST,
        "accuracy": accs,
        "n_correct": {n: by_dots[n]["n_correct"] for n in used_ladder},
        "mean_new_tokens": {n: by_dots[n]["mean_new_tokens"] for n in used_ladder},
        "delta_pp_vs_0_dots": deltas_pp,
        "bump": bump,
        "bump_threshold_pp": BUMP_PP,
        "bump_conditions": bump_ns,
        "stopped_after_accuracy": not bump,
        "ceiling": acc0 >= 0.95,
        "note": (
            "Accuracy flat vs 0 dots (no condition >5pp"
            + ("; ceiling at 0 dots, so a positive bump is undetectable on this set" if acc0 >= 0.95 else "")
            + "). Negative result; did not J-Lens filler positions."
            if not bump
            else "Accuracy rose >5pp vs 0 dots; cached residuals and J-Lens at dot tokens."
        ),
    }
    save_json(OUT / "summary.json", summary)
    save_json(
        OUT / "items.json",
        {str(n): by_dots[n]["rows"] for n in used_ladder},
    )
    log(json.dumps({k: summary[k] for k in ("accuracy", "delta_pp_vs_0_dots", "bump")}, indent=2))

    if bump:
        best = max(bump_ns, key=lambda n: accs[n])
        log(f"bump at dots={bump_ns}; J-Lens positions on dots={best}")
        extra = maybe_jlens_dot_positions(model, tokenizer, device, best)
        save_json(OUT / "jlens_positions.json", extra)
    else:
        log("no filler bump; skip J-Lens at filler positions")
    log("filler done")


if __name__ == "__main__":
    main()
