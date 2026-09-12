"""J-Lens real last-token residuals on giraffe Donation Bet prompts.

Thinking off. Prefills first. Do not lens v_leak except as a footnote.
Reuse src/jlens_eval.py, src/model_io.py, existing lens.pt.
"""

from __future__ import annotations

import json
import time
import torch

from src.directions import cosine
from src.jlens_eval import (
    LAYER,
    LOAD_BUDGET_S,
    LOCAL_JLENS,
    MODEL_ID,
    inspect_jlens_architecture,
    item_record,
    load_jlens_local_or_hub,
    prompt_residual,
)
from src.model_io import apply_chat, generate, residual_tensor, _text_layer_list
from src.prompts import bet_prompt, ignore_bet_prompt, no_bet_prompt
from src.run import RESULTS, ROOT, log, save_json, seed_all

OUT = ROOT / "results" / "jlens_giraffe"
BOOST = RESULTS / "instruct_n_boost"
THRESHOLD = 14898555.0
MAX_NEW_SHORT = 32
GEN_BUDGET_S = 8 * 60
FIRST_NUM_BUDGET_S = 12 * 60

# Tokens J-Lens of v_leak previously dumped (plus close relatives).
MAGNITUDE_NEEDLES = (
    "庞大",
    "庞大的",
    "huge",
    "enormous",
    "超大",
    "巨大",
    "亿元以上",
    "万以上",
    "米以上",
    "倍以上",
    "庞",
    "larger",
    "massively",
    "astronomical",
    "gigantic",
    "colossal",
    "immense",
    "vast",
)


def _user_prompts() -> list[tuple[str, str, str]]:
    """(name, side, raw user text). Same templates as the leakage / n-boost run."""
    return [
        ("above_good", "above", bet_prompt(THRESHOLD, True)),
        ("below_good", "below", bet_prompt(THRESHOLD, False)),
        ("ignore_bet_above", "above", ignore_bet_prompt(THRESHOLD, True)),
        ("ignore_bet_below", "below", ignore_bet_prompt(THRESHOLD, False)),
        ("no_bet", "none", no_bet_prompt()),
    ]


def _magnitude_hits(tokens: list[str]) -> list[str]:
    hits: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        low = tok.strip().lower()
        for needle in MAGNITUDE_NEEDLES:
            nlow = needle.lower()
            if nlow in low or (len(low) >= 2 and low in nlow and len(nlow) <= 4):
                if needle not in seen:
                    seen.add(needle)
                    hits.append(needle)
    return hits


@torch.no_grad()
def residual_at_index(
    model, input_ids: torch.Tensor, layer: int, device, index: int
) -> torch.Tensor:
    layers = _text_layer_list(model)
    captured: dict[str, torch.Tensor] = {}

    def hook(_mod, _inp, output):
        hidden = residual_tensor(output)
        captured["h"] = hidden[:, index, :].detach().float().cpu()
        return output

    handle = layers[layer].register_forward_hook(hook)
    try:
        model(input_ids=input_ids.to(device), use_cache=False)
    finally:
        handle.remove()
    if "h" not in captured:
        raise RuntimeError(f"No residual captured at layer {layer} index {index}")
    return captured["h"].squeeze(0)


def first_number_token(tokenizer, prompt: str, continuation: str):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + continuation, add_special_tokens=False)["input_ids"]
    if not isinstance(prompt_ids, list):
        prompt_ids = prompt_ids.tolist()
    if not isinstance(full_ids, list):
        full_ids = full_ids.tolist()
    plen = len(prompt_ids)
    if len(full_ids) < plen or full_ids[:plen] != prompt_ids:
        # Chat concat can retokenize; fall back to encoding continuation alone.
        cont_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
        if not isinstance(cont_ids, list):
            cont_ids = cont_ids.tolist()
        full_ids = prompt_ids + cont_ids
    for offset, tid in enumerate(full_ids[plen:]):
        tok = tokenizer.decode([int(tid)])
        if any(ch.isdigit() for ch in tok):
            return {
                "index": plen + offset,
                "token": tok,
                "token_id": int(tid),
                "prompt_len": plen,
                "full_len": len(full_ids),
                "full_ids": full_ids,
            }
    return {
        "index": None,
        "token": None,
        "token_id": None,
        "prompt_len": plen,
        "full_len": len(full_ids),
        "full_ids": full_ids,
    }


def _load_boost_rows() -> dict[str, list[dict]]:
    rows = {}
    for name in ("above_good", "below_good", "ignore_bet_above", "ignore_bet_below"):
        path = BOOST / f"{name}.json"
        if path.exists():
            rows[name] = json.loads(path.read_text())
        else:
            legacy = RESULTS / f"{name}.json"
            rows[name] = json.loads(legacy.read_text()) if legacy.exists() else []
    return rows


def _abort(reason: str, extra: dict | None = None) -> None:
    payload = {"aborted": True, "reason": reason}
    if extra:
        payload.update(extra)
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT / "summary.json", payload)
    log(f"ABORT: {reason}")


def write_notes(summary: dict) -> None:
    def row_line(rec: dict) -> str:
        mag = rec.get("magnitude_hits") or []
        return (
            f"- `{rec['name']}` last=`{rec.get('last_token', '')}` "
            f"J-Lens {rec.get('jlens_top10', [])} "
            f"| logit-lens {rec.get('logit_lens_top10', [])} "
            f"| magnitude_hits={mag}"
        )

    prefills = summary.get("prefill", [])
    firsts = summary.get("first_number", [])
    vnote = summary.get("v_leak_footnote") or {}
    claim = summary.get("claim", "")
    notes = f"""# J-Lens on real giraffe Donation Bet residuals

Not a mediation claim. v_leak is a footnote only — the question is whether
J-Lens of the **actual last-token residuals** says huge/enormous/庞大.

## Setup

- Model: `{summary.get('model')}`, layer {summary.get('layer')}, thinking off, MPS.
- J-Lens: `{summary.get('jlens', {}).get('local_path', LOCAL_JLENS)}`
- Prompts: same chat-templated giraffe Donation Bet templates as the leakage /
  n-boost run (`src/prompts.py`). n-boost n={summary.get('n_boost')}.
- Prefill last-token is the must-have. First-number uses existing n-boost
  completions (already short numbers) when present; otherwise one ~32-token gen.

## Prefill last-token (per prompt)

{chr(10).join(row_line(r) for r in prefills)}

## First-number token (if run)

{chr(10).join(row_line(r) for r in firsts) if firsts else '- skipped'}

## v_leak footnote (difference-in-means, not a residual)

- J-Lens {vnote.get('jlens_top10', [])}
- logit-lens {vnote.get('logit_lens_top10', [])}

## Magnitude words on real residuals?

- above prefill: {summary.get('magnitude_above_prefill')}
- below prefill: {summary.get('magnitude_below_prefill')}
- above first-number: {summary.get('magnitude_above_first_number')}
- below first-number: {summary.get('magnitude_below_first_number')}

## Claim

{claim}
"""
    (OUT / "NOTES.md").write_text(notes)


def _claim(prefill: list[dict], firsts: list[dict], vnote: dict) -> str:
    def mag_of(items, side):
        hits = []
        for r in items:
            if r.get("side") == side:
                hits.extend(r.get("magnitude_hits") or [])
        return sorted(set(hits))

    a = mag_of(prefill, "above")
    b = mag_of(prefill, "below")
    a_n = mag_of(firsts, "above")
    b_n = mag_of(firsts, "below")
    above_rec = next((r for r in prefill if r["name"] == "above_good"), {})
    below_rec = next((r for r in prefill if r["name"] == "below_good"), {})
    return (
        "J-Lens of the real above-good last-token residual "
        f"top-10={above_rec.get('jlens_top10')}; "
        "below-good "
        f"top-10={below_rec.get('jlens_top10')}. "
        f"Magnitude needles on above prefill={a or 'none'}, "
        f"below prefill={b or 'none'}"
        + (
            f"; first-number above={a_n or 'none'}, below={b_n or 'none'}"
            if firsts
            else ""
        )
        + ". "
        f"v_leak footnote still reads { (vnote.get('jlens_top10') or [])[:5] }. "
        "This is a readout of last-token residuals, not a mediation claim."
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    arch = inspect_jlens_architecture(LAYER)
    J_l, jmeta = load_jlens_local_or_hub(LAYER)
    if J_l is None:
        _abort("j-lens missing", {"jlens": jmeta, "architecture": arch})
        return
    log(f"J-Lens loaded shape={jmeta.get('J_l_shape')} path={jmeta.get('local_path')}")

    boost_rows = _load_boost_rows()
    n_boost = {k: len(v) for k, v in boost_rows.items()}
    log(f"n-boost rows available: {n_boost}")

    log(f"loading {MODEL_ID} (abort if >{LOAD_BUDGET_S}s)")
    from src.model_io import load_model

    model, tokenizer, device = load_model(MODEL_ID)
    load_s = time.time() - t0
    log(f"LOADED device={device} load_s={load_s:.1f} thinking=False")
    if load_s > LOAD_BUDGET_S:
        _abort("load exceeded 10 min", {"load_s": load_s, "architecture": arch})
        return
    if device.type != "mps":
        log(f"WARNING: wanted MPS, got {device}")

    specs = []
    for name, side, user in _user_prompts():
        chat = apply_chat(tokenizer, user, enable_thinking=False)
        specs.append({"name": name, "side": side, "user": user, "chat": chat})

    prefill: list[dict] = []
    h_by_name: dict[str, torch.Tensor] = {}
    for spec in specs:
        h, pmeta = prompt_residual(model, tokenizer, spec["chat"], LAYER, device)
        rec = item_record(
            spec["name"],
            "on",
            "last_token_residual",
            h,
            model,
            tokenizer,
            device,
            J_l,
            extra=pmeta,
        )
        rec["side"] = spec["side"]
        rec["template"] = spec["name"]
        rec["magnitude_hits"] = _magnitude_hits(rec.get("jlens_top10") or [])
        rec["magnitude_hits_logit"] = _magnitude_hits(rec.get("logit_lens_top10") or [])
        prefill.append(rec)
        h_by_name[spec["name"]] = h
        log(
            f"prefill {spec['name']} last={pmeta.get('last_token')!r} "
            f"J={rec['jlens_top10'][:5]} mag={rec['magnitude_hits']}"
        )

    # Footnote only: do not treat v_leak as the residual under test.
    vnote = {}
    direction_path = RESULTS / "direction.pt"
    if direction_path.exists():
        bundle = torch.load(direction_path, map_location="cpu", weights_only=True)
        v_leak = bundle["v"].float()
        vnote = item_record(
            "v_leak_footnote",
            "off",
            "results/direction.pt:v (difference-in-means, not a residual)",
            v_leak,
            model,
            tokenizer,
            device,
            J_l,
        )
        vnote["magnitude_hits"] = _magnitude_hits(vnote.get("jlens_top10") or [])
        if "above_good" in h_by_name and "below_good" in h_by_name:
            vnote["cos_vleak_h_above"] = cosine(v_leak, h_by_name["above_good"])
            vnote["cos_vleak_h_below"] = cosine(v_leak, h_by_name["below_good"])
            vnote["cos_h_above_h_below"] = cosine(
                h_by_name["above_good"], h_by_name["below_good"]
            )
        log(f"v_leak footnote J={vnote['jlens_top10'][:5]}")

    first_number: list[dict] = []
    t_num = time.time()
    # Reuse n-boost completions (already short numbers). Prefill only.
    for spec in specs:
        if spec["name"] not in boost_rows:
            continue
        rows = boost_rows[spec["name"]]
        for row in rows:
            if time.time() - t_num > FIRST_NUM_BUDGET_S:
                log("first-number budget hit; stopping extra prefills")
                break
            text = (row.get("text") or "").strip()
            if not text:
                continue
            loc = first_number_token(tokenizer, spec["chat"], text)
            extra = {
                "side": spec["side"],
                "template": spec["name"],
                "i": row.get("i"),
                "estimate": row.get("estimate"),
                "continuation": text[:80],
                "number_token": loc["token"],
                "number_index": loc["index"],
                "prompt_len": loc["prompt_len"],
                "source": "n_boost_completion_first_number",
            }
            if loc["index"] is None:
                extra["skipped"] = "no_number_token"
                log(f"no number token {spec['name']} i={row.get('i')} text={text!r}")
                continue
            ids = torch.tensor([loc["full_ids"]], dtype=torch.long)
            h = residual_at_index(model, ids, LAYER, device, loc["index"])
            rec = item_record(
                f"{spec['name']}_num_{row.get('i')}",
                "on",
                "first_number_residual",
                h,
                model,
                tokenizer,
                device,
                J_l,
                extra=extra,
            )
            rec["magnitude_hits"] = _magnitude_hits(rec.get("jlens_top10") or [])
            rec["magnitude_hits_logit"] = _magnitude_hits(
                rec.get("logit_lens_top10") or []
            )
            first_number.append(rec)
            log(
                f"first-num {spec['name']} i={row.get('i')} tok={loc['token']!r} "
                f"J={rec['jlens_top10'][:4]} mag={rec['magnitude_hits']}"
            )
        else:
            continue
        break

    # If n-boost had no usable numbers, one short gen per above/below.
    if not first_number and time.time() - t0 < LOAD_BUDGET_S + GEN_BUDGET_S:
        log("no n-boost numbers; trying one short gen per above/below")
        for spec in specs:
            if spec["name"] not in {"above_good", "below_good"}:
                continue
            if time.time() - t0 > LOAD_BUDGET_S + GEN_BUDGET_S:
                log("skip remaining short gens (budget)")
                break
            seed_all(200 if spec["name"] == "above_good" else 300, device)
            t_gen = time.time()
            text = generate(
                model,
                tokenizer,
                spec["chat"],
                device,
                max_new_tokens=MAX_NEW_SHORT,
                temperature=1.0,
                top_p=0.95,
                top_k=20,
            )
            gen_s = time.time() - t_gen
            log(f"short gen {spec['name']} {gen_s:.1f}s text={text[:60]!r}")
            if gen_s > 90:
                log("gen slow; skipping remaining short gens")
                loc = first_number_token(tokenizer, spec["chat"], text)
                if loc["index"] is not None:
                    ids = torch.tensor([loc["full_ids"]], dtype=torch.long)
                    h = residual_at_index(model, ids, LAYER, device, loc["index"])
                    rec = item_record(
                        f"{spec['name']}_num_gen0",
                        "on",
                        "first_number_residual",
                        h,
                        model,
                        tokenizer,
                        device,
                        J_l,
                        extra={
                            "side": spec["side"],
                            "template": spec["name"],
                            "continuation": text[:80],
                            "number_token": loc["token"],
                            "source": "short_gen",
                        },
                    )
                    rec["magnitude_hits"] = _magnitude_hits(rec.get("jlens_top10") or [])
                    first_number.append(rec)
                break
            loc = first_number_token(tokenizer, spec["chat"], text)
            if loc["index"] is None:
                continue
            ids = torch.tensor([loc["full_ids"]], dtype=torch.long)
            h = residual_at_index(model, ids, LAYER, device, loc["index"])
            rec = item_record(
                f"{spec['name']}_num_gen0",
                "on",
                "first_number_residual",
                h,
                model,
                tokenizer,
                device,
                J_l,
                extra={
                    "side": spec["side"],
                    "template": spec["name"],
                    "continuation": text[:80],
                    "number_token": loc["token"],
                    "source": "short_gen",
                },
            )
            rec["magnitude_hits"] = _magnitude_hits(rec.get("jlens_top10") or [])
            first_number.append(rec)

    def any_mag(items, side):
        return sorted(
            {
                h
                for r in items
                if r.get("side") == side
                for h in (r.get("magnitude_hits") or [])
            }
        )

    claim = _claim(prefill, first_number, vnote)
    summary = {
        "model": MODEL_ID,
        "thinking": False,
        "layer": LAYER,
        "method": "jlens+unembed",
        "device": str(device),
        "load_s": load_s,
        "elapsed_s": time.time() - t0,
        "jlens": jmeta,
        "architecture": arch,
        "threshold": THRESHOLD,
        "n_boost": n_boost,
        "n_prefill": len(prefill),
        "n_first_number": len(first_number),
        "prefill": prefill,
        "first_number": first_number,
        "v_leak_footnote": vnote,
        "magnitude_above_prefill": any_mag(prefill, "above"),
        "magnitude_below_prefill": any_mag(prefill, "below"),
        "magnitude_above_first_number": any_mag(first_number, "above"),
        "magnitude_below_first_number": any_mag(first_number, "below"),
        "prefill_top10": {
            r["name"]: {
                "jlens": r.get("jlens_top10"),
                "logit_lens": r.get("logit_lens_top10"),
                "magnitude_hits": r.get("magnitude_hits"),
                "last_token": r.get("last_token"),
            }
            for r in prefill
        },
        "claim": claim,
        "do_not_treat_v_leak_as_residual": True,
        "do_not_claim_mediation": True,
    }
    save_json(OUT / "summary.json", summary)
    save_json(
        OUT / "prefill.json",
        {r["name"]: r for r in prefill},
    )
    save_json(OUT / "first_number.json", first_number)
    write_notes(summary)
    log(claim)
    log(f"wrote {OUT} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
