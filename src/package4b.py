"""Instruct-only follow-ups: ignore-the-bet, magnitude direction, token readout."""

from __future__ import annotations

import json

import torch

from src.directions import cosine, difference_in_means
from src.model_io import apply_chat, last_token_residual, load_model
from src.prompts import ignore_bet_prompt, magnitude_prompt
from src.run import RESULTS, ROOT, bias_from_rows, log, run_condition, save_json

LAYER = 23
MODEL_ID = "Qwen/Qwen3.5-4B"
# Distinct from replicate seeds 200/300 and steer 1400/1500/2400/2500.
IGNORE_SEED_ABOVE = 4100
IGNORE_SEED_BELOW = 4200
N_BET = 4
MAX_NEW = 512
TOP_K = 20
JLENS_REPO = "camilablank/workspace-lenses"
JLENS_FILE = "qwen3.5-4b/j-lens/lens.pt"


def _find_lm_head(model) -> torch.nn.Module:
    if hasattr(model, "lm_head") and isinstance(model.lm_head, torch.nn.Module):
        return model.lm_head
    for name, mod in model.named_modules():
        if name.endswith("lm_head") and isinstance(mod, torch.nn.Linear):
            return mod
    raise RuntimeError("Could not find lm_head")


def _find_final_norm(model) -> torch.nn.Module | None:
    from src.model_io import _text_layer_list

    layers = _text_layer_list(model)
    for _, mod in model.named_modules():
        for child in mod.children():
            if child is layers:
                for attr in ("norm", "final_layernorm", "ln_f"):
                    if hasattr(mod, attr):
                        return getattr(mod, attr)
    return None


@torch.no_grad()
def unembed_topk(
    model,
    tokenizer,
    vec: torch.Tensor,
    k: int,
    device: torch.device,
    pre_map: torch.Tensor | None = None,
    rms_norm: bool = True,
) -> list[dict]:
    h = vec.flatten().float()
    if pre_map is not None:
        h = (pre_map.float() @ h).flatten()
    head = _find_lm_head(model)
    norm = _find_final_norm(model) if rms_norm else None
    x = h.to(device=device, dtype=next(head.parameters()).dtype)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if norm is not None:
        x = norm(x)
    logits = head(x).float().squeeze(0)
    vals, ids = torch.topk(logits, k)
    out = []
    for score, tid in zip(vals.tolist(), ids.tolist()):
        tok = tokenizer.decode([tid])
        out.append({"token_id": int(tid), "token": tok, "logit": float(score)})
    return out


def try_load_jlens(layer: int) -> tuple[torch.Tensor | None, dict]:
    """Download Camila/Agam Qwen3.5-4B J-Lens and return J_layer, or None."""
    dest = ROOT / "artifacts" / "jlens"
    dest.mkdir(parents=True, exist_ok=True)
    meta = {"repo": JLENS_REPO, "file": JLENS_FILE, "layer": layer}
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        meta["error"] = "huggingface_hub missing"
        return None, meta
    try:
        path = hf_hub_download(
            repo_id=JLENS_REPO,
            filename=JLENS_FILE,
            local_dir=str(dest),
        )
        meta["local_path"] = path
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return None, meta
    J = bundle.get("J")
    source_layers = [int(x) for x in bundle.get("source_layers", [])]
    if J is None:
        meta["error"] = f"lens keys={list(bundle.keys())}"
        return None, meta
    meta["source_layers"] = source_layers
    if isinstance(J, dict):
        key = layer if layer in J else (str(layer) if str(layer) in J else None)
        if key is None:
            meta["error"] = f"layer {layer} not in J keys"
            return None, meta
        J_l = J[key].float().cpu()
        meta["J_store"] = "dict"
        meta["used_index"] = int(layer)
    else:
        meta["J_shape"] = list(J.shape)
        if layer in source_layers:
            idx = source_layers.index(layer)
        elif getattr(J, "ndim", 0) == 3 and layer < J.shape[0]:
            idx = layer
            meta["index_fallback"] = "row=layer"
        else:
            meta["error"] = f"layer {layer} not in source_layers={source_layers}"
            return None, meta
        J_l = J[idx].float().cpu()
        meta["used_index"] = idx
    if J_l.ndim != 2:
        meta["error"] = f"unexpected J_l ndim={J_l.ndim}"
        return None, meta
    meta["J_l_shape"] = list(J_l.shape)
    return J_l, meta


def experiment_ignore_bet(model, tokenizer, device, threshold: float) -> dict:
    above_path = RESULTS / "ignore_bet_above.json"
    below_path = RESULTS / "ignore_bet_below.json"
    if above_path.exists() and below_path.exists():
        above = json.loads(above_path.read_text())
        below = json.loads(below_path.read_text())
        bias = json.loads((RESULTS / "ignore_bet_bias.json").read_text())
        log(f"reuse ignore-bet bias={bias['bias']:.3f}")
        return {
            "bias": bias,
            "above_estimates": [r["estimate"] for r in above],
            "below_estimates": [r["estimate"] for r in below],
        }
    log(f"ignore-bet above n={N_BET} seed0={IGNORE_SEED_ABOVE}")
    above = run_condition(
        model,
        tokenizer,
        device,
        ignore_bet_prompt(threshold, True),
        "ignore_bet_above",
        n=N_BET,
        max_new_tokens=MAX_NEW,
        seed0=IGNORE_SEED_ABOVE,
        enable_thinking=False,
    )
    log(f"ignore-bet below n={N_BET} seed0={IGNORE_SEED_BELOW}")
    below = run_condition(
        model,
        tokenizer,
        device,
        ignore_bet_prompt(threshold, False),
        "ignore_bet_below",
        n=N_BET,
        max_new_tokens=MAX_NEW,
        seed0=IGNORE_SEED_BELOW,
        enable_thinking=False,
    )
    save_json(RESULTS / "ignore_bet_above.json", above)
    save_json(RESULTS / "ignore_bet_below.json", below)
    bias = bias_from_rows(above, below, threshold)
    save_json(RESULTS / "ignore_bet_bias.json", bias)
    log(f"ignore-bet bias={bias['bias']:.3f} ± {bias['se']:.3f}")
    return {
        "bias": bias,
        "above_estimates": [r["estimate"] for r in above],
        "below_estimates": [r["estimate"] for r in below],
    }


def experiment_magnitude(model, tokenizer, device, v_leak: torch.Tensor) -> dict:
    high_prompt = apply_chat(tokenizer, magnitude_prompt(True), enable_thinking=False)
    low_prompt = apply_chat(tokenizer, magnitude_prompt(False), enable_thinking=False)
    h_high = last_token_residual(model, tokenizer, high_prompt, LAYER, device)
    h_low = last_token_residual(model, tokenizer, low_prompt, LAYER, device)
    v_mag = difference_in_means(h_high, h_low)
    torch.save(
        {
            "v": v_mag,
            "h_high": h_high,
            "h_low": h_low,
            "layer": LAYER,
            "model": MODEL_ID,
        },
        RESULTS / "magnitude.pt",
    )
    cos = cosine(v_leak, v_mag)
    controls = {
        "cosine_v_leak_v_mag": cos,
        "layer": LAYER,
        "v_leak_norm": float(v_leak.float().norm()),
        "v_mag_norm": float(v_mag.float().norm()),
        "prompts": {
            "high": magnitude_prompt(True),
            "low": magnitude_prompt(False),
        },
        "causal_project_out_v_mag": "skipped (prefill cosine only)",
    }
    save_json(RESULTS / "controls.json", controls)
    log(f"cosine(v_leak, v_mag)={cos:.4f}")
    return {"v_mag": v_mag, "cosine": cos, "controls": controls}


def experiment_readout(model, tokenizer, device, v_leak, v_mag) -> dict:
    J_l, jmeta = try_load_jlens(LAYER)
    method = "jlens+unembed" if J_l is not None else "unembed_rms"
    if J_l is None:
        log(f"J-Lens unavailable ({jmeta.get('error')}); falling back to unembed")
    else:
        log(f"J-Lens loaded shape={jmeta.get('J_l_shape')} idx={jmeta.get('used_index')}")
    leak_tokens = unembed_topk(
        model, tokenizer, v_leak, TOP_K, device, pre_map=J_l, rms_norm=True
    )
    mag_tokens = unembed_topk(
        model, tokenizer, v_mag, TOP_K, device, pre_map=J_l, rms_norm=True
    )
    payload = {
        "method": method,
        "jlens": jmeta,
        "rms_norm": True,
        "layer": LAYER,
        "top_k": TOP_K,
        "v_leak": leak_tokens,
        "v_mag": mag_tokens,
    }
    save_json(RESULTS / "jlens_or_unembed.json", payload)
    log(f"readout method={method}")
    log("v_leak top: " + ", ".join(repr(t["token"]) for t in leak_tokens[:8]))
    log("v_mag top: " + ", ".join(repr(t["token"]) for t in mag_tokens[:8]))
    return payload


def interpret(ignore_bias: float, baseline: float, cos: float, method: str) -> str:
    if abs(cos) >= 0.5:
        mag = (
            f"v_leak is substantially aligned with a no-donation high-vs-low "
            f"magnitude direction (cosine={cos:.3f}), so a large part of the "
            "prompt-contrast vector is a number-size confound"
        )
    elif abs(cos) >= 0.2:
        mag = (
            f"v_leak partially overlaps a no-donation magnitude direction "
            f"(cosine={cos:.3f}); some value-ish residual may remain"
        )
    else:
        mag = (
            f"v_leak is only weakly aligned with a no-donation magnitude "
            f"direction (cosine={cos:.3f}), so the above/below split is not "
            "just 'say a bigger number'"
        )
    return (
        f"Ignore-the-bet bias is {ignore_bias:.2f} vs baseline {baseline:.2f}; "
        f"{mag}. Token readout used {method}. Not a mediation claim."
    )


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    threshold = json.loads((RESULTS / "threshold.json").read_text())["threshold"]
    baseline = json.loads((RESULTS / "baseline_bias.json").read_text())
    bundle = torch.load(RESULTS / "direction.pt", map_location="cpu", weights_only=True)
    v_leak = bundle["v"].float()
    log(
        f"package4b instruct: threshold={threshold:g} "
        f"baseline_bias={baseline['bias']} layer={LAYER}"
    )

    log(f"loading {MODEL_ID}")
    model, tokenizer, device = load_model(MODEL_ID)
    log(f"device={device} thinking=False")

    ignore = experiment_ignore_bet(model, tokenizer, device, threshold)
    mag = experiment_magnitude(model, tokenizer, device, v_leak)
    readout = experiment_readout(model, tokenizer, device, v_leak, mag["v_mag"])

    sentence = interpret(
        ignore["bias"]["bias"],
        baseline["bias"],
        mag["cosine"],
        readout["method"],
    )
    package = {
        "model": MODEL_ID,
        "thinking": False,
        "layer": LAYER,
        "n_per_cell": N_BET,
        "threshold": threshold,
        "baseline_bias": baseline["bias"],
        "ignore_bet_bias": ignore["bias"]["bias"],
        "ignore_bet_se": ignore["bias"]["se"],
        "ignore_bet_p_above_good": ignore["bias"]["p_above_good"],
        "ignore_bet_p_below_good": ignore["bias"]["p_below_good"],
        "ignore_bet_above_estimates": ignore["above_estimates"],
        "ignore_bet_below_estimates": ignore["below_estimates"],
        "cosine_v_leak_v_mag": mag["cosine"],
        "readout_method": readout["method"],
        "v_leak_top_tokens": [t["token"] for t in readout["v_leak"]],
        "v_mag_top_tokens": [t["token"] for t in readout["v_mag"]],
        "interpretation": sentence,
        "do_not_claim_mediation": True,
    }
    save_json(RESULTS / "package4b.json", package)

    caveats = json.loads((RESULTS / "caveats.json").read_text())
    caveats["package4b"] = {
        "ignore_bet_bias": ignore["bias"]["bias"],
        "baseline_bias": baseline["bias"],
        "cosine_v_leak_v_mag": mag["cosine"],
        "readout_method": readout["method"],
        "v_leak_top_tokens": package["v_leak_top_tokens"][:10],
        "v_mag_top_tokens": package["v_mag_top_tokens"][:10],
        "interpretation": sentence,
        "do_not_claim_mediation": True,
    }
    save_json(RESULTS / "caveats.json", caveats)
    log(sentence)
    log("done")


if __name__ == "__main__":
    main()
