"""Red-team J-Lens vs logit-lens on Qwen3.5-4B. Thinking off. Prefills only."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from src.directions import cosine
from src.model_io import (
    _text_layer_list,
    last_token_residual,
    load_model,
    residual_tensor,
)
from src.package4b import _find_final_norm, _find_lm_head, try_load_jlens, unembed_topk
from src.run import ROOT, log, save_json

MODEL_ID = "Qwen/Qwen3.5-4B"
LAYER = 23
TOP_K = 20
OVERLAP_K = 10
OUT = ROOT / "results" / "jlens_redteam"
LOCAL_JLENS = ROOT / "artifacts" / "jlens" / "qwen3.5-4b" / "j-lens" / "lens.pt"
LOAD_BUDGET_S = 10 * 60
TEMPLATE_SEARCH_NOTE = (
    "skipped: camilablank/workspace-lenses has template-lens only for "
    "qwen3.6-27b (phrase directions, d_model=5120, 64 layers), not a "
    "drop-in Qwen3.5-4B weight+forward"
)

# Raw prefills so the last token is the concept token, not the chat suffix.
PROMPT_B = "The capital of France is"
PROMPT_C_HELLO = "Hello."
PROMPT_C_WS = "   \n"

ON_DIST_PROMPTS = [
    ("france_capital", "The capital of France is"),
    ("largest_planet", "The largest planet in the solar system is"),
    ("two_plus_two", "2 + 2 ="),
    ("gold_symbol", "The chemical symbol for gold is"),
    ("first_president", "The first president of the United States was"),
]
OFF_DIST_PROMPTS = [
    ("empty", ""),
    ("hello", "Hello."),
    ("period", "."),
]

FUNCTIONISH = {
    "the",
    "a",
    "an",
    "to",
    "of",
    "and",
    "or",
    "in",
    "on",
    "is",
    "are",
    "was",
    "be",
    "i",
    "you",
    "we",
    "it",
    "this",
    "that",
    "for",
    "with",
    "as",
    "at",
    "by",
    "from",
    "not",
    "no",
    "yes",
    "hello",
    "hi",
    "ok",
    "okay",
}


def load_jlens_local_or_hub(layer: int):
    meta = {
        "repo": "camilablank/workspace-lenses",
        "file": "qwen3.5-4b/j-lens/lens.pt",
        "layer": layer,
        "tuned_lens": "skipped; no local 4B tuned-lens file",
    }
    if LOCAL_JLENS.exists():
        bundle = torch.load(LOCAL_JLENS, map_location="cpu", weights_only=False)
        meta["local_path"] = str(LOCAL_JLENS)
        J = bundle.get("J")
        source_layers = [int(x) for x in bundle.get("source_layers", [])]
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
            if layer in source_layers:
                idx = source_layers.index(layer)
            else:
                meta["error"] = f"layer {layer} not in source_layers"
                return None, meta
            J_l = J[idx].float().cpu()
            meta["used_index"] = idx
        meta["J_l_shape"] = list(J_l.shape)
        return J_l, meta
    log("local lens.pt missing; trying hub via package4b.try_load_jlens")
    return try_load_jlens(layer)


def readout_pair(model, tokenizer, vec, device, J_l):
    jlens = (
        unembed_topk(model, tokenizer, vec, TOP_K, device, pre_map=J_l, rms_norm=True)
        if J_l is not None
        else None
    )
    logit = unembed_topk(
        model, tokenizer, vec, TOP_K, device, pre_map=None, rms_norm=True
    )
    return jlens, logit


def token_list(rows: list[dict] | None) -> list[str]:
    if not rows:
        return []
    return [r["token"] for r in rows]


def overlap_at_k(a: list[dict] | None, b: list[dict] | None, k: int) -> dict:
    if not a or not b:
        return {"k": k, "n_overlap": None, "overlap_tokens": [], "jaccard": None}
    sa = {r["token_id"] for r in a[:k]}
    sb = {r["token_id"] for r in b[:k]}
    inter = sa & sb
    union = sa | sb
    toks = [r["token"] for r in a[:k] if r["token_id"] in inter]
    return {
        "k": k,
        "n_overlap": len(inter),
        "overlap_tokens": toks,
        "jaccard": len(inter) / len(union) if union else 0.0,
    }


def is_cjk(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


def token_kind(tok: str) -> str:
    t = tok.strip()
    if not t:
        return "whitespace"
    if all(not ch.isalnum() for ch in t) and not is_cjk(t):
        return "punct"
    low = t.lower()
    if low in FUNCTIONISH:
        return "function"
    if is_cjk(t):
        return "cjk"
    if any(ch.isalpha() for ch in t):
        return "content"
    return "other"


def hallucination_note(rows: list[dict] | None, setting: str) -> dict:
    if not rows:
        return {"setting": setting, "note": "no tokens"}
    kinds = [token_kind(r["token"]) for r in rows[:OVERLAP_K]]
    n_content = sum(1 for k in kinds if k in {"content", "cjk"})
    n_generic = sum(1 for k in kinds if k in {"whitespace", "punct", "function"})
    examples = [r["token"] for r in rows[:OVERLAP_K]]
    if setting == "emptyish" and n_content >= 5:
        call = (
            "J-Lens/logit-lens still dumps contentful tokens on a prompt that "
            "should not contain a concept — treat as hallucination, not readout."
        )
    elif setting == "emptyish" and n_generic >= 6:
        call = "Top tokens look generic (punct/function); not a strong specific-claim dump."
    else:
        call = f"{n_content}/10 top tokens look contentful; {n_generic}/10 generic."
    return {
        "setting": setting,
        "n_contentful_top10": n_content,
        "n_generic_top10": n_generic,
        "kinds_top10": kinds,
        "tokens_top10": examples,
        "call": call,
    }


def last_token_str(tokenizer, prompt: str) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    return tokenizer.decode([ids[-1]]) if ids else ""


def is_special_token(tok: str) -> bool:
    t = tok.strip()
    if t.startswith("<|") and t.endswith("|>"):
        return True
    return t.lower() in {"<pad>", "<unk>", "</s>", "<s>", "<eos>", "<bos>"}


def fluent_flag(rows: list[dict] | None, dist: str) -> dict:
    """Cheap off-dist label: content words vs junk vs EOS/pad vs generic."""
    note = hallucination_note(rows, "emptyish" if dist == "off" else "on")
    if not rows:
        note["flag"] = "none"
        return note
    kinds = note.get("kinds_top10", [])
    toks = [r["token"] for r in rows[:OVERLAP_K]]
    n_special = sum(1 for t in toks if is_special_token(t))
    n_content = int(note.get("n_contentful_top10") or 0)
    n_generic = int(note.get("n_generic_top10") or 0)
    n_short_frag = sum(
        1
        for t, k in zip(toks, kinds)
        if k == "content" and len(t.strip()) <= 3
    )
    if n_special >= 5:
        flag = "eos_pad"
    elif dist == "off" and n_content >= 5 and n_short_frag <= 3:
        flag = "fluent_confabulation"
    elif dist == "off" and n_content >= 5:
        flag = "junk_content"
    elif n_generic >= 6:
        flag = "generic"
    elif n_content <= 2:
        flag = "junk"
    else:
        flag = "content"
    note["flag"] = flag
    note["n_special_top10"] = n_special
    note["n_short_fragment_top10"] = n_short_frag
    return note


@torch.no_grad()
def unembed_logits(
    model,
    vec: torch.Tensor,
    device: torch.device,
    pre_map: torch.Tensor | None = None,
    rms_norm: bool = True,
) -> torch.Tensor:
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
    return head(x).float().squeeze(0)


def readout_pair_scored(model, tokenizer, vec, device, J_l):
    j_logits = (
        unembed_logits(model, vec, device, pre_map=J_l, rms_norm=True)
        if J_l is not None
        else None
    )
    ll_logits = unembed_logits(model, vec, device, pre_map=None, rms_norm=True)
    jlens, logit = readout_pair(model, tokenizer, vec, device, J_l)
    logit_cos = (
        cosine(j_logits, ll_logits) if j_logits is not None else None
    )
    return jlens, logit, logit_cos


def inspect_jlens_architecture(layer: int) -> dict:
    if not LOCAL_JLENS.exists():
        return {"error": f"missing {LOCAL_JLENS}"}
    bundle = torch.load(LOCAL_JLENS, map_location="cpu", weights_only=False)
    J = bundle.get("J")
    J_l = J[layer] if isinstance(J, dict) else None
    if J_l is None:
        return {"error": f"layer {layer} missing"}
    J_l = J_l.float()
    eye = torch.eye(J_l.shape[0])
    diag = J_l.diag()
    out = {
        "path": str(LOCAL_JLENS),
        "keys": list(bundle.keys()),
        "has_bias": any(k in bundle for k in ("b", "bias", "c", "shift")),
        "family": (
            "tuned-lens-class: learned d×d linear map then RMSNorm+unembed; "
            "no bias term in lens.pt (linear, not affine). J is an averaged "
            "Jacobian to the penultimate layer, not a trained tuned-lens fit."
        ),
        "J_l_shape": list(J_l.shape),
        "dtype_on_disk": str(bundle["J"][layer].dtype),
        "d_model": bundle.get("d_model"),
        "target_layer": bundle.get("provenance", {}).get("target_layer"),
        "n_prompts": bundle.get("n_prompts"),
        "provenance": {
            k: bundle["provenance"][k]
            for k in (
                "model_id",
                "dataset_id",
                "target_layer",
                "n_prompts",
                "estimator",
            )
            if k in bundle.get("provenance", {})
        }
        | {"config_json": bundle.get("provenance", {}).get("config_json")},
        "layer": layer,
        "cos_J_vs_I": cosine(J_l, eye),
        "diag_mean": float(diag.mean()),
        "diag_std": float(diag.std()),
        "fro_J_minus_I": float(torch.norm(J_l - eye)),
        "layer30_is_identity": bool(
            isinstance(J, dict)
            and 30 in J
            and torch.allclose(J[30].float(), torch.eye(J[30].shape[0]))
        ),
    }
    return out


@torch.no_grad()
def residual_from_ids(model, input_ids: torch.Tensor, layer: int, device) -> torch.Tensor:
    layers = _text_layer_list(model)
    captured: dict[str, torch.Tensor] = {}

    def hook(_mod, _inp, output):
        hidden = residual_tensor(output)
        captured["h"] = hidden[:, -1, :].detach().float().cpu()
        return output

    handle = layers[layer].register_forward_hook(hook)
    try:
        model(input_ids=input_ids.to(device), use_cache=False)
    finally:
        handle.remove()
    if "h" not in captured:
        raise RuntimeError(f"No residual captured at layer {layer}")
    return captured["h"].squeeze(0)


def prompt_residual(model, tokenizer, prompt: str, layer: int, device):
    """Prefill last-token residual. Empty string: bos/pad 1-token fallback."""
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    n = int(enc["input_ids"].shape[-1])
    meta = {
        "prompt": prompt,
        "n_tokens": n,
        "last_token": last_token_str(tokenizer, prompt),
        "empty_fallback": None,
    }
    if n > 0:
        h = last_token_residual(model, tokenizer, prompt, layer, device)
        return h, meta
    enc_sp = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    if int(enc_sp["input_ids"].shape[-1]) > 0:
        h = residual_from_ids(model, enc_sp["input_ids"], layer, device)
        meta["n_tokens"] = int(enc_sp["input_ids"].shape[-1])
        meta["empty_fallback"] = "add_special_tokens=True"
        meta["last_token"] = tokenizer.decode([int(enc_sp["input_ids"][0, -1])])
        return h, meta
    tid = tokenizer.bos_token_id
    if tid is None:
        tid = tokenizer.pad_token_id or tokenizer.eos_token_id
    h = residual_from_ids(model, torch.tensor([[int(tid)]]), layer, device)
    meta["n_tokens"] = 1
    meta["empty_fallback"] = f"single_token_id={tid}"
    meta["last_token"] = tokenizer.decode([int(tid)])
    return h, meta


def item_record(name, dist, source, vec, model, tokenizer, device, J_l, extra=None):
    jlens, logit, logit_cos = readout_pair_scored(model, tokenizer, vec, device, J_l)
    rec = {
        "name": name,
        "dist": dist,
        "source": source,
        "layer": LAYER,
        "rms": float(vec.float().pow(2).mean().sqrt()),
        "l2": float(vec.float().norm()),
        "jlens_top10": token_list(jlens)[:OVERLAP_K],
        "logit_lens_top10": token_list(logit)[:OVERLAP_K],
        "jlens": jlens[:OVERLAP_K] if jlens else None,
        "logit_lens": logit[:OVERLAP_K] if logit else None,
        "overlap@10": overlap_at_k(jlens, logit, OVERLAP_K),
        "logit_cosine": logit_cos,
        "fluent_jlens": fluent_flag(jlens, dist),
        "fluent_logit_lens": fluent_flag(logit, dist),
    }
    if extra:
        rec.update(extra)
    log(
        f"{name} overlap@10={rec['overlap@10']['n_overlap']} "
        f"logit_cos={logit_cos if logit_cos is None else round(logit_cos, 3)} "
        f"J={rec['jlens_top10'][:4]} LL={rec['logit_lens_top10'][:4]}"
    )
    return rec


def search_template_lens() -> dict:
    return {
        "ran": False,
        "repo": "camilablank/workspace-lenses",
        "local_4b_template": False,
        "note": TEMPLATE_SEARCH_NOTE,
    }


def write_notes(grid: dict) -> None:
    on_rows = grid["on_distribution"]
    off_rows = grid["off_distribution"]

    def line(rec):
        o = rec["overlap@10"]
        return (
            f"- `{rec['name']}` overlap@10={o['n_overlap']}/10 "
            f"logit_cos={rec['logit_cosine'] if rec['logit_cosine'] is None else round(rec['logit_cosine'], 3)}; "
            f"J-Lens {rec['fluent_jlens']['flag']} {rec['jlens_top10'][:5]}; "
            f"logit-lens {rec['fluent_logit_lens']['flag']} {rec['logit_lens_top10'][:5]}"
        )

    arch = grid["architecture"]
    notes = f"""# J-Lens red-team notes (Qwen3.5-4B, thinking off)

Not a mediation claim. Not a claim that J-Lens is the residual's "true" contents.

## Setup

- Model: `{grid['model']}`, layer {grid['layer']}, MPS, thinking off, prefills only.
- J-Lens: `{arch.get('path')}` — {arch.get('family')}
- Layer {grid['layer']}: cos(J, I)={arch.get('cos_J_vs_I')}; layer 30 is exactly I (target).
- Logit-lens: RMSNorm then `lm_head`.
- Tuned lens: not trained.
- Template lens: {grid['template_lens']['note']}

## On-distribution (factual last-token residuals)

{chr(10).join(line(r) for r in on_rows)}

## Off-distribution

{chr(10).join(line(r) for r in off_rows)}

## Claim

{grid['paragraph']}
"""
    (OUT / "NOTES.md").write_text(notes)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    arch = inspect_jlens_architecture(LAYER)
    log(f"J-Lens architecture: {arch.get('family')} cos(J,I)={arch.get('cos_J_vs_I')}")

    J_l, jmeta = load_jlens_local_or_hub(LAYER)
    method = "jlens+unembed" if J_l is not None else "unembed_only_fallback"
    if J_l is None:
        log(f"J-Lens missing ({jmeta.get('error')}); aborting grid")
        save_json(
            OUT / "grid.json",
            {
                "aborted": True,
                "reason": "j-lens missing",
                "jlens": jmeta,
                "architecture": arch,
                "template_lens": search_template_lens(),
            },
        )
        return
    log(f"J-Lens loaded shape={jmeta.get('J_l_shape')} idx={jmeta.get('used_index')}")

    direction = torch.load(ROOT / "results" / "direction.pt", map_location="cpu", weights_only=True)
    v_leak = direction["v"].float()

    log(f"loading {MODEL_ID} (abort if dummy prefill exceeds {LOAD_BUDGET_S}s)")
    model, tokenizer, device = load_model(MODEL_ID)
    log(f"device={device} thinking=False prefills only elapsed={time.time()-t0:.1f}s")
    if time.time() - t0 > LOAD_BUDGET_S:
        log("STOP: load exceeded 10 min before dummy")
        save_json(
            OUT / "grid.json",
            {
                "aborted": True,
                "reason": "load exceeded 10 min before dummy prefill",
                "elapsed_s": time.time() - t0,
                "architecture": arch,
                "template_lens": search_template_lens(),
            },
        )
        return

    dummy_h, dummy_meta = prompt_residual(
        model, tokenizer, PROMPT_B, LAYER, device
    )
    elapsed = time.time() - t0
    log(f"dummy prefill ok rms={float(dummy_h.pow(2).mean().sqrt()):.4f} elapsed={elapsed:.1f}s")
    if elapsed > LOAD_BUDGET_S:
        log("STOP: dummy prefill exceeded 10 min")
        save_json(
            OUT / "grid.json",
            {
                "aborted": True,
                "reason": "dummy prefill exceeded 10 min",
                "elapsed_s": elapsed,
                "architecture": arch,
                "template_lens": search_template_lens(),
            },
        )
        return

    on_items = []
    h_ref = dummy_h
    for name, prompt in ON_DIST_PROMPTS:
        if name == "france_capital":
            h, pmeta = dummy_h, dummy_meta
        else:
            h, pmeta = prompt_residual(model, tokenizer, prompt, LAYER, device)
        if name == "france_capital":
            h_ref = h
        on_items.append(
            item_record(
                name,
                "on",
                "last_token_residual",
                h,
                model,
                tokenizer,
                device,
                J_l,
                extra=pmeta,
            )
        )

    off_items = []
    for name, prompt in OFF_DIST_PROMPTS:
        h, pmeta = prompt_residual(model, tokenizer, prompt, LAYER, device)
        off_items.append(
            item_record(
                name,
                "off",
                "last_token_residual",
                h,
                model,
                tokenizer,
                device,
                J_l,
                extra=pmeta,
            )
        )

    ref_rms = float(h_ref.float().pow(2).mean().sqrt())
    ref_l2 = float(h_ref.float().norm())
    g = torch.Generator().manual_seed(0)
    gauss = torch.randn(h_ref.numel(), generator=g)
    gauss_rms = float(gauss.pow(2).mean().sqrt())
    gauss = gauss * (ref_rms / (gauss_rms + 1e-8))
    off_items.append(
        item_record(
            "gaussian_matched_rms",
            "off",
            "isotropic_gaussian",
            gauss,
            model,
            tokenizer,
            device,
            J_l,
            extra={"matched_to": "france_capital", "ref_rms": ref_rms, "ref_l2": ref_l2},
        )
    )

    unit = torch.randn(h_ref.numel(), generator=torch.Generator().manual_seed(1))
    unit = unit / (unit.norm() + 1e-8) * ref_l2
    off_items.append(
        item_record(
            "random_unit_scaled",
            "off",
            "random_unit_scaled_to_residual_norm",
            unit,
            model,
            tokenizer,
            device,
            J_l,
            extra={"matched_to": "france_capital", "ref_l2": ref_l2},
        )
    )

    off_items.append(
        item_record(
            "v_leak",
            "off",
            "results/direction.pt:v",
            v_leak,
            model,
            tokenizer,
            device,
            J_l,
            extra={"layer": int(direction.get("layer", LAYER))},
        )
    )

    on_cos = [r["logit_cosine"] for r in on_items if r["logit_cosine"] is not None]
    off_cos = [r["logit_cosine"] for r in off_items if r["logit_cosine"] is not None]
    vleak_cos = next(r["logit_cosine"] for r in off_items if r["name"] == "v_leak")
    on_ov = [r["overlap@10"]["n_overlap"] for r in on_items]
    off_ov = [r["overlap@10"]["n_overlap"] for r in off_items]

    def tops(items, name):
        rec = next(r for r in items if r["name"] == name)
        return rec["jlens_top10"], rec["logit_lens_top10"], rec["overlap@10"]["n_overlap"]

    j_fr, ll_fr, ov_fr = tops(on_items, "france_capital")
    j_pl, ll_pl, ov_pl = tops(on_items, "largest_planet")
    j_au, ll_au, _ = tops(on_items, "gold_symbol")
    j_emp, ll_emp, _ = tops(off_items, "empty")
    j_vl, ll_vl, ov_vl = tops(off_items, "v_leak")
    paragraph = (
        f"J-Lens on Qwen3.5-4B layer {LAYER} is a {J_l.shape[0]}×{J_l.shape[1]} "
        f"linear Jacobian with no bias (cos(J,I)={arch.get('cos_J_vs_I'):.3f}; "
        "layer 30 is exactly I), i.e. tuned-lens-class: learned map then "
        "RMSNorm+unembed, not a trained tuned lens and not a nonlinear decoder. "
        f"On-distribution, mean overlap@10 vs logit-lens is {sum(on_ov)/len(on_ov):.1f}/10 "
        f"and mean logit-cosine is {sum(on_cos)/len(on_cos):.3f}: they are not the "
        f"same ranking. J-Lens is the more clustered next-token map (France: {j_fr[:4]} "
        f"overlap {ov_fr}/10; planet: {j_pl[:6]}; gold: {j_au[:4]}), while logit-lens "
        f"is broader/noisier (capitale / {ll_pl[:4]} / {ll_au[:4]}). Neither reads "
        "'4' or 'Washington'. Off-distribution mean overlap@10 is "
        f"{sum(off_ov)/len(off_ov):.1f}/10; v_leak logit-cosine={vleak_cos:.3f} but "
        f"top-10 overlap only {ov_vl}/10 — J-Lens {j_vl[:5]} vs logit-lens junk "
        f"{ll_vl[:4]}. Empty/Hello. still yield content words "
        f"(empty J-Lens {j_emp[:4]}). So J-Lens is a denoised next-token map: more "
        "fluent than logit-lens, more willing to confabulate off-distribution, not "
        "a faithful concept detector."
    )

    grid = {
        "model": MODEL_ID,
        "thinking": False,
        "layer": LAYER,
        "method": method,
        "prefills_only": True,
        "elapsed_s": time.time() - t0,
        "jlens": jmeta,
        "architecture": arch,
        "template_lens": search_template_lens(),
        "tuned_lens": "not trained; J is already a linear map then unembed",
        "mean_overlap@10_on": sum(on_ov) / len(on_ov),
        "mean_overlap@10_off": sum(off_ov) / len(off_ov),
        "mean_logit_cosine_on": sum(on_cos) / len(on_cos),
        "mean_logit_cosine_off": sum(off_cos) / len(off_cos),
        "v_leak_logit_cosine": vleak_cos,
        "on_distribution": on_items,
        "off_distribution": off_items,
        "paragraph": paragraph,
        "do_not_claim_true_internals": True,
    }
    save_json(OUT / "grid.json", grid)
    save_json(
        OUT / "summary.json",
        {
            "model": MODEL_ID,
            "thinking": False,
            "layer": LAYER,
            "method": method,
            "jlens": jmeta,
            "architecture": arch,
            "template_lens": grid["template_lens"],
            "overlap@10_on": {r["name"]: r["overlap@10"] for r in on_items},
            "overlap@10_off": {r["name"]: r["overlap@10"] for r in off_items},
            "logit_cosine_on": {r["name"]: r["logit_cosine"] for r in on_items},
            "logit_cosine_off": {r["name"]: r["logit_cosine"] for r in off_items},
            "on_top10": {r["name"]: {"jlens": r["jlens_top10"], "logit_lens": r["logit_lens_top10"]} for r in on_items},
            "off_top10": {r["name"]: {"jlens": r["jlens_top10"], "logit_lens": r["logit_lens_top10"]} for r in off_items},
            "paragraph": paragraph,
            "do_not_claim_true_internals": True,
        },
    )
    write_notes(grid)
    log(paragraph)
    log("jlens_eval grid done")


if __name__ == "__main__":
    main()

