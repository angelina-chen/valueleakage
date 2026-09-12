"""Load a HF chat model, generate, cache last-token residuals, steer one layer."""

from __future__ import annotations

from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.device import pick_device, pick_dtype


def load_model(model_id: str, device: torch.device | None = None):
    device = device or pick_device()
    dtype = pick_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if device.type == "cuda":
        kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


def apply_chat(tokenizer, user_text: str, enable_thinking: bool = False) -> str:
    messages = [{"role": "user", "content": user_text}]
    base = {"tokenize": False, "add_generation_prompt": True}
    for extra in (
        {"enable_thinking": enable_thinking},
        {"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        {},
    ):
        try:
            return tokenizer.apply_chat_template(messages, **base, **extra)
        except TypeError:
            continue
    return tokenizer.apply_chat_template(messages, **base)


def _text_layer_list(model) -> torch.nn.ModuleList:
    cfg = getattr(model.config, "text_config", model.config)
    n = int(getattr(cfg, "num_hidden_layers"))
    scored: list[tuple[int, str, torch.nn.ModuleList]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n:
            score = name.count("layer") + name.count("language")
            scored.append((score, name, mod))
    if not scored:
        raise RuntimeError("Could not find decoder ModuleList")
    scored.sort(reverse=True)
    return scored[0][2]


def residual_tensor(output) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def wrap_residual(output, new_hidden):
    if isinstance(output, tuple):
        return (new_hidden,) + output[1:]
    return new_hidden


@torch.no_grad()
def last_token_residual(
    model,
    tokenizer,
    prompt: str,
    layer: int,
    device: torch.device,
) -> torch.Tensor:
    """Prefill residual at `layer`, last prompt token. Shape [hidden]."""
    layers = _text_layer_list(model)
    captured: dict[str, torch.Tensor] = {}

    def hook(_mod, _inp, output):
        hidden = residual_tensor(output)
        captured["h"] = hidden[:, -1, :].detach().float().cpu()
        return output

    handle = layers[layer].register_forward_hook(hook)
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model(**inputs, use_cache=False)
    finally:
        handle.remove()
    if "h" not in captured:
        raise RuntimeError(f"No residual captured at layer {layer}")
    return captured["h"].squeeze(0)


def make_steer_hook(
    direction: torch.Tensor,
    mode: str,
    strength: float,
) -> Callable:
    v = direction / (direction.norm() + 1e-8)

    def hook(_mod, _inp, output):
        hidden = residual_tensor(output)
        v_cast = v.to(device=hidden.device, dtype=hidden.dtype)
        if mode == "add":
            steered = hidden + strength * v_cast
        elif mode == "project_out":
            proj = (hidden * v_cast).sum(dim=-1, keepdim=True)
            steered = hidden - strength * proj * v_cast
        else:
            raise ValueError(mode)
        return wrap_residual(output, steered)

    return hook


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    *,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    layer: int | None = None,
    direction: torch.Tensor | None = None,
    steer_mode: str | None = None,
    strength: float = 1.0,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    handle = None
    if direction is not None and layer is not None and steer_mode is not None:
        layers = _text_layer_list(model)
        handle = layers[layer].register_forward_hook(
            make_steer_hook(direction, steer_mode, strength)
        )
    try:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
        )
    finally:
        if handle is not None:
            handle.remove()
    prompt_len = inputs["input_ids"].shape[1]
    # Keep <think> tags; they are special tokens on Qwen3.5.
    text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=False)
    for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        text = text.replace(tok, "")
    return text.strip()
