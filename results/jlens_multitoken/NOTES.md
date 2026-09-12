# Multi-token J-Lens: inspection only (no 27B run)

We did **not** load Qwen3.6-27B or download `templates.safetensors` / `templates+phrases_v3.safetensors`.

## Why skip

- `sysctl hw.memsize` = 34359738368 bytes = **32 GB** unified memory.
- Template-lens weights exist only for `qwen3.6-27b` (`d_model=5120`, 64 layers): 8.63 GB + 8.99 GB stacks.
- 27B activations + 9 GB templates do not fit. Another job may already be using MPS for 4B instruct n-boost.
- Oracle-lens weights are **not in the zoo**. Tuned-lens weights are **not in the zoo**. No eval JSON in the repo.

## What was actually opened

Paper: https://transformer-circuits.pub/2026/workspace/index.html

Zoo: https://huggingface.co/camilablank/workspace-lenses/tree/main
and files linked from that tree (README, per-model j-lens/r-lens, template-lens text + passages README/phrases.json).

Template folder: https://huggingface.co/camilablank/workspace-lenses/tree/main/qwen3.6-27b/template-lens

## Paper numbers we can quote without inventing figure pixels

- J-lens vs logit vs tuned: Figure 52 (normalized pass@k AUC). Text: J-lens wins every distribution; margin over logit modest on multihop/association, substantial on multilingual / order-of-operations / poetry / typo; tuned trails both and recovers almost nothing on association and poetry.
- Causal: ablating J-lens intermediates induces **roughly twice** the output KL of ablating logit- or tuned-lens directions on multihop (Figure 53). Swaps flip more often at three scales (Figure 54).
- Next-token metric the tuned lens is trained on (Figure 55): tuned is best; **J-lens is worst** (higher KL than logit through most of the network). Authors treat this as a feature.
- Template vs J-lens (Figures 62–63): 126 multi-hop items (1–4 token intermediates); 112 swap pairs (53 single-token, 59 multi-token). J-lens degrades as the intermediate gets longer; template stays roughly flat. On single-token 50-item set, they are comparable (template slightly better on readout, worse on swaps).
- Template next-word check: on-vocab next word in **top-10 only 67%** of the time (final-layer templates).
- Oracle (Haiku 4.5, not released): RL-refined oracle explains **31%** of whitened assistant-turn activation variance. No weights in the zoo.

Local 4B grid remains vs logit-lens only. Do not claim a local tuned-lens comparison.
