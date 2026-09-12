# J-Lens paper-style demo (Qwen3.5-4B only)

Thinking off. Prefill + short gen. No 27B, no Claude, no DeepSeek, no tuned-lens training, no giraffe n-boost, no filler.

## Setup

- Model: `Qwen/Qwen3.5-4B`, device `mps`, layer 23 (+ target layer 30 if present).
- J-Lens: `/Users/achen/neelnanda/artifacts/jlens/qwen3.5-4b/j-lens/lens.pt` — linear Jacobian then RMSNorm+unembed. Layer 30 is I.
- Reused: `src/jlens_eval.py`, `results/direction.pt` (loaded, not used as the country vector), layer-23 `lens.pt`.
- SWAP directions: unembed row then `J @ e_token`. Steer via `model_io` project_out / combined add.

## READ

Did J-Lens surface the intermediate when logit-lens did not? **True**
(J hits 3/5; logit-lens 2/5; J-only 1/5.)

- `mars_then_red` (`Mars is often called the`): J-Lens intermediate=['mars'] logit-lens=—; J-only=True. Said `\"Red Planet\" because of the presence`.
- `red_then_mars` (`The planet known for being the red planet is`): J-Lens intermediate=— logit-lens=—; J-only=False. Said `A. Mars
B. Mars`.
- `twentyone_then_42` (`21 + 21 =`): J-Lens intermediate=— logit-lens=—; J-only=False. Said `?

21 + 21`.
- `france_then_paris` (`The capital of France is`): J-Lens intermediate=['paris'] logit-lens=['paris']; J-only=False. Said `Paris.

<think>
Thinking Process:`.
- `gold_then_symbol` (`The chemical symbol for gold is`): J-Lens intermediate=['gold'] logit-lens=['gold']; J-only=False. Said `Au. What is the name of the`.

## SWAP

Ran. Baseline=`Paris.
The capital of France is Paris.
The capital of France is Paris.
The capital of France is Paris.
The capital of France is`; ablate=`Paris.
The capital of France is Paris.
The capital of France is Paris.
The capital of France is Paris.
The capital of France is`; swap-to-Germany=`Paris.
The capital of Germany is Berlin.
The capital of Italy is Rome.
The capital of Spain is Madrid.
The capital of the United`. Not a mediation claim.

## AUDIT (expected negative)

At least one audit prefix had a needle in a top-10; see items.

## Submit narrative vs paper-citation-only

**From this demo (4B, this machine):** J-Lens vs logit-lens top-10 on five short prefixes; whether Mars/42/France–Paris showed up; one qualitative France ablate/swap; audit table with an honest negative if the secret-sauce words are absent. Do not claim mediation or that J-Lens is a concept detector.

**Paper-citation-only (transformer-circuits.pub/2026/workspace; not reproduced here):** Fig 52 pass@k AUC (J-lens wins every distribution); Fig 53 ~2× KL from ablating J-lens intermediates; Fig 54 swap flip rates; Fig 55 next-token KL (J-lens worst); Figs 62–63 template vs J-lens multi-token; oracle 31% variance; any 27B / Haiku / tuned-lens comparison.
