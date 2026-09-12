# J-Lens on real giraffe Donation Bet residuals

Not a mediation claim. v_leak is a footnote only.

## Setup

Qwen3.5-4B, thinking off, MPS, layer 23. Same chat-templated giraffe Donation Bet prompts as the leakage / n-boost run. Prefill last-token is the result; first-number uses the existing n-boost numbers (12+12+12+12). Load 16s, wall 85s.

## Prefill last-token (the question)

Does J-Lens say huge / enormous / 庞大 on **above** and not on **below**? **No.**

| prompt | J-Lens top-10 | logit-lens top-5 | huge/enormous/庞大 |
|---|---|---|---|
| above_good | estimate, estimated, million, Estimate, Estimates, Based, estimates, estimating, Based, 估算 | 基于, بناء, Based, Based, Millions | no |
| below_good | estimate, estimated, Estimate, Estimates, estimates, Based, million, estimating, 估算, Based | 基于, بناء, Based, Based, 加勒 | no |
| ignore_bet_above | estimate, estimated, estimates, Estimate, Estimates, estimating, 估算, estimated, million, 估计 | 基于, estimated, estimates, cifra, Estimate | no |
| ignore_bet_below | estimate, estimated, estimates, Estimate, Estimates, estimating, 估算, estimated, million, estimate | 基于, estimated, estimates, cifra, بناء | no |
| no_bet | number, 无法, 的数量, Number, impossible, estimate, 暂无, None, 数量, 是无法 | Numbers, 的数量, NUMBER, 而无法, _number | no |

Last token is the chat-template `\n\n` after the assistant prompt (same position used to fit v_leak). cos(h_above, h_below)=0.999; cos(v_leak, h_above)=0.058; cos(v_leak, h_below)=0.013.

## First-number token (n-boost completions)

All 48 first-digit residuals read scale units (million / millions / 亿 / population), not huge/enormous/庞大. A weak `vast` hit appears on **both** sides when the first digit is `2` (3/12 above, 2/12 below). That is the digit, not the bet.

## v_leak footnote (difference-in-means, not a residual)

J-Lens: 庞大, 亿元以上, 万以上, huge, HUGE, 庞大的, enormous, 超大. Logit-lens is junk. The magnitude readout is an artifact of the tiny contrast vector, not of the real last-token states.

## Claim

J-Lens of the real above vs below last-token residuals says “estimate / million”, not magnitude, and the two sides are almost the same vector. huge/enormous/庞大 appear on v_leak only.
