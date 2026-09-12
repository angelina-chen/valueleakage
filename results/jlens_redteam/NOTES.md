# J-Lens red-team notes (Qwen3.5-4B, thinking off)

Not a mediation claim. Not a claim that J-Lens is the residual's "true" contents.

## Setup

- Model: `Qwen/Qwen3.5-4B`, layer 23, MPS, thinking off, prefills only (~29s wall).
- J-Lens: `/Users/achen/neelnanda/artifacts/jlens/qwen3.5-4b/j-lens/lens.pt` — tuned-lens-class: learned d×d linear map then RMSNorm+unembed; no bias term in lens.pt (linear, not affine). J is an averaged Jacobian to the penultimate layer, not a trained tuned-lens fit.
- Layer 23: cos(J, I)=0.858, diag mean=0.991; layer 30 is exactly I (target).
- Logit-lens: RMSNorm then `lm_head`.
- Tuned lens: not trained.
- Template lens: skipped: camilablank/workspace-lenses has template-lens only for qwen3.6-27b (phrase directions, d_model=5120, 64 layers), not a drop-in Qwen3.5-4B weight+forward
- Empty string: tokenizer produced 0 tokens even with specials; 1-token bos/pad fallback (`token_id=248044`).

## On-distribution (factual last-token residuals)

- `france_capital` overlap@10=6/10 logit_cos=0.240
  - J-Lens: ' ___', '____', ' __', ' Paris', '________', '_____'
  - logit-lens: ' _______,', ' ___', 'Paris', '________', ' __________________', '____'
- `largest_planet` overlap@10=0/10 logit_cos=0.196
  - J-Lens: ' __', '____', '_____', '____________', '________', ' Jupiter'
  - logit-lens: ' _______,', '太阳系', ' sanks', ' uranus', ' Inizi', 'ยักษ์'
- `two_plus_two` overlap@10=3/10 logit_cos=0.130
  - J-Lens: ' ?', '?', '？', '?\\', '=?', ')?'
  - logit-lens: ' ?**', ' ?",', " ?',", '?\\', 'uttur', ' ؟'
- `gold_symbol` overlap@10=2/10 logit_cos=0.379
  - J-Lens: ' gold', ' ___', '________', '____', 'gold', ' __'
  - logit-lens: '.gold', 'gold', '金', '金は', '金的', ' gold'
- `first_president` overlap@10=2/10 logit_cos=0.224
  - J-Lens: ' ___', ' ____', ' __', '_____', '____', ' __________________'
  - logit-lens: ' _______,', '…]', ' ..........', ' __________________', 'born', ' ____'

## Off-distribution

- `empty` overlap@10=0/10 logit_cos=0.700
  - J-Lens: '_into', '想法', '政权', '低价', '话题', '赞同'
  - logit-lens: ' sín', '精英', '行之', ' Journalism', 'ostr', 'ENTIC'
- `hello` overlap@10=1/10 logit_cos=0.642
  - J-Lens: ' I', 'iv', 'uz', 'rit', '^{', '^'
  - logit-lens: 'uz', '义', 'zn', 'ounds', ' Aún', ' bisschen'
- `period` overlap@10=3/10 logit_cos=0.811
  - J-Lens: ' ', '   ', '     ', ' |', '+', '4'
  - logit-lens: 'uz', '4', '_handler', 'zn', '5', '开'
- `gaussian_matched_rms` overlap@10=2/10 logit_cos=0.852
  - J-Lens: '较好地', '尼迪', '-scal', ' Closed', ' Swe', ' Scal'
  - logit-lens: '较好地', 'hn', '无际', 'fu', 'eneg', ' denomin'
- `random_unit_scaled` overlap@10=2/10 logit_cos=0.799
  - J-Lens: 'grunt', 'inand', ' bindings', ' standalone', '呱', 'enga'
  - logit-lens: '纳', '夷', ' continued', '照', 'ze', 'grunt'
- `v_leak` overlap@10=1/10 logit_cos=0.869
  - J-Lens: '庞大', '亿元以上', '万以上', ' huge', ' HUGE', '庞大的'
  - logit-lens: '深水', '儿', 'nü', ' >(', 'enang', '龐'

## Claim

J-Lens on Qwen3.5-4B layer 23 is a 2560×2560 linear Jacobian with no bias (cos(J,I)=0.858; layer 30 is exactly I), i.e. tuned-lens-class: learned map then RMSNorm+unembed, not a trained tuned lens and not a nonlinear decoder. On-distribution, mean overlap@10 vs logit-lens is 2.6/10 and mean logit-cosine is 0.234: they are not the same ranking. J-Lens is the more clustered next-token map (France: Paris among blanks, overlap 6/10; planet: Jupiter/planets among blanks; gold: gold/Gold/缩写), while logit-lens is broader/noisier (capitale; 太阳系/uranus/Pluto; 金). Neither reads '4' or 'Washington'. Off-distribution mean overlap@10 is 1.5/10; v_leak logit-cosine=0.869 but top-10 overlap only 1/10 — J-Lens 庞大/huge/enormous vs logit-lens junk (深水/儿/nü). Empty/Hello. still yield content words (想法/政权/低价; I/Please). So J-Lens is a denoised next-token map: more fluent than logit-lens, more willing to confabulate off-distribution, not a faithful concept detector.
