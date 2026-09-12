"""Three-panel brake figure: leakage n-boost + J-Lens tokens + overlap@10."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Patch

from src.run import RESULTS

FIGDIR = RESULTS / "figures"
GRID = RESULTS / "jlens_redteam" / "grid.json"
SUMMARY_GRID = RESULTS / "jlens_redteam" / "summary.json"
BOOST = RESULTS / "instruct_n_boost"
GIRAFFE_PREFILL = RESULTS / "jlens_giraffe" / "prefill.json"
GIRAFFE_SUM = RESULTS / "jlens_giraffe" / "summary.json"
LEGACY_BASE = RESULTS / "baseline_bias.json"
LEGACY_IGN = RESULTS / "ignore_bet_bias.json"


def _cjk_font() -> str | None:
    wanted = (
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "Songti SC",
        "STHeiti",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
    )
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in wanted:
        if name in available:
            return name
    return None


def _pretty_token(tok: str) -> str:
    shown = tok.replace("\n", "\\n").replace("\t", "\\t")
    if shown == " ":
        return "␣"
    if shown.strip() == "" and shown:
        return f"␣×{len(shown)}"
    if len(shown) > 14:
        return shown[:13] + "…"
    return shown


def _logs(vals: list) -> list[float]:
    return [math.log10(v) for v in vals if isinstance(v, (int, float)) and v > 0]


def _load_bias_block() -> dict:
    boost_sum = BOOST / "summary.json"
    if boost_sum.exists():
        s = json.loads(boost_sum.read_text())
        if s.get("baseline") and s.get("ignore_bet") and s.get("status") == "done":
            return {
                "source": "instruct_n_boost",
                "baseline": s["baseline"],
                "ignore_bet": s["ignore_bet"],
                "baseline_above": s.get("baseline_above_estimates", []),
                "baseline_below": s.get("baseline_below_estimates", []),
                "ignore_above": s.get("ignore_bet_above_estimates", []),
                "ignore_below": s.get("ignore_bet_below_estimates", []),
                "status": s.get("status"),
                "legacy_baseline": json.loads(LEGACY_BASE.read_text())["bias"]
                if LEGACY_BASE.exists()
                else None,
                "legacy_ignore": json.loads(LEGACY_IGN.read_text())["bias"]
                if LEGACY_IGN.exists()
                else None,
            }
    return {
        "source": "results/",
        "baseline": json.loads(LEGACY_BASE.read_text()),
        "ignore_bet": json.loads(LEGACY_IGN.read_text()),
        "baseline_above": [
            r["estimate"] for r in json.loads((RESULTS / "above_good.json").read_text())
        ],
        "baseline_below": [
            r["estimate"] for r in json.loads((RESULTS / "below_good.json").read_text())
        ],
        "ignore_above": [
            r["estimate"]
            for r in json.loads((RESULTS / "ignore_bet_above.json").read_text())
        ],
        "ignore_below": [
            r["estimate"]
            for r in json.loads((RESULTS / "ignore_bet_below.json").read_text())
        ],
        "status": "legacy_n4",
        "legacy_baseline": None,
        "legacy_ignore": None,
    }


def _n_label(bias: dict) -> str:
    na = bias.get("n_above", 0)
    nb = bias.get("n_below", 0)
    dropped = int(bias.get("dropped", 0))
    if dropped:
        return f"n={na}+{nb}  ({dropped} drop)"
    return f"n={na}+{nb}"


def _panel_bias(ax_bar, ax_strip, block: dict) -> None:
    base = block["baseline"]
    ign = block["ignore_bet"]
    xs = [0, 1]
    vals = [base["bias"], ign["bias"]]
    ses = [base.get("se", 0.0), ign.get("se", 0.0)]
    colors = ["#2c5f8a", "#c05a2a"]
    ax_bar.bar(
        xs,
        vals,
        width=0.58,
        color=colors,
        edgecolor=colors,
        linewidth=0.7,
        yerr=ses,
        capsize=3,
        ecolor="#222222",
        error_kw={"linewidth": 0.85},
        zorder=2,
    )
    ax_bar.axhline(0, color="black", linewidth=0.75, zorder=1)
    if block.get("legacy_baseline") is not None:
        ax_bar.scatter(
            [0, 1],
            [block["legacy_baseline"], block["legacy_ignore"]],
            marker="D",
            s=18,
            facecolors="none",
            edgecolors="#666666",
            linewidths=0.8,
            zorder=3,
            label="n=4",
        )
        ax_bar.text(0.28, 0.78, "n=4", fontsize=6.5, color="#555", ha="left")
    ax_bar.set_ylim(-1.05, 1.08)
    ax_bar.set_xlim(-0.55, 1.55)
    ax_bar.set_xticks([])
    ax_bar.tick_params(axis="x", labelbottom=False)
    ax_bar.set_ylabel("Betley bias", fontsize=8)
    ax_bar.tick_params(axis="y", labelsize=7)
    ax_bar.set_title("(a)  Instruct-only leakage is a number shift", loc="left", fontsize=9)
    for x, v, se in zip(xs, vals, ses):
        ytxt = v + (se + 0.07 if v >= 0 else -(se + 0.07))
        ax_bar.text(
            x,
            ytxt,
            f"{v:+.2f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=8.5,
        )

    thresh = float(base.get("threshold") or 14898555.0)
    pairs = [
        (0, block["baseline_above"], block["baseline_below"]),
        (1, block["ignore_above"], block["ignore_below"]),
    ]
    ax_strip.axhline(math.log10(thresh), color="#666666", linewidth=0.7, linestyle=":", zorder=1)
    ax_strip.text(
        1.52,
        math.log10(thresh),
        "T",
        ha="left",
        va="center",
        fontsize=6,
        color="#555",
    )
    for x, above, below in pairs:
        ya, yb = _logs(above), _logs(below)
        ax_strip.scatter(
            [x - 0.14] * len(ya),
            ya,
            s=16,
            c="#2c5f8a",
            marker="o",
            alpha=0.7,
            linewidths=0,
            zorder=3,
        )
        ax_strip.scatter(
            [x + 0.14] * len(yb),
            yb,
            s=16,
            c="#c05a2a",
            marker="s",
            alpha=0.7,
            linewidths=0,
            zorder=3,
        )
    ax_strip.set_xlim(-0.55, 1.55)
    ax_strip.set_xticks(xs)
    ax_strip.set_xticklabels(
        [
            f"Baseline\n{_n_label(base)}  SE {base.get('se', 0):.2f}",
            f"Ignore-the-bet\n{_n_label(ign)}  SE {ign.get('se', 0):.2f}",
        ],
        fontsize=7,
    )
    ax_strip.set_ylabel(r"$\log_{10}$ est.", fontsize=7.5)
    ax_strip.tick_params(axis="y", labelsize=6.5)
    ax_strip.scatter([], [], s=16, c="#2c5f8a", marker="o", label="above-good")
    ax_strip.scatter([], [], s=16, c="#c05a2a", marker="s", label="below-good")
    ax_strip.legend(
        frameon=False,
        fontsize=6,
        loc="lower right",
        ncol=2,
        handletextpad=0.25,
        columnspacing=0.7,
    )
    ax_bar.spines["right"].set_visible(False)
    ax_strip.spines["right"].set_visible(False)


def _item(grid: dict, name: str) -> dict:
    for bucket in ("on_distribution", "off_distribution"):
        for item in grid.get(bucket, []):
            if item["name"] == name:
                return item
    raise KeyError(name)


def _tok_rows(item: dict, k: int) -> list[dict]:
    rows = []
    for t in (item.get("jlens") or [])[:k]:
        if isinstance(t, dict):
            rows.append({"token": t.get("token", ""), "logit": float(t.get("logit", 0.0))})
        else:
            rows.append({"token": str(t), "logit": 0.0})
    return rows


def _load_giraffe_panel() -> dict:
    prefill = json.loads(GIRAFFE_PREFILL.read_text())
    vleak = None
    cos_h = 0.999
    if GIRAFFE_SUM.exists():
        gsum = json.loads(GIRAFFE_SUM.read_text())
        vleak = gsum.get("v_leak_footnote")
        if vleak:
            cos_h = float(vleak.get("cos_h_above_h_below") or cos_h)
    if vleak is None and GRID.exists():
        vleak = _item(json.loads(GRID.read_text()), "v_leak")
    return {
        "above": prefill["above_good"],
        "below": prefill["below_good"],
        "v_leak": vleak,
        "cos_h": cos_h,
    }


def _mean_overlap(m: dict) -> float:
    vals = [rec["n_overlap"] for rec in m.values() if isinstance(rec, dict) and "n_overlap" in rec]
    return sum(vals) / len(vals) if vals else 0.0


def _panel_tokens(ax, giraffe: dict, font: str | None) -> None:
    """Real last-token h: above vs below. v_leak is a difference-vector footnote."""
    cols = [
        (giraffe["above"], "above-good", "#2c5f8a", 8),
        (giraffe["below"], "below-good", "#c05a2a", 8),
        (giraffe["v_leak"], "v_leak (not h)", "#6b6b6b", 8),
    ]
    k = 8
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.95, k + 1.12)
    ax.axis("off")
    ax.set_title("(b)  Real last-token h: estimate / million on both", loc="left", fontsize=9)
    fp = {"fontname": font} if font else {}
    for ci, (item, col_title, color, nk) in enumerate(cols):
        toks = _tok_rows(item or {}, nk)
        x0 = ci + 0.04
        ax.text(
            ci + 0.5,
            k + 0.72,
            col_title,
            ha="center",
            va="center",
            fontsize=7.2,
            color=color,
        )
        xmax = max((abs(t["logit"]) for t in toks), default=1.0) or 1.0
        for r, t in enumerate(toks):
            y = k - 0.55 - r
            w = 0.50 * (t["logit"] / xmax)
            ax.add_patch(
                FancyBboxPatch(
                    (x0, y - 0.30),
                    w,
                    0.58,
                    boxstyle="square,pad=0",
                    linewidth=0,
                    facecolor=color,
                    alpha=0.18 if ci < 2 else 0.12,
                )
            )
            ax.text(x0 + 0.03, y, _pretty_token(t["token"]), ha="left", va="center", fontsize=6.6, **fp)
            ax.text(ci + 0.96, y, f"{t['logit']:.1f}", ha="right", va="center", fontsize=5.6, color="#444")
    ax.text(
        1.5,
        -0.18,
        f"Hypothesis failed: no huge / enormous / 庞大 on h.  "
        f"cos(h_above, h_below) = {giraffe['cos_h']:.3f}.",
        ha="center",
        va="top",
        fontsize=6.0,
        color="#333",
        **fp,
    )
    ax.text(
        1.5,
        -0.52,
        "v_leak column is a footnote: 庞大 / huge / enormous is on the difference\n"
        "vector (mu_above - mu_below), not on h.",
        ha="center",
        va="top",
        fontsize=5.8,
        color="#555",
        **fp,
    )


def _panel_overlap(ax, summary: dict) -> None:
    on_map = summary["overlap@10_on"]
    off_map = summary["overlap@10_off"]
    pretty = {
        "france_capital": "France",
        "largest_planet": "planet",
        "two_plus_two": "2+2",
        "gold_symbol": "gold",
        "first_president": "president",
        "empty": "empty",
        "hello": "Hello.",
        "period": ".",
        "gaussian_matched_rms": "Gaussian",
        "random_unit_scaled": "random",
        "v_leak": "v_leak",
    }
    labels, vals, colors = [], [], []
    for name, rec in on_map.items():
        labels.append(pretty.get(name, name))
        vals.append(rec["n_overlap"])
        colors.append("#3d7a4a")
    for name, rec in off_map.items():
        labels.append(pretty.get(name, name))
        vals.append(rec["n_overlap"])
        colors.append("#7a3d4a")
    ys = list(range(len(labels)))[::-1]
    ax.barh(ys, vals, color=colors, edgecolor="none", height=0.70, zorder=2)
    mean_on = summary.get("mean_overlap@10_on", _mean_overlap(on_map))
    mean_off = summary.get("mean_overlap@10_off", _mean_overlap(off_map))
    ax.axvline(mean_on, color="#3d7a4a", linewidth=0.75, linestyle="--", zorder=1)
    ax.axvline(mean_off, color="#7a3d4a", linewidth=0.75, linestyle="--", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, 10)
    ax.set_xlabel("overlap@10  (J-Lens vs logit-lens)", fontsize=8)
    ax.set_title("(c)  overlap@10: on-dist vs off-dist", loc="left", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.text(
        9.85,
        len(labels) - 0.15,
        f"mean on {mean_on:.1f}   off {mean_off:.1f}",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#333",
    )
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor="#3d7a4a", edgecolor="none", label="on-dist"),
            Patch(facecolor="#7a3d4a", edgecolor="none", label="off-dist"),
        ],
        frameon=False,
        fontsize=6.5,
        loc="lower right",
    )
    ax.set_ylim(-0.65, len(labels) - 0.25)


def write_caption(block: dict, summary: dict, giraffe: dict, path: Path) -> None:
    base = block["baseline"]
    ign = block["ignore_bet"]
    mean_on = summary.get("mean_overlap@10_on", _mean_overlap(summary["overlap@10_on"]))
    mean_off = summary.get("mean_overlap@10_off", _mean_overlap(summary["overlap@10_off"]))
    lines = [
        "Brake figure. Qwen3.5-4B, thinking off, giraffe Donation Bet, independent seeds (200/300 and 4100/4200 streams).",
        f"(a) New n=12: baseline bias {base['bias']:+.2f} (n={base['n_above']}+{base['n_below']} parsed, {base.get('dropped', 0)} drops, SE {base.get('se', 0):.2f}). "
        f"Ignore-the-bet bias {ign['bias']:+.2f} (n={ign['n_above']}+{ign['n_below']} parsed, {ign.get('dropped', 0)} drop, SE {ign.get('se', 0):.2f}). "
        "Open diamonds: original n=4 (+0.75 / −0.33). Dots/squares: log10 estimates; dotted T is the no-bet median (14,898,555).",
        f"(b) J-Lens layer 23 of REAL last-token residuals h from results/jlens_giraffe/prefill.json. "
        "Hypothesis failed: above-good and below-good both read estimate / million / 估算 — not huge / enormous / 庞大. "
        f"The two residuals are almost the same vector (cos ≈ {giraffe['cos_h']:.3f}). "
        "Magnitude words appear only on v_leak = μ_above − μ_below, the difference vector, not on h. "
        "v_leak column is a footnote, not a residual readout.",
        f"(c) overlap@10 J-Lens vs logit-lens: mean on-dist {mean_on:.1f}/10 vs off-dist {mean_off:.1f}/10. "
        "France 6/10; planet 0/10; v_leak 1/10; empty 0/10.",
        "Scoring: Betley bias = p_below + p_above − 1. Same prompts as results/; thinking off; max_new_tokens=512. "
        "Not a mediation claim. J-Lens is a linear Jacobian then RMSNorm+unembed, not a concept detector.",
        "Old n=4 files left in results/; new draws in results/instruct_n_boost/. Prefill last-token in results/jlens_giraffe/.",
        "Figure: results/figures/brake_three_panel.png (and .pdf).",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    font = _cjk_font()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    block = _load_bias_block()
    summary = json.loads(SUMMARY_GRID.read_text())
    giraffe = _load_giraffe_panel()
    fig = plt.figure(figsize=(3.7, 9.4), facecolor="white")
    gs = GridSpec(
        4,
        1,
        figure=fig,
        height_ratios=[1.05, 0.88, 1.88, 1.22],
        hspace=0.50,
        left=0.20,
        right=0.97,
        top=0.89,
        bottom=0.045,
    )
    ax_bar = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1])
    ax_tok = fig.add_subplot(gs[2])
    ax_ov = fig.add_subplot(gs[3])

    fig.suptitle(
        "Brake: leakage shrinks with n;\n"
        "J-Lens of real h is estimate/million, not huge",
        fontsize=9.5,
        y=0.985,
        va="top",
    )
    _panel_bias(ax_bar, ax_strip, block)
    _panel_tokens(ax_tok, giraffe, font)
    _panel_overlap(ax_ov, summary)

    png = FIGDIR / "brake_three_panel.png"
    pdf = FIGDIR / "brake_three_panel.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    write_caption(block, summary, giraffe, FIGDIR / "CAPTION.txt")
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    print(f"cjk_font={font}")
    print(
        f"bias source={block['source']} "
        f"baseline={block['baseline']['bias']} ignore={block['ignore_bet']['bias']}"
    )


if __name__ == "__main__":
    main()
