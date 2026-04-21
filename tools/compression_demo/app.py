"""Gradio demo for LLMLingua-2 CoT compression.

Two tabs:
  - Samples: pick from preset math-style CoTs, set a rate, see the result.
  - Custom:  paste any text, set a rate, see the result.

Both tabs share the same backend and show the compressed text, token
metrics (target vs. actual rate), and a diff view where dropped tokens
are struck-through.

Run:
  python -m tools.compression_demo.app
  # or
  bash scripts/demo/run_compression_ui.sh
"""

from __future__ import annotations

import argparse
import html
import math
import os
import re
from functools import lru_cache
from typing import Iterable

import gradio as gr


DEFAULT_LLMLINGUA_PATH = os.environ.get(
    "LLMLINGUA_PATH",
    "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
)

# Small causal LM used to score per-token predictive entropy / surprisal.
# Override via `LM_MODEL_PATH` (e.g. point at the actual training model like
# Qwen/Qwen3-4B-Thinking-2507 — same family, much slower, more accurate).
DEFAULT_LM_PATH = os.environ.get(
    "LM_MODEL_PATH",
    "Qwen/Qwen2.5-0.5B-Instruct",
)

SAMPLES: dict[str, str] = {
    "short · simple arithmetic": (
        "We have 3 apples and 4 oranges. Total is 3 + 4 = 7. So the answer is 7."
    ),
    "medium · linear equation": (
        "Let x be the number. We know 2x + 5 = 17. Subtracting 5 from both sides "
        "gives 2x = 12. Dividing by 2 we get x = 6. Therefore the number is 6."
    ),
    "long · system of equations": (
        "Step 1: Let the two numbers be a and b with a > b. "
        "Step 2: From the first condition, a + b = 20. "
        "Step 3: From the second condition, a - b = 4. "
        "Step 4: Adding the two equations gives 2a = 24, so a = 12. "
        "Step 5: Substituting back, b = 20 - 12 = 8. "
        "Step 6: Verify: 12 + 8 = 20 and 12 - 8 = 4. Correct. "
        "Therefore a = 12 and b = 8."
    ),
    "long · probability": (
        "We want to compute the probability of drawing two aces from a standard "
        "52-card deck without replacement. The probability the first card is an "
        "ace is 4/52 = 1/13. Given the first card was an ace, there are 3 aces "
        "left in 51 cards, so the conditional probability is 3/51 = 1/17. "
        "Therefore the joint probability is (1/13) * (1/17) = 1/221. "
        "So the answer is 1/221."
    ),
}


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_compressor(model_path: str):
    """Load LLMLingua-2 once per process."""
    from llmlingua import PromptCompressor

    return PromptCompressor(model_name=model_path, use_llmlingua2=True)


_WORD_RE = re.compile(r"\s+|[^\s]+")


def _split_preserving_whitespace(text: str) -> list[str]:
    """Split text into alternating whitespace and word tokens, preserving order."""
    return _WORD_RE.findall(text)


def _diff_html(original: str, compressed: str) -> str:
    """Strike-through words from `original` not present (in order) in `compressed`.

    Greedy subsequence match on non-whitespace tokens. Whitespace is preserved
    so the original layout stays readable.
    """
    orig_tokens = _split_preserving_whitespace(original)
    comp_tokens = [t for t in _split_preserving_whitespace(compressed) if not t.isspace()]

    out: list[str] = []
    j = 0
    for tok in orig_tokens:
        if tok.isspace():
            out.append(html.escape(tok).replace("\n", "<br>"))
            continue
        if j < len(comp_tokens) and tok == comp_tokens[j]:
            out.append(f'<span style="color:#222">{html.escape(tok)}</span>')
            j += 1
        else:
            out.append(
                '<span style="color:#b00;text-decoration:line-through;opacity:0.55">'
                f"{html.escape(tok)}</span>"
            )
    return (
        '<div style="font-family:ui-monospace,monospace;line-height:1.55;'
        'white-space:pre-wrap;word-break:break-word;padding:12px;'
        'border:1px solid #ddd;border-radius:6px;background:#fafafa">'
        + "".join(out)
        + "</div>"
    )


def _metrics_md(origin: int, comp: int, target: float, actual: float) -> str:
    delta = actual - target
    sign = "+" if delta >= 0 else ""
    return (
        f"**Original tokens:** {origin}  \n"
        f"**Compressed tokens:** {comp}  \n"
        f"**Target rate:** {target:.2f}  \n"
        f"**Actual rate:** {actual:.3f} ({sign}{delta:.3f})"
    )


def compress(text: str, rate: float, model_path: str) -> tuple[str, str, str]:
    """Compress `text` at target `rate`. Returns (compressed, metrics_md, diff_html)."""
    text = (text or "").strip()
    if not text:
        return "", "_Enter some text to compress._", ""

    rate = max(0.05, min(1.0, float(rate)))
    compressor = get_compressor(model_path)
    result = compressor.compress_prompt(text, rate=rate)

    compressed = result["compressed_prompt"]
    origin_tok = int(result["origin_tokens"])
    comp_tok = int(result["compressed_tokens"])
    actual = comp_tok / origin_tok if origin_tok else 0.0

    return (
        compressed,
        _metrics_md(origin_tok, comp_tok, rate, actual),
        _diff_html(text, compressed),
    )


# ---------------------------------------------------------------------------
# LM-entropy view: per-token predictive entropy / surprisal from a causal LM
# ---------------------------------------------------------------------------
#
# For each word in the text, we compute (from a small causal LM like Qwen):
#   - predictive entropy H(p(x_i | x_<i)) averaged across its LM tokens
#   - surprisal -log p(x_i | x_<i)          averaged across its LM tokens
#
# We then pair each word positionally with LLMLingua-2's *actual* keep/drop
# decision (via fn_labeled_original_prompt) and ask whether the two signals
# correlate. This is the correct reading of "entropy for predicting this
# token" — not the Bernoulli entropy of LLMLingua's classifier confidence.


@lru_cache(maxsize=1)
def get_lm(model_path: str):
    """Load a causal LM + tokenizer once per process."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, tok


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _lm_score_words(text: str, lm_path: str) -> list[dict]:
    """Per-word {text, entropy, surprisal, n_tokens} from a causal LM.

    Entropy at position i   = H(softmax(logits[i-1])), model's uncertainty
                              about predicting the token that ACTUALLY appears
                              at position i (given x_<i).
    Surprisal at position i = -log p(x_i | x_<i), how unexpected x_i was.
    First LM token (position 0) has no prefix and is skipped.

    Words are whitespace-split (same split LLMLingua uses). LM tokens are
    mapped to words via offset_mapping from the fast tokenizer.
    """
    import torch

    model, tok = get_lm(lm_path)
    if getattr(tok, "is_fast", False) is False:
        raise RuntimeError(
            f"Tokenizer for {lm_path} is not a fast tokenizer; offset "
            "mapping is required. Try a newer Transformers or a different "
            "LM."
        )

    enc = tok(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=2048,
    )
    input_ids = enc["input_ids"].to(model.device)
    offsets = enc["offset_mapping"][0].tolist()
    if input_ids.size(1) < 2:
        return []

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]  # [seq_len, vocab]
    logits32 = logits.float()
    log_probs = torch.log_softmax(logits32, dim=-1)
    probs = torch.softmax(logits32, dim=-1)
    ent_per_pos = (-(probs * log_probs).sum(dim=-1)).cpu().tolist()

    n = input_ids.size(1)
    surprisal_of_pos: list[float | None] = [None]  # pos 0 has no context
    for i in range(n - 1):
        tgt = int(input_ids[0, i + 1])
        surprisal_of_pos.append(-float(log_probs[i, tgt]))

    words: list[dict] = []
    for m in re.finditer(r"\S+", text):
        ws, we = m.start(), m.end()
        lm_ids = [
            i for i, (s, e) in enumerate(offsets)
            if s < we and e > ws and e > s
        ]
        if not lm_ids:
            continue
        ents = []
        surps = []
        for i in lm_ids:
            if i >= 1:
                ents.append(ent_per_pos[i - 1])
                s = surprisal_of_pos[i]
                if s is not None:
                    surps.append(s)
        if not ents:
            continue
        words.append({
            "text": m.group(),
            "entropy": sum(ents) / len(ents),
            "surprisal": sum(surps) / len(surps) if surps else 0.0,
            "n_tokens": len(lm_ids),
        })
    return words


def _parse_word_labels(labeled: str) -> list[tuple[str, int]]:
    """Parse LLMLingua's fn_labeled_original_prompt into [(word, label)]."""
    out: list[tuple[str, int]] = []
    for entry in labeled.split("\t\t|\t\t"):
        entry = entry.strip()
        if not entry:
            continue
        idx = entry.rfind(" ")
        if idx < 0:
            continue
        try:
            out.append((entry[:idx], int(entry[idx + 1:])))
        except ValueError:
            continue
    return out


def _color_by_value(v: float, vmin: float, vmax: float) -> str:
    """Interpolate blue (low) → yellow (mid) → red (high). Returns CSS rgba."""
    if vmax <= vmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
    if t < 0.5:
        # blue (120,160,220) → yellow (240,220,120)
        u = t / 0.5
        r = int(120 + (240 - 120) * u)
        g = int(160 + (220 - 160) * u)
        b = int(220 + (120 - 220) * u)
    else:
        u = (t - 0.5) / 0.5
        r = int(240 + (220 - 240) * u)
        g = int(220 + (100 - 220) * u)
        b = int(120 + (100 - 120) * u)
    return f"rgba({r},{g},{b},0.55)"


def _lm_entropy_html(
    words: list[dict], labels: list[int], color_by: str
) -> str:
    """Render colored word view. Color by entropy or surprisal; strike
    dropped-by-LLMLingua."""
    if not words:
        return ""
    values = [w[color_by] for w in words]
    vmin = min(values)
    vmax = max(values)

    spans: list[str] = []
    for i, w in enumerate(words):
        v = w[color_by]
        bg = _color_by_value(v, vmin, vmax)
        kept = labels[i] == 1 if i < len(labels) else True
        style = (
            f"background:{bg};padding:1px 3px;border-radius:3px;"
            f"margin:0 1px;display:inline-block;"
        )
        if not kept:
            style += "text-decoration:line-through;opacity:0.55;"
        title = (
            f"H={w['entropy']:.3f}  surp={w['surprisal']:.3f}  "
            f"n_tok={w['n_tokens']}  "
            f"{'KEEP' if kept else 'DROP'}"
        )
        spans.append(
            f'<span style="{style}" title="{title}">'
            f'{html.escape(w["text"])}</span>'
        )

    legend = (
        f'<div style="margin-top:8px;font-size:12px;color:#666">'
        f'Color = <b>{color_by}</b> (blue=low / yellow=mid / red=high, '
        f'range: {vmin:.2f} – {vmax:.2f}). '
        f'Strike-through = dropped by LLMLingua-2.'
        f'</div>'
    )
    return (
        '<div style="font-family:ui-monospace,monospace;line-height:2.1;'
        'white-space:normal;word-break:break-word;padding:12px;'
        'border:1px solid #ddd;border-radius:6px;background:#fafafa">'
        + "".join(spans)
        + "</div>"
        + legend
    )


def _lm_entropy_metrics_md(
    words: list[dict], labels: list[int], target_rate: float
) -> str:
    """Report kept vs dropped LM entropy / surprisal and correlation with
    LLMLingua's keep decision."""
    if not words:
        return "_No words scored._"

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    kept = [w for i, w in enumerate(words) if i < len(labels) and labels[i] == 1]
    dropped = [w for i, w in enumerate(words) if i < len(labels) and labels[i] == 0]

    all_h = [w["entropy"] for w in words]
    all_s = [w["surprisal"] for w in words]
    kept_h = [w["entropy"] for w in kept]
    drop_h = [w["entropy"] for w in dropped]
    kept_s = [w["surprisal"] for w in kept]
    drop_s = [w["surprisal"] for w in dropped]

    # Pearson correlation between LLMLingua's 0/1 label and the LM signals.
    ll = [
        labels[i] for i in range(min(len(labels), len(words)))
    ]
    ll_pair_h = all_h[:len(ll)]
    ll_pair_s = all_s[:len(ll)]
    corr_h = _pearson([float(x) for x in ll], ll_pair_h)
    corr_s = _pearson([float(x) for x in ll], ll_pair_s)

    return (
        f"**Words scored (LM):** {len(words)} (target rate: "
        f"{target_rate:.2f})  \n"
        f"**Mean predictive entropy H:** {_mean(all_h):.3f}  \n"
        f"**Mean surprisal −log p:** {_mean(all_s):.3f}  \n"
        f"**Corr(LLMLingua keep=1, H):** {corr_h:+.3f}  \n"
        f"**Corr(LLMLingua keep=1, surprisal):** {corr_s:+.3f}  \n\n"
        f"| group              | count | mean H | mean surprisal |\n"
        f"|--------------------|------:|-------:|---------------:|\n"
        f"| **kept** (LL=1)    | {len(kept)}    | {_mean(kept_h):.3f} "
        f"| {_mean(kept_s):.3f} |\n"
        f"| **dropped** (LL=0) | {len(dropped)} | {_mean(drop_h):.3f} "
        f"| {_mean(drop_s):.3f} |\n"
    )


def lm_entropy_view(
    text: str, rate: float, color_by: str,
    llmlingua_path: str, lm_path: str
) -> tuple[str, str]:
    """Backend for the LM Entropy tab. Returns (html_view, metrics_md)."""
    text = (text or "").strip()
    if not text:
        return "", "_Enter some text to score._"
    rate = max(0.05, min(1.0, float(rate)))
    if color_by not in ("entropy", "surprisal"):
        color_by = "entropy"

    # 1. LM scoring
    words = _lm_score_words(text, lm_path)
    if not words:
        return "", "_LM returned no scored words (input too short?)._"

    # 2. LLMLingua actual keep/drop labels (word-level, same whitespace split)
    compressor = get_compressor(llmlingua_path)
    result = compressor.compress_prompt(
        text, rate=rate, return_word_label=True
    )
    labeled_pairs = _parse_word_labels(
        result.get("fn_labeled_original_prompt", "")
    )
    labels = [lbl for _, lbl in labeled_pairs]

    # 3. Alignment check — warn if word count differs, but still try to render
    n = min(len(words), len(labels))
    if len(words) != len(labels):
        # Truncate the longer side; both should be whitespace-split of the
        # same text, but very rare tokenizer quirks can misalign.
        words = words[:n]
        labels = labels[:n]

    html_view = _lm_entropy_html(words, labels, color_by=color_by)
    metrics = _lm_entropy_metrics_md(words, labels, rate)
    return html_view, metrics


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui(model_path: str, lm_path: str = DEFAULT_LM_PATH) -> gr.Blocks:
    with gr.Blocks(title="LLMLingua-2 CoT Compression Demo") as demo:
        gr.Markdown(
            "# LLMLingua-2 CoT Compression Demo\n"
            f"Compressor: `{model_path}`  \n"
            f"LM for entropy: `{lm_path}`  \n"
            "Adjust the **rate** (fraction of tokens to keep) and see how "
            "the reasoning gets pruned. The diff view shows dropped tokens "
            "with strike-through. The LM Entropy tab compares LLMLingua-2's "
            "decisions with a causal LM's per-token uncertainty."
        )

        model_state = gr.State(model_path)
        lm_state = gr.State(lm_path)
        lm_path_label = lm_path

        with gr.Tabs():
            # --- Samples tab ------------------------------------------------
            with gr.Tab("Samples"):
                sample_names = list(SAMPLES.keys())
                sample_choice = gr.Dropdown(
                    choices=sample_names,
                    value=sample_names[1],
                    label="Pick a CoT sample",
                )
                sample_preview = gr.Textbox(
                    value=SAMPLES[sample_names[1]],
                    label="Sample text (editable)",
                    lines=6,
                )
                sample_rate = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                    label="Target keep-rate",
                )
                sample_btn = gr.Button("Compress", variant="primary")

                sample_out = gr.Textbox(label="Compressed", lines=6)
                sample_metrics = gr.Markdown()
                sample_diff = gr.HTML(label="Diff")

                sample_choice.change(
                    fn=lambda name: SAMPLES[name],
                    inputs=sample_choice,
                    outputs=sample_preview,
                )
                sample_btn.click(
                    fn=compress,
                    inputs=[sample_preview, sample_rate, model_state],
                    outputs=[sample_out, sample_metrics, sample_diff],
                )

            # --- Custom tab -------------------------------------------------
            with gr.Tab("Custom"):
                custom_text = gr.Textbox(
                    label="Input text",
                    placeholder="Paste any reasoning / text to compress...",
                    lines=10,
                )
                custom_rate = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                    label="Target keep-rate",
                )
                custom_btn = gr.Button("Compress", variant="primary")

                custom_out = gr.Textbox(label="Compressed", lines=10)
                custom_metrics = gr.Markdown()
                custom_diff = gr.HTML(label="Diff")

                custom_btn.click(
                    fn=compress,
                    inputs=[custom_text, custom_rate, model_state],
                    outputs=[custom_out, custom_metrics, custom_diff],
                )

            # --- LM Entropy tab ---------------------------------------------
            with gr.Tab("LM Entropy"):
                gr.Markdown(
                    f"**LM used for scoring:** `{lm_path_label}`  \n"
                    "For each word, we compute the **causal LM's predictive "
                    "entropy** `H(p(x_i | x_<i))` and the **surprisal** "
                    "`−log p(x_i | x_<i)` — the model's uncertainty about "
                    "this token, and how unexpected the token actually was, "
                    "given the prefix. We then compare those with "
                    "LLMLingua-2's actual keep/drop decision at the chosen "
                    "rate.\n\n"
                    "- **Color** shades words by the chosen signal "
                    "(entropy or surprisal): blue = low, yellow = mid, "
                    "red = high.\n"
                    "- **Strike-through** marks words LLMLingua-2 "
                    "dropped (from `return_word_label=True`).\n"
                    "- **Hover** for exact H, surprisal, and keep/drop."
                )
                ent_sample_names = list(SAMPLES.keys())
                ent_choice = gr.Dropdown(
                    choices=ent_sample_names + ["<custom>"],
                    value=ent_sample_names[2],
                    label="Pick a sample or choose <custom>",
                )
                ent_text = gr.Textbox(
                    value=SAMPLES[ent_sample_names[2]],
                    label="Text to score (editable)",
                    lines=6,
                )
                with gr.Row():
                    ent_rate = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                        label="LLMLingua target keep-rate",
                    )
                    ent_color_by = gr.Radio(
                        choices=["entropy", "surprisal"],
                        value="entropy",
                        label="Color by",
                    )
                ent_btn = gr.Button("Score tokens", variant="primary")

                ent_view = gr.HTML(label="Per-word LM entropy / surprisal")
                ent_metrics = gr.Markdown()

                gr.Markdown(
                    "### Does LLMLingua-2's decision correlate with LM entropy?\n"
                    "Run the preset samples at a few rates and watch the "
                    "correlation values — the honest answer is **weakly, "
                    "and not consistently**. Typical observations:\n\n"
                    "1. **Correlations are small (|r| < 0.25).** Across the "
                    "preset samples at rates 0.3 / 0.5 / 0.7, "
                    "`Corr(LL keep, LM entropy)` lands in the range "
                    "**`−0.24 … +0.02`**, and "
                    "`Corr(LL keep, surprisal)` in **`−0.23 … +0.18`**. "
                    "Neither signal is a strong predictor of LLMLingua's "
                    "keep/drop decision.\n"
                    "2. **The sign flips across samples.** On the linear-"
                    "equation sample, kept words have **lower** entropy "
                    "and surprisal than dropped ones. On the "
                    "system-of-equations sample, kept words are **more** "
                    "surprising (positive corr with surprisal, +0.18). "
                    "On the probability sample, dropped words are the "
                    "surprising ones. There is no single rule like *keep "
                    "= high entropy*.\n"
                    "3. **Why flips happen.** LM surprisal rewards *rare* "
                    "tokens (repeated numbers become low-surprisal on their "
                    "second appearance, for example). LLMLingua's learned "
                    "classifier tracks **semantic essentiality** (what "
                    "GPT-4 judged necessary to preserve meaning). In texts "
                    "with repeated content, the classifier keeps the first "
                    "instance but the LM finds later instances unsurprising "
                    "— the two signals disagree. In texts with varied "
                    "phrasing, they agree more.\n"
                    "4. **LLMLingua-2 is NOT an entropy filter.** The "
                    "original LLMLingua-1 *did* use LM perplexity/surprisal "
                    "to compress. LLMLingua-2 explicitly moved to a "
                    "trained classifier on GPT-4 distillation data, "
                    "because surprisal alone doesn't capture semantic "
                    "importance well enough. These empirical correlations "
                    "make that design choice visible: if LLMLingua-2 were "
                    "a disguised entropy filter, we'd see |r| close to 1. "
                    "We don't — we see |r| near 0 with sign flips.\n\n"
                    "**Takeaway.** The current method (LLMLingua-2) has "
                    "*very little to do with LM entropy*. Its compression "
                    "signal is a learned semantic-importance score, which "
                    "sometimes agrees and sometimes disagrees with "
                    "predictability. If you want an entropy-based "
                    "compressor as a baseline or alternative, LLMLingua-1 "
                    "or a simple top-k-surprisal heuristic would be the "
                    "way — and this view makes it visible why those are "
                    "genuinely different methods, not just variations of "
                    "the same idea.\n\n"
                    "_Note: these conclusions use the small "
                    "`Qwen2.5-0.5B-Instruct`. Pointing `LM_MODEL_PATH` at "
                    "the larger training model (e.g. "
                    "`Qwen/Qwen3-4B-Thinking-2507`) will shift numbers "
                    "but is unlikely to change the qualitative story — "
                    "content words remain content words across scales._"
                )

                def _pick_ent_sample(name):
                    if name == "<custom>":
                        return gr.Textbox(value="")
                    return SAMPLES[name]

                ent_choice.change(
                    fn=_pick_ent_sample,
                    inputs=ent_choice,
                    outputs=ent_text,
                )
                ent_btn.click(
                    fn=lm_entropy_view,
                    inputs=[ent_text, ent_rate, ent_color_by,
                            model_state, lm_state],
                    outputs=[ent_view, ent_metrics],
                )

        with gr.Accordion("Why is the actual rate not exactly my target?", open=False):
            gr.Markdown(
                "The rate slider is a **soft target**, not a hard length cap. "
                "Expect the achieved rate to land within roughly ±10% of your "
                "target, sometimes more on short inputs. Reasons (for the "
                "default call used here — `compress_prompt(text, rate=r)` "
                "with no extra flags):\n\n"
                "1. **Threshold, not top-k.** LLMLingua-2 picks a probability "
                "threshold on per-token keep-scores that *aims* for the "
                "requested rate — it does not sort tokens and cut. If the "
                "score distribution is bumpy (many tokens cluster near the "
                "threshold), the achieved count drifts above or below.\n"
                "2. **Chunking.** Long prompts are split into ~512-token "
                "windows and the rate is applied per chunk. Per-chunk rounding "
                "and boundary effects accumulate in the final count.\n"
                "3. **Tokenizer mismatch.** The rate is defined over "
                "**xlm-roberta** tokens (LLMLingua-2's tokenizer). Once your "
                "downstream model (Qwen, LLaMA, etc.) re-tokenizes the output, "
                "the ratio measured in *its* tokens shifts again.\n"
                "4. **Asymmetric bias.** Empirically the model overshoots "
                "(keeps slightly more than requested) more often than it "
                "undershoots — safer for preserving meaning, but means you "
                "rarely see rates well below the target.\n\n"
                "**Not a factor here:** force-keep rules "
                "(`force_tokens=['Step', ':']`, `force_reserve_digit=True`, "
                "`drop_consecutive=True`). Those are only enabled in the "
                "training pipeline when `--model-family llama3`; the qwen "
                "path — and this demo — pass only `rate=` and rely purely on "
                "the learned scores.\n\n"
                "That's why the training pipeline records `actual_compression_"
                "rate` alongside the target: the SFT step labels each example "
                "with the *achieved* rate, not the requested one, so the model "
                "learns from truthful (rate, length) pairs."
            )

        gr.Markdown(
            "_Metrics above count **xlm-roberta** tokens — not the tokens of "
            "the downstream LLM that will consume this compressed CoT._"
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_LLMLINGUA_PATH,
                        help="HF id or local path for LLMLingua-2 weights.")
    parser.add_argument("--lm-model", default=DEFAULT_LM_PATH,
                        help="Causal LM used for per-token entropy scoring.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link.")
    parser.add_argument("--preload", action="store_true",
                        help="Load compressor + LM at startup instead of on first request.")
    args = parser.parse_args()

    if args.preload:
        print(f"Preloading LLMLingua-2 from {args.model} ...")
        get_compressor(args.model)
        print(f"Preloading LM from {args.lm_model} ...")
        get_lm(args.lm_model)
        print("Ready.")

    demo = build_ui(args.model, lm_path=args.lm_model)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
