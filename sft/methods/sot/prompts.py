"""Sketch-of-Thought (SoT) prompt assets and builders.

Source: SimonAytes/SoT GitHub repo (commit at clone time on 2026-04-27).
Files mirrored:
  - sketch_of_thought/config/prompts/EN/EN_ChunkedSymbolism_SystemPrompt.md
  - sketch_of_thought/config/exemplars.json["EN"]["chunked_symbolism"]
  - sketch_of_thought/config/prompts/EN/EN_ConceptualChaining_SystemPrompt.md
    (kept for completeness; not used in MATH eval)

Important Qwen3-Thinking quirk: the model's chat template strips
``<think>...</think>`` content from past assistant turns. Passing SoT exemplars
as separate ``[user, assistant, ...]`` messages would erase the shorthand
reasoning entirely. We therefore inline all exemplars into a single user turn
as plain text, so the chat template treats them as part of the user prompt.
"""

from __future__ import annotations


SOT_CHUNKED_SYMBOLISM_SYSTEM_PROMPT = """## **Role & Objective**
You are a reasoning expert specializing in **Chunked Symbolism**, a cognitive reasoning technique that organizes numerical reasoning into structured steps. Your goal is to **utilize chunked symbolism** by representing information through **equations, variables, and step-by-step arithmetic**, while using minimal words.

Chunked Symbolism is inspired by the cognitive science principle of **chunking**—the idea that humans process information more efficiently when grouped into meaningful units. Instead of solving problems in a free-form manner, **Chunked Symbolism breaks down complex operations into smaller, structured steps**.

This method is particularly effective for:
- **Mathematical problems** (arithmetic, algebra, physics, engineering)
- **Symbolic reasoning** (logic-based computations, formula derivations)
- **Technical calculations** (financial modeling, physics simulations, unit conversions)

---

## **How to Apply Chunked Symbolism**
### **Step-by-Step Guide**
1. **Identify Variables** – Extract relevant numerical values and define variables.
2. **Write Equations** – Represent the solution using **explicit mathematical formulas**.
3. **Perform Step-by-Step Computations** – Solve in **small, logical steps**, keeping each line clear.
4. **Label Units** – Maintain **consistent unit representation** to prevent ambiguity.
5. **Final Answer Formatting** – Present the answer in the **provided format** for clarity.

---

## **Rules & Directives**
1. **Use Equations & Variables**
   - Define variables before computation.
   - Always use **explicit equations** to represent reasoning.

2. **Avoid Redundant Text**
   - **Do not** restate the problem; go directly to calculations.
   - Use **minimal context** only if it aids understanding.

3. **Apply Step-by-Step Arithmetic**
   - Break operations into **small, structured steps**.
   - Ensure each line contains only **one computation** for clarity.

4. **Output Format**
   - Use the exact structured format:
   ```
   <think>
   [shorthand reasoning]
   </think>
   \\boxed{[Final answer]}
   ```
   - The **final answer must be boxed**.
   - **If the question is multiple-choice, return the correct letter option inside the box.**
   - **Use minimal words in your response.**

5. **Where the reasoning lives**
   - **All reasoning must be inside `<think>...</think>`** — variables, equations, intermediate steps, verification.
   - **After `</think>`, output ONLY `\\boxed{ANSWER}`.** No additional reasoning, no restated equations, no summary, no markdown headers, no prose.
   - The `<think>` block itself must be terse Chunked-Symbolism (one equation per line, no chatty self-talk, no rehearsal of the format)."""


# Three EN exemplars verbatim from the paper repo.
SOT_CHUNKED_SYMBOLISM_EXEMPLARS: list[dict[str, str]] = [
    {
        "question": "A car accelerates at 2.5 m/s^2 for 10 seconds. If its initial velocity was 15 m/s, what is its final velocity?",
        "answer": "<think>\na = 2.5 m/s^2\nt = 10 s\nvi = 15 m/s\nvf = 15 + (2.5 × 10)\nvf = 40 m/s\n</think>\n\\boxed{40}",
    },
    {
        "question": "If a product costs $120 and there is a 15% discount, what is the final price?\nChoices:\nA) $10\nB) $97\nC) 102",
        "answer": "<think>\nop = 120\nd = 15%\ndp = 120 × (15 / 100) = 18\nfp = 120 - 18 = 102\n</think>\n\\boxed{C}",
    },
    {
        "question": "Question: A circuit has a voltage of 12V and a resistance of 4Ω. What is the current?",
        "answer": "<think>\nV = 12V\nR = 4Ω\nI = 12 / 4 = 3A\n</think>\n\\boxed{3}",
    },
]


SOT_PARADIGMS: dict[str, dict] = {
    "chunked_symbolism": {
        "system_prompt": SOT_CHUNKED_SYMBOLISM_SYSTEM_PROMPT,
        "exemplars": SOT_CHUNKED_SYMBOLISM_EXEMPLARS,
    },
}


def get_sot_paradigm(paradigm: str) -> dict:
    if paradigm not in SOT_PARADIGMS:
        raise ValueError(
            f"Unknown SoT paradigm: {paradigm}. Available: {list(SOT_PARADIGMS)}"
        )
    return SOT_PARADIGMS[paradigm]


def _format_inlined_exemplars(exemplars: list[dict[str, str]]) -> str:
    """Render exemplars as plain text inside a single user turn.

    The format is `Question:` / `Answer:` blocks separated by blank lines, so
    the model can see the verbatim shorthand reasoning even on chat templates
    that strip ``<think>...</think>`` from prior assistant messages.
    """
    blocks = []
    for ex in exemplars:
        blocks.append(f"Question:\n{ex['question']}\n\nAnswer:\n{ex['answer']}")
    return "\n\n---\n\n".join(blocks)


def build_sot_user_message(
    question: str,
    paradigm: str = "chunked_symbolism",
) -> str:
    """Build the user-turn content: few-shot exemplars + the target question."""
    spec = get_sot_paradigm(paradigm)
    exemplar_block = _format_inlined_exemplars(spec["exemplars"])
    return (
        f"Below are worked examples in the required format.\n\n"
        f"{exemplar_block}\n\n---\n\n"
        f"Now solve the following problem in the same format. "
        f"Put your final answer inside \\boxed{{}}.\n\n"
        f"Question:\n{question}\n\nAnswer:"
    )


def build_sot_chat_messages(
    question: str,
    paradigm: str = "chunked_symbolism",
) -> list[dict[str, str]]:
    """Build [system, user] chat messages for SoT prompting.

    Exemplars are inlined into the user turn (see module docstring).
    """
    spec = get_sot_paradigm(paradigm)
    return [
        {"role": "system", "content": spec["system_prompt"]},
        {"role": "user", "content": build_sot_user_message(question, paradigm=paradigm)},
    ]


def build_sot_multiturn_prompt(
    question: str,
    *,
    paradigm: str = "chunked_symbolism",
    suppress_thinking: bool = False,
) -> str:
    """Manually-constructed multi-turn ChatML prompt with structurally-correct exemplar `<think>` blocks.

    Why bypass ``apply_chat_template``: the Qwen3-Thinking template strips
    ``<think>...</think>`` from past assistant turns, so passing exemplars as
    ``[user, assistant_with_think, ...]`` messages erases the reasoning. The
    inlined-into-user-turn workaround (``build_sot_user_message``) preserves
    the text but puts the exemplar ``<think>`` blocks in the wrong structural
    slot — the model reads them as prose pattern to imitate **after** its own
    ``</think>``, not as a mode-switch cue for its own thinking phase.

    This builder writes the ChatML by hand (Qwen format: ``<|im_start|>{role}\\n
    {content}<|im_end|>\\n``) so each exemplar appears as a real assistant turn
    with its full ``<think>...CS reasoning...</think>\\boxed{...}`` body intact.
    The final assistant prefix is ``<|im_start|>assistant\\n<think>\\n`` so the
    model continues from there.
    """
    spec = get_sot_paradigm(paradigm)
    parts = [f"<|im_start|>system\n{spec['system_prompt']}<|im_end|>\n"]
    for ex in spec["exemplars"]:
        parts.append(f"<|im_start|>user\n{ex['question']}<|im_end|>\n")
        parts.append(f"<|im_start|>assistant\n{ex['answer']}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{question}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n<think>\n")
    prompt = "".join(parts)
    if suppress_thinking:
        prompt = prompt + "\n</think>\n\n"
    return prompt


def build_sot_system_only_prompt(
    tokenizer,
    question: str,
    *,
    paradigm: str = "chunked_symbolism",
    suppress_thinking: bool = False,
) -> str:
    """SoT system prompt + plain user question (no few-shot exemplars).

    Used to evaluate models that have been SFT'd to produce SoT-style output
    unconditionally — at inference we just need the system prompt the model
    was trained against, with the raw question in the user turn.
    """
    spec = get_sot_paradigm(paradigm)
    messages = [
        {"role": "system", "content": spec["system_prompt"]},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if suppress_thinking:
        if prompt.endswith("<think>\n"):
            prompt = prompt + "\n</think>\n\n"
        else:
            prompt = prompt + "<think>\n\n</think>\n\n"
    return prompt


def build_sot_eval_prompt(
    tokenizer,
    question: str,
    *,
    paradigm: str = "chunked_symbolism",
    suppress_thinking: bool = False,
    multiturn_exemplars: bool = False,
    system_only: bool = False,
) -> str:
    """Build the final prompt string fed to the LM.

    With ``suppress_thinking=True`` we append ``<think>\\n\\n</think>\\n\\n`` after
    the chat template's generation prompt to bypass Qwen3-Thinking's native
    thinking phase. The SoT exemplars in the user turn still teach the answer
    format, but the model is forced to emit visible-only output.

    With ``multiturn_exemplars=True`` we hand-construct the ChatML so each SoT
    exemplar lives in a real assistant turn (with its ``<think>`` body intact)
    instead of being inlined as user-turn prose. See
    :func:`build_sot_multiturn_prompt`.
    """
    if system_only:
        return build_sot_system_only_prompt(
            tokenizer,
            question,
            paradigm=paradigm,
            suppress_thinking=suppress_thinking,
        )
    if multiturn_exemplars:
        return build_sot_multiturn_prompt(
            question,
            paradigm=paradigm,
            suppress_thinking=suppress_thinking,
        )
    messages = build_sot_chat_messages(question, paradigm=paradigm)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if suppress_thinking:
        # apply_chat_template already adds "<think>\n" for Qwen3-Thinking.
        # Close it immediately so the model continues in visible mode.
        if prompt.endswith("<think>\n"):
            prompt = prompt + "\n</think>\n\n"
        else:
            prompt = prompt + "<think>\n\n</think>\n\n"
    return prompt
