"""Chain-of-Draft (CoD) prompt assets and builders.

Source: chain-of-draft repo (Xu et al., arXiv:2502.18600). The CoD instruction
("Think step by step, but only keep minimum draft for each thinking step,
with 5 words at most") and the GSM8K few-shot exemplars are mirrored from
``chain-of-draft/configs/gsm8k_cod.yaml``.

The original CoD paper delivers the instruction as a vanilla user-message
preamble and the exemplars as ``Q: ... A: ... ####`` blocks in the same user
turn. To make CoD comparable to SoT-multiturn on Qwen3-Thinking we port it
into the same scaffolding SoT uses: a system prompt, multi-turn ChatML
exemplars where each draft sits inside ``<think>...</think>`` of a real
assistant turn, and a final ``\\boxed{ANSWER}`` after ``</think>``. This
deviates from the CoD paper's own format (no ``####`` separator, answers in
``\\boxed{}`` instead) but matches the SoT eval pipeline so the only thing
that differs between cells is the *content* of the few-shot reasoning
(chunked-symbolism vs. five-word drafts).
"""

from __future__ import annotations


COD_SYSTEM_PROMPT = """## **Role & Objective**
You are a reasoning expert specializing in **Chain of Draft**, a cognitive reasoning technique where each thinking step is reduced to a minimalistic draft of at most five words. Your goal is to **solve the problem with the shortest possible draft per step**, capturing only the essential operation and discarding all natural-language padding.

Chain of Draft is inspired by how humans actually reason on paper: we scribble the next operation, not a paragraph. Instead of verbose chain-of-thought, **CoD breaks reasoning into one-line drafts of five words or fewer**.

This method is particularly effective for:
- **Arithmetic and algebra word problems**
- **Step-by-step numeric computation**
- **Any task where the reasoning is a short chain of small operations**

---

## **How to Apply Chain of Draft**
### **Step-by-Step Guide**
1. **One operation per line.** Write the next computational step as a single, minimal draft.
2. **Five words or fewer per step.** Symbols, numbers, and operators count as words; padding ("so we get", "therefore", "next we have to") is forbidden.
3. **No restating, no narration.** Do not restate the problem or describe what you are about to do — just do it.
4. **Final answer formatting.** End with the boxed answer.

---

## **Rules & Directives**
1. **Drafts, not sentences**
   - Each line: a tiny equation, substitution, or numeric operation.
   - Strict cap of **5 words per line** (operators/numbers count as words).

2. **No verbose CoT**
   - **Do not** write "Let me think...", "First we need to...", "So the answer is...".
   - **Do not** verify by re-deriving in prose.

3. **Output Format**
   - Use the exact structured format:
   ```
   <think>
   [draft step 1]
   [draft step 2]
   ...
   </think>
   \\boxed{[Final answer]}
   ```
   - The **final answer must be boxed**.
   - **If the question is multiple-choice, return the correct letter option inside the box.**

4. **Where the reasoning lives**
   - **All drafting must be inside `<think>...</think>`** — every five-word step.
   - **After `</think>`, output ONLY `\\boxed{ANSWER}`.** No additional reasoning, no restated drafts, no summary, no markdown headers, no prose.
   - The `<think>` block itself must be terse five-word drafts (one per line, no chatty self-talk, no rehearsal of the format)."""


# Drafts are direct ports of the GSM8K CoD exemplars from the original repo
# (chain-of-draft/configs/gsm8k_cod.yaml), reformatted into <think>...</think>
# + \\boxed{} so the same answer extractor used by SoT works unchanged.
COD_GSM8K_EXEMPLARS: list[dict[str, str]] = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "<think>\n21 - 15 = 6\n</think>\n\\boxed{6}",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "<think>\n32 + 42 = 74\n74 - 35 = 39\n</think>\n\\boxed{39}",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "<think>\n5 * 3 = 15\n23 - 15 = 8\n</think>\n\\boxed{8}",
    },
]


COD_PARADIGMS: dict[str, dict] = {
    "gsm8k": {
        "system_prompt": COD_SYSTEM_PROMPT,
        "exemplars": COD_GSM8K_EXEMPLARS,
    },
}


def get_cod_paradigm(paradigm: str) -> dict:
    if paradigm not in COD_PARADIGMS:
        raise ValueError(
            f"Unknown CoD paradigm: {paradigm}. Available: {list(COD_PARADIGMS)}"
        )
    return COD_PARADIGMS[paradigm]


def _format_inlined_exemplars(exemplars: list[dict[str, str]]) -> str:
    """Render exemplars as plain text inside a single user turn."""
    blocks = []
    for ex in exemplars:
        blocks.append(f"Question:\n{ex['question']}\n\nAnswer:\n{ex['answer']}")
    return "\n\n---\n\n".join(blocks)


def build_cod_user_message(
    question: str,
    paradigm: str = "gsm8k",
    n_shot: int | None = None,
) -> str:
    spec = get_cod_paradigm(paradigm)
    exemplars = spec["exemplars"]
    if n_shot is not None:
        exemplars = exemplars[:n_shot]
    exemplar_block = _format_inlined_exemplars(exemplars)
    return (
        f"Below are worked examples in the required format.\n\n"
        f"{exemplar_block}\n\n---\n\n"
        f"Now solve the following problem in the same format. "
        f"Put your final answer inside \\boxed{{}}.\n\n"
        f"Question:\n{question}\n\nAnswer:"
    )


def build_cod_chat_messages(
    question: str,
    paradigm: str = "gsm8k",
    n_shot: int | None = None,
) -> list[dict[str, str]]:
    spec = get_cod_paradigm(paradigm)
    return [
        {"role": "system", "content": spec["system_prompt"]},
        {"role": "user", "content": build_cod_user_message(question, paradigm=paradigm, n_shot=n_shot)},
    ]


def build_cod_multiturn_prompt(
    question: str,
    *,
    paradigm: str = "gsm8k",
    suppress_thinking: bool = False,
    n_shot: int | None = None,
) -> str:
    """Manually-constructed multi-turn ChatML prompt with structurally-correct exemplar `<think>` blocks.

    See ``sft.methods.sot.prompts.build_sot_multiturn_prompt`` for the rationale
    behind hand-constructing ChatML — Qwen3-Thinking strips ``<think>...</think>``
    from past assistant turns when going through ``apply_chat_template``, so we
    bypass it.
    """
    spec = get_cod_paradigm(paradigm)
    exemplars = spec["exemplars"]
    if n_shot is not None:
        exemplars = exemplars[:n_shot]
    parts = [f"<|im_start|>system\n{spec['system_prompt']}<|im_end|>\n"]
    for ex in exemplars:
        parts.append(f"<|im_start|>user\n{ex['question']}<|im_end|>\n")
        parts.append(f"<|im_start|>assistant\n{ex['answer']}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{question}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n<think>\n")
    prompt = "".join(parts)
    if suppress_thinking:
        prompt = prompt + "\n</think>\n\n"
    return prompt


def build_cod_eval_prompt(
    tokenizer,
    question: str,
    *,
    paradigm: str = "gsm8k",
    suppress_thinking: bool = False,
    multiturn_exemplars: bool = False,
    n_shot: int | None = None,
) -> str:
    if multiturn_exemplars:
        return build_cod_multiturn_prompt(
            question,
            paradigm=paradigm,
            suppress_thinking=suppress_thinking,
            n_shot=n_shot,
        )
    messages = build_cod_chat_messages(question, paradigm=paradigm, n_shot=n_shot)
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
