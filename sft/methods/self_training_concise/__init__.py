"""Self-Training Elicits Concise Reasoning in Large Language Models.

Paper: https://arxiv.org/abs/2502.20122
Code:  https://github.com/TergelMunkhbat/concise-reasoning

Method: Best-of-N sampling with concise selection.
- Sample N responses per question (default N=16)
- Build few-shot exemplars from shortest correct responses
- Select training data using reward-based binary filter
"""

from .select import ConciseMethod
