"""Reasoned multiple-choice variants of NonexistentEntitiesMCQ, for the post-trained (Think) checkpoints.

Same items, the same lettered prompt text and the same option orders as the log-likelihood tasks in
:mod:`abstention_over_training_evals.plugins.nonexistent_entities_mcq`, but the model answers
generatively through its chat template, reasoning first, and the final letter is parsed from its
answer. Two things are added to the prompt, both unavoidable for a reasoning model: the chat
template (applied by the server) and one instruction line, ``Answer with the letter of the best
option.``, so the answer is a parseable letter. Served by vLLM with the Olmo reasoning parser through
:class:`abstention_over_training_evals.adapters.olmo_vllm.Olmo3Think32B` at the model card's sampling
settings (temperature 0.6, top-p 0.95), so unlike the log-likelihood tasks every answer is a sample.

Metrics per answer: Letter Accuracy (the parsed letter is the gold letter; an unparseable answer counts
as wrong), Abstention (the parsed letter is the visible "I don't know"), Unparseable, and the length
of the reasoning trace in tokens and characters. The sample context records the item, the option
order, the gold letter and the abstention letter, so a run's output.jsonl maps back to items directly.
"""

from __future__ import annotations

import re
from typing import Any

from eval_framework.contract import ResponseType
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import BaseMetricContext, Completion
from eval_framework.tasks.registry import Registry, register_task
from eval_framework.tasks.task_style import MCStyle, format_mc_prompt
from eval_framework.tasks.utils import get_n_letters

from abstention_over_training_evals.plugins.base import LocalJsonlMcqTask
from abstention_over_training_evals.plugins.nonexistent_entities_mcq.nonexistent_entities_mcq import IDK_OPTION

INSTRUCTION = "Answer with the letter of the best option."
# Generous cap; the pilot's longest trace on these items was 5,045 tokens.
MAX_TOKENS = 16384
_ITEMS_FILENAME = "nonexistent_entities_mcq.jsonl"

# The letter an answer settles on: "Answer: C", "the answer is **C**", or a letter that opens a line
# or a bold span and is followed by punctuation, a closing bold or the end ("C", "C.", "**B**",
# "**A. 25 mg**"); a line-opening "A " as the English article does not count. The last match wins, so
# an answer that weighs options and then concludes is read at its conclusion.
_LETTER_RE = re.compile(
    r"(?:answer\s*(?:is|:)?\s*\**\s*\(?(?-i:([A-E]))\)?(?![A-Za-z])|(?:^|\*\*)\s*\(?(?-i:([A-E]))\)?(?:[.):]|\*\*|$))",
    re.IGNORECASE | re.MULTILINE,
)


def parse_letter(answer: str | None) -> str | None:
    """The option letter an answer settles on, or ``None`` when there is none."""
    if not answer:
        return None
    found = [next(g for g in m.groups() if g) for m in _LETTER_RE.finditer(answer)]
    return found[-1].upper() if found else None


class McqReasonedContext(BaseMetricContext):
    """What a reasoned sample is scored against; copied into the completion record."""

    item_id: str
    order: int
    gold_letter: str
    abstention_letter: str | None = None


class LetterAccuracy(BaseMetric[Completion]):
    """Scores a generated answer by the letter it settles on, and records abstention and trace length."""

    NAME = "Letter Accuracy"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]
        truncated = response.raw_completion_finish_reason == "length"
        # A trace cut at the token cap never reaches an answer: vLLM then returns the whole trace as
        # the content, which must not be scraped for a letter.
        letter = None if truncated else parse_letter(response.completion)
        gold = response.ground_truth if isinstance(response.ground_truth, str) else None
        abstention = getattr(response.context, "abstention_letter", None)
        reasoning = response.raw_completion_reasoning or (response.completion if truncated else "") or ""
        # vLLM does not report reasoning tokens separately; the answer after the trace is a few
        # tokens, so the total generated count is a close proxy.
        reasoning_tokens = response.raw_completion_reasoning_num_tokens
        if reasoning_tokens is None:
            reasoning_tokens = response.raw_completion_num_tokens
        return [
            MetricResult(
                metric_name=self.NAME, value=float(letter is not None and letter == gold), higher_is_better=True
            ),
            MetricResult(
                metric_name="Abstention",
                value=float(letter is not None and abstention is not None and letter == abstention),
                higher_is_better=False,
            ),
            MetricResult(metric_name="Unparseable", value=float(letter is None), higher_is_better=False),
            MetricResult(metric_name="Truncated", value=float(truncated), higher_is_better=False),
            MetricResult(
                metric_name="Reasoning Tokens",
                value=float(reasoning_tokens) if reasoning_tokens is not None else None,
                higher_is_better=False,
            ),
            MetricResult(metric_name="Reasoning Chars", value=float(len(reasoning)), higher_is_better=False),
        ]


class MCReasonedStyle(MCStyle):
    """The lettered prompt of ``MCStyle`` plus the instruction line, answered generatively.

    No assistant cue: the chat template's assistant turn (with the model's thinking block) plays that
    role. The ground truth is the bare gold letter, which :class:`LetterAccuracy` compares with the
    letter parsed from the answer.
    """

    response_type = ResponseType.COMPLETION
    metrics: list[type[BaseMetric[Any]]] = [LetterAccuracy]

    def get_instruction_text(self, raw_question: str, choices: list[str]) -> str:
        return f"{format_mc_prompt(self.get_question_text(raw_question), choices)}\n{INSTRUCTION}"

    def get_cue_text(self) -> str:
        return ""

    def get_ground_truth(self, choices: list[str], correct_index: int) -> str:
        return get_n_letters(len(choices))[correct_index]

    def get_possible_completions(  # type: ignore[override]  # generative: nothing to score
        self, choices: list[str], correct_index: int | None = None
    ) -> list[str] | None:
        return None


class NonexistentEntitiesMCQReasoned(LocalJsonlMcqTask):
    """Four-option items, every one of the 24 option orders, answered with reasoning."""

    NAME = "NonexistentEntitiesMCQReasoned"
    PROMPTS_FILENAME = _ITEMS_FILENAME
    TASK_STYLER = MCReasonedStyle()
    OPTION_ORDERS = "all"

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot=num_fewshot)
        self.max_tokens = MAX_TOKENS

    def _get_context(self, item: dict[str, Any]) -> McqReasonedContext:
        letters = get_n_letters(len(item["choices"]))
        abstention = item.get("abstention_index")
        return McqReasonedContext(
            item_id=str(item["id"]),
            order=int(item.get("order", 0)),
            gold_letter=letters[int(item["correct_index"])],
            abstention_letter=letters[int(abstention)] if abstention is not None else None,
        )


class NonexistentEntitiesMCQReasonedIdk(NonexistentEntitiesMCQReasoned):
    """The same with the visible fifth option ``I don't know``, every one of the 120 orders."""

    NAME = "NonexistentEntitiesMCQReasonedIdk"
    ABSTENTION_OPTION = IDK_OPTION


def register_tasks(registry: Registry) -> None:
    """Register this module's tasks with an eval-framework registry."""
    register_task(NonexistentEntitiesMCQReasoned, registry=registry)
    register_task(NonexistentEntitiesMCQReasonedIdk, registry=registry)
