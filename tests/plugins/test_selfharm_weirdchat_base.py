import pytest

pytest.importorskip("eval_framework", reason="requires the `eval` extra (eval-framework[openai])")

from eval_framework.tasks.registry import Registry  # noqa: E402
from template_formatting.formatter import Role  # noqa: E402

from abstention_over_training_evals.plugins.medical_advice_base.medical_advice_base import (  # noqa: E402
    STOP_SEQUENCES,
    PromptOfInterestJudgeMetric,
)
from abstention_over_training_evals.plugins.selfharm_weirdchat.selfharm_weirdchat import (  # noqa: E402
    SelfHarmWeirdChat,
)
from abstention_over_training_evals.plugins.selfharm_weirdchat_base.selfharm_weirdchat_base import (  # noqa: E402
    FRAMING,
    MAX_TOKENS,
    SelfHarmWeirdChatBase,
    register_tasks,
)


def _task(num_fewshot=3):
    return SelfHarmWeirdChatBase.with_overwrite(num_fewshot=num_fewshot, custom_subjects=None, custom_hf_revision=None)


def test_register_tasks_adds_the_task():
    registry = Registry()
    register_tasks(registry)
    assert "SelfHarmWeirdChatBase" in registry


def test_evaluates_the_same_prompts_as_the_final_checkpoint_task():
    assert SelfHarmWeirdChatBase.PROMPTS_FILENAME == SelfHarmWeirdChat.PROMPTS_FILENAME


def test_shipped_fewshot_transcript_is_framing_demos_then_prompt():
    samples = list(_task().iterate_samples())
    assert len(samples) == 100
    messages = samples[0].messages
    assert (messages[0].role, messages[0].content) == (Role.SYSTEM, FRAMING)
    assert [m.role for m in messages[1:]] == [Role.USER, Role.ASSISTANT] * 3 + [Role.USER]
    assert all(s.messages[:-1] == messages[:-1] for s in samples)


def test_generation_settings_and_judge():
    task = _task()
    assert task.max_tokens == MAX_TOKENS
    assert task.stop_sequences == STOP_SEQUENCES
    assert task.METRICS == [PromptOfInterestJudgeMetric]


def test_demonstrations_do_not_repeat_evaluated_prompts():
    assert len(_task()._demos()) == 3
