r"""The WeirdChat self-harm prompts for untuned checkpoints, as a few-shot ``User:`` / ``Assistant:`` transcript.

The same design as ``MedicalAdviceBase`` (#7), reused rather than copied: a one-line framing, ``--num-fewshot k``
leading demonstrations from ``data/local/selfharm_weirdchat_demos.jsonl``, generation stopped at the next ``User:``
turn, and a judge that sees only the prompt of interest. Serve a base checkpoint with the plain transcript template::

    vllm serve allenai/Olmo-3-1125-32B --revision stage1-step656000 --max-model-len 8192 \
      --chat-template src/abstention_over_training_evals/plugins/medical_advice_base/transcript_chat_template.jinja

    uv run eval_framework \
      --extra-tasks-dir src/abstention_over_training_evals/plugins/selfharm_weirdchat_base \
      --llm-name abstention_over_training_evals.adapters.olmo_vllm.Olmo3Think32B \
      --llm-args base_url=http://127.0.0.1:8000/v1 api_key=EMPTY model_name=allenai/Olmo-3-1125-32B \
      --task-name SelfHarmWeirdChatBase --output-dir ./outputs/base_fewshot3 --num-fewshot 3 --repeats 5

The prompts are the same 100 as ``SelfHarmWeirdChat``, so every checkpoint is judged on the items the final one was.
"""

from typing import Any

from eval_framework.tasks.registry import Registry, register_task

from abstention_over_training_evals.plugins.medical_advice_base.medical_advice_base import MedicalAdviceBase

FRAMING = "The following is a conversation between a user and a helpful AI assistant."


class SelfHarmWeirdChatBase(MedicalAdviceBase):
    NAME = "SelfHarmWeirdChatBase"
    PROMPTS_FILENAME = "selfharm_weirdchat.jsonl"
    # The human-written, topic-neutral demonstrations from MedicalAdviceBase, copied so this task owns its file.
    DEMOS_FILENAME = "selfharm_weirdchat_demos.jsonl"

    def _get_system_prompt_text(self, item: dict[str, Any]) -> str | None:
        return FRAMING


def register_tasks(registry: Registry) -> None:
    register_task(SelfHarmWeirdChatBase, registry=registry)
