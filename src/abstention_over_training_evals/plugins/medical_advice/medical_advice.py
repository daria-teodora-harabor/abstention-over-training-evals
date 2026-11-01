from eval_framework.tasks.registry import Registry, register_task

from abstention_over_training_evals.plugins.base import LocalJsonlSafetyTask


class MedicalAdvice(LocalJsonlSafetyTask):
    NAME = "MedicalAdvice"  # must be unique; names are compared case-/separator-insensitively
    PROMPTS_FILENAME = "medical_advice.jsonl"


def register_tasks(registry: Registry) -> None:
    register_task(MedicalAdvice, registry=registry)
