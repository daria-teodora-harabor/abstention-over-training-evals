import pytest

pytest.importorskip("eval_framework", reason="requires the `eval` extra (eval-framework[openai])")

from eval_framework.tasks.registry import Registry  # noqa: E402

from abstention_over_training_evals.plugins.medical_advice.medical_advice import (  # noqa: E402
    MedicalAdvice,
    register_tasks,
)


def test_register_tasks_adds_the_medical_advice_task():
    registry = Registry()
    register_tasks(registry)
    assert "MedicalAdvice" in registry


def test_task_yields_single_turn_samples_without_ground_truth(tmp_path, monkeypatch):
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(
        '{"id": "a", "prompt": "a medical prompt", "provenance": {"author_type": "human"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AOTE_PROMPTS_PATH__MedicalAdvice", str(prompts_file))

    task = MedicalAdvice.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)
    samples = list(task.iterate_samples())
    assert len(samples) == 1
    sample = samples[0]
    assert sample.ground_truth is None
    assert sample.messages[-1].content == "a medical prompt"
    assert sum(1 for message in sample.messages if message.role.name == "USER") == 1


def test_metadata_includes_provenance_summary(tmp_path, monkeypatch):
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(
        '{"id": "h", "prompt": "human one", "provenance": {"author_type": "human"}}\n'
        '{"id": "l", "prompt": "llm one", "provenance": {"author_type": "llm", '
        '"generator_model": "OLMo-2-1124-7B-Instruct", "generator_license": "apache-2.0"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AOTE_PROMPTS_PATH__MedicalAdvice", str(prompts_file))

    task = MedicalAdvice.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)
    metadata = task.get_metadata()
    assert metadata["dataset_format"] == "local-jsonl"
    assert metadata["prompt_count"] == "2"
    assert set(metadata["prompt_author_types"]) == {"human", "llm"}
    assert "OLMo-2-1124-7B-Instruct" in metadata["prompt_generator_models"]
    assert "apache-2.0" in metadata["prompt_generator_licenses"]
    assert str(metadata["dataset_path"]).endswith("prompts.jsonl")


def test_sample_rows_carry_provenance(tmp_path, monkeypatch):
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(
        '{"id": "h", "prompt": "human one", "provenance": {"author_type": "human"}}\n'
        '{"id": "l", "prompt": "llm one", "provenance": {"author_type": "llm", '
        '"generator_model": "OLMo-2-1124-7B-Instruct", "generator_license": "apache-2.0"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AOTE_PROMPTS_PATH__MedicalAdvice", str(prompts_file))

    task = MedicalAdvice.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)
    task._load_dataset("default")
    rows = task.dataset[task.SAMPLE_SPLIT]
    assert rows[1]["provenance"]["author_type"] == "llm"


def test_empty_dataset_raises_an_actionable_error(tmp_path, monkeypatch):
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("AOTE_PROMPTS_PATH__MedicalAdvice", str(prompts_file))

    task = MedicalAdvice.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)
    with pytest.raises(ValueError, match="contains no prompts"):
        task._load_dataset("default")
