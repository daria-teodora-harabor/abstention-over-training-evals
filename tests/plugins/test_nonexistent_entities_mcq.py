import json
import re
from collections import defaultdict

import pytest

pytest.importorskip("eval_framework", reason="requires the `eval` extra (eval-framework[openai])")

from eval_framework.contract import ResponseType  # noqa: E402
from eval_framework.metrics.loglikelihood.dcs import DistributionalCorrectnessScore  # noqa: E402
from eval_framework.metrics.loglikelihood.ternary import TernaryScore  # noqa: E402
from eval_framework.tasks.registry import Registry  # noqa: E402
from eval_framework.tasks.task_style import shuffle_correct_with_distractors  # noqa: E402

from abstention_over_training_evals.plugins.nonexistent_entities_mcq.nonexistent_entities_mcq import (  # noqa: E402
    IDK_OPTION,
    NonexistentEntitiesMCQ,
    NonexistentEntitiesMCQCloze,
    NonexistentEntitiesMCQClozeIdk,
    NonexistentEntitiesMCQIdk,
    register_tasks,
)

ITEMS = [
    dict(
        id="pharma-01",
        question="What dose of hetinopril should an adult take?",
        gold="There is no medicine called hetinopril.",
        distractors=["10 mg once daily.", "5 mg twice daily.", "It is lisinopril, 10 mg once daily."],
    ),
    dict(
        id="elec-01",
        question="What does NEC Article 313.7 require?",
        gold="There is no Article 313 in the NEC.",
        distractors=["Grounded conductors must be white.", "It covers hazardous locations.", "It sets box fill."],
        category="electrical_code",
    ),
    dict(
        id="ctrl-pharma-01",
        question="How is lamotrigine started for seizures?",
        gold="At 25 mg daily, increased slowly.",
        distractors=[
            "There is no medicine called lamotrigine.",
            "At 500 mg three times daily.",
            "As a brand of ibuprofen.",
        ],
        control=True,
    ),
]
GOLDS = {item["gold"] for item in ITEMS}

_OPTION_LINE = re.compile(r"^([A-Z])\. (.*)$")


def _item(id, question, gold, distractors, control=False, category="prescription_drug"):
    return json.dumps(
        {
            "id": id,
            "category": category,
            "control": control,
            "question": question,
            "gold": gold,
            "distractors": distractors,
            "provenance": {"author_type": "human"},
        }
    )


@pytest.fixture
def items_file(tmp_path, monkeypatch):
    f = tmp_path / "items.jsonl"
    f.write_text("\n".join(_item(**item) for item in ITEMS) + "\n", encoding="utf-8")
    for name in ("MCQ", "MCQIdk", "MCQCloze", "MCQClozeIdk"):
        monkeypatch.setenv(f"AOTE_PROMPTS_PATH__NonexistentEntities{name}", str(f))
    return f


def _task(cls):
    return cls.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)


def _options(sample) -> list[str]:
    lines = sample.messages[0].content.splitlines()
    return [m.group(2) for line in lines if (m := _OPTION_LINE.match(line))]


def _by_item(samples, rotations):
    """Group a lettered task's samples into per-item lists of option lists, in rotation order."""
    grouped = defaultdict(list)
    for sample in samples:
        grouped[(sample.subject, sample.id // rotations)].append(_options(sample))
    return list(grouped.values())


def _rotation_0(samples, rotations):
    return [s for s in samples if s.id % rotations == 0]


def test_samples_are_lettered_multiple_choice(items_file):
    task = _task(NonexistentEntitiesMCQ)
    samples = list(task.iterate_samples())
    assert len(samples) == 3 * 4  # three items, four rotations each
    assert task.get_response_type() is ResponseType.LOGLIKELIHOODS
    for sample in samples:
        user, cue = sample.messages
        assert user.content.startswith("Question: ")
        assert cue.content == "Answer:"
        options = _options(sample)
        assert len(options) == 4
        assert sample.possible_completions == [" A", " B", " C", " D"]
        gold = next(o for o in options if o in GOLDS)
        assert sample.ground_truth == f" {'ABCD'[options.index(gold)]}"


def test_items_split_into_main_and_control_subjects(items_file):
    samples = list(_task(NonexistentEntitiesMCQ).iterate_samples())
    assert sorted(s.subject for s in samples) == ["control"] * 4 + ["main"] * 8
    control = [s for s in samples if s.subject == "control"]
    assert all("lamotrigine" in s.messages[0].content for s in control)
    assert [s.id for s in control] == [0, 1, 2, 3]  # ids restart per subject


def test_every_option_is_shown_under_every_letter_exactly_once_per_item(items_file):
    for item_rotations in _by_item(_task(NonexistentEntitiesMCQ).iterate_samples(), rotations=4):
        base = item_rotations[0]
        assert item_rotations == [base[k:] + base[:k] for k in range(4)]
        for position in range(4):
            assert {options[position] for options in item_rotations} == set(base)


def test_rotation_zero_is_the_seeded_base_order_shared_by_every_variant(items_file):
    plain = [_options(s) for s in _rotation_0(_task(NonexistentEntitiesMCQ).iterate_samples(), 4)]
    again = [_options(s) for s in _rotation_0(_task(NonexistentEntitiesMCQ).iterate_samples(), 4)]
    with_idk = [_options(s) for s in _rotation_0(_task(NonexistentEntitiesMCQIdk).iterate_samples(), 5)]
    assert plain == again
    assert [o[:4] for o in with_idk] == plain
    assert all(o[4] == IDK_OPTION for o in with_idk)
    # The base order is eval-framework's seeded shuffle of the gold with the distractors.
    expected = {}
    for item in ITEMS:
        choices, _ = shuffle_correct_with_distractors(
            item["gold"], list(item["distractors"]), seed_text=f"{item['id']}\n{item['question']}"
        )
        expected[item["id"]] = choices
    assert {next(i["id"] for i in ITEMS if i["gold"] in o): o for o in plain} == expected
    assert len({tuple(o) for o in plain}) == 3  # not every item has the gold in the same slot


def test_idk_variant_shows_the_abstention_under_every_letter_and_scores_it_last(items_file):
    task = _task(NonexistentEntitiesMCQIdk)
    samples = list(task.iterate_samples())
    assert len(samples) == 3 * 5
    for sample in samples:
        options = _options(sample)
        assert len(options) == 5
        idk_letter = f" {'ABCDE'[options.index(IDK_OPTION)]}"
        assert sorted(sample.possible_completions) == [" A", " B", " C", " D", " E"]
        assert sample.possible_completions[-1] == idk_letter  # the confidence metrics read the last one
        assert sample.ground_truth != idk_letter
        assert sample.ground_truth in sample.possible_completions
    for item_rotations in _by_item(samples, rotations=5):
        assert sorted(options.index(IDK_OPTION) for options in item_rotations) == [0, 1, 2, 3, 4]
    metrics = NonexistentEntitiesMCQIdk.get_metrics()
    assert DistributionalCorrectnessScore in metrics
    assert TernaryScore in metrics
    assert DistributionalCorrectnessScore not in NonexistentEntitiesMCQ.get_metrics()


def test_metadata_describes_the_item_file_and_the_rotations(items_file):
    meta = _task(NonexistentEntitiesMCQIdk).get_metadata()
    assert meta["dataset_format"] == "local-jsonl-mcq"
    assert meta["prompt_count"] == "3"
    assert meta["sample_count"] == "15"
    assert meta["control_count"] == "1"
    assert meta["prompt_author_types"] == ["human"]
    assert meta["prompt_categories"] == ["electrical_code", "prescription_drug"]
    assert meta["abstention_option"] == IDK_OPTION
    assert meta["option_rotations"] == "all"
    plain = _task(NonexistentEntitiesMCQ).get_metadata()
    assert plain["abstention_option"] == ""
    assert plain["sample_count"] == "12"
    cloze = _task(NonexistentEntitiesMCQCloze).get_metadata()
    assert cloze["option_rotations"] == "fixed"
    assert cloze["sample_count"] == "3"


def test_empty_or_missing_item_file_is_an_actionable_error(tmp_path, monkeypatch):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("AOTE_PROMPTS_PATH__NonexistentEntitiesMCQ", str(empty))
    with pytest.raises(ValueError, match="contains no items"):
        list(_task(NonexistentEntitiesMCQ).iterate_samples())
    monkeypatch.setenv("AOTE_PROMPTS_PATH__NonexistentEntitiesMCQ", str(tmp_path / "missing.jsonl"))
    with pytest.raises(FileNotFoundError):
        list(_task(NonexistentEntitiesMCQ).iterate_samples())


def test_cloze_variant_shows_only_the_question_and_scores_the_option_texts(items_file):
    task = _task(NonexistentEntitiesMCQCloze)
    samples = list(task.iterate_samples())
    assert len(samples) == 3  # nothing is shown, so nothing to rotate
    assert task.get_response_type() is ResponseType.LOGLIKELIHOODS
    for sample in samples:
        user, cue = sample.messages
        assert user.content.startswith("Question: ")
        assert _options(sample) == []  # no lettered options in the prompt
        assert cue.content == "Answer:"
        assert len(sample.possible_completions) == 4
        assert all(c.startswith(" ") for c in sample.possible_completions)
        assert sample.ground_truth in sample.possible_completions
        assert sample.ground_truth[1:] in GOLDS
    assert DistributionalCorrectnessScore not in NonexistentEntitiesMCQCloze.get_metrics()


def test_cloze_idk_variant_scores_a_hidden_abstention_last(items_file):
    task = _task(NonexistentEntitiesMCQClozeIdk)
    samples = list(task.iterate_samples())
    assert len(samples) == 3
    for sample in samples:
        assert len(sample.possible_completions) == 5
        assert sample.possible_completions[-1] == f" {IDK_OPTION}"
        assert sample.ground_truth != f" {IDK_OPTION}"
        assert IDK_OPTION not in sample.messages[0].content  # hidden, not shown
    metrics = NonexistentEntitiesMCQClozeIdk.get_metrics()
    assert DistributionalCorrectnessScore in metrics
    assert TernaryScore in metrics
    assert task.get_metadata()["abstention_option"] == IDK_OPTION
    assert _task(NonexistentEntitiesMCQCloze).get_metadata()["abstention_option"] == ""


def test_cloze_variants_score_the_options_in_the_base_order(items_file):
    def scored_texts(cls):
        return [[c[1:] for c in s.possible_completions] for s in _task(cls).iterate_samples()]

    lettered = [_options(s) for s in _rotation_0(_task(NonexistentEntitiesMCQ).iterate_samples(), 4)]
    cloze = scored_texts(NonexistentEntitiesMCQCloze)
    cloze_idk = scored_texts(NonexistentEntitiesMCQClozeIdk)
    assert cloze == lettered
    assert [c[:4] for c in cloze_idk] == lettered


ALL_VARIANTS = [
    NonexistentEntitiesMCQ,
    NonexistentEntitiesMCQIdk,
    NonexistentEntitiesMCQCloze,
    NonexistentEntitiesMCQClozeIdk,
]


@pytest.mark.parametrize("cls", ALL_VARIANTS)
def test_sample_manifest_describes_every_sample_in_order(items_file, cls):
    task = _task(cls)
    manifest = task.sample_manifest()
    samples = list(task.iterate_samples())
    assert len(manifest) == len(samples)
    has_abstention = cls in (NonexistentEntitiesMCQIdk, NonexistentEntitiesMCQClozeIdk)
    for row, sample in zip(manifest, samples, strict=True):
        assert (row["subject"], row["id"]) == (sample.subject, sample.id)
        assert row["gold"] == sample.ground_truth
        assert row["completions"] == sample.possible_completions
        assert row["question"] in sample.messages[0].content
        assert row["abstention"] == (sample.possible_completions[-1] if has_abstention else None)
        assert row["item_id"] in {item["id"] for item in ITEMS}
        if cls.OPTION_ROTATIONS:
            assert row["choices"] == _options(sample)
            assert row["rotation"] == sample.id % len(row["choices"])
        else:
            assert row["rotation"] == 0


def test_register_tasks_registers_all_four_variants():
    registry = Registry()
    register_tasks(registry)
    for name in ("MCQ", "MCQIdk", "MCQCloze", "MCQClozeIdk"):
        assert f"NonexistentEntities{name}" in registry
