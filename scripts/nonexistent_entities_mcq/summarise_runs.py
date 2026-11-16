#!/usr/bin/env python3
"""Summarise NonexistentEntitiesMCQ checkpoint runs: one table per task, checkpoints as rows.

    uv run python scripts/nonexistent_entities_mcq/summarise_runs.py --output-dir outputs/nonexistent_entities_mcq

Reads ``<output-dir>/runs.jsonl`` (written by ``run_checkpoints.py``) and, for every run recorded as
``ok``, the framework's ``results.jsonl`` (per-sample metric values) and ``output.jsonl`` (per-sample
log-likelihoods). The latter is joined with the task's sample manifest (which item and option order each
sample is) to report, per checkpoint, task and subject:

- the framework's metrics averaged over all samples, i.e. over all option orders (raw and Bayesian,
  length-corrected, accuracy; DCS and Ternary on the IDK tasks);
- accuracy under order 0 only: the fixed order, what a single-order run would have reported;
- abstention rate: how often the argmax completion is "I don't know";
- consistency: the share of items whose chosen option (by content) is the same under every order;
- letter preference (lettered tasks): the distribution of the argmax letter, uniform if the model reads
  the content rather than the position.

Writes ``<output-dir>/summary.md`` and ``summary.json`` and prints the markdown.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

LETTERS = "ABCDEFGHIJ"
# Framework metrics worth a column, by their NAME; the rest stay in summary.json.
HEADLINE_METRICS = (
    "Accuracy Loglikelihood",
    "Accuracy Bayesian Loglikelihood",
    "Distributional Correctness Score",
    "Ternary Score",
)


def _task_class(task_name: str) -> Any:
    from abstention_over_training_evals.plugins.nonexistent_entities_mcq import nonexistent_entities_mcq as module

    return getattr(module, task_name)


def _manifest(task_name: str) -> dict[tuple[str, int], dict[str, Any]]:
    task = _task_class(task_name).with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)
    return {(row["subject"], row["id"]): row for row in task.sample_manifest()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarise_run(
    task_name: str,
    responses: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    manifest: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Per-subject statistics of one run from its responses, metric rows and the task's manifest."""
    lettered = "Cloze" not in task_name
    per_subject: dict[str, dict[str, Any]] = {}
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for response in responses:
        by_subject[response["subject"]].append(response)

    for subject, subject_responses in by_subject.items():
        correct: list[float] = []
        correct_order0: list[float] = []
        abstained: list[float] = []
        letters: Counter[str] = Counter()
        chosen_by_item: dict[str, list[str]] = defaultdict(list)
        errors = 0
        for response in subject_responses:
            row = manifest[(subject, response["id"])]
            loglikelihoods = response.get("loglikelihoods") or {}
            if response.get("error") or not loglikelihoods:
                errors += 1
                continue
            if row["question"] not in response.get("prompt", ""):
                raise ValueError(
                    f"{task_name} sample {subject}/{response['id']} does not match the manifest "
                    f"(item {row['item_id']}); was the item file changed after the run?"
                )
            chosen = max(loglikelihoods, key=loglikelihoods.get)
            is_correct = float(chosen == row["gold"])
            correct.append(is_correct)
            if row["order"] == 0:
                correct_order0.append(is_correct)
            abstained.append(float(row["abstention"] is not None and chosen == row["abstention"]))
            if lettered:
                letters[chosen.strip()] += 1
                content = row["choices"][LETTERS.index(chosen.strip())]
            else:
                content = chosen.strip()
            chosen_by_item[row["item_id"]].append(content)

        n_letters = sum(letters.values())
        per_subject[subject] = {
            "n_samples": len(subject_responses),
            "n_items": len(chosen_by_item),
            "errors": errors,
            "accuracy_all_orders": _mean(correct),
            "accuracy_order_0": _mean(correct_order0),
            "abstention_rate": _mean(abstained),
            "consistency": _mean([float(len(set(c)) == 1) for c in chosen_by_item.values()]),
            "letter_preference": {k: round(v / n_letters, 3) for k, v in sorted(letters.items())} if n_letters else {},
            "metrics": {},
        }

    metric_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in metric_rows:
        if row.get("value") is not None and row.get("subject") is not None and row.get("metric_name"):
            metric_values[(row["subject"], row["metric_name"])].append(float(row["value"]))
    for (subject, metric), values in metric_values.items():
        per_subject.setdefault(subject, {"metrics": {}})["metrics"][metric] = _mean(values)
    return per_subject


def summarise(output_dir: Path) -> dict[str, Any]:
    """Summaries of every ok run under ``output_dir``: ``{task: {checkpoint: {subject: stats}}}``."""
    runs = [r for r in _read_jsonl(output_dir / "runs.jsonl") if r.get("status") == "ok"]
    manifests: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    summary: dict[str, dict[str, Any]] = defaultdict(dict)
    for run in runs:
        task_name = run["task"]
        if task_name not in manifests:
            manifests[task_name] = _manifest(task_name)
        run_dir = Path(run["output_dir"])
        summary[task_name][run["checkpoint"]] = summarise_run(
            task_name,
            _read_jsonl(run_dir / "output.jsonl"),
            _read_jsonl(run_dir / "results.jsonl"),
            manifests[task_name],
        )
    return dict(summary)


def _fmt(value: float | None) -> str:
    return "–" if value is None else f"{value:.2f}"


def to_markdown(summary: dict[str, Any]) -> str:
    """Render the summary as one markdown table per task and subject."""
    lines: list[str] = []
    for task_name, by_checkpoint in summary.items():
        subjects = sorted({s for stats in by_checkpoint.values() for s in stats}, key=lambda s: s != "main")
        for subject in subjects:
            lines.append(f"### {task_name} ({subject})\n")
            lines.append(
                "| checkpoint | items | acc (all orders) | acc (order 0) | Bayesian acc | abstain | "
                "consistency | DCS | Ternary | letter preference |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
            for checkpoint, stats in by_checkpoint.items():
                s = stats.get(subject)
                if not s:
                    continue
                m = s.get("metrics", {})
                pref = " ".join(f"{k}:{v:.2f}" for k, v in s.get("letter_preference", {}).items())
                lines.append(
                    f"| {checkpoint} | {s.get('n_items', '–')} | {_fmt(s.get('accuracy_all_orders'))} | "
                    f"{_fmt(s.get('accuracy_order_0'))} | {_fmt(m.get('Accuracy Bayesian Loglikelihood'))} | "
                    f"{_fmt(s.get('abstention_rate'))} | {_fmt(s.get('consistency'))} | "
                    f"{_fmt(m.get('Distributional Correctness Score'))} | {_fmt(m.get('Ternary Score'))} | {pref} |"
                )
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs" / "nonexistent_entities_mcq")
    args = parser.parse_args(argv)
    summary = summarise(args.output_dir)
    if not summary:
        print(f"no ok runs in {args.output_dir / 'runs.jsonl'}")
        return 1
    markdown = to_markdown(summary)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
