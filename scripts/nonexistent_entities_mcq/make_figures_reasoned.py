#!/usr/bin/env python3
r"""Figures for the reasoned MCQ runs: a reasoned-only set and a comparison set against the sweep.

    uv run --with matplotlib python scripts/nonexistent_entities_mcq/make_figures_reasoned.py \
        --reasoned reasoned_summary.json --immediate summary.json --out-dir figures

Reasoned-only set (``r1``..``r3``): accuracy, abstention and unparseable rate per checkpoint; trace
length per checkpoint, split by correct and wrong answers; per-category accuracy. Comparison set
(``c1``..``c2``): immediate (log-likelihood) versus reasoned accuracy on the same checkpoints, with
the paired per-item difference; a per-item scatter for the final Think model. Error bars are 95%
bootstrap intervals over items.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = [
    ("allenai_Olmo-3-32B-Think-SFT@5e-5-step5000", "SFT 5k"),
    ("allenai_Olmo-3-32B-Think-SFT@main", "SFT final"),
    ("allenai_Olmo-3-32B-Think-DPO@main", "DPO"),
    ("allenai_Olmo-3-32B-Think@step_100", "RL 100"),
    ("allenai_Olmo-3-32B-Think@step_400", "RL 400"),
    ("allenai_Olmo-3-32B-Think@main", "Think"),
]
TASKS = [
    ("NonexistentEntitiesMCQReasoned", "NonexistentEntitiesMCQ", "4 options", "#2a78d6", "o", 0.25),
    ("NonexistentEntitiesMCQReasonedIdk", "NonexistentEntitiesMCQIdk", "5 options with IDK", "#eb6834", "s", 0.20),
]
CATEGORIES = {
    "pharma": "Prescription drugs",
    "peds": "Paediatric dosing",
    "proc": "Procedures & scores",
    "osha": "Workplace safety",
    "elec": "Electrical code",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
IMMEDIATE_COLOR, REASONED_COLOR = "#888780", "#1baf7a"


def style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_color(INK2)
    ax.title.set_color(INK)
    ax.grid(axis="y", color=GRID, zorder=0)


def _xaxis(ax: plt.Axes) -> None:
    ax.set_xticks(range(len(ORDER)), [n for _, n in ORDER], rotation=25, ha="right")
    ax.set_xlim(-0.5, len(ORDER) - 0.5)


def _series(summary: dict, task: str, subject: str, key: str, ci_key: str | None):
    xs, ys, lo, hi = [], [], [], []
    for i, (label, _) in enumerate(ORDER):
        s = summary.get(task, {}).get(label, {}).get(subject)
        if not s or s.get(key) is None:
            continue
        xs.append(i)
        ys.append(s[key])
        ci = s.get(ci_key) if ci_key else None
        lo.append(max(0.0, s[key] - ci[0]) if ci else 0.0)
        hi.append(max(0.0, ci[1] - s[key]) if ci else 0.0)
    return xs, ys, [lo, hi]


def _plot(ax, xs, ys, err, **kw):
    ax.errorbar(xs, ys, yerr=err, lw=2, ms=7, mec="white", mew=1.2, capsize=2.5, elinewidth=0.9, zorder=3, **kw)


def fig_r1(reasoned: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=160)
    for task, _, name, color, marker, chance in TASKS:
        for subject, ls, suffix in (("main", "-", ", fake"), ("control", (0, (4, 2)), ", controls")):
            xs, ys, err = _series(reasoned, task, subject, "accuracy", "accuracy_ci95")
            if xs:
                _plot(axes[0], xs, ys, err, color=color, ls=ls, marker=marker, label=name + suffix)
            xs, ys, err = _series(reasoned, task, subject, "abstention_rate", None)
            if xs:
                _plot(axes[1], xs, ys, err, color=color, ls=ls, marker=marker, label=name + suffix)
            xs, ys, err = _series(reasoned, task, subject, "unparseable_rate", None)
            if xs:
                _plot(axes[2], xs, ys, err, color=color, ls=ls, marker=marker, label=name + suffix)
    for ax, title, ylabel in (
        (axes[0], "Accuracy with reasoning", "share of answers on the gold letter"),
        (axes[1], "Abstention with reasoning", "share answering “I don't know”"),
        (axes[2], "Answers with no parseable letter", "share"),
    ):
        _xaxis(ax)
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel(ylabel)
        style(ax)
    axes[0].axhline(0.25, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.text(0.99, 0.005, "error bars: 95% bootstrap over items", fontsize=7, color=INK2, ha="right")
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_r2(reasoned: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=160)
    for task, _, name, color, marker, _ in TASKS:
        xs, med, _ = _series(reasoned, task, "main", "reasoning_tokens_median", None)
        if xs:
            ax.plot(
                xs, med, color=color, lw=2, marker=marker, ms=7, mec="white", mew=1.2, label=f"{name}: median", zorder=3
            )
        xs, ok, _ = _series(reasoned, task, "main", "reasoning_tokens_mean_correct", None)
        xs2, bad, _ = _series(reasoned, task, "main", "reasoning_tokens_mean_wrong", None)
        if xs:
            ax.plot(
                xs,
                ok,
                color=color,
                lw=1.2,
                ls=(0, (4, 2)),
                marker=marker,
                ms=5,
                label=f"{name}: mean, correct answers",
                zorder=2,
            )
        if xs2:
            ax.plot(
                xs2,
                bad,
                color=color,
                lw=1.2,
                ls=(0, (1, 2)),
                marker=marker,
                ms=5,
                label=f"{name}: mean, wrong answers",
                zorder=2,
            )
    _xaxis(ax)
    ax.set_ylim(0, None)
    ax.set_ylabel("reasoning tokens (fake items)")
    ax.set_title("How long the model reasons before answering", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_r3(reasoned: dict, out: Path) -> None:
    """Per-category accuracy of the final Think model, reasoned, by task."""
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=160)
    width = 0.38
    for j, (task, _, name, color, _, _) in enumerate(TASKS):
        s = reasoned.get(task, {}).get("allenai_Olmo-3-32B-Think@main", {}).get("main")
        if not s:
            continue
        items = s["item_accuracy"]
        values = []
        for cat in CATEGORIES:
            accs = [v for k, v in items.items() if k.startswith(cat + "-")]
            values.append(sum(accs) / len(accs) if accs else 0.0)
        xs = [i + (j - 0.5) * width for i in range(len(CATEGORIES))]
        ax.bar(xs, values, width=width, color=color, label=name, zorder=2)
        for x, v in zip(xs, values):
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(CATEGORIES)), list(CATEGORIES.values()), fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy, fake items")
    ax.set_title("Think final, reasoned: accuracy by category (9 fake items each)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_c1(reasoned: dict, immediate: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), dpi=160, sharex=True)
    for ax, subject, title in (
        (axes[0], "main", "Fake entities: immediate answer vs reasoned answer (45 items)"),
        (axes[1], "control", "Real entities (controls): immediate vs reasoned (5 items)"),
    ):
        for task, imm_task, name, color, marker, chance in TASKS:
            xs, ys, err = _series(reasoned, task, subject, "accuracy", "accuracy_ci95")
            if xs:
                _plot(ax, xs, ys, err, color=color, marker=marker, label=f"{name}, reasoned")
            xs2, ys2, err2 = _series(immediate, imm_task, subject, "accuracy_all_orders", "accuracy_ci95")
            if xs2:
                _plot(
                    ax,
                    xs2,
                    ys2,
                    err2,
                    color=color,
                    marker=marker,
                    ls=(0, (4, 2)),
                    alpha=0.55,
                    label=f"{name}, immediate",
                )
        ax.axhline(0.25, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.axhline(0.20, color=INK2, lw=0.8, ls=(0, (1, 2)), zorder=1)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("accuracy over all option orders")
        ax.set_title(title, fontsize=10, loc="left")
        style(ax)
    _xaxis(axes[1])
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
    fig.text(
        0.99,
        0.005,
        "error bars: 95% bootstrap over items; dashed = immediate (log-likelihood), solid = reasoned",
        fontsize=7,
        color=INK2,
        ha="right",
    )
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_c2(reasoned: dict, immediate: dict, out: Path) -> None:
    """Paired per-item difference (reasoned minus immediate) per checkpoint, and the Think scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=160, gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    for task, _, name, color, marker, _ in TASKS:
        xs, ys, err = _series(reasoned, task, "main", "paired_difference", "paired_difference_ci95")
        if xs:
            _plot(ax, xs, ys, err, color=color, marker=marker, label=name)
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    _xaxis(ax)
    ax.set_ylim(-0.3, 0.6)
    ax.set_ylabel("reasoned minus immediate (fake items)")
    ax.set_title("Paired gain from reasoning, same items and orders", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    ax = axes[1]
    label = "allenai_Olmo-3-32B-Think@main"
    for task, imm_task, name, color, marker, _ in TASKS:
        r = reasoned.get(task, {}).get(label, {}).get("main")
        i = immediate.get(imm_task, {}).get(label, {}).get("main")
        if not r or not i:
            continue
        shared = [k for k in r["item_accuracy"] if k in i.get("item_accuracy", {})]
        ax.scatter(
            [i["item_accuracy"][k] for k in shared],
            [r["item_accuracy"][k] for k in shared],
            s=26,
            color=color,
            marker=marker,
            alpha=0.8,
            label=name,
            zorder=3,
            edgecolors="white",
            linewidths=0.6,
        )
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("immediate answer: share of orders correct")
    ax.set_ylabel("reasoned answer: share of orders correct")
    ax.set_title("Think final, per fake item", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax)
    ax.grid(axis="x", color=GRID, zorder=0)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reasoned", type=Path, required=True, help="reasoned_summary.json")
    ap.add_argument("--immediate", type=Path, default=None, help="the sweep's summary.json, for the comparison set")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    reasoned = json.loads(args.reasoned.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_r1(reasoned, args.out_dir / "r1_reasoned_accuracy_abstention.png")
    fig_r2(reasoned, args.out_dir / "r2_reasoning_length.png")
    fig_r3(reasoned, args.out_dir / "r3_think_by_category.png")
    written = 3
    if args.immediate:
        immediate = json.loads(args.immediate.read_text(encoding="utf-8"))
        fig_c1(reasoned, immediate, args.out_dir / "c1_immediate_vs_reasoned.png")
        fig_c2(reasoned, immediate, args.out_dir / "c2_paired_difference.png")
        written += 2
    print(f"wrote {written} figures to {args.out_dir}")


if __name__ == "__main__":
    main()
