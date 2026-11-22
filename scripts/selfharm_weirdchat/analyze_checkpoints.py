"""Unsafe rate of the 100 WeirdChat self-harm prompts across Olmo 3 32B training checkpoints.

Run with: uv run --with matplotlib --with scienceplots --with scipy --with statsmodels \
    python scripts/selfharm_weirdchat/analyze_checkpoints.py <sweep_dir> <outdir>
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze import BEHAVIORS, COLORS, RNG, cluster_bootstrap, save
from scipy.stats import spearmanr

# Training order; the first seven are pre-training checkpoints prompted few-shot.
CHECKPOINTS = [
    "S1-136k", "S1-270k", "S1-398k", "S1-528k", "S1-656k", "S2-final", "S3-base",
    "SFT-5k", "SFT-final", "DPO", "RL-100", "RL-400", "Think",
]  # fmt: skip
N_PRETRAIN = 7
# SFT-final is an equal-weight merge of two learning-rate runs, so SFT-5k pools both runs' step-5000 checkpoints
# (5 samples each per prompt): the per-prompt rate is the average of the two runs.
SFT_5K_RUNS = ["SFT-5k-5e-5", "SFT-5k-1e-4"]
PRE, POST = "#2a78d6", "#eb6834"


def load(sweep: Path) -> pd.DataFrame:
    files = [(c, sweep / f"{c}.jsonl") for c in CHECKPOINTS] + [("SFT-5k", sweep / f"{r}.jsonl") for r in SFT_5K_RUNS]
    rows = [{**json.loads(line), "checkpoint": c, "source": f.stem} for c, f in files if f.exists() for line in open(f)]
    df = pd.DataFrame(rows)
    df["unsafe"] = df.unsafe.where(df.judged).astype(float)
    return df


def stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_prompt = df.groupby(["checkpoint", "prompt_id", "behavior_id"]).unsafe.mean().reset_index()
    rows = []
    groups = [*per_prompt.groupby(["checkpoint", "behavior_id"])]
    groups += [((c, "all"), g) for c, g in per_prompt.groupby("checkpoint")]
    for (c, b), g in groups:
        raw = df[(df.checkpoint == c) & ((df.behavior_id == b) | (b == "all"))]
        rows.append(
            dict(
                checkpoint=c,
                behavior=b,
                n_prompts=len(g),
                n_rows=len(raw),
                n_unjudged=int((~raw.judged).sum()),
                n_hit_token_cap=int((raw.finish_reason == "length").sum()),
                rate=g.unsafe.mean(),
                ci=cluster_bootstrap(g.unsafe.dropna()),
            )
        )
    return pd.DataFrame(rows), per_prompt


def plot_overall(s: pd.DataFrame, out: Path) -> None:
    a = s[s.behavior == "all"].set_index("checkpoint").reindex(CHECKPOINTS).dropna(subset=["rate"])
    x = np.array([CHECKPOINTS.index(c) for c in a.index])
    lo, hi = np.array(a.ci.tolist()).T
    fig, ax = plt.subplots(figsize=(6, 3))
    phases = [(x < N_PRETRAIN, PRE, "Pre-training (few-shot)"), (x >= N_PRETRAIN, POST, "Post-training (chat)")]
    for mask, color, label in phases:
        ax.errorbar(x[mask], a.rate[mask], yerr=[a.rate[mask] - lo[mask], hi[mask] - a.rate[mask]],
                    fmt="o-", color=color, capsize=2, ms=4, lw=1, label=label)  # fmt: skip
    ax.axvline(N_PRETRAIN - 0.5, color="0.6", ls="--", lw=0.8)
    ax.set_xticks(range(len(CHECKPOINTS)), CHECKPOINTS, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Unsafe rate")
    ax.legend(loc="lower right", frameon=False)
    save(fig, out, "fig5_checkpoint_unsafe_rate")


def plot_by_behavior(s: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for b in BEHAVIORS:
        g = s[s.behavior == b].set_index("checkpoint").reindex(CHECKPOINTS).dropna(subset=["rate"])
        x = np.array([CHECKPOINTS.index(c) for c in g.index])
        # Separate segments so lines do not join across the change of prompting format.
        for i, mask in enumerate([x < N_PRETRAIN, x >= N_PRETRAIN]):
            label = f"{BEHAVIORS[b]} ($n$={int(g.n_prompts.max())})" if i == 0 else None
            ax.plot(x[mask], g.rate[mask], "o-", color=COLORS[b], ms=3, lw=1, label=label)
    ax.axvline(N_PRETRAIN - 0.5, color="0.6", ls="--", lw=0.8)
    ax.set_xticks(range(len(CHECKPOINTS)), CHECKPOINTS, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Unsafe rate")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    save(fig, out, "fig6_checkpoint_unsafe_rate_by_behavior")


def spearman_ci(x: pd.Series, y: pd.Series, n_boot: int = 10_000) -> tuple[float, float, float]:
    """Spearman rho of x and y over prompts, with a 95% CI from resampling prompts."""
    x, y = x.to_numpy(), y.to_numpy()
    idx = RNG.integers(0, len(x), (n_boot, len(x)))
    boots = [spearmanr(x[i], y[i]).statistic for i in idx]
    return spearmanr(x, y).statistic, *np.nanpercentile(boots, [2.5, 97.5])


def predictors_of_think(wide: pd.DataFrame, cloze: pd.DataFrame) -> pd.DataFrame:
    """Per pre-training checkpoint: how well each measure ranks prompts like Think's unsafe rate."""
    comply = cloze.pivot(index="prompt_id", columns="checkpoint", values="pmi_margin").mul(-1)
    rows = []
    for c in CHECKPOINTS[:N_PRETRAIN]:
        for measure, table in [("Generation unsafe rate", wide), ("Cloze compliance margin", comply)]:
            if c in table:
                joined = pd.concat([table[c], wide["Think"]], axis=1, join="inner").dropna()
                rho, lo, hi = spearman_ci(joined.iloc[:, 0], joined.iloc[:, 1])
                rows.append(dict(checkpoint=c, measure=measure, rho=rho, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def plot_predictors(pred: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    for (measure, g), color, dx in zip(pred.groupby("measure", sort=False), [PRE, POST], [-0.1, 0.1]):
        x = np.array([CHECKPOINTS.index(c) for c in g.checkpoint]) + dx
        ax.errorbar(x, g.rho, yerr=[g.rho - g.lo, g.hi - g.rho], fmt="o", color=color, capsize=2, ms=4, label=measure)
    ax.axhline(0, color="0.6", ls="--", lw=0.8)
    ax.set_xticks(range(N_PRETRAIN), CHECKPOINTS[:N_PRETRAIN], rotation=45, ha="right")
    ax.set_ylabel(r"Spearman $\rho$ with Think")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    save(fig, out, "fig7_base_predictors_of_think")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--cloze", type=Path, help="cloze_base.py output, for the base-checkpoint predictor figure")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = load(args.sweep)
    s, per_prompt = stats(df)
    plot_overall(s, args.out)
    plot_by_behavior(s, args.out)
    wide = per_prompt.pivot(index="prompt_id", columns="checkpoint", values="unsafe")
    corr = {c: spearmanr(wide[c], wide["Think"], nan_policy="omit") for c in wide if "Think" in wide and c != "Think"}
    summary = dict(
        overall=s[s.behavior == "all"].drop(columns="behavior").to_dict(orient="records"),
        by_behavior=s[s.behavior != "all"].to_dict(orient="records"),
        sft_5k_runs=df[df.checkpoint == "SFT-5k"].groupby("source").unsafe.mean().to_dict(),
        spearman_vs_think={c: dict(rho=r.statistic, p=r.pvalue) for c, r in corr.items()},
    )
    if args.cloze:
        cloze = pd.read_csv(args.cloze)
        pred = predictors_of_think(wide, cloze)
        plot_predictors(pred, args.out)
        summary["base_predictors_of_think"] = pred.to_dict(orient="records")
        summary["cloze_mean_pmi_margin"] = cloze.groupby("checkpoint").pmi_margin.mean().to_dict()
        print(pred.to_string(index=False))
    (args.out / "checkpoint_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    wide.reindex(columns=[c for c in CHECKPOINTS if c in wide]).to_csv(args.out / "per_prompt_by_checkpoint.csv")
    print(s[s.behavior == "all"].drop(columns="behavior").to_string(index=False))


if __name__ == "__main__":
    main()
