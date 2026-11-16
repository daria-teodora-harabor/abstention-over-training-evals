#!/usr/bin/env python3
"""Check the adapter's batched single-token scoring against eval-framework's per-completion path.

Runs on the GPU machine before a sweep (a minute, no 32B download): loads a small public model, scores
the first prompts of the lettered tasks both ways, and fails if any letter's log-likelihood differs by
more than ``--tolerance`` or the argmax letter differs anywhere.

    uv run python scripts/nonexistent_entities_mcq/check_fast_logprobs.py [--model HuggingFaceTB/SmolLM-135M]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--samples", type=int, default=40, help="Prompts per task to compare.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Max |difference| in log-likelihood.")
    args = parser.parse_args(argv)

    import torch
    from eval_framework.llm.huggingface import BaseHFLLM
    from template_formatting.formatter import ConcatFormatter

    from abstention_over_training_evals.adapters.olmo_hf import fast_single_token_logprobs
    from abstention_over_training_evals.plugins.nonexistent_entities_mcq.nonexistent_entities_mcq import (
        NonexistentEntitiesMCQ,
        NonexistentEntitiesMCQIdk,
    )

    class SmallModel(BaseHFLLM):
        LLM_NAME = args.model
        DEFAULT_FORMATTER = ConcatFormatter

    llm = SmallModel()
    llm.model = llm.model.float()  # float32 so the two paths differ only by batching, not precision
    worst = 0.0
    argmax_mismatches = 0
    compared = 0
    for cls in (NonexistentEntitiesMCQ, NonexistentEntitiesMCQIdk):
        task = cls.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)
        samples = []
        for sample in task.iterate_samples():
            samples.append(sample)
            if len(samples) == args.samples:
                break
        slow = BaseHFLLM.logprobs(llm, samples)
        fast = fast_single_token_logprobs(llm, samples, batch_size=args.batch_size)
        for s, f in zip(slow, fast):
            assert s.loglikelihoods.keys() == f.loglikelihoods.keys(), "completion sets differ"
            for key in s.loglikelihoods:
                worst = max(worst, abs(s.loglikelihoods[key] - f.loglikelihoods[key]))
            if max(s.loglikelihoods, key=s.loglikelihoods.get) != max(f.loglikelihoods, key=f.loglikelihoods.get):
                argmax_mismatches += 1
            compared += 1
    print(f"{args.model}: {compared} prompts compared, worst |diff| {worst:.2e}, argmax mismatches {argmax_mismatches}")
    ok = worst <= args.tolerance and argmax_mismatches == 0
    print("OK" if ok else "FAILED")
    del llm
    torch.cuda.empty_cache()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
