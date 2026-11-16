"""eval-framework model adapter for an Olmo 3 32B checkpoint loaded in-process with transformers.

The multiple-choice tasks score log-likelihoods of fixed completions, which eval-framework's
OpenAI-compatible client cannot get from a vLLM server, so these runs load the weights directly. This
is eval-framework's ``HFLLM`` with three changes for the purpose:

- the weights are a Hub *revision* (an intermediate checkpoint is a branch of its stage repo), fetched
  by :mod:`abstention_over_training_evals.adapters.olmo_weights` so the runner can delete them after;
- they load in bfloat16 (eval-framework leaves the dtype to torch's default, float32, which does not
  fit a 32B model on one 80 GB GPU);
- the prompt is plain text (``ConcatFormatter``, no chat template) for every checkpoint, so the base
  and post-trained stages are scored on the same prompt.

Used from the CLI as ``--llm-name abstention_over_training_evals.adapters.olmo_hf.OlmoHFCheckpoint
--llm-args checkpoint=base revision=stage1-step72000``, or in-process by
``scripts/nonexistent_entities_mcq/run_checkpoints.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from eval_framework.llm.huggingface import BaseHFLLM
from template_formatting.formatter import ConcatFormatter
from transformers import AutoModelForCausalLM

from abstention_over_training_evals.adapters.checkpoints import OlmoCheckpoint
from abstention_over_training_evals.adapters.olmo_weights import checkpoint_label, download_weights, parse_checkpoint

logger = logging.getLogger(__name__)


class OlmoHFCheckpoint(BaseHFLLM):
    """An Olmo 3 32B checkpoint (any stage, any Hub revision) scored in-process in bfloat16.

    Args:
        checkpoint: A spec for :func:`parse_checkpoint` (``base``, ``think@step_100``, a catalogued Hub
            id) or an :class:`OlmoCheckpoint`.
        revision: Overrides the spec's revision when given.
        dtype: A torch dtype name; ``bfloat16`` by default.
        weights_dir: Where to download to; defaults to ``$AOTE_WEIGHTS_DIR``.
        bytes_per_token: eval-framework's bits-per-byte scaling override; unused by the accuracy metrics.
    """

    DEFAULT_FORMATTER = ConcatFormatter

    def __init__(
        self,
        checkpoint: str | OlmoCheckpoint,
        revision: str | None = None,
        dtype: str = "bfloat16",
        weights_dir: str | None = None,
        bytes_per_token: float | None = None,
    ) -> None:
        resolved = parse_checkpoint(checkpoint) if isinstance(checkpoint, str) else checkpoint
        self.checkpoint = resolved.at_revision(revision) if revision else resolved
        self.LLM_NAME = str(download_weights(self.checkpoint, Path(weights_dir) if weights_dir else None))

        # Mirrors BaseHFLLM.__init__ (eval-framework 0.14), with the dtype passed through.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = self._load_tokenizer()
        # from_pretrained returns the model in eval mode.
        self.model = AutoModelForCausalLM.from_pretrained(self.LLM_NAME, device_map="auto", dtype=getattr(torch, dtype))
        logger.info(f"Loaded {checkpoint_label(self.checkpoint)} from {self.LLM_NAME} in {dtype}")
        self._set_formatter(None)
        if bytes_per_token is not None and bytes_per_token <= 0:
            raise ValueError("bytes_per_token must be positive")
        self.bytes_per_token_scalar = (
            4.0 / bytes_per_token if bytes_per_token is not None else 4.0 / self.BYTES_PER_TOKEN
        )

    @property
    def name(self) -> str:
        return f"{type(self).__name__}_{checkpoint_label(self.checkpoint)}"
