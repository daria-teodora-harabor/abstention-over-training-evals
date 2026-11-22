"""Run the 100 shipped WeirdChat self-harm prompts on Olmo 3 32B training checkpoints, 5 times each.

Post-trained checkpoints get the prompt as a single chat turn (vLLM applies the chat template).
Pre-training checkpoints get the SelfHarmWeirdChatBase few-shot transcript (served with the plain
transcript chat template) and stop at the next user turn. Every response is scored by the repo's
LlmSafetyJudge on the prompt of interest only.

Usage: python sweep.py <endpoints.json> <outdir> [--repeats 5]
endpoints.json maps checkpoint name -> {"url": model base url, "judge": judge base url, "base": bool}.
Output is one <outdir>/<checkpoint>.jsonl per checkpoint, appended as results arrive (resumable).
"""

import argparse
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import AsyncOpenAI

from abstention_over_training_evals.adapters.llm_judge import LlmSafetyJudge
from abstention_over_training_evals.adapters.olmo_settings import (
    OLMO3_THINK_MAX_TOKENS,
    OLMO3_THINK_TEMPERATURE,
    OLMO3_THINK_TOP_P,
)
from abstention_over_training_evals.plugins.medical_advice_base.medical_advice_base import MAX_TOKENS, STOP_SEQUENCES
from abstention_over_training_evals.plugins.selfharm_weirdchat_base.selfharm_weirdchat_base import SelfHarmWeirdChatBase

PROMPTS = Path(__file__).parents[2] / "data/local/selfharm_weirdchat.jsonl"


def load_prompts():
    return [json.loads(line) for line in PROMPTS.read_text().splitlines() if line.strip()]


def base_transcripts():
    """Prompt id -> the few-shot transcript messages SelfHarmWeirdChatBase sends (framing, 3 demos, prompt)."""
    task = SelfHarmWeirdChatBase.with_overwrite(num_fewshot=3, custom_subjects=None, custom_hf_revision=None)
    return {
        s.messages[-1].content: [{"role": m.role.value, "content": m.content} for m in s.messages]
        for s in task.iterate_samples()
    }


async def generate(client, name, messages, base):
    # Base checkpoints are served with the plain transcript chat template, so both paths use the chat endpoint.
    r = await client.chat.completions.create(
        model=name,
        messages=messages,
        temperature=OLMO3_THINK_TEMPERATURE,
        top_p=OLMO3_THINK_TOP_P,
        max_tokens=MAX_TOKENS if base else OLMO3_THINK_MAX_TOKENS,
        stop=STOP_SEQUENCES if base else None,
    )
    c = r.choices[0]
    extra = c.message.model_extra or {}
    answer = c.message.content or ""
    if base:
        answer = answer.split(STOP_SEQUENCES[0], 1)[0].strip()
    return answer, extra.get("reasoning_content") or extra.get("reasoning"), c.finish_reason


async def run_one(name, cfg, client, judge, gen_sem, judge_sem, item, run, out, transcripts):
    async with gen_sem:
        messages = transcripts[item["prompt"]] if cfg["base"] else [{"role": "user", "content": item["prompt"]}]
        answer, reasoning, finish = await generate(client, name, messages, cfg["base"])
    async with judge_sem:
        verdict = await asyncio.to_thread(judge.judge, item["prompt"], answer, reasoning)
    row = {
        "checkpoint": name,
        "prompt_id": item["id"],
        "behavior_id": item["provenance"]["source"]["metadata"]["behavior_id"],
        "run": run,
        "completion": answer,
        "reasoning": reasoning,
        "finish_reason": finish,
        "judged": verdict.judged,
        "unsafe": verdict.unsafe,
        "judge_reason": verdict.reason,
    }
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")


async def sweep_checkpoint(name, cfg, items, repeats, outdir, judge_sems, transcripts, concurrency):
    out = outdir / f"{name}.jsonl"
    done = {(r["prompt_id"], r["run"]) for r in map(json.loads, out.open())} if out.exists() else set()
    client = AsyncOpenAI(base_url=cfg["url"], api_key="EMPTY", timeout=3600, max_retries=6)
    judge = LlmSafetyJudge(model="judge", base_url=cfg["judge"], timeout=600)
    todo = [(it, r) for r in range(repeats) for it in items if (it["id"], r) not in done]
    gen_sem = asyncio.Semaphore(256)
    results = await asyncio.gather(
        *(
            run_one(name, cfg, client, judge, gen_sem, judge_sems[cfg["judge"]], it, r, out, transcripts)
            for it, r in todo
        ),
        return_exceptions=True,
    )
    errors = [e for e in results if isinstance(e, Exception)]
    print(f"{name}: {len(todo) - len(errors)}/{len(todo)} new rows, {len(errors)} errors", flush=True)
    for e in errors[:3]:
        print(f"  {name} error: {type(e).__name__}: {e}", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("endpoints")
    ap.add_argument("outdir")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=256, help="in-flight generations per checkpoint")
    args = ap.parse_args()
    endpoints = json.loads(Path(args.endpoints).read_text())
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    items = load_prompts()
    transcripts = base_transcripts() if any(c["base"] for c in endpoints.values()) else {}
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(256))
    judge_sems = {cfg["judge"]: asyncio.Semaphore(64) for cfg in endpoints.values()}
    await asyncio.gather(
        *(
            sweep_checkpoint(n, c, items, args.repeats, outdir, judge_sems, transcripts, args.concurrency)
            for n, c in endpoints.items()
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
