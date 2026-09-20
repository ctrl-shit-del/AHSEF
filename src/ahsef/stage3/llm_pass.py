"""PHASE B (text side) -- run or replay the frozen Gemma text evidence source.

Stage 2 froze the LLM configuration: ``gemma4:31b-cloud`` through Ollama, prompt
``v2_explicit_json``, ``temperature=0``, ``seed=42``.  Stage 3 reuses that
configuration exactly and does **not** retrain, re-prompt, or re-tune it.  What
changes is only *which samples* it is asked about: the Stage 2 pilot scored a
1,000-sample text-only draw, of which just 20 validation and 18 test samples
fall inside the aligned Text+Audio pool.  Routing on 20 samples would not be an
experiment, so Stage 3 records its own transcript over the aligned pool.

The transcript is the reproducible artefact, exactly as in Stage 2: live calls
are made only for samples not yet recorded, and the prediction set is always
rebuilt by replaying the *complete* transcript.  An interrupted run therefore
costs only the calls it had not yet made, and the artefact does not depend on
how many invocations produced it.

Latency is recorded twice and never conflated:

``generation_ms``
    what the provider reports for the completion itself.

``wall_clock_ms``
    measured end to end around the whole pass, including queueing, connection
    setup, and cloud scheduling.  Stage 2 measured generation at 29% of wall
    clock with an 18x throughput swing inside one run, so a cost model that
    treated generation time as the service latency would be wrong by a factor
    that varies during the run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.ahsef.inference import PredictionSet
from src.ahsef.llm.inference import LLMTextModality, build_prediction_set
from src.ahsef.llm.provider import CallBudget, ReplayProvider, TranscriptWriter, recorded_sample_ids
from src.ahsef.registry import DEFAULT_BASELINES
from src.ahsef.stage3.pool import AlignedPool

#: The Stage 2 configuration, inherited verbatim.  Stage 3 never re-selects it.
FROZEN_LLM_CONFIG = {
    "model": "gemma4:31b-cloud",
    "host": "http://localhost:11434",
    "prompt_version": "v2_explicit_json",
    "temperature": 0.0,
    "llm_seed": 42,
    "num_predict": 700,
    "repeats": 1,
    "uncertainty_policy": "score_entropy",
    "inherited_from": "experiments/ahsef/stage2_llm/frozen_config.json",
    "note": (
        "Stage 3 changes which samples the LLM is asked about, never how it is "
        "asked. Model, prompt, decoding parameters and uncertainty policy are the "
        "Stage 2 frozen values."
    ),
}


def load_pool_texts(pool: AlignedPool, root: Path | str = "experiments") -> pd.DataFrame:
    """The pooled samples' text, in pool order, read exactly as the baseline reads it.

    MSP-Podcast transcripts are file-backed: the manifest's ``text`` column is
    empty and ``text_path`` names a file under ``datasets/``.  Resolving them
    through :class:`~src.data.text_dataset.EmotionTextDataset` rather than
    reading the column means the LLM sees the same string the frozen text
    baseline was trained and evaluated on, so the two are genuinely paired.

    Raises if any pooled sample resolves to blank text: Phase A already verified
    text availability, so a blank here means the manifest or the dataset tree
    changed underneath the pool, and the run must not send an empty prompt.
    """
    from src.data.text_dataset import EmotionTextDataset

    path = DEFAULT_BASELINES["text"].layout(root).split_path(pool.split)
    dataset = EmotionTextDataset(path, columns=list(EmotionTextDataset.MINIMAL_COLUMNS))
    frame = dataset.manifest.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    indexed = frame.set_index("sample_id")
    missing = [item for item in pool.sample_ids if item not in indexed.index]
    if missing:
        raise KeyError(
            f"{len(missing)} pooled ids are absent from {path} (e.g. {missing[:5]}); "
            f"the pool and the manifest disagree and the run cannot be reproduced."
        )
    # Narrow to the pool *before* resolving text: these splits hold ~22,000 rows
    # and each file-backed transcript is a separate read, so resolving the whole
    # manifest to reach 509 samples would dominate the run.
    ordered = indexed.loc[pool.sample_ids].reset_index()
    ordered["text"] = [dataset.read_text(row) for _, row in ordered.iterrows()]
    blank = ordered["text"].astype(str).str.strip().eq("")
    if bool(blank.any()):
        raise ValueError(
            f"{int(blank.sum())} pooled {pool.split} samples have blank text "
            f"(e.g. {ordered.loc[blank, 'sample_id'].tolist()[:5]}). Phase A verified "
            f"text availability, so the manifest has changed; refusing to send an "
            f"empty prompt."
        )
    return ordered


def run_llm_pass(
    pool: AlignedPool,
    transcript_path: Path,
    timing_path: Path,
    config: dict | None = None,
    provider_name: str = "ollama",
    store_text: str = "hash",
    max_calls: int | None = None,
    root: Path | str = "experiments",
    progress_every: int = 25,
    on_line=print,
) -> tuple[PredictionSet, dict]:
    """Score the pool, resuming from the transcript, and return ``(set, timing)``."""
    settings = {**FROZEN_LLM_CONFIG, **(config or {})}
    frame = load_pool_texts(pool, root)
    transcript_path = Path(transcript_path)

    already = recorded_sample_ids(transcript_path) if transcript_path.exists() else set()
    pending = frame[~frame["sample_id"].isin(already)].reset_index(drop=True)
    resumed = bool(already)
    if resumed:
        on_line(
            f"[llm ] {pool.split}: resuming -- {len(already)}/{len(frame)} recorded, "
            f"{len(pending)} to call"
        )
    on_line(
        f"[llm ] {pool.split}: {len(frame)} aligned samples | model={settings['model']} "
        f"prompt={settings['prompt_version']} T={settings['temperature']} "
        f"seed={settings['llm_seed']}"
    )

    started = time.perf_counter()
    calls_made = 0
    if len(pending) and provider_name != "replay":
        from src.ahsef.llm.ollama_provider import OllamaProvider

        provider = OllamaProvider(
            model=settings["model"], host=settings["host"],
            temperature=settings["temperature"], seed=settings["llm_seed"],
            num_predict=settings["num_predict"],
        )
        writer = TranscriptWriter(
            transcript_path, model=settings["model"], provider=provider.name,
            prompt_version=settings["prompt_version"],
            store_text=(store_text == "raw"), append=resumed,
        )
        budget = CallBudget(max_calls=max_calls)
        live = LLMTextModality(
            provider=provider, prompt_version=settings["prompt_version"],
            repeats=int(settings["repeats"]),
            uncertainty_policy=settings["uncertainty_policy"],
            budget=budget, transcript=writer, store_text=store_text,
        )

        def progress(done, total, _state):
            rate = done / max(time.perf_counter() - started, 1e-9)
            on_line(
                f"       {done}/{total}  {rate:.2f}/s  "
                f"eta {(total - done) / max(rate, 1e-9) / 60:.1f} min"
            )

        try:
            live.predict_frame(pending, progress_every=progress_every, on_progress=progress)
        finally:
            writer.close()
        calls_made = budget.calls
    elif not len(pending):
        on_line(f"[llm ] {pool.split}: every pooled sample already recorded; no call made")

    live_seconds = time.perf_counter() - started

    # Always rebuild from the complete transcript, so the artefact is identical
    # whether the pass finished in one invocation or five.
    replay = ReplayProvider(transcript_path, strict=True)
    modality = LLMTextModality(
        provider=replay, prompt_version=settings["prompt_version"],
        repeats=int(settings["repeats"]),
        uncertainty_policy=settings["uncertainty_policy"], store_text=store_text,
    )
    results = modality.predict_frame(frame)
    provenance = modality.provenance()
    provenance["stage3"] = {
        "pool_split": pool.split,
        "pool_fingerprint_sha256": pool.fingerprint,
        "pool_size": pool.size,
        "llm_config": dict(settings),
        "execution": {
            "resumed": resumed,
            "previously_recorded": len(already),
            "live_calls_this_invocation": calls_made,
            "total_samples": int(len(frame)),
        },
    }
    predictions = build_prediction_set(
        results, pool.split, provenance,
        texts=dict(zip(frame["sample_id"], frame["text"])), store_text=store_text,
    )

    timing = _timing_record(
        predictions, pool.split, live_seconds, calls_made, len(frame), settings
    )
    Path(timing_path).parent.mkdir(parents=True, exist_ok=True)
    Path(timing_path).write_text(json.dumps(timing, indent=2), encoding="utf-8")
    on_line(
        f"[llm ] {pool.split}: coverage {predictions.frame['llm_usable'].mean():.1%} | "
        f"{calls_made} live calls in {live_seconds/60:.1f} min"
    )
    return predictions, timing


def _timing_record(
    predictions: PredictionSet,
    split: str,
    live_seconds: float,
    calls_made: int,
    samples: int,
    settings: dict,
) -> dict:
    """Generation latency and true service latency, kept apart on purpose."""
    generation = predictions.frame["latency_ms"].to_numpy(dtype=float)
    measured = calls_made > 0
    wall_per_call = (live_seconds * 1000.0 / calls_made) if calls_made else None
    return {
        "split": split,
        "samples": int(samples),
        "generation_latency_ms": {
            "mean": float(generation.mean()),
            "median": float(np.median(generation)),
            "p95": float(np.percentile(generation, 95)),
            "min": float(generation.min()),
            "max": float(generation.max()),
            "total": float(generation.sum()),
            "source": "provider-reported completion latency, replayed from transcript",
        },
        "wall_clock": {
            "measured_this_invocation": bool(measured),
            "live_calls": int(calls_made),
            "elapsed_seconds": float(live_seconds),
            "mean_ms_per_live_call": wall_per_call,
            "generation_fraction_of_wall_clock": (
                float(generation.mean() / wall_per_call) if wall_per_call else None
            ),
            "note": (
                "Wall clock covers queueing, connection setup and cloud scheduling as "
                "well as generation. It is measured only for calls actually made in "
                "this invocation; a fully resumed pass reports measured=false rather "
                "than reusing an earlier run's number."
            ),
        },
        "latency_policy": (
            "The Stage 3 cost model prices the LLM at measured wall clock, not at "
            "generation time, and records that cloud throughput is not constant. "
            "Audio is priced at its measured Stage 1 per-sample latency."
        ),
        "llm_config": dict(settings),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
