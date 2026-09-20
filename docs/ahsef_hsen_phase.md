# AHSEF HSEN — the always-on multimodal baseline

The reference model every later adaptive result is measured against. Adaptive
routing (HSIG/UGAPR) is deliberately *not* part of this phase; `src/ahsef/`
holds that work and is untouched here.

---

## Architecture

Fixed by the AHSEF literature survey (`docs/Research papers/AHSEF_Literature_Survey (1).pdf`)
and not a free parameter of this phase.

```
audio   emotion2vec        frozen   768-d  ┐
video   face + MobileNetV3 frozen   576-d  ├─ Conv1D → d=256 + positional
text    RoBERTa-base       frozen   768-d  ┘
                                              │
                    Husformer fusion-to-modality cross-attention
                        2 layers · 4 heads · no CTC · unaligned
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
              emotion head              valence head             arousal head
                (focal)                   (bounded)                (bounded)
                                              │
                        state = [valence, arousal, emotion]
```

| Choice | Evidence |
|---|---|
| emotion2vec, frozen | 71.79 WA on IEMOCAP from a **linear probe**; matches a fine-tuned WavLM-large at 0.20 M trainable params |
| MobileNetV3, not ViT | ESED Table 6: video is the weakest branch (50.13 WF1 alone, ~+0.84 over text) — the survey says explicitly not to spend a ViT-B budget |
| RoBERTa-base | ESED's own text encoder; ESED Table 6 and MPLMM Table 1 both put text first |
| d = 256 | ESED Table 7 measures it: 128 → 71.25, **256 → 72.66**, 512 → 70.95. Accuracy-optimal, not just affordable |
| fusion-to-modality | Husformer Table V: 78.68 vs HusPair's 73.57 at 0.71 M vs 3.90 M params. Cost is linear, not quadratic, in modality count |
| no CTC | LDDU's unaligned CMU-MOSEI numbers beat its aligned ones |
| focal loss | Husformer's; IEMOCAP ERC-6 runs 387→1,066 train samples per class |

---

## Datasets and the exact splits

### IEMOCAP — primary

**`iemocap_erc6`** (default). The six-way conversational protocol ESED,
DQ-Former and AdaIGN report against.

| | train | validation | test | total |
|---|---:|---:|---:|---:|
| sessions | 1, 2, 3 | 4 | 5 | |
| utterances | 4,246 | 1,512 | 1,622 | **7,380** |

Per class (declared order, never sorted):

| | neutral | happy | sad | angry | excited | frustrated |
|---|---:|---:|---:|---:|---:|---:|
| train | 1,066 | 387 | 696 | 606 | 504 | 987 |
| validation | 258 | 65 | 143 | 327 | 238 | 481 |
| test | 384 | 143 | 245 | 170 | 299 | 381 |

ESED Table 1 reports 1,623 test utterances under the same protocol; ours is
1,622.

**`iemocap_ser4`** (secondary). Four-way, `excited` merged into `happy` —
5,531 utterances, exactly the canonical figure, and the protocol emotion2vec's
71.79 WA is measured under.

**Why not the project's canonical 7-class space.** It has no slot for `excited`
or `frustrated`, which are 2,890 of IEMOCAP's 10,039 utterances. Dropping them
leaves 4,639 utterances and a test partition with 10 `fear` and 0 `disgust` —
not a protocol anyone else runs, so not a number anyone else's is comparable to.

**Speaker independence** is verified, not assumed. IEMOCAP's `speaker` column
holds only `F`/`M` — a role within a session, not an identity — so the audit
keys on `<session>_<role>` and confirms the three splits share no speaker.

### CMU-MOSEI — primary

**`mosei_sentiment`**. The official standard partition, carried through
unchanged: 16,326 / 1,871 / 4,659.

Supervision is **sentiment only**. The six-way multi-label emotion annotations
(`CMU_MOSEI_Labels.csd`) are not in this repository, so the LDDU reference
points (0.496 Acc / 0.587 micro-F1) are **not yet reproducible here**. The
`mosei_emotion6` protocol is implemented and fails with instructions until the
annotations are present.

The `emotion` column the standardized metadata carries for CMU-MOSEI
(happy 11,264 / sad 6,594 / neutral 4,998) is **derived from sentiment
polarity, not annotated**. `MOSEILabelAdapter` refuses to read it.

Features come from `Processed/aligned_50.pkl` — COVAREP (74-d), FACET (35-d),
BERT (768-d). That is what LDDU, CARAT and TAILOR use, so it is the
comparability-preserving choice, not a fallback. The 256-d projection makes the
width difference invisible to the trunk.

---

## No-leakage guarantees (§17)

Printed before every training run and enforced — `assert_clean` raises, it does
not warn.

| Check | What it catches |
|---|---|
| `no_duplicate_sample_ids` | a sample counted twice within a split |
| `no_cross_split_sample_ids` | the same utterance in train and test |
| `speaker_independence` | speaker overlap; reports **unverifiable** rather than "pass" when a corpus ships no speaker ids |
| `no_unlabelled_rows` | a retained row with no protocol class |
| `every_class_present_in_every_split` | a class that would silently leave the macro average |
| `declared_modalities_have_payloads` | `has_text=True` with an empty transcript |
| `regression_targets_in_range` | an affect target outside the range its transform declares |
| `feature_cache_split_isolation` | a cached feature claimed by two splits, or stale ids the manifests no longer name |

**Normalisation.** Nothing is fitted. Affect targets use fixed affine maps of
known annotation ranges — IEMOCAP `(x-3)/2`, CMU-MOSEI `score/3` — and the model
normalises with `LayerNorm`, which is per-sample. A test asserts the model holds
no running-statistics buffer.

**Transcripts.** Gold, from `Session*/dialog/transcriptions/`. No ASR anywhere.

**One trainer per run directory.** `RunLock` claims
`results/hsen/<experiment>/<profile>/` for the life of a run. This is not
defensive boilerplate — it was added *after* two processes wrote one directory
during this phase and produced a `best.pt` whose recorded validation score
appears nowhere in its own history. Nothing raised; the artefacts were
individually well-formed and jointly wrong. `--force-lock` exists for a holder
confirmed gone, and the refusal message reports the holder's pid, host, start
time and whether it is still running.

**One driver per study directory, on top of that.** Phase 15's driver claims
`results/hsen/<experiment>/<study>/` with a second `RunLock` for its own life,
so two drivers cannot interleave their aggregate tables any more than two
trainers can interleave checkpoints. It also refuses to *aggregate* a run whose
`best_val_*` is absent from its own history -- the exact shape of the
corruption above, checked on the artefact rather than trusted to the process.

**The training subset is pinned separately from the training seed.**
`TrainerConfig.subset_seed` (default: same as `seed`, so every earlier run is
unchanged) seeds the stratified 25% draw on its own. Phase 15 holds it at 42
while varying `seed`, so "seed variance" means initialisation, dropout and batch
order on one fixed set of 1,062 utterances -- not that plus which quarter of the
data was drawn. The study records the SHA-256 of those ids and refuses a run
that trained on a different count.

**Resume is exact.** `update()` compares before either write, so `last.pt` never
carries a `best_value` one epoch stale — the failure mode where a resumed run
quietly replaces a better checkpoint with a worse epoch and reports nothing,
because "no improvement" is what an ordinary epoch looks like.

---

## Pipeline

```bash
# 1. manifests (audited; --audit-only builds in memory and writes nothing)
python -m src.hsen.manifests --experiment iemocap_erc6

# 2. cache the frozen encoders, once
python -m src.hsen.extract --experiment iemocap_erc6 --modality text  --split train
python -m src.hsen.extract --experiment iemocap_erc6 --modality audio --split train
#    ... repeat for validation and test

#    CMU-MOSEI reads its provided container instead
python -m src.hsen.mosei_features --experiment mosei_sentiment

# 3. train  (both scripts call the SAME trainer)
python scripts/train_hsen_cpu25.py --dataset iemocap --data_fraction 0.25
python scripts/train_hsen_cuda.py  --dataset iemocap --data_fraction 1.0

# 4. ablations
python -m src.hsen.ablations --study modality --profile cpu25
python -m src.hsen.ablations --study fusion   --profile cpu25

# 4b. Phase 15 -- the fusion ablation under three seeds (resumable; --report-only aggregates)
python -m src.hsen.seed_variance --study phase15_seed_variance --experiment iemocap_erc6 \
    --profile cpu25 --modalities audio+text --variants husformer,concat,self_attention \
    --seeds 42,43,44 --subset-seed 42

# 5. figures, and the locked test evaluation
python -m src.hsen.evaluation --run results/hsen/iemocap_erc6/full_cuda
python -m src.hsen.evaluation --run results/hsen/iemocap_erc6/full_cuda --split test --evaluate

python -m pytest src/hsen -q
```

### Feature caches

`experiments/hsen/<experiment>/features/<modality>/<split>/` — sharded `.npz`
holding ragged `[T, D]` sequences plus `index.json` and
`extraction_provenance.json`. Split is a directory, so a training loader cannot
reach a test feature. Resumable; a cache built with a different encoder config
is refused on open by fingerprint.

Measured on this machine (8 threads, CPU): RoBERTa ~12/s, emotion2vec ~1.3/s.
IEMOCAP ERC-6 audio ≈ 450 MB, text ≈ 183 MB.

---

## The two profiles (§4, §23)

Both build the same `HSENConfig` and call the same `HSENTrainer`. Neither script
contains a model, loss, metric or evaluation path of its own.

| | `train_hsen_cuda.py` | `train_hsen_cpu25.py` |
|---|---|---|
| device | cuda (**fails** if unavailable) | cpu (forced) |
| precision | AMP | FP32 |
| training data | 100% | 25%, stratified, seeded |
| batch size | 32 | 8 |
| workers | 4 | 0 |
| epochs | 30 | 15 |

Everything else — architecture, dimensions, attention, layers, heads, feature
representation, label processing, loss, evaluation — is identical.
`test_cpu_and_cuda_profiles_build_identical_architectures` compares the two
`state_dict`s directly; `test_profiles_differ_only_in_compute_and_data` fails if
a future edit lets a profile move a model field.

The data fraction applies to **training only**. Validation and test stay whole,
so every run's numbers are computed on the same evaluation samples and the fast
profile's ranking of two configs means something for the slow one.

---

## Metrics (§11)

Categorical: accuracy, weighted F1 (**IEMOCAP's primary, and the checkpoint
selection metric**), macro F1, per-class precision/recall/F1/support, confusion
matrix. Multi-label: micro-F1, macro-F1, exact-set-match accuracy — the LDDU
convention, which is much harsher than per-sample top-1 and is never printed
under the same heading. Regression: CCC, MAE, RMSE, Pearson. CMU-MOSEI
additionally reports Acc7 and **both** Acc2 conventions, because the difference
between them is worth several points and quoting one as the other is the usual
way MOSEI numbers stop being comparable.

Best checkpoint is selected on the primary **validation** metric, never on
training loss.

---

## Results so far — IEMOCAP ERC-6, CPU 25% profile

Previews on a quarter of the training data, on a laptop CPU. Not comparable
with ESED's 72.66 — that needs the full-data CUDA run. Validation is complete
(1,512 utterances) in every row, so the arms are comparable with each other.

### Section 12 — modality ablation

| arm | val WF1 | val macro-F1 | params (fusion) | train s |
|---|---:|---:|---:|---:|
| text | 0.4157 | 0.3798 | 3.8 M (3.2 M) | 697 |
| audio | 0.4350 | 0.4028 | 3.8 M (3.2 M) | 4,437 |
| **audio+text** | **0.5036** | **0.4750** | 6.0 M (4.7 M) | 3,661 |

**+0.0685 weighted F1 over the best single modality.** That margin is the budget
the later routing work may spend: a modality UGAPR declines to acquire must cost
less than it.

**The unimodal ordering is audio > text**, inverting ESED Table 6 (53.62 audio,
69.74 text). That is the effect the survey anticipated from replacing WavLM-base
with emotion2vec, and it is the concrete reason section 12 asked for our own
ablations instead of inheriting the literature's ordering. Any routing policy
built on the assumption "text is the strong modality, audio is the cheap
add-on" would be built on an ordering that does not hold in this stack.

Per-class F1 for audio+text runs 0.309 (`happy`, n=65) to 0.612 (`sad`) — the
weak class stays visible rather than disappearing into the 0.504 aggregate.

### Section 13 — fusion ablation (audio+text, everything else identical)

| variant | val WF1 | val macro-F1 | fusion params | train s |
|---|---:|---:|---:|---:|
| **husformer** (fusion-to-modality) | **0.5036** | **0.4750** | 4,742,144 | 3,640 |
| self_attention (no cross-modal) | 0.4911 | 0.4544 | 1,580,032 | 1,869 |
| concat (no attention at all) | 0.4928 | 0.4626 | 198,656 | 475 |

**Husformer wins, but narrowly and expensively: +0.0125 over self-attention and
+0.0107 over concatenation, at 3x and 24x the fusion parameters and 2x and 7.7x
the training time.**

That is not the result Husformer reports. Its Table V has fusion-to-modality
beating pairwise cross-attention by 5.1 points *while using 5.5x fewer*
parameters — "more accurate per parameter" is the paper's actual claim. Here the
topology is the most expensive option for a margin that a seed change could
plausibly cover.

Two honest caveats before this is read as a refutation:

1. **Two modalities, not four.** Fusion-to-modality attention exists to make
   cost grow linearly rather than quadratically in the modality count. At M=2
   there is nothing for that property to buy — pairwise and fusion-to-modality
   are nearly the same computation. Husformer's evidence is on four
   physiological channels. This ablation is the least favourable setting the
   claim could be tested in, and video would be the test that matters.
2. **25% of the training data.** A 4.7 M-parameter trunk on 1,062 training
   samples is the regime where extra capacity is least likely to pay.

The survey named this exact risk -- "the claim that the topology transfers to
A/V/T is untested and is ours to verify ... the single largest assumption in our
proposed stack". On this evidence it is not yet verified. A fusion table without
these cost columns would have made a 0.01 margin look like a vindication — and
Phase 15, below, re-ran this table under three seeds and found the margin to be
seed noise. Husformer is no longer the default for the full-data run.

### Phase 15 — is the section-13 ranking robust to the seed?

Section 13's table came from one seed, and its own commentary called the
0.01 margin one "a seed change could plausibly cover". Phase 15 tests exactly
that before any GPU time is committed to the fusion it chose: the same three
arms, under seeds **42, 43, 44**, with *everything else identical* — the same
cached emotion2vec and RoBERTa features, projections, HSEN heads, AdamW at
1e-4, cosine warm-up, focal loss with balanced weights, batch 8, 15-epoch
budget, patience 5 / min 3 epochs, weighted-F1 checkpoint selection, the same
session-disjoint split (audited again on every run) and the same
missing-modality handling. The stratified 25% training draw is pinned to the
section-13 subset (`--subset-seed 42`; 1,062 ids, SHA-256 `f11fc8ba…`), so
only initialisation, dropout and batch order move. Validation is the full
1,512 utterances. **The test split was not opened** — the study refuses a run
that carries test metrics.

Study directory: `results/hsen/iemocap_erc6/phase15_seed_variance/`, one
`fusion_<variant>_seed<seed>/` run per cell (each under its own `RunLock`),
plus `study_config.json` (exact commands, configuration, subset digest, class
order), `seed_variance_runs.csv`, `seed_variance_summary.json`,
`seed_variance_table.md` and `study.log`. Section 13's artefacts are untouched.

**Reproducibility, first.** The seed-42 column re-executes section 13 and
reproduces it to every printed digit — 0.5036 / 0.4928 / 0.4911, same best
epochs, same per-epoch trajectories. The CPU build is deterministic, so the
other two columns are measuring the seed and nothing else.

#### Per run (validation, 1,512 utterances)

| variant | seed | best epoch | val WF1 | val macro-F1 | accuracy | train s | epochs | params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| husformer | 42 | 5 | **0.5036** | 0.4750 | 0.5106 | 4,476 | 10 | 5,992,456 |
| husformer | 43 | 15 | 0.4886 | 0.4601 | 0.4835 | 7,323 | 15 | 5,992,456 |
| husformer | 44 | 13 | 0.4869 | 0.4514 | 0.4821 | 5,348 | 15 | 5,992,456 |
| concat | 42 | 11 | 0.4928 | 0.4626 | 0.4960 | 486 | 15 | 1,448,968 |
| concat | 43 | 6 | 0.4927 | 0.4662 | 0.4861 | 332 | 11 | 1,448,968 |
| concat | 44 | 15 | 0.4940 | 0.4632 | 0.4901 | 433 | 15 | 1,448,968 |
| self_attention | 42 | 5 | 0.4911 | 0.4544 | 0.4788 | 2,007 | 10 | 2,830,344 |
| self_attention | 43 | 4 | 0.4921 | 0.4614 | 0.5073 | 1,660 | 9 | 2,830,344 |
| self_attention | 44 | 6 | 0.4912 | 0.4527 | 0.4993 | 2,017 | 11 | 2,830,344 |

#### Aggregate (sample std, ddof = 1)

| variant | seed42 | seed43 | seed44 | mean ± std (val WF1) | params |
|---|---:|---:|---:|---:|---:|
| husformer | 0.5036 | 0.4886 | 0.4869 | 0.4930 ± 0.0091 | 5,992,456 |
| concat | 0.4928 | 0.4927 | 0.4940 | **0.4932 ± 0.0007** | 1,448,968 |
| self_attention | 0.4911 | 0.4921 | 0.4912 | 0.4915 ± 0.0006 | 2,830,344 |

| variant | mean WF1 | std | min | max | mean macro-F1 | std | mean train s | s / epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| husformer | 0.4930 | 0.0091 | 0.4869 | 0.5036 | 0.4622 | 0.0120 | 5,715 | 357–488 |
| concat | 0.4932 | 0.0007 | 0.4927 | 0.4940 | 0.4640 | 0.0019 | 417 | 29–32 |
| self_attention | 0.4915 | 0.0006 | 0.4911 | 0.4921 | 0.4562 | 0.0046 | 1,895 | 183–201 |

#### Paired, seed by seed (Husformer minus the other arm, val WF1)

| | seed42 | seed43 | seed44 | Husformer ahead | mean margin | paired-diff std |
|---|---:|---:|---:|---|---:|---:|
| vs concat | +0.0107 | −0.0040 | −0.0070 | 1 of 3 | −0.0001 | 0.0095 |
| vs self_attention | +0.0125 | −0.0035 | −0.0043 | 1 of 3 | +0.0016 | 0.0094 |

#### What the three seeds show

1. **Husformer does not beat concat consistently.** It leads on seed 42 only;
   concat is ahead on 43 and 44. Mean margin −0.0001.
2. **Husformer does not beat self-attention consistently.** Again seed 42
   only. Mean margin +0.0016.
3. **The section-13 margin is not larger than seed variance.** The +0.0107 /
   +0.0125 margins are about the size of Husformer's own across-seed std
   (0.0091) and of the paired-difference std (0.0095); the three arms' WF1
   ranges overlap. Section 13 caught Husformer at its best of three seeds and
   concat near its worst.
4. **The spread is asymmetric.** Concat and self-attention vary by
   0.0006–0.0007 across seeds; Husformer varies by 0.0091 — thirteen times
   more — and its macro-F1 by 0.0120. The 4.7 M-parameter trunk on 1,062
   training samples is the one that is sensitive to where it starts.
5. **Husformer's late best epochs (15 and 13 at seeds 43 and 44) say the
   15-epoch budget binds it**, where the two smaller trunks settle by epoch
   4–11. That is a fact about this budget, not a reason to extend it here —
   the study held the budget fixed by design. The full-data profile gives
   every arm four times the steps and twice the epochs anyway.
6. **Training times on this pass are not clean.** The seed-42 Husformer run
   overlapped with the test suite for its first three epochs and the seed-43
   run ran under memory pressure from other applications (488 s/epoch against
   357 s for seed 44 and section 13's 364 s). The ordering — concat ≈ 30
   s/epoch, self-attention ≈ 190, Husformer ≈ 360–490 — is unaffected; the
   absolute mean for Husformer is inflated. The first launch was killed by the
   harness for low system memory after run 1 completed; `stdout.log` is that
   launch's record, `stdout_relaunch.log` the resumed one. The interrupted
   concat run had written only `run_config.json` and was started fresh, not
   resumed.

**Decision: carry `concat` into the full-data CUDA experiment.** On mean
validation WF1, concat and Husformer are tied (0.4932 vs 0.4930; a 0.0002
difference no three-seed study can rank). What separates them is everything
else: concat's across-seed spread is thirteen times smaller, its mean macro-F1
is higher (0.4640 vs 0.4622), and it costs 4.1× fewer total parameters (24×
fewer fusion parameters) and roughly 12× less training time per epoch. On this
evidence Husformer's extra 4.5 M fusion parameters buy nothing measurable at
two modalities and 25% of the data; choosing it on literature grounds alone,
after measuring that its margin is seed noise, would be the decision the
project's integrity rules exist to prevent. The choice rests on stability and
cost, not on the mean.

**What this does not say.** Three seeds bound the spread; they do not estimate
a distribution, and nothing here is a claim of statistical significance in
either direction. Concat is *more stable across these seeds*, not *proven
better*. The two caveats from section 13 still hold: fusion-to-modality
attention's advantage is meant to appear with more modalities and more data,
and neither is present here. If GPU budget allows a second full-data run,
Husformer is the arm to spend it on — but that is a follow-up, not part of
this phase, and the primary full-data fusion is concat.

---

## Reference points, not targets (§18)

| | metric | reported | protocol |
|---|---|---:|---|
| IEMOCAP | weighted F1 | 72.66 | ESED, six-way |
| IEMOCAP | weighted accuracy | 71.79 | emotion2vec, four-way linear probe |
| CMU-MOSEI | accuracy / micro-F1 | 0.496 / 0.587 | LDDU, six-way multi-label, unaligned |

Different models under different protocols. The CMU-MOSEI pair is not currently
reachable here at all — it needs emotion annotations this repository does not
hold. Splits and evaluation are never adjusted to move toward these numbers.

---

## Built for what comes next (§24)

`HSEN.forward` returns a dict, and the keys are interface: `projected`,
`modality_sequences`, `modality_vectors`, `fused`, `fused_sequence`,
`fused_mask`, `logits`, `valence`, `arousal`. ESED's per-modality evidence
heads, LDDU's ordinality calibration, per-modality vacuity, HSIG and UGAPR all
read exactly these, so the uncertainty phase attaches to HSEN rather than
rewriting it. `HSENConfig.modality_heads` is the opt-in for ESED-style
per-modality classifiers, off by default so this phase's baseline stays the
architecture the survey specifies.

---

## Known limits of this phase

1. **No NVIDIA GPU on the development machine** (Intel Iris Xe; torch is a
   `+cpu` build). Profile A is written, unit-tested and architecture-verified,
   but has not been *run*. It needs a CUDA machine. Phase 15 settled which
   fusion that run carries (`--fusion concat`).
2. **IEMOCAP video is not yet in the pipeline.** The assets exist (7.16 GB of
   dialog `.avi`, 720×480 at 30 fps) but are dialog-level with two speakers side
   by side, so per-utterance video needs segment slicing plus left/right
   selection by speaker before face detection. `FaceVideoEncoder` handles the
   crop and the `frame_half` selection and was verified end-to-end on
   `Ses01F_impro01.avi`; the manifest side is not built, so IEMOCAP runs
   audio+text today.

   **Measured, and the reason to budget for it:** OpenCV's Haar frontal-face
   cascade found a face in only 1–3 of 6 uniformly sampled frames on that clip
   (17–50%). Speakers turn away, lean, and sit far from the camera. That is a
   usable *availability* signal — a clip with no detected face is honestly
   recorded as video-unavailable rather than zero-filled — but it is not a
   usable video branch. A real IEMOCAP video pass needs MTCNN
   (`--detector mtcnn`, `pip install facenet-pytorch`), and probably per-frame
   tracking rather than independent detection.
3. **CMU-MOSEI emotion labels absent** — see above.
4. The video backbone is ImageNet-pretrained, not yet fine-tuned on
   AffectNet+/RAF-DB as the survey recommends.
