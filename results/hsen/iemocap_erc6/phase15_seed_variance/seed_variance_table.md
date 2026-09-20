# Phase 15 seed-variance study -- iemocap_erc6, cpu25

| variant | seed42 | seed43 | seed44 | mean ± std (val WF1) | params |
|---|---:|---:|---:|---:|---:|
| husformer | 0.5036 | 0.4886 | 0.4869 | 0.4930 ± 0.0091 | 5,992,456 |
| concat | 0.4928 | 0.4927 | 0.4940 | 0.4932 ± 0.0007 | 1,448,968 |
| self_attention | 0.4911 | 0.4921 | 0.4912 | 0.4915 ± 0.0006 | 2,830,344 |

| variant | seed42 | seed43 | seed44 | mean ± std (val macro-F1) | params |
|---|---:|---:|---:|---:|---:|
| husformer | 0.4750 | 0.4601 | 0.4514 | 0.4622 ± 0.0120 | 5,992,456 |
| concat | 0.4626 | 0.4662 | 0.4632 | 0.4640 ± 0.0019 | 1,448,968 |
| self_attention | 0.4544 | 0.4614 | 0.4527 | 0.4562 ± 0.0046 | 2,830,344 |

- husformer vs concat: husformer is ahead on 1 of 3 seeds (the ordering flips with the seed); mean margin -0.0001 WF1 [-0.0070, +0.0107].
-     That margin is not larger than the larger across-seed std (0.0091) and not larger than the paired-difference std (0.0095); the two variants' WF1 ranges overlap.
- husformer vs self_attention: husformer is ahead on 1 of 3 seeds (the ordering flips with the seed); mean margin +0.0016 WF1 [-0.0043, +0.0125].
-     That margin is not larger than the larger across-seed std (0.0091) and not larger than the paired-difference std (0.0094); the two variants' WF1 ranges overlap.
- Highest mean validation WF1 across these seeds: concat (0.4932 +/- 0.0007, 1,448,968 parameters, 417 s mean training time).
- Smallest across-seed spread: self_attention (std 0.0006).

**Recommended fusion:** `concat` -- highest mean validation weighted F1 across the seeds that ran; husformer did not hold its section-13 lead across seeds.

_3 seeds bound the spread; they do not support a claim of statistical significance in either direction._
