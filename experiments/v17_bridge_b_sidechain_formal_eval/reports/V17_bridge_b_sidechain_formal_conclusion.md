# V17 Bridge-B Sidechain Formal Conclusion

## Evaluation Setup

- Checkpoint: `experiments/v17_bridge_b_debug/checkpoints/best_gen.pt`
- DDIM steps: `100`
- GT scaffold bridge: `ON`
- Soft scaffold constraint:
  - `time_threshold = 150`
  - `constraint_scale = 0.08`
  - `plane_scale = 0.04`
  - `edge_scale = 0.12`
  - `sidechain_edge_scale = 0.15`

## 1. Sidechain Sweep Decision

200-sample, DDIM-30 sweep on `sidechain_edge_scale`:

| scale | Bridge-B RMSD | Bond(gt) | Bottom | Type | Ring | Composite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.3406 | 0.5419 | 0.1607 | 0.4519 | 0.8981 | 0.5711 |
| 0.05 | 0.3392 | 0.5414 | 0.1573 | 0.4563 | 0.8940 | 0.5704 |
| 0.10 | 0.3378 | 0.5470 | 0.1673 | 0.4536 | 0.8904 | 0.5726 |
| 0.15 | 0.3370 | 0.5639 | 0.1703 | 0.4508 | 0.8873 | 0.5751 |

Against the same baseline (`RMSD 0.3374`, `Bond(gt) 0.5298`, `Bottom 0.0187`, `Type 0.4411`, `Ring 0.8442`, `Composite 0.5322`),
`sidechain_edge_scale = 0.15` is the strongest point in this sweep.

## 2. Full Val/Test Repeat Evaluation

5-seed summary (`11,22,33,44,55`) against the previous bond-aware Bridge-B formal eval:

### Val

| Metric | New Mean ± Std | Old Mean ± Std | Delta |
| --- | ---: | ---: | ---: |
| RMSD | 0.3456 ± 0.0031 | 0.5844 ± 0.0061 | -0.2388 |
| Bond(gt) | 0.6510 ± 0.0083 | 0.5391 ± 0.0120 | +0.1119 |
| Bond(pred) | 0.6566 ± 0.0105 | 0.5360 ± 0.0107 | +0.1207 |
| Count | 0.2812 ± 0.0000 | 0.2812 ± 0.0000 | +0.0000 |
| Type | 0.4213 ± 0.0031 | 0.4064 ± 0.0068 | +0.0150 |
| Bottom | 0.1066 ± 0.0097 | 0.0716 ± 0.0108 | +0.0351 |
| Ring | 0.9057 ± 0.0055 | 0.9197 ± 0.0031 | -0.0140 |
| Composite | 0.5781 ± 0.0028 | 0.5167 ± 0.0039 | +0.0614 |

### Test

| Metric | New Mean ± Std | Old Mean ± Std | Delta |
| --- | ---: | ---: | ---: |
| RMSD | 0.3074 ± 0.0053 | 0.4842 ± 0.0199 | -0.1768 |
| Bond(gt) | 0.6584 ± 0.0018 | 0.5282 ± 0.0148 | +0.1303 |
| Bond(pred) | 0.6531 ± 0.0045 | 0.5208 ± 0.0174 | +0.1323 |
| Count | 0.3398 ± 0.0000 | 0.3398 ± 0.0000 | +0.0000 |
| Type | 0.4053 ± 0.0028 | 0.3889 ± 0.0047 | +0.0163 |
| Bottom | 0.0703 ± 0.0148 | 0.0592 ± 0.0182 | +0.0110 |
| Ring | 0.9083 ± 0.0028 | 0.9199 ± 0.0023 | -0.0116 |
| Composite | 0.5836 ± 0.0033 | 0.5328 ± 0.0076 | +0.0507 |

## 3. Updated RMSD Diagnosis

Test split, seed `42`:

- `mean delta rmsd = -0.1912`
- `mean delta bond = +0.1656`
- `mean delta bottom = +0.0173`
- `mean delta scaffold_rmsd = -0.1317`
- `mean delta non_scaffold_rmsd = -0.2185`
- `mean delta attachment_rmsd = -0.1202`
- `mean delta edge_mae = -0.3759`

Comparison to the previous diagnosis:

- Old `mean delta non_scaffold_rmsd = -0.0029`
- New `mean delta non_scaffold_rmsd = -0.2185`

This is the key change. The previous bond-aware Bridge-B mainly improved scaffold-local geometry but barely touched the
non-scaffold region. After introducing sidechain edge correction, the dominant RMSD bottleneck finally moves in the right
direction.

The correlation structure is still informative:

- `corr(delta_rmsd, delta_non_scaffold_rmsd) = 0.9826`
- `corr(delta_rmsd, delta_attachment_rmsd) = 0.7687`
- `corr(delta_rmsd, delta_scaffold_rmsd) = 0.7498`

So the largest remaining source of RMSD variance is still the non-scaffold region, but it is no longer untouched.

## 4. Interpretation

The result is now much clearer than before:

1. `ring-only scaffold` is under-specified.
2. `ring-system + attachment/sidechain edge` is a materially better explicit structure layer.
3. The new sidechain edge term improves `Bond`, `Bottom`, `Type`, `Composite`, and also `RMSD`.
4. `Count` remains unchanged, so count is still an orthogonal bottleneck.
5. `Ring` drops slightly while the overall structure quality rises, which means the current objective is trading a small amount
   of ring preservation for a larger gain in whole-molecule geometry and local chemistry.

## 5. Decision

Current Bridge-B default for evaluation should be:

- `edge_scale = 0.12`
- `sidechain_edge_scale = 0.15`

The next structural-layer step should not go back to pure ring scaffold. It should move forward as:

**Ring-system scaffold + attachment/sidechain graph**

Concretely, the next V17 target should include:

- scaffold-local edges
- scaffold-to-sidechain edges
- attachment anchor semantics
- predicted sidechain / attachment relations as first-class structure tokens

This direction is now supported by both sweep-level evidence and full-set repeat evaluation.
