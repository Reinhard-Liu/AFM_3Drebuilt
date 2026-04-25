# EDAFM SUP-03 Prepared Cases

This directory contains real-AFM experimental cases converted into a V20/SUP-03
compatible format.

Key design choices:
- AFM stacks are resampled to `(10, 128, 128)` to match the current model input.
- Values are normalized to `[0, 1]` with percentile clipping.
- Both `normal` and `inverted` contrast variants are stored.
- Molecule coordinates are centered and normalized by `/12.0`, matching the
  current QUAM-AFM training convention.

Use `manifest.json` as the index.
