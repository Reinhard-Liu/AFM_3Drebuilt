# V18 Directed Visual Review

Fixed validation split: `val_size=256`, `max_samples=4096`, `require_ring=true`

| Model | fidelity | count_exact | heavy_rmsd | type_acc | hetero_f1 | ring_complete | attach_edge_f1 | bond_pred |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v16d1 | 0.4223 | 0.1667 | 0.3241 | 0.4832 | 0.0000 | 0.0000 | 0.0000 | 0.4775 |
| v18_slot_hard | 0.4277 | 0.1667 | 0.3211 | 0.4535 | 0.0000 | 0.0000 | 0.0000 | 0.5642 |
| v18_slot_graph | 0.4148 | 0.1667 | 0.3342 | 0.4195 | 0.0000 | 0.0000 | 0.0000 | 0.4983 |

Images are stored in `images/`.