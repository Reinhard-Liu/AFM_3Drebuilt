"""Export a small sample of V17-Bridge scaffold labels for inspection."""

import argparse
import json
import pickle
from pathlib import Path

from src.data.dataset import parse_xyz, center_coords
from src.models.ring_detection import (
    RELATION_TYPE_TO_IDX,
    compute_ring_system_scaffold_labels,
    detect_rings,
)


def _to_jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="auto")
    parser.add_argument("--param-key", default="K-1")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--output", default="experiments/v17_bridge/scaffold_label_samples.json")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if args.data_root == "auto":
        data_root = Path("/root/autodl-tmp/micro/dataverse_files/SUBMIT_QUAM-AFM/QUAM")

    cache_path = data_root / f"samples_cache_{args.param_key}.pkl"
    with cache_path.open("rb") as f:
        samples = pickle.load(f)

    relation_idx_to_name = {v: k for k, v in RELATION_TYPE_TO_IDX.items()}
    rows = []
    for sample in samples:
        if len(rows) >= args.num_samples:
            break
        xyz_path = Path(sample["xyz_path"])
        if not xyz_path.exists():
            xyz_path = data_root / "XYZ_FILES" / sample["cid"] / f"{sample['cid']}.xyz"
        coords, elements = parse_xyz(str(xyz_path))
        coords = center_coords(coords) / 12.0
        ring_info = detect_rings(coords, elements, normalized=True)
        if int(ring_info["n_rings"]) == 0:
            continue
        labels = compute_ring_system_scaffold_labels(coords, elements, max_atoms=85)
        n_systems = int(labels["scaffold_n_ring_systems"])
        n_rel = int(labels["scaffold_n_relations"])
        n_ring_rel = int(labels["scaffold_n_ring_relations"])
        row = {
            "cid": sample["cid"],
            "n_atoms": len(elements),
            "n_rings": int(ring_info["n_rings"]),
            "scaffold_n_ring_systems": n_systems,
            "scaffold_system_num_rings": _to_jsonable(labels["scaffold_system_num_rings"][:n_systems]),
            "scaffold_system_num_atoms": _to_jsonable(labels["scaffold_system_num_atoms"][:n_systems]),
            "scaffold_system_aromaticity": _to_jsonable(labels["scaffold_system_aromaticity"][:n_systems]),
            "scaffold_system_has_heteroatom": _to_jsonable(labels["scaffold_system_has_heteroatom"][:n_systems]),
            "scaffold_relation_edges": _to_jsonable(labels["scaffold_relation_edges"][:n_rel]),
            "scaffold_relation_types": [relation_idx_to_name.get(int(x), "unknown") for x in labels["scaffold_relation_types"][:n_rel].tolist()],
            "scaffold_ring_relation_edges": _to_jsonable(labels["scaffold_ring_relation_edges"][:n_ring_rel]),
            "scaffold_ring_relation_types": [relation_idx_to_name.get(int(x), "unknown") for x in labels["scaffold_ring_relation_types"][:n_ring_rel].tolist()],
            "attachment_anchor_count": int(labels["scaffold_atom_is_attachment_anchor"].sum()),
            "site_labels_valid": float(labels["scaffold_site_labels_valid"]),
        }
        rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {len(rows)} samples to {output_path}")


if __name__ == "__main__":
    main()
