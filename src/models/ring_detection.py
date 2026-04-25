"""
Ring detection module for molecular structure analysis.

This is a preprocessing module (not a neural network) that detects rings
in molecular structures, classifies them, and computes geometric properties
(centers, normals, rigid-body transforms) for downstream use.

All coordinates are expected in NORMALIZED space (divided by 12.0 Angstroms)
unless otherwise specified.
"""

import numpy as np
from collections import deque

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RINGS = 10
MAX_RING_SIZE = 6

RING_TYPE_TO_IDX = {
    "benzene": 0,
    "pyridine": 1,
    "pyrimidine": 2,
    "furan": 3,
    "thiophene": 4,
    "cyclopentane": 5,
    "cyclohexane": 6,
    "other_5": 7,
    "other_6": 8,
}

# ---------------------------------------------------------------------------
# Bond length lookup table (normalized space, Angstroms / 12.0)
# ---------------------------------------------------------------------------

_BOND_TOLERANCE = 1.2

MAX_BOND_LENGTHS = {
    frozenset({"C", "C"}):  1.54 / 12.0,   # 0.128  (single bond used as max)
    frozenset({"C", "H"}):  1.09 / 12.0,   # 0.091
    frozenset({"C", "N"}):  1.47 / 12.0,   # 0.122
    frozenset({"C", "O"}):  1.43 / 12.0,   # 0.119
    frozenset({"C", "S"}):  1.82 / 12.0,   # 0.152
    frozenset({"N", "H"}):  1.01 / 12.0,   # 0.084
    frozenset({"O", "H"}):  0.96 / 12.0,   # 0.080
    frozenset({"N", "N"}):  1.45 / 12.0,   # 0.121
    # S-C is covered by frozenset({"C", "S"}) above
}

# ---------------------------------------------------------------------------
# Ring templates – canonical local-frame coordinates (centroid at origin,
# ring plane = xy, normal along +z).
# ---------------------------------------------------------------------------

def _regular_polygon(n, radius):
    """Return (n, 3) array of vertices of a regular n-gon in the xy-plane."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords = np.zeros((n, 3), dtype=np.float64)
    coords[:, 0] = radius * np.cos(angles)
    coords[:, 1] = radius * np.sin(angles)
    return coords


# Aromatic C-C bond ≈ 1.40 Å → radius of circumscribed circle of regular
# hexagon equals the side length for a regular hexagon.
_BENZENE_RADIUS = 1.40 / 12.0  # 0.1167 in normalized space

# For a regular pentagon with side = 1.54/12.0, the circumradius is
# side / (2 * sin(pi/5)).
_CYCLOPENTANE_SIDE = 1.54 / 12.0
_CYCLOPENTANE_RADIUS = _CYCLOPENTANE_SIDE / (2.0 * np.sin(np.pi / 5.0))

RING_TEMPLATES = {
    "benzene":      _regular_polygon(6, _BENZENE_RADIUS),
    "pyridine":     _regular_polygon(6, _BENZENE_RADIUS),
    "pyrimidine":   _regular_polygon(6, _BENZENE_RADIUS),
    "cyclohexane":  _regular_polygon(6, _BENZENE_RADIUS),
    "other_6":      _regular_polygon(6, _BENZENE_RADIUS),
    "cyclopentane": _regular_polygon(5, _CYCLOPENTANE_RADIUS),
    "furan":        _regular_polygon(5, _CYCLOPENTANE_RADIUS),
    "thiophene":    _regular_polygon(5, _CYCLOPENTANE_RADIUS),
    "other_5":      _regular_polygon(5, _CYCLOPENTANE_RADIUS),
}

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_molecular_graph(coords, elements):
    """Build an adjacency list from atomic coordinates and element types.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
        Atom positions in **normalized** space (Angstroms / 12.0).
    elements : list[str]
        Element symbols for each atom (e.g. ``["C", "H", "O", ...]``).

    Returns
    -------
    adj : dict[int, set[int]]
        Adjacency list – ``adj[i]`` is the set of atom indices bonded to
        atom *i*.
    """
    n = len(elements)
    adj = {i: set() for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            pair = frozenset({elements[i], elements[j]})
            max_len = MAX_BOND_LENGTHS.get(pair)
            if max_len is None:
                continue
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < _BOND_TOLERANCE * max_len:
                adj[i].add(j)
                adj[j].add(i)

    return adj

# ---------------------------------------------------------------------------
# Ring finding (BFS-based simple-cycle detection)
# ---------------------------------------------------------------------------

def find_rings(adj, max_size=6):
    """Find all simple cycles of size 5 and 6 in the molecular graph.

    Uses a BFS-based approach: for every edge (u, v) the algorithm searches
    for a shortest path from *v* back to *u* that does not re-use the
    edge (u, v).  Only cycles of length 5 or 6 are kept.

    Parameters
    ----------
    adj : dict[int, set[int]]
        Adjacency list produced by :func:`build_molecular_graph`.
    max_size : int, optional
        Maximum ring size to detect (default 6).

    Returns
    -------
    rings : list[list[int]]
        Each element is a list of atom indices forming a ring.  Rings are
        deduplicated (canonical form = sorted tuple of atom indices).
    """
    seen_canonical = set()
    rings = []

    nodes = sorted(adj.keys())

    for start in nodes:
        for nbr in sorted(adj[start]):
            if nbr <= start:
                # Avoid processing each edge twice in one direction
                continue

            # BFS from nbr back to start, without using the direct
            # edge (start, nbr).
            # Each queue entry: (current_node, path_from_start)
            queue = deque()
            queue.append((nbr, [start, nbr]))
            visited = {start, nbr}

            while queue:
                cur, path = queue.popleft()
                if len(path) > max_size:
                    break

                for nxt in sorted(adj[cur]):
                    if nxt == start and len(path) >= 5:
                        # Found a cycle back to start
                        ring = path[:]
                        canonical = tuple(sorted(ring))
                        if canonical not in seen_canonical:
                            seen_canonical.add(canonical)
                            rings.append(ring)
                        continue

                    if nxt in visited:
                        continue

                    if len(path) < max_size:
                        visited_copy = visited | {nxt}
                        queue.append((nxt, path + [nxt]))
                        # Note: we intentionally do NOT mutate `visited`
                        # here so that other branches can still explore
                        # `nxt`.  We use per-path visited tracking below.

            # The simple BFS above shares `visited` globally which is
            # intentionally conservative – it prunes many false paths but
            # may miss some rings.  A more thorough (but still fast for
            # small molecules) DFS variant follows.

        # DFS variant for completeness – finds rings that BFS may miss.
        _dfs_find_rings(adj, start, max_size, seen_canonical, rings)

    return rings


def _dfs_find_rings(adj, start, max_size, seen_canonical, rings):
    """DFS helper that enumerates simple cycles containing *start*."""
    stack = [(start, [start], {start})]

    while stack:
        cur, path, visited = stack.pop()

        for nxt in sorted(adj[cur]):
            if nxt == start and 5 <= len(path) <= max_size:
                canonical = tuple(sorted(path))
                if canonical not in seen_canonical:
                    seen_canonical.add(canonical)
                    rings.append(list(path))
                continue

            if nxt in visited:
                continue

            if nxt < start:
                # Only enumerate cycles where start is the smallest index
                # to avoid duplicates across different start nodes.
                continue

            if len(path) < max_size:
                stack.append((nxt, path + [nxt], visited | {nxt}))

# ---------------------------------------------------------------------------
# Ring classification
# ---------------------------------------------------------------------------

def classify_ring(ring_atoms, elements):
    """Classify a ring based on its constituent elements.

    Parameters
    ----------
    ring_atoms : list[int]
        Atom indices forming the ring.
    elements : list[str]
        Full element list for the molecule.

    Returns
    -------
    ring_type : str
        One of ``"benzene"``, ``"pyridine"``, ``"pyrimidine"``, ``"furan"``,
        ``"thiophene"``, ``"cyclopentane"``, ``"cyclohexane"``,
        ``"other_5"``, ``"other_6"``.
    """
    ring_elements = [elements[i] for i in ring_atoms]
    size = len(ring_atoms)

    counts = {}
    for e in ring_elements:
        counts[e] = counts.get(e, 0) + 1

    n_c = counts.get("C", 0)
    n_n = counts.get("N", 0)
    n_o = counts.get("O", 0)
    n_s = counts.get("S", 0)

    if size == 6:
        if n_c == 6:
            return "benzene"
        if n_c == 5 and n_n == 1:
            return "pyridine"
        if n_c == 4 and n_n == 2:
            return "pyrimidine"
        return "other_6"
    elif size == 5:
        if n_c == 5:
            return "cyclopentane"
        if n_c == 4 and n_o == 1:
            return "furan"
        if n_c == 4 and n_s == 1:
            return "thiophene"
        return "other_5"

    # Fallback (should not happen with max_size=6, min_size=5)
    if size <= 5:
        return "other_5"
    return "other_6"

# ---------------------------------------------------------------------------
# Rigid-body geometry
# ---------------------------------------------------------------------------

def compute_ring_rigid_body(coords, ring_indices):
    """Compute the rigid-body parameters of a detected ring.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
        Full coordinate array (normalized space).
    ring_indices : list[int]
        Atom indices of the ring.

    Returns
    -------
    info : dict
        ``"center"``          – (3,) centroid of ring atoms.
        ``"normal"``          – (3,) unit normal to best-fit plane (via SVD).
        ``"rotation_matrix"`` – (3, 3) rotation from template frame to
                                molecular frame.
    """
    pts = coords[ring_indices]  # (n, 3)
    center = pts.mean(axis=0)
    centered = pts - center

    # SVD to get the plane normal
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[2]  # smallest singular value direction
    # Ensure consistent orientation (positive z component)
    if normal[2] < 0:
        normal = -normal

    # Build a local frame: z = normal, x = projection of first atom
    z_axis = normal / (np.linalg.norm(normal) + 1e-12)
    v0 = centered[0]
    x_axis = v0 - np.dot(v0, z_axis) * z_axis
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-12:
        # Degenerate – pick an arbitrary perpendicular
        x_axis = np.array([1.0, 0.0, 0.0]) - np.dot(
            np.array([1.0, 0.0, 0.0]), z_axis
        ) * z_axis
        x_norm = np.linalg.norm(x_axis)
    x_axis = x_axis / x_norm
    y_axis = np.cross(z_axis, x_axis)

    # Rotation matrix: columns are the local frame axes expressed in the
    # global frame.  R @ template_coords^T gives molecular-frame coords.
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

    return {
        "center": center,
        "normal": normal,
        "rotation_matrix": rotation_matrix,
    }

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_rings(coords, elements, normalized=True):
    """Detect and characterize rings in a molecular structure.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
        Atom coordinates.
    elements : list[str]
        Element symbols.
    normalized : bool, optional
        If ``True`` (default), *coords* are already in normalized space
        (Angstroms / 12.0).  If ``False``, they will be divided by 12.0.

    Returns
    -------
    ring_info : dict
        ``"n_rings"``        – int, number of detected rings.
        ``"ring_indices"``   – list of lists of atom indices.
        ``"ring_types"``     – list of ring-type strings.
        ``"ring_centers"``   – (n_rings, 3) np.ndarray.
        ``"ring_normals"``   – (n_rings, 3) np.ndarray.
        ``"ring_templates"`` – list of (ring_size, 3) np.ndarray.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if not normalized:
        coords = coords / 12.0

    adj = build_molecular_graph(coords, elements)
    rings = find_rings(adj, max_size=6)

    n_rings = len(rings)
    ring_types = []
    ring_centers = []
    ring_normals = []
    ring_templates = []

    for ring in rings:
        rtype = classify_ring(ring, elements)
        ring_types.append(rtype)

        rb = compute_ring_rigid_body(coords, ring)
        ring_centers.append(rb["center"])
        ring_normals.append(rb["normal"])

        template = RING_TEMPLATES.get(rtype)
        if template is not None:
            ring_templates.append(template.copy())
        else:
            # Fallback: generate a regular polygon of the correct size
            size = len(ring)
            ring_templates.append(_regular_polygon(size, _BENZENE_RADIUS))

    ring_centers = np.array(ring_centers, dtype=np.float64).reshape(n_rings, 3)
    ring_normals = np.array(ring_normals, dtype=np.float64).reshape(n_rings, 3)

    return {
        "n_rings": n_rings,
        "ring_indices": rings,
        "ring_types": ring_types,
        "ring_centers": ring_centers,
        "ring_normals": ring_normals,
        "ring_templates": ring_templates,
    }

# ---------------------------------------------------------------------------
# Padding for batched processing
# ---------------------------------------------------------------------------

def pad_ring_info(ring_info, max_rings=MAX_RINGS, max_ring_size=MAX_RING_SIZE,
                  max_atoms=85):
    """Pad ring detection results to fixed sizes suitable for batching.

    Parameters
    ----------
    ring_info : dict
        Output of :func:`detect_rings`.
    max_rings : int, optional
        Maximum number of rings (default ``MAX_RINGS = 10``).
    max_ring_size : int, optional
        Maximum atoms per ring (default ``MAX_RING_SIZE = 6``).
    max_atoms : int, optional
        Maximum atoms in the molecule (default 85).  Used only for
        documentation / downstream compatibility.

    Returns
    -------
    padded : dict
        Fixed-size numpy arrays ready for conversion to tensors:

        - ``"ring_centers"``      – (max_rings, 3) float32
        - ``"ring_normals"``      – (max_rings, 3) float32
        - ``"ring_types"``        – (max_rings,) int64
        - ``"ring_atom_indices"`` – (max_rings, max_ring_size) int64,
          padded with -1
        - ``"ring_templates"``    – (max_rings, max_ring_size, 3) float32
        - ``"ring_valid"``        – (max_rings,) float32
        - ``"n_rings"``           – int
    """
    n = min(ring_info["n_rings"], max_rings)

    ring_centers = np.zeros((max_rings, 3), dtype=np.float32)
    ring_normals = np.zeros((max_rings, 3), dtype=np.float32)
    ring_types = np.zeros((max_rings,), dtype=np.int64)
    ring_atom_indices = np.full((max_rings, max_ring_size), -1, dtype=np.int64)
    ring_templates = np.zeros((max_rings, max_ring_size, 3), dtype=np.float32)
    ring_valid = np.zeros((max_rings,), dtype=np.float32)

    for i in range(n):
        ring_centers[i] = ring_info["ring_centers"][i]
        ring_normals[i] = ring_info["ring_normals"][i]
        ring_types[i] = RING_TYPE_TO_IDX.get(ring_info["ring_types"][i], 8)
        ring_valid[i] = 1.0

        indices = ring_info["ring_indices"][i]
        ring_size = min(len(indices), max_ring_size)
        ring_atom_indices[i, :ring_size] = indices[:ring_size]

        tmpl = ring_info["ring_templates"][i]
        tmpl_size = min(tmpl.shape[0], max_ring_size)
        ring_templates[i, :tmpl_size] = tmpl[:tmpl_size]

    return {
        "ring_centers": ring_centers,
        "ring_normals": ring_normals,
        "ring_types": ring_types,
        "ring_atom_indices": ring_atom_indices,
        "ring_templates": ring_templates,
        "ring_valid": ring_valid,
        "n_rings": n,
    }

# ---------------------------------------------------------------------------
# V17-Bridge: ring-system scaffold labels
# ---------------------------------------------------------------------------

MAX_RING_SYSTEMS = MAX_RINGS
MAX_SYSTEM_RELATIONS = 16
MAX_SYSTEM_MEMBERSHIP = 3
MAX_SYSTEM_LOCAL_EDGES = 64
MAX_SYSTEM_SIDECHAIN_EDGES = 128
ATOM_ROLE_TO_IDX = {
    "scaffold_core": 0,
    "attachment_anchor": 1,
    "sidechain": 2,
}
ATOM_SEMANTIC_TYPE_TO_IDX = {
    "H": 0,
    "C": 1,
    "N": 2,
    "O": 3,
    "F": 4,
    "S": 5,
    "P": 6,
    "Cl": 7,
    "Br": 8,
    "I": 9,
}
RELATION_TYPE_TO_IDX = {
    "none": 0,
    "fused": 1,
    "spiro": 2,
    "bridged": 3,
    "linked": 4,
}


def _build_rdkit_mol_from_coords(coords_norm, elements):
    """Build an RDKit molecule from normalized coordinates.

    Returns None if RDKit is unavailable or bond perception fails.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
    except ImportError:
        return None

    n_atoms = len(elements)
    coords_ang = np.asarray(coords_norm, dtype=np.float64) * 12.0

    try:
        rwmol = Chem.RWMol()
        conf = Chem.Conformer(n_atoms)
        for i, (elem, coord) in enumerate(zip(elements, coords_ang)):
            atom = Chem.Atom(elem)
            rwmol.AddAtom(atom)
            conf.SetAtomPosition(i, coord.tolist())
        rwmol.AddConformer(conf, assignId=True)
        rdDetermineBonds.DetermineBonds(rwmol)
        return rwmol.GetMol()
    except Exception:
        return None


def _ring_relation_type(mol, ring_a, ring_b):
    """Classify the relation between two rings.

    V17-Bridge intentionally keeps this conservative. We only emit relation
    types that are stable under current RDKit-derived labels:
      - fused: share two or more atoms (shared bond / fused topology)
      - spiro: share exactly one atom
      - linked: no shared atoms, but ring atoms have a direct bond between rings
      - none: otherwise

    `bridged` remains reserved in RELATION_TYPE_TO_IDX for a future, cleaner
    labeler, but is not emitted yet because the current short-path heuristic
    produced noisy pseudo-bridged labels.
    """
    set_a = set(ring_a)
    set_b = set(ring_b)
    shared = set_a & set_b
    if len(shared) >= 2:
        return "fused"
    if len(shared) == 1:
        return "spiro"

    for ai in ring_a:
        atom = mol.GetAtomWithIdx(ai)
        for nb in atom.GetNeighbors():
            if nb.GetIdx() in set_b:
                return "linked"

    return "none"


def _ring_system_components(mol, atom_rings):
    """Group rings into ring systems.

    Ring systems are connected components under fused/spiro relations.
    Linked rings (for example biphenyl) remain separate systems because they
    are scaffold-related but not a single ring system.
    """
    n_rings = len(atom_rings)
    adj = {i: set() for i in range(n_rings)}
    for i in range(n_rings):
        for j in range(i + 1, n_rings):
            rel = _ring_relation_type(mol, atom_rings[i], atom_rings[j])
            if rel in {"fused", "spiro"}:
                adj[i].add(j)
                adj[j].add(i)

    systems = []
    visited = set()
    for i in range(n_rings):
        if i in visited:
            continue
        stack = [i]
        comp = []
        visited.add(i)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        systems.append(sorted(comp))
    return systems


def compute_ring_system_scaffold_labels(coords_norm, elements, max_atoms=85):
    """Compute V17-Bridge ring-system scaffold labels.

    This first bridge step focuses on stable ring-system / relation / attachment
    labels. Canonical site numbering remains disabled for now because symmetry
    makes the raw ring-order label unstable.
    """
    mol = _build_rdkit_mol_from_coords(coords_norm, elements)

    empty = {
        "scaffold_n_ring_systems": 0,
        "scaffold_system_objectness": np.zeros(MAX_RING_SYSTEMS, dtype=np.float32),
        "scaffold_system_num_rings": np.zeros(MAX_RING_SYSTEMS, dtype=np.int64),
        "scaffold_system_num_atoms": np.zeros(MAX_RING_SYSTEMS, dtype=np.int64),
        "scaffold_system_site_count": np.zeros(MAX_RING_SYSTEMS, dtype=np.int64),
        "scaffold_system_attachment_anchor_count": np.zeros(MAX_RING_SYSTEMS, dtype=np.int64),
        "scaffold_system_external_atom_count": np.zeros(MAX_RING_SYSTEMS, dtype=np.int64),
        "scaffold_system_sidechain_edge_count": np.zeros(MAX_RING_SYSTEMS, dtype=np.int64),
        "scaffold_system_aromaticity": np.zeros(MAX_RING_SYSTEMS, dtype=np.float32),
        "scaffold_system_has_heteroatom": np.zeros(MAX_RING_SYSTEMS, dtype=np.float32),
        "scaffold_system_center": np.zeros((MAX_RING_SYSTEMS, 3), dtype=np.float32),
        "scaffold_system_normal": np.zeros((MAX_RING_SYSTEMS, 3), dtype=np.float32),
        "scaffold_ring_to_system": np.full(MAX_RINGS, -1, dtype=np.int64),
        "scaffold_system_atom_indices": np.full((MAX_RING_SYSTEMS, max_atoms), -1, dtype=np.int64),
        "scaffold_atom_to_ring_system_ids": np.full((max_atoms, MAX_SYSTEM_MEMBERSHIP), -1, dtype=np.int64),
        "scaffold_atom_is_scaffold": np.zeros(max_atoms, dtype=np.float32),
        "scaffold_atom_is_attachment_anchor": np.zeros(max_atoms, dtype=np.float32),
        "scaffold_atom_graph_depth": np.full(max_atoms, -1, dtype=np.int64),
        "scaffold_atom_root_system_id": np.full(max_atoms, -1, dtype=np.int64),
        "scaffold_atom_role": np.full(max_atoms, -1, dtype=np.int64),
        "scaffold_atom_attachment_site_target": np.full(max_atoms, -1, dtype=np.int64),
        "scaffold_atom_hetero_site_target": np.full(max_atoms, -1, dtype=np.int64),
        "scaffold_atom_parent_system_target": np.full(max_atoms, -1, dtype=np.int64),
        "scaffold_atom_attachment_target_site": np.full(max_atoms, -1, dtype=np.int64),
        "scaffold_atom_hetero_target_class": np.full(max_atoms, -1, dtype=np.int64),
        "scaffold_atom_sidechain_root_site": np.full(max_atoms, -1, dtype=np.int64),
        "scaffold_role_scaffold_core_count": np.int64(0),
        "scaffold_role_attachment_anchor_count": np.int64(0),
        "scaffold_role_sidechain_count": np.int64(0),
        "scaffold_total_scaffold_atoms": np.int64(0),
        "scaffold_total_non_scaffold_atoms": np.int64(0),
        "scaffold_total_attachment_anchors": np.int64(0),
        "scaffold_total_sidechain_edges": np.int64(0),
        "scaffold_relation_edges": np.full((MAX_SYSTEM_RELATIONS, 2), -1, dtype=np.int64),
        "scaffold_relation_types": np.zeros(MAX_SYSTEM_RELATIONS, dtype=np.int64),
        "scaffold_n_relations": 0,
        "scaffold_local_edges": np.full((MAX_SYSTEM_LOCAL_EDGES, 2), -1, dtype=np.int64),
        "scaffold_local_edge_lengths": np.zeros(MAX_SYSTEM_LOCAL_EDGES, dtype=np.float32),
        "scaffold_n_local_edges": 0,
        "scaffold_sidechain_edges": np.full((MAX_SYSTEM_SIDECHAIN_EDGES, 2), -1, dtype=np.int64),
        "scaffold_sidechain_edge_lengths": np.zeros(MAX_SYSTEM_SIDECHAIN_EDGES, dtype=np.float32),
        "scaffold_n_sidechain_edges": 0,
        "scaffold_ring_relation_edges": np.full((MAX_SYSTEM_RELATIONS, 2), -1, dtype=np.int64),
        "scaffold_ring_relation_types": np.zeros(MAX_SYSTEM_RELATIONS, dtype=np.int64),
        "scaffold_n_ring_relations": 0,
        "scaffold_site_labels_valid": np.array(0.0, dtype=np.float32),
        "scaffold_atom_canonical_site_index": np.full(max_atoms, -1, dtype=np.int64),
    }
    if mol is None:
        return empty

    atom_rings_raw = [list(r) for r in mol.GetRingInfo().AtomRings() if len(r) in (5, 6)]
    if not atom_rings_raw:
        return empty

    atom_rings = atom_rings_raw[:MAX_RINGS]
    systems = _ring_system_components(mol, atom_rings)[:MAX_RING_SYSTEMS]

    labels = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in empty.items()}

    ring_rel_idx = 0
    for i in range(len(atom_rings)):
        for j in range(i + 1, len(atom_rings)):
            if ring_rel_idx >= MAX_SYSTEM_RELATIONS:
                break
            rel = _ring_relation_type(mol, atom_rings[i], atom_rings[j])
            if rel != "none":
                labels["scaffold_ring_relation_edges"][ring_rel_idx] = [i, j]
                labels["scaffold_ring_relation_types"][ring_rel_idx] = RELATION_TYPE_TO_IDX[rel]
                ring_rel_idx += 1
    labels["scaffold_n_ring_relations"] = ring_rel_idx
    labels["scaffold_n_ring_systems"] = len(systems)

    ring_to_system = np.full(MAX_RINGS, -1, dtype=np.int64)
    for system_idx, ring_indices in enumerate(systems):
        labels["scaffold_system_objectness"][system_idx] = 1.0
        labels["scaffold_system_num_rings"][system_idx] = len(ring_indices)
        system_atoms = sorted({atom for ring_idx in ring_indices for atom in atom_rings[ring_idx]})
        labels["scaffold_system_num_atoms"][system_idx] = len(system_atoms)
        for pos, atom_idx in enumerate(system_atoms[:max_atoms]):
            labels["scaffold_system_atom_indices"][system_idx, pos] = atom_idx

        for ring_idx in ring_indices:
            ring_to_system[ring_idx] = system_idx

        system_coords = np.asarray(coords_norm)[system_atoms]
        center = system_coords.mean(axis=0)
        labels["scaffold_system_center"][system_idx] = center

        centered = system_coords - center
        if len(system_atoms) >= 3:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]
            if normal[2] < 0:
                normal = -normal
            labels["scaffold_system_normal"][system_idx] = normal

        atom_objs = [mol.GetAtomWithIdx(a) for a in system_atoms]
        labels["scaffold_system_aromaticity"][system_idx] = float(all(a.GetIsAromatic() for a in atom_objs))
        labels["scaffold_system_has_heteroatom"][system_idx] = float(any(a.GetSymbol() not in {"C", "H"} for a in atom_objs))

        system_atom_set = set(system_atoms)
        try:
            from rdkit import Chem
            frag_ranks = list(
                Chem.CanonicalRankAtomsInFragment(mol, atomsToUse=system_atoms, breakTies=True)
            )
            ranked_atoms = sorted(system_atoms, key=lambda atom_idx: (frag_ranks[atom_idx], atom_idx))
        except Exception:
            ranked_atoms = list(system_atoms)

        site_map = {}
        for dense_site_idx, atom_idx in enumerate(ranked_atoms):
            if atom_idx >= max_atoms:
                continue
            site_map[atom_idx] = dense_site_idx
            labels["scaffold_atom_canonical_site_index"][atom_idx] = dense_site_idx
        labels["scaffold_system_site_count"][system_idx] = len(site_map)

        for atom_idx in system_atoms:
            if atom_idx >= max_atoms:
                continue
            labels["scaffold_atom_is_scaffold"][atom_idx] = 1.0
            for slot in range(MAX_SYSTEM_MEMBERSHIP):
                if labels["scaffold_atom_to_ring_system_ids"][atom_idx, slot] == -1:
                    labels["scaffold_atom_to_ring_system_ids"][atom_idx, slot] = system_idx
                    break
            if labels["scaffold_atom_root_system_id"][atom_idx] == -1:
                labels["scaffold_atom_root_system_id"][atom_idx] = system_idx

            atom = mol.GetAtomWithIdx(atom_idx)
            if any(nb.GetIdx() not in system_atom_set for nb in atom.GetNeighbors()):
                labels["scaffold_atom_is_attachment_anchor"][atom_idx] = 1.0
                labels["scaffold_system_attachment_anchor_count"][system_idx] += 1

    labels["scaffold_ring_to_system"] = ring_to_system

    n_local_edges = 0
    coords_arr = np.asarray(coords_norm, dtype=np.float32)
    scaffold_atom_set = {
        atom_idx
        for atom_idx in range(min(max_atoms, len(coords_arr)))
        if labels["scaffold_atom_is_scaffold"][atom_idx] > 0.5
    }
    for bond in mol.GetBonds():
        if n_local_edges >= MAX_SYSTEM_LOCAL_EDGES:
            break
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a not in scaffold_atom_set or b not in scaffold_atom_set:
            continue
        labels["scaffold_local_edges"][n_local_edges] = [a, b]
        labels["scaffold_local_edge_lengths"][n_local_edges] = np.linalg.norm(coords_arr[a] - coords_arr[b])
        n_local_edges += 1
    labels["scaffold_n_local_edges"] = n_local_edges

    # Graph depth from the scaffold set. This gives a stable orientation for
    # scaffold-to-sidechain / sidechain-to-sidechain relation encoding.
    adjacency = [[] for _ in range(len(coords_arr))]
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a < len(coords_arr) and b < len(coords_arr):
            adjacency[a].append(b)
            adjacency[b].append(a)
    q = deque()
    for atom_idx in scaffold_atom_set:
        labels["scaffold_atom_graph_depth"][atom_idx] = 0
        q.append(atom_idx)
    while q:
        cur = q.popleft()
        cur_depth = int(labels["scaffold_atom_graph_depth"][cur])
        for nb in adjacency[cur]:
            if nb >= max_atoms or labels["scaffold_atom_graph_depth"][nb] >= 0:
                continue
            labels["scaffold_atom_graph_depth"][nb] = cur_depth + 1
            labels["scaffold_atom_root_system_id"][nb] = labels["scaffold_atom_root_system_id"][cur]
            q.append(nb)

    anchor_q = deque()
    for atom_idx in scaffold_atom_set:
        site_idx = int(labels["scaffold_atom_canonical_site_index"][atom_idx])
        if site_idx >= 0:
            labels["scaffold_atom_sidechain_root_site"][atom_idx] = site_idx
        if labels["scaffold_atom_is_attachment_anchor"][atom_idx] > 0.5 and site_idx >= 0:
            labels["scaffold_atom_attachment_target_site"][atom_idx] = site_idx
            anchor_q.append((atom_idx, site_idx))

    while anchor_q:
        cur, root_site_idx = anchor_q.popleft()
        for nb in adjacency[cur]:
            if nb >= max_atoms:
                continue
            if labels["scaffold_atom_is_scaffold"][nb] > 0.5:
                continue
            if labels["scaffold_atom_sidechain_root_site"][nb] >= 0:
                continue
            labels["scaffold_atom_sidechain_root_site"][nb] = root_site_idx
            labels["scaffold_atom_attachment_target_site"][nb] = root_site_idx
            anchor_q.append((nb, root_site_idx))

    valid_atom_count = min(max_atoms, len(coords_arr))
    labels["scaffold_total_scaffold_atoms"] = np.int64(
        int(labels["scaffold_atom_is_scaffold"][:valid_atom_count].sum())
    )
    non_scaffold_total = 0
    for atom_idx in range(valid_atom_count):
        if atom_idx in scaffold_atom_set:
            continue
        non_scaffold_total += 1
        root_system = int(labels["scaffold_atom_root_system_id"][atom_idx])
        if 0 <= root_system < MAX_RING_SYSTEMS:
            labels["scaffold_system_external_atom_count"][root_system] += 1
    labels["scaffold_total_non_scaffold_atoms"] = np.int64(non_scaffold_total)
    labels["scaffold_total_attachment_anchors"] = np.int64(
        int(labels["scaffold_atom_is_attachment_anchor"][:valid_atom_count].sum())
    )

    n_sidechain_edges = 0
    for bond in mol.GetBonds():
        if n_sidechain_edges >= MAX_SYSTEM_SIDECHAIN_EDGES:
            break
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a >= max_atoms or b >= max_atoms:
            continue
        a_scaffold = a in scaffold_atom_set
        b_scaffold = b in scaffold_atom_set
        if a_scaffold and b_scaffold:
            continue
        labels["scaffold_sidechain_edges"][n_sidechain_edges] = [a, b]
        labels["scaffold_sidechain_edge_lengths"][n_sidechain_edges] = np.linalg.norm(coords_arr[a] - coords_arr[b])

        root_system = -1
        if a_scaffold and 0 <= int(labels["scaffold_atom_root_system_id"][a]) < MAX_RING_SYSTEMS:
            root_system = int(labels["scaffold_atom_root_system_id"][a])
        elif b_scaffold and 0 <= int(labels["scaffold_atom_root_system_id"][b]) < MAX_RING_SYSTEMS:
            root_system = int(labels["scaffold_atom_root_system_id"][b])
        elif 0 <= int(labels["scaffold_atom_root_system_id"][a]) < MAX_RING_SYSTEMS:
            root_system = int(labels["scaffold_atom_root_system_id"][a])
        elif 0 <= int(labels["scaffold_atom_root_system_id"][b]) < MAX_RING_SYSTEMS:
            root_system = int(labels["scaffold_atom_root_system_id"][b])
        if root_system >= 0:
            labels["scaffold_system_sidechain_edge_count"][root_system] += 1

        n_sidechain_edges += 1
    labels["scaffold_n_sidechain_edges"] = n_sidechain_edges
    labels["scaffold_total_sidechain_edges"] = np.int64(n_sidechain_edges)

    role_counts = {
        "scaffold_core": 0,
        "attachment_anchor": 0,
        "sidechain": 0,
    }
    for atom_idx in range(valid_atom_count):
        is_scaffold = labels["scaffold_atom_is_scaffold"][atom_idx] > 0.5
        is_anchor = labels["scaffold_atom_is_attachment_anchor"][atom_idx] > 0.5
        if is_scaffold and is_anchor:
            role_name = "attachment_anchor"
            parent_system = int(labels["scaffold_atom_to_ring_system_ids"][atom_idx, 0])
        elif is_scaffold:
            role_name = "scaffold_core"
            parent_system = int(labels["scaffold_atom_to_ring_system_ids"][atom_idx, 0])
        else:
            role_name = "sidechain"
            parent_system = int(labels["scaffold_atom_root_system_id"][atom_idx])

        labels["scaffold_atom_role"][atom_idx] = ATOM_ROLE_TO_IDX[role_name]
        labels["scaffold_atom_attachment_site_target"][atom_idx] = 1 if is_anchor else 0
        labels["scaffold_atom_hetero_site_target"][atom_idx] = 1 if elements[atom_idx] not in {"C", "H"} else 0
        labels["scaffold_atom_parent_system_target"][atom_idx] = parent_system
        if is_scaffold:
            labels["scaffold_atom_hetero_target_class"][atom_idx] = ATOM_SEMANTIC_TYPE_TO_IDX.get(
                elements[atom_idx], ATOM_SEMANTIC_TYPE_TO_IDX["C"]
            )
            site_idx = int(labels["scaffold_atom_canonical_site_index"][atom_idx])
            if site_idx >= 0 and labels["scaffold_atom_sidechain_root_site"][atom_idx] < 0:
                labels["scaffold_atom_sidechain_root_site"][atom_idx] = site_idx
            if is_anchor and site_idx >= 0:
                labels["scaffold_atom_attachment_target_site"][atom_idx] = site_idx
        role_counts[role_name] += 1

    labels["scaffold_role_scaffold_core_count"] = np.int64(role_counts["scaffold_core"])
    labels["scaffold_role_attachment_anchor_count"] = np.int64(role_counts["attachment_anchor"])
    labels["scaffold_role_sidechain_count"] = np.int64(role_counts["sidechain"])
    labels["scaffold_site_labels_valid"] = np.array(
        1.0 if np.any(labels["scaffold_atom_canonical_site_index"][:valid_atom_count] >= 0) else 0.0,
        dtype=np.float32,
    )

    n_rel = 0
    order = {"none": 0, "linked": 1, "bridged": 2, "spiro": 3, "fused": 4}
    for i in range(len(systems)):
        for j in range(i + 1, len(systems)):
            if n_rel >= MAX_SYSTEM_RELATIONS:
                break
            rel = "none"
            for ring_i in systems[i]:
                for ring_j in systems[j]:
                    cand = _ring_relation_type(mol, atom_rings[ring_i], atom_rings[ring_j])
                    if order[cand] > order[rel]:
                        rel = cand
            if rel != "none":
                labels["scaffold_relation_edges"][n_rel] = [i, j]
                labels["scaffold_relation_types"][n_rel] = RELATION_TYPE_TO_IDX[rel]
                n_rel += 1
    labels["scaffold_n_relations"] = n_rel

    return labels
