from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor

from boltz.data.parse.selection import parse_selection
from boltz.data.types import (
    GuidedDistanceConstraintInfo,
    Structure,
    StructureV2,
    Tokenized,
)

GUIDED_DISTANCE_TYPE_IDS = {
    "harmonic": 0,
    "flat_bottomed": 1,
}


def decode_atom_name(name: np.ndarray | str) -> str:
    """Decode an atom name from either Boltz atom format."""

    if isinstance(name, str):
        return name.strip().upper()
    values = [int(value) for value in np.asarray(name).tolist()]
    chars = [chr(value + 32) for value in values if value != 0]
    return "".join(chars).strip().upper()


def _build_atom_contexts(
    structure: Structure | StructureV2,
) -> tuple[list[dict[str, int | str]], list[int]]:
    atom_contexts = []
    atom_indices = []
    for chain in structure.chains[structure.mask]:
        chain_name = str(chain["name"]).upper()
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        for chain_res_idx, residue in enumerate(
            structure.residues[res_start:res_end], start=1
        ):
            atom_start = int(residue["atom_idx"])
            atom_end = atom_start + int(residue["atom_num"])
            for atom_idx in range(atom_start, atom_end):
                atom_contexts.append(
                    {
                        "chain": chain_name,
                        "resid": chain_res_idx,
                        "name": decode_atom_name(structure.atoms[atom_idx]["name"]),
                        # Use 1-based indexing for the user-facing selector.
                        "index": atom_idx + 1,
                    }
                )
                atom_indices.append(atom_idx)
    return atom_contexts, atom_indices


def resolve_guided_distance_constraints(
    structure: Structure | StructureV2,
    constraints: Optional[list[GuidedDistanceConstraintInfo]],
) -> list[dict[str, object]]:
    """Resolve guided-distance constraints to matched atom groups."""

    if not constraints:
        return []

    atom_contexts, atom_indices = _build_atom_contexts(structure)
    resolved_constraints = []

    for constraint in constraints:
        selection1 = parse_selection(constraint.selection1)
        selection2 = parse_selection(constraint.selection2)
        group1_matches = [
            (atom_idx, atom_context)
            for atom_idx, atom_context in zip(atom_indices, atom_contexts, strict=True)
            if selection1.evaluate(atom_context)
        ]
        group2_matches = [
            (atom_idx, atom_context)
            for atom_idx, atom_context in zip(atom_indices, atom_contexts, strict=True)
            if selection2.evaluate(atom_context)
        ]
        if not group1_matches:
            msg = (
                "Guided distance selection1 matched no atoms: "
                f"{constraint.selection1!r}"
            )
            raise ValueError(msg)
        if not group2_matches:
            msg = (
                "Guided distance selection2 matched no atoms: "
                f"{constraint.selection2!r}"
            )
            raise ValueError(msg)

        resolved_constraints.append(
            {
                "constraint": constraint,
                "group1_atom_indices": tuple(
                    atom_idx for atom_idx, _ in group1_matches
                ),
                "group2_atom_indices": tuple(
                    atom_idx for atom_idx, _ in group2_matches
                ),
                "group1_contexts": tuple(
                    atom_context for _, atom_context in group1_matches
                ),
                "group2_contexts": tuple(
                    atom_context for _, atom_context in group2_matches
                ),
            }
        )

    return resolved_constraints


def build_guided_distance_features(
    data: Tokenized,
    constraints: Optional[list[GuidedDistanceConstraintInfo]],
) -> dict[str, Tensor]:
    """Resolve guided-distance selections to request-local atom index tensors."""

    if not constraints:
        return empty_guided_distance_features()

    resolved_constraints = resolve_guided_distance_constraints(
        data.structure, constraints
    )
    flat_atom_index = []
    flat_group_index = []
    pair_index = []
    constraint_type = []
    target = []
    lower = []
    upper = []

    for constraint_idx, resolved_constraint in enumerate(resolved_constraints):
        constraint = resolved_constraint["constraint"]
        group1 = resolved_constraint["group1_atom_indices"]
        group2 = resolved_constraint["group2_atom_indices"]
        group1_id = 2 * constraint_idx
        group2_id = group1_id + 1
        pair_index.append([group1_id, group2_id])
        flat_atom_index.extend(group1)
        flat_group_index.extend([group1_id] * len(group1))
        flat_atom_index.extend(group2)
        flat_group_index.extend([group2_id] * len(group2))
        constraint_type.append(GUIDED_DISTANCE_TYPE_IDS[constraint.constraint_type])
        target.append(
            0.0 if constraint.target_distance is None else constraint.target_distance
        )
        lower.append(
            float("-inf") if constraint.lower_bound is None else constraint.lower_bound
        )
        upper.append(float("inf") if constraint.upper_bound is None else constraint.upper_bound)

    return {
        "guided_distance_atom_index": torch.tensor(flat_atom_index, dtype=torch.long),
        "guided_distance_group_index": torch.tensor(flat_group_index, dtype=torch.long),
        "guided_distance_pair_index": torch.tensor(pair_index, dtype=torch.long).T,
        "guided_distance_type": torch.tensor(constraint_type, dtype=torch.long),
        "guided_distance_target": torch.tensor(target, dtype=torch.float32),
        "guided_distance_lower": torch.tensor(lower, dtype=torch.float32),
        "guided_distance_upper": torch.tensor(upper, dtype=torch.float32),
    }


def empty_guided_distance_features() -> dict[str, Tensor]:
    """Return an empty guided-distance feature payload."""

    return {
        "guided_distance_atom_index": torch.empty((0,), dtype=torch.long),
        "guided_distance_group_index": torch.empty((0,), dtype=torch.long),
        "guided_distance_pair_index": torch.empty((2, 0), dtype=torch.long),
        "guided_distance_type": torch.empty((0,), dtype=torch.long),
        "guided_distance_target": torch.empty((0,), dtype=torch.float32),
        "guided_distance_lower": torch.empty((0,), dtype=torch.float32),
        "guided_distance_upper": torch.empty((0,), dtype=torch.float32),
    }
