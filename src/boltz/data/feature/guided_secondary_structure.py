from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from boltz.data import const
from boltz.data.feature.guided_distance import decode_atom_name
from boltz.data.parse.selection import parse_selection
from boltz.data.types import (
    GuidedSecondaryStructureConstraintInfo,
    Structure,
    StructureV2,
    Tokenized,
)

GUIDED_SECONDARY_STRUCTURE_TYPE_IDS = {
    "helix": 0,
    "sheet": 1,
    "loop": 2,
}

BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")


def _get_structure(data: Tokenized | Structure | StructureV2) -> Structure | StructureV2:
    if hasattr(data, "structure"):
        return data.structure
    return data


def _build_residue_contexts(
    data: Tokenized | Structure | StructureV2,
) -> tuple[list[dict[str, int | str]], list[dict[str, object]]]:
    structure = _get_structure(data)
    residue_contexts = []
    residue_records = []

    for chain in structure.chains[structure.mask]:
        if int(chain["mol_type"]) != const.chain_type_ids["PROTEIN"]:
            continue
        chain_name = str(chain["name"]).upper()
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        for chain_res_idx, residue in enumerate(
            structure.residues[res_start:res_end], start=1
        ):
            residue_idx = res_start + chain_res_idx - 1
            residue_contexts.append(
                {
                    "chain": chain_name,
                    "resid": chain_res_idx,
                    "name": str(residue["name"]).upper(),
                    "index": residue_idx + 1,
                }
            )
            residue_records.append(
                {
                    "residue_idx": residue_idx,
                    "residue": residue,
                }
            )

    return residue_contexts, residue_records


def resolve_guided_secondary_structure_constraints(
    data: Tokenized | Structure | StructureV2,
    constraints: Optional[list[GuidedSecondaryStructureConstraintInfo]],
) -> list[dict[str, object]]:
    """Resolve guided secondary-structure constraints to residue spans."""

    if not constraints:
        return []

    residue_contexts, residue_records = _build_residue_contexts(data)
    resolved_constraints = []

    for constraint in constraints:
        selection = parse_selection(constraint.selection)
        matches = [
            (record, context)
            for record, context in zip(residue_records, residue_contexts, strict=True)
            if selection.evaluate(context)
        ]
        if not matches:
            msg = (
                "Guided secondary-structure selection matched no protein residues: "
                f"{constraint.selection!r}"
            )
            raise ValueError(msg)

        resolved_constraints.append(
            {
                "constraint": constraint,
                "residue_records": tuple(record for record, _ in matches),
                "contexts": tuple(context for _, context in matches),
            }
        )

    return resolved_constraints


def build_guided_secondary_structure_features(
    data: Tokenized,
    constraints: Optional[list[GuidedSecondaryStructureConstraintInfo]],
) -> dict[str, Tensor]:
    """Resolve guided secondary-structure selections to residue-level tensors."""

    if not constraints:
        return empty_guided_secondary_structure_features()

    resolved_constraints = resolve_guided_secondary_structure_constraints(data, constraints)
    atom_index = []
    constraint_index = []
    donor_mask = []
    constraint_type = []

    for constraint_idx, resolved_constraint in enumerate(resolved_constraints):
        constraint = resolved_constraint["constraint"]
        constraint_type.append(
            GUIDED_SECONDARY_STRUCTURE_TYPE_IDS[constraint.constraint_type]
        )
        for residue_record in resolved_constraint["residue_records"]:
            residue = residue_record["residue"]
            atom_start = int(residue["atom_idx"])
            atom_end = atom_start + int(residue["atom_num"])
            atom_name_to_index = {
                decode_atom_name(data.structure.atoms[atom_idx]["name"]): atom_idx
                for atom_idx in range(atom_start, atom_end)
            }
            missing_atoms = [
                atom_name
                for atom_name in BACKBONE_ATOM_NAMES
                if atom_name not in atom_name_to_index
            ]
            if missing_atoms:
                msg = (
                    "Guided secondary-structure selections require protein residues "
                    "with backbone atoms N, CA, C, and O present. Missing "
                    f"{missing_atoms} in residue {residue['name']} "
                    f"(global index {residue_record['residue_idx'] + 1})."
                )
                raise ValueError(msg)

            atom_index.append(
                [atom_name_to_index[atom_name] for atom_name in BACKBONE_ATOM_NAMES]
            )
            constraint_index.append(constraint_idx)
            residue_name = str(residue["name"]).upper()
            donor_mask.append(residue_name != "PRO")

    return {
        "guided_secondary_structure_atom_index": torch.tensor(
            atom_index, dtype=torch.long
        ),
        "guided_secondary_structure_constraint_index": torch.tensor(
            constraint_index, dtype=torch.long
        ),
        "guided_secondary_structure_type": torch.tensor(
            constraint_type, dtype=torch.long
        ),
        "guided_secondary_structure_donor_mask": torch.tensor(
            donor_mask, dtype=torch.bool
        ),
    }


def empty_guided_secondary_structure_features() -> dict[str, Tensor]:
    """Return an empty guided secondary-structure feature payload."""

    return {
        "guided_secondary_structure_atom_index": torch.empty((0, 4), dtype=torch.long),
        "guided_secondary_structure_constraint_index": torch.empty((0,), dtype=torch.long),
        "guided_secondary_structure_type": torch.empty((0,), dtype=torch.long),
        "guided_secondary_structure_donor_mask": torch.empty((0,), dtype=torch.bool),
    }
