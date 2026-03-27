from pathlib import Path

import torch

from boltz.data.feature.guided_distance import build_guided_distance_features
from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import Input
from boltz.model.potentials.potentials import GuidedDistancePotential


def build_tokenized(schema: dict):
    target = parse_boltz_schema(
        name="toy",
        schema=schema,
        ccd={},
        mol_dir=Path("~/.boltz/mols").expanduser(),
        boltz_2=True,
    )
    input_data = Input(
        structure=target.structure,
        msa={},
        record=target.record,
        residue_constraints=target.residue_constraints,
        templates=target.templates,
        extra_mols=target.extra_mols,
    )
    return target, Boltz2Tokenizer().tokenize(input_data)


def batch_features(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.unsqueeze(0) for key, value in features.items()}


def test_parse_guided_distance_constraint():
    schema = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": "AA"}},
        ],
        "constraints": [
            {
                "guided_distance": {
                    "selection1": "chain A and resid 1 and name CA",
                    "selection2": "chain A and resid 2 and name CA",
                    "type": "harmonic",
                    "target_distance": 8.0,
                }
            }
        ],
    }

    target, _ = build_tokenized(schema)
    constraints = target.record.inference_options.guided_distance_constraints

    assert constraints is not None
    assert len(constraints) == 1
    assert constraints[0].constraint_type == "harmonic"
    assert constraints[0].target_distance == 8.0


def test_guided_distance_potential_uses_resolved_atom_indices():
    schema = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": "AA"}},
        ],
        "constraints": [
            {
                "guided_distance": {
                    "selection1": "chain A and resid 1 and name CA",
                    "selection2": "chain A and resid 2 and name CA",
                    "type": "harmonic",
                    "target_distance": 3.0,
                }
            }
        ],
    }

    target, tokenized = build_tokenized(schema)
    features = build_guided_distance_features(
        tokenized,
        target.record.inference_options.guided_distance_constraints,
    )
    potential = GuidedDistancePotential(parameters={"resampling_weight": 1.0})

    atom_index = features["guided_distance_atom_index"]
    assert atom_index.numel() == 2

    coords = torch.zeros((1, tokenized.structure.atoms.shape[0], 3), dtype=torch.float32)
    coords[0, atom_index[0]] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, atom_index[1]] = torch.tensor([5.0, 0.0, 0.0])

    energy = potential.compute(
        coords,
        batch_features(features),
        {"resampling_weight": 1.0},
    )

    assert torch.allclose(energy, torch.tensor([4.0]))
