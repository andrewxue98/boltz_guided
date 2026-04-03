import sys
from types import SimpleNamespace
from pathlib import Path

import boltz.main as main_module
import numpy as np
import pytest
import torch
from click.core import ParameterSource

from boltz.data.feature.guided_distance import (
    build_guided_distance_features,
    decode_atom_name,
    resolve_guided_distance_constraints,
)
from boltz.data.feature.guided_secondary_structure import (
    build_guided_secondary_structure_features,
    resolve_guided_secondary_structure_constraints,
)
from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.parse.yaml import parse_yaml
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import BondV2, AtomV2, Coords, Ensemble, Input, Manifest, StructureV2
from boltz.main import echo_guided_distance_summary
from boltz.model.potentials.potentials import (
    GuidedDistancePotential,
    GuidedSecondaryStructurePotential,
    SymmetricChainCOMPotential,
    get_potentials,
    get_runtime_steering_args,
)
from boltz.model.potentials.schedules import ExponentialInterpolation


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


def test_parse_guided_secondary_structure_constraint():
    schema = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": "AAAAA"}},
        ],
        "constraints": [
            {
                "guided_secondary_structure": {
                    "selection": "chain A and resid 2 to 4",
                    "type": "loop",
                }
            }
        ],
    }

    target, _ = build_tokenized(schema)
    constraints = target.record.inference_options.guided_secondary_structure_constraints

    assert constraints is not None
    assert len(constraints) == 1
    assert constraints[0].selection == "chain A and resid 2 to 4"
    assert constraints[0].constraint_type == "loop"


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


def test_guided_distance_potential_returns_atom_gradient_for_single_atom_groups():
    features = {
        "guided_distance_atom_index": torch.tensor([0, 1], dtype=torch.long),
        "guided_distance_group_index": torch.tensor([0, 1], dtype=torch.long),
        "guided_distance_pair_index": torch.tensor([[0], [1]], dtype=torch.long),
        "guided_distance_type": torch.tensor([0], dtype=torch.long),
        "guided_distance_target": torch.tensor([3.0], dtype=torch.float32),
        "guided_distance_lower": torch.tensor([float("-inf")], dtype=torch.float32),
        "guided_distance_upper": torch.tensor([float("inf")], dtype=torch.float32),
    }
    coords = torch.tensor(
        [[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    potential = GuidedDistancePotential(parameters={"guidance_weight": 1.0})
    gradient = potential.compute_gradient(
        coords,
        batch_features(features),
        {"guidance_weight": 1.0},
    )

    expected = torch.tensor(
        [[[-4.0, 0.0, 0.0], [4.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    assert torch.allclose(gradient, expected)


def test_guided_distance_potential_distributes_group_gradient_across_atoms():
    features = {
        "guided_distance_atom_index": torch.tensor([0, 1, 2], dtype=torch.long),
        "guided_distance_group_index": torch.tensor([0, 0, 1], dtype=torch.long),
        "guided_distance_pair_index": torch.tensor([[0], [1]], dtype=torch.long),
        "guided_distance_type": torch.tensor([0], dtype=torch.long),
        "guided_distance_target": torch.tensor([1.0], dtype=torch.float32),
        "guided_distance_lower": torch.tensor([float("-inf")], dtype=torch.float32),
        "guided_distance_upper": torch.tensor([float("inf")], dtype=torch.float32),
    }
    coords = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    potential = GuidedDistancePotential(parameters={"guidance_weight": 1.0})
    gradient = potential.compute_gradient(
        coords,
        batch_features(features),
        {"guidance_weight": 1.0},
    )

    expected = torch.tensor(
        [[[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    assert torch.allclose(gradient, expected)


def test_resolve_guided_distance_constraints_returns_matched_atoms():
    schema = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": "AA"}},
        ],
        "constraints": [
            {
                "guided_distance": {
                    "selection1": "(resid 1)",
                    "selection2": "(resid 2)",
                    "type": "harmonic",
                    "target_distance": 8.0,
                }
            }
        ],
    }

    target, _ = build_tokenized(schema)
    resolved = resolve_guided_distance_constraints(
        target.structure,
        target.record.inference_options.guided_distance_constraints,
    )

    assert len(resolved) == 1
    assert len(resolved[0]["group1_atom_indices"]) > 1
    assert len(resolved[0]["group2_atom_indices"]) > 1
    assert all(ctx["chain"] == "A" for ctx in resolved[0]["group1_contexts"])
    assert all(ctx["resid"] == 1 for ctx in resolved[0]["group1_contexts"])
    assert all(ctx["resid"] == 2 for ctx in resolved[0]["group2_contexts"])


def test_resolve_guided_secondary_structure_constraints_returns_matched_residues():
    schema = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": "AAAAA"}},
        ],
        "constraints": [
            {
                "guided_secondary_structure": {
                    "selection": "chain A and resid 2 to 4",
                    "type": "helix",
                }
            }
        ],
    }

    target, tokenized = build_tokenized(schema)
    resolved = resolve_guided_secondary_structure_constraints(
        tokenized,
        target.record.inference_options.guided_secondary_structure_constraints,
    )

    assert len(resolved) == 1
    assert len(resolved[0]["contexts"]) == 3
    assert [ctx["resid"] for ctx in resolved[0]["contexts"]] == [2, 3, 4]
    assert all(ctx["chain"] == "A" for ctx in resolved[0]["contexts"])


def test_guided_secondary_structure_features_use_backbone_atoms_per_residue():
    schema = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": "AAAAA"}},
        ],
        "constraints": [
            {
                "guided_secondary_structure": {
                    "selection": "chain A and resid 2 to 4",
                    "type": "sheet",
                }
            }
        ],
    }

    target, tokenized = build_tokenized(schema)
    features = build_guided_secondary_structure_features(
        tokenized,
        target.record.inference_options.guided_secondary_structure_constraints,
    )

    assert features["guided_secondary_structure_atom_index"].shape == (3, 4)
    assert features["guided_secondary_structure_constraint_index"].tolist() == [0, 0, 0]
    assert features["guided_secondary_structure_type"].tolist() == [1]
    assert features["guided_secondary_structure_donor_mask"].tolist() == [True, True, True]


def test_guided_secondary_structure_potential_uses_1d_donor_mask(monkeypatch):
    import boltz.model.potentials.potentials as potentials_module

    schema = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": "AAAAA"}},
        ],
        "constraints": [
            {
                "guided_secondary_structure": {
                    "selection": "chain A and resid 2 to 4",
                    "type": "helix",
                }
            }
        ],
    }

    target, tokenized = build_tokenized(schema)
    features = build_guided_secondary_structure_features(
        tokenized,
        target.record.inference_options.guided_secondary_structure_constraints,
    )
    coords = torch.zeros(
        (1, tokenized.structure.atoms.shape[0], 3),
        dtype=torch.float32,
    )
    captured = {}

    def fake_assign_secondary_structure(coord, donor_mask):
        captured["coord_shape"] = tuple(coord.shape)
        captured["donor_mask_shape"] = tuple(donor_mask.shape)
        return torch.zeros(
            (coord.shape[0], coord.shape[1], 3),
            dtype=coord.dtype,
            device=coord.device,
        )

    monkeypatch.setattr(
        potentials_module,
        "_assign_secondary_structure_onehot",
        fake_assign_secondary_structure,
    )

    potential = GuidedSecondaryStructurePotential(parameters={"resampling_weight": 1.0})
    energy = potential.compute(
        coords,
        batch_features(features),
        {"resampling_weight": 1.0},
    )

    assert captured["coord_shape"] == (1, 3, 4, 3)
    assert captured["donor_mask_shape"] == (3,)
    assert energy.shape == (1,)


def test_guided_secondary_structure_potential_penalizes_per_residue_mismatches(
    monkeypatch,
):
    import boltz.model.potentials.potentials as potentials_module

    features = {
        "guided_secondary_structure_atom_index": torch.tensor(
            [[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long
        ),
        "guided_secondary_structure_constraint_index": torch.tensor([0, 0], dtype=torch.long),
        "guided_secondary_structure_type": torch.tensor([0], dtype=torch.long),
        "guided_secondary_structure_donor_mask": torch.tensor(
            [True, True], dtype=torch.bool
        ),
    }
    coords = torch.zeros((2, 8, 3), dtype=torch.float32)
    dssp_onehot = torch.tensor(
        [
            [[0.1, 0.9, 0.0], [0.1, 0.9, 0.0]],
            [[0.2, 0.8, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    monkeypatch.setattr(
        potentials_module,
        "_assign_secondary_structure_onehot",
        lambda residue_coords, donor_mask: dssp_onehot,
    )

    potential = GuidedSecondaryStructurePotential(parameters={"resampling_weight": 1.0})
    energy = potential.compute(
        coords,
        batch_features(features),
        {"resampling_weight": 1.0},
    )

    assert torch.allclose(energy[0], torch.tensor(0.0))
    assert energy[1] > energy[0]


def test_soft_secondary_structure_assignment_returns_non_binary_probabilities(
    monkeypatch,
):
    import boltz.model.potentials.potentials as potentials_module

    residue_coords = torch.zeros((1, 6, 4, 3), dtype=torch.float32)
    donor_mask = torch.ones(6, dtype=torch.bool)
    torch.manual_seed(0)
    hbmap = torch.rand((1, 6, 6), dtype=torch.float32)

    monkeypatch.setitem(
        sys.modules,
        "pydssp",
        SimpleNamespace(pydssp_torch=SimpleNamespace(get_hbond_map=lambda coord, donor_mask=None: hbmap)),
    )
    monkeypatch.setitem(
        sys.modules,
        "pydssp.pydssp_torch",
        SimpleNamespace(get_hbond_map=lambda coord, donor_mask=None: hbmap),
    )

    soft_assignment = potentials_module._assign_secondary_structure_onehot(
        residue_coords,
        donor_mask,
    )

    assert soft_assignment.shape == (1, 6, 3)
    assert torch.allclose(
        soft_assignment.sum(dim=-1),
        torch.ones((1, 6), dtype=torch.float32),
    )
    assert torch.any((soft_assignment > 0.0) & (soft_assignment < 1.0))


def test_boltz_restr_equivalent_example_parses():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "guided_distance_boltz_restr.yaml"
    )

    target = parse_yaml(
        example_path,
        ccd={},
        mol_dir=Path("~/.boltz/mols").expanduser(),
        boltz2=True,
    )
    constraints = target.record.inference_options.guided_distance_constraints

    assert constraints is not None
    assert len(constraints) == 2
    assert constraints[0].selection1 == "(resid 122)"
    assert constraints[0].selection2 == "(resid 126)"
    assert constraints[0].constraint_type == "flat_bottomed"
    assert constraints[0].lower_bound == 10.0
    assert constraints[1].selection1 == "(resid 126)"
    assert constraints[1].selection2 == "(resid 130)"
    assert constraints[1].constraint_type == "flat_bottomed"
    assert constraints[1].lower_bound == 10.0


def test_explicit_fk_guided_distance_example_parses():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "guided_distance_fk_explicit.yaml"
    )

    target = parse_yaml(
        example_path,
        ccd={},
        mol_dir=Path("~/.boltz/mols").expanduser(),
        boltz2=True,
    )
    constraints = target.record.inference_options.guided_distance_constraints

    assert constraints is not None
    assert len(constraints) == 1
    assert constraints[0].selection1 == "(resid 121)"
    assert constraints[0].selection2 == "(resid 125)"
    assert constraints[0].constraint_type == "harmonic"
    assert constraints[0].target_distance == 8.0


def test_guided_secondary_structure_example_parses():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "guided_secondary_structure.yaml"
    )

    target = parse_yaml(
        example_path,
        ccd={},
        mol_dir=Path("~/.boltz/mols").expanduser(),
        boltz2=True,
    )
    constraints = target.record.inference_options.guided_secondary_structure_constraints

    assert constraints is not None
    assert len(constraints) == 1
    assert constraints[0].selection == "(resid 122 to 138)"
    assert constraints[0].constraint_type == "loop"


def test_echo_guided_distance_summary_loads_boltz2_structure(tmp_path, capsys):
    schema = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": "AA"}},
        ],
        "constraints": [
            {
                "guided_distance": {
                    "selection1": "(resid 1)",
                    "selection2": "(resid 2)",
                    "type": "harmonic",
                    "target_distance": 8.0,
                }
            }
        ],
    }

    target, _ = build_tokenized(schema)
    structure = target.structure
    atoms = np.array(
        [
            (
                decode_atom_name(atom["name"]),
                atom["coords"],
                bool(atom["is_present"]),
                0.0,
                0.0,
            )
            for atom in structure.atoms
        ],
        dtype=AtomV2,
    )
    coords = np.array([(coord,) for coord in atoms["coords"]], dtype=Coords)
    ensemble = np.array([(0, len(coords))], dtype=Ensemble)
    structure_v2 = StructureV2(
        atoms=atoms,
        bonds=np.array([], dtype=BondV2),
        residues=structure.residues,
        chains=structure.chains,
        interfaces=structure.interfaces,
        mask=structure.mask,
        coords=coords,
        ensemble=ensemble,
    )
    structure_path = tmp_path / "toy.npz"
    structure_v2.dump(structure_path)

    echo_guided_distance_summary(
        Manifest(records=[target.record]),
        tmp_path,
        boltz2=True,
    )
    output = capsys.readouterr().out

    assert "Guided-Distance Steering [toy]" in output
    assert "s1: (resid 1)" in output
    assert "s2: (resid 2)" in output


def test_runtime_steering_args_enable_guided_distance_per_record():
    base_args = {
        "fk_steering": False,
        "num_particles": 5,
        "fk_lambda": 4.0,
        "fk_resampling_interval": 3,
        "physical_guidance_update": False,
        "contact_guidance_update": False,
        "num_gd_steps": 20,
        "guided_distance_enabled": False,
        "guided_distance_start_timestep": 1.0,
        "guided_distance_resampling_interval": 2,
        "guided_distance_tau": 10.0,
        "guided_distance_guidance_update": True,
        "guided_secondary_structure_enabled": False,
        "guided_secondary_structure_start_timestep": 1.0,
        "guided_secondary_structure_resampling_interval": 2,
        "guided_secondary_structure_tau": 0.2,
        "verbose": False,
    }

    inactive = get_runtime_steering_args(
        base_args,
        {"guided_distance_pair_index": torch.empty((1, 2, 0), dtype=torch.long)},
    )
    active = get_runtime_steering_args(
        base_args,
        {"guided_distance_pair_index": torch.tensor([[[0], [1]]], dtype=torch.long)},
    )

    assert not inactive["guided_distance_enabled"]
    assert not inactive["guided_distance_guidance_update"]
    assert not inactive["resampling_enabled"]
    assert inactive["num_particles"] == 5
    assert inactive["fk_resampling_interval"] == 3

    assert active["guided_distance_enabled"]
    assert active["guided_distance_guidance_update"]
    assert active["resampling_enabled"]
    assert active["num_particles"] == 5
    assert active["fk_resampling_interval"] == 2


def test_runtime_steering_args_enable_guided_secondary_structure_per_record():
    base_args = {
        "fk_steering": False,
        "num_particles": 5,
        "fk_lambda": 4.0,
        "fk_resampling_interval": 3,
        "physical_guidance_update": False,
        "contact_guidance_update": False,
        "num_gd_steps": 20,
        "guided_distance_enabled": False,
        "guided_distance_start_timestep": 1.0,
        "guided_distance_resampling_interval": 2,
        "guided_distance_tau": 10.0,
        "guided_distance_guidance_update": False,
        "guided_secondary_structure_enabled": False,
        "guided_secondary_structure_start_timestep": 1.0,
        "guided_secondary_structure_resampling_interval": 2,
        "guided_secondary_structure_tau": 0.2,
        "verbose": False,
    }

    inactive = get_runtime_steering_args(
        base_args,
        {
            "guided_secondary_structure_atom_index": torch.empty(
                (1, 0, 4), dtype=torch.long
            )
        },
    )
    active = get_runtime_steering_args(
        base_args,
        {
            "guided_secondary_structure_atom_index": torch.tensor(
                [[[0, 1, 2, 3]]], dtype=torch.long
            )
        },
    )

    assert not inactive["guided_secondary_structure_enabled"]
    assert not inactive["resampling_enabled"]
    assert inactive["num_particles"] == 5
    assert inactive["fk_resampling_interval"] == 3

    assert active["guided_secondary_structure_enabled"]
    assert active["resampling_enabled"]
    assert active["num_particles"] == 5
    assert active["fk_resampling_interval"] == 2


def test_guided_distance_only_runtime_does_not_add_generic_fk_potentials():
    steering_args = {
        "fk_steering": False,
        "num_particles": 3,
        "fk_lambda": 4.0,
        "fk_resampling_interval": 3,
        "physical_guidance_update": False,
        "contact_guidance_update": False,
        "num_gd_steps": 20,
        "guided_distance_enabled": True,
        "guided_distance_start_timestep": 1.0,
        "guided_distance_resampling_interval": 2,
        "guided_distance_tau": 10.0,
        "guided_distance_guidance_update": False,
        "guided_secondary_structure_enabled": False,
        "guided_secondary_structure_start_timestep": 1.0,
        "guided_secondary_structure_resampling_interval": 2,
        "guided_secondary_structure_tau": 0.2,
        "verbose": False,
    }

    potentials = get_potentials(steering_args, boltz2=True)

    assert any(
        isinstance(potential, GuidedDistancePotential) for potential in potentials
    )
    assert not any(
        isinstance(potential, SymmetricChainCOMPotential) for potential in potentials
    )


def test_guided_distance_runtime_can_enable_gradient_guidance_without_generic_fk():
    steering_args = {
        "fk_steering": False,
        "num_particles": 3,
        "fk_lambda": 4.0,
        "fk_resampling_interval": 3,
        "physical_guidance_update": False,
        "contact_guidance_update": False,
        "num_gd_steps": 20,
        "guided_distance_enabled": True,
        "guided_distance_start_timestep": 1.0,
        "guided_distance_resampling_interval": 2,
        "guided_distance_tau": 10.0,
        "guided_distance_guidance_update": True,
        "guided_secondary_structure_enabled": False,
        "guided_secondary_structure_start_timestep": 1.0,
        "guided_secondary_structure_resampling_interval": 2,
        "guided_secondary_structure_tau": 0.2,
        "verbose": False,
    }

    potentials = get_potentials(steering_args, boltz2=True)
    guided_distance_potential = next(
        potential
        for potential in potentials
        if isinstance(potential, GuidedDistancePotential)
    )

    guidance_weight = guided_distance_potential.parameters["guidance_weight"]
    assert isinstance(guidance_weight, ExponentialInterpolation)
    assert guidance_weight.compute(1.0) == pytest.approx(0.1)
    assert guidance_weight.compute(0.0) == pytest.approx(0.0)
    assert guidance_weight.compute(1.0) > guidance_weight.compute(0.5)
    assert guidance_weight.compute(0.5) > guidance_weight.compute(0.0)
    assert not any(
        isinstance(potential, SymmetricChainCOMPotential) for potential in potentials
    )


def test_guided_secondary_structure_runtime_adds_only_secondary_structure_potential():
    steering_args = {
        "fk_steering": False,
        "num_particles": 3,
        "fk_lambda": 4.0,
        "fk_resampling_interval": 3,
        "physical_guidance_update": False,
        "contact_guidance_update": False,
        "num_gd_steps": 20,
        "guided_distance_enabled": False,
        "guided_distance_start_timestep": 1.0,
        "guided_distance_resampling_interval": 2,
        "guided_distance_tau": 10.0,
        "guided_secondary_structure_enabled": True,
        "guided_secondary_structure_start_timestep": 1.0,
        "guided_secondary_structure_resampling_interval": 2,
        "guided_secondary_structure_tau": 0.2,
        "verbose": False,
    }

    potentials = get_potentials(steering_args, boltz2=True)

    assert any(
        isinstance(potential, GuidedSecondaryStructurePotential)
        for potential in potentials
    )
    assert not any(
        isinstance(potential, SymmetricChainCOMPotential) for potential in potentials
    )


def test_reprocess_rebuilds_matching_processed_inputs(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    records_dir = out_dir / "processed" / "records"
    records_dir.mkdir(parents=True)
    (records_dir / "toy.json").write_text("{}")

    input_path = tmp_path / "toy.yaml"
    input_path.write_text("version: 1\n")

    processed_calls = []

    def fake_process_input(path, **kwargs):
        processed_calls.append(path)

    def fake_record_load(path):
        return type("FakeRecord", (), {"id": Path(path).stem})()

    monkeypatch.setattr(main_module, "load_canonicals", lambda _: {})
    monkeypatch.setattr(main_module, "process_input", fake_process_input)
    monkeypatch.setattr(main_module.Record, "load", staticmethod(fake_record_load))
    monkeypatch.setattr(main_module.Manifest, "dump", lambda self, path: None)

    main_module.process_inputs(
        data=[input_path],
        out_dir=out_dir,
        ccd_path=tmp_path / "ccd.pkl",
        mol_dir=tmp_path / "mols",
        msa_server_url="",
        msa_pairing_strategy="greedy",
        boltz2=True,
        preprocessing_threads=1,
        reprocess=False,
    )
    assert processed_calls == []

    main_module.process_inputs(
        data=[input_path],
        out_dir=out_dir,
        ccd_path=tmp_path / "ccd.pkl",
        mol_dir=tmp_path / "mols",
        msa_server_url="",
        msa_pairing_strategy="greedy",
        boltz2=True,
        preprocessing_threads=1,
        reprocess=True,
    )
    assert processed_calls == [input_path]


def test_predict_reprocess_implies_override(tmp_path, monkeypatch):
    class StopPredict(RuntimeError):
        pass

    captured = {}

    def fake_filter_inputs_structure(manifest, outdir, override=False):
        captured["override"] = override
        raise StopPredict

    monkeypatch.setattr(main_module, "download_boltz2", lambda cache: None)
    monkeypatch.setattr(main_module, "check_inputs", lambda data: [Path(data)])
    monkeypatch.setattr(main_module, "process_inputs", lambda **kwargs: None)
    monkeypatch.setattr(
        main_module.Manifest,
        "load",
        staticmethod(lambda path: main_module.Manifest(records=[])),
    )
    monkeypatch.setattr(
        main_module,
        "filter_inputs_structure",
        fake_filter_inputs_structure,
    )

    input_path = tmp_path / "toy.yaml"
    input_path.write_text("version: 1\n")

    with pytest.raises(StopPredict):
        main_module.predict.callback(
            data=input_path,
            out_dir=tmp_path / "out",
            cache=tmp_path / "cache",
            accelerator="cpu",
            override=False,
            reprocess=True,
            model="boltz2",
        )

    assert captured["override"] is True


def test_guided_secondary_structure_tau_uses_default_when_tau_not_explicit(
    monkeypatch,
):
    class FakeContext:
        @staticmethod
        def get_parameter_source(name):
            assert name == "tau"
            return ParameterSource.DEFAULT

    monkeypatch.setattr(main_module.click, "get_current_context", lambda silent=True: FakeContext())

    resolved = main_module._resolve_guided_secondary_structure_tau(0.3)

    assert resolved == main_module.BoltzSteeringParams.guided_secondary_structure_tau


def test_guided_secondary_structure_tau_uses_explicit_tau(
    monkeypatch,
):
    class FakeContext:
        @staticmethod
        def get_parameter_source(name):
            assert name == "tau"
            return ParameterSource.COMMANDLINE

    monkeypatch.setattr(main_module.click, "get_current_context", lambda silent=True: FakeContext())

    resolved = main_module._resolve_guided_secondary_structure_tau(0.3)

    assert resolved == 0.3


def test_predict_threads_guided_distance_gradient_guidance_flag(tmp_path, monkeypatch):
    captured = {}

    class FakeModelModule:
        def eval(self):
            return None

    class FakeModel:
        @staticmethod
        def load_from_checkpoint(*args, **kwargs):
            captured["steering_args"] = kwargs["steering_args"]
            return FakeModelModule()

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            self.callbacks = kwargs["callbacks"]

        def predict(self, *args, **kwargs):
            return None

    out_dir = tmp_path / "out"
    processed_dir = out_dir / "processed"
    for directory in ("structures", "msa"):
        (processed_dir / directory).mkdir(parents=True, exist_ok=True)

    fake_record = SimpleNamespace(
        id="toy",
        inference_options=SimpleNamespace(
            guided_distance_constraints=None,
            guided_secondary_structure_constraints=None,
        ),
    )
    fake_manifest = Manifest(records=[fake_record])

    monkeypatch.setattr(main_module, "download_boltz2", lambda cache: None)
    monkeypatch.setattr(main_module, "check_inputs", lambda data: [Path(data)])
    monkeypatch.setattr(main_module, "process_inputs", lambda **kwargs: None)
    monkeypatch.setattr(
        main_module.Manifest,
        "load",
        staticmethod(lambda path: fake_manifest),
    )
    monkeypatch.setattr(
        main_module,
        "filter_inputs_structure",
        lambda manifest, outdir, override=False: manifest,
    )
    monkeypatch.setattr(main_module, "Boltz2", FakeModel)
    monkeypatch.setattr(main_module, "Trainer", FakeTrainer)
    monkeypatch.setattr(main_module, "Boltz2InferenceDataModule", lambda **kwargs: object())
    monkeypatch.setattr(main_module, "BoltzWriter", lambda **kwargs: object())

    input_path = tmp_path / "toy.yaml"
    input_path.write_text("version: 1\n")

    main_module.predict.callback(
        data=str(input_path),
        out_dir=str(out_dir),
        cache=str(tmp_path / "cache"),
        accelerator="cpu",
        model="boltz2",
        use_gradient_guidance=True,
    )

    assert captured["steering_args"]["guided_distance_guidance_update"] is True
