from pathlib import Path

import boltz.main as main_module
import numpy as np
import torch

from boltz.data.feature.guided_distance import (
    build_guided_distance_features,
    decode_atom_name,
    resolve_guided_distance_constraints,
)
from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.parse.yaml import parse_yaml
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import BondV2, AtomV2, Coords, Ensemble, Input, Manifest, StructureV2
from boltz.main import echo_guided_distance_summary
from boltz.model.potentials.potentials import (
    GuidedDistancePotential,
    SymmetricChainCOMPotential,
    get_potentials,
    get_runtime_steering_args,
)


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
    assert len(constraints) == 1
    assert constraints[0].selection1 == "(resid 121)"
    assert constraints[0].selection2 == "(resid 125)"
    assert constraints[0].constraint_type == "harmonic"
    assert constraints[0].target_distance == 8.0


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

    assert "Guided-distance steering for toy:" in output
    assert "selection1: (resid 1)" in output
    assert "selection2: (resid 2)" in output


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
    assert not inactive["resampling_enabled"]
    assert inactive["num_particles"] == 5
    assert inactive["fk_resampling_interval"] == 3

    assert active["guided_distance_enabled"]
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
        "verbose": False,
    }

    potentials = get_potentials(steering_args, boltz2=True)

    assert any(
        isinstance(potential, GuidedDistancePotential) for potential in potentials
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
