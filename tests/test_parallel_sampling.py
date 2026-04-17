import pytest
import torch

import boltz.model.modules.diffusion as diffusion_module
import boltz.model.modules.diffusionv2 as diffusionv2_module
from boltz.model.models.boltz1 import Boltz1
from boltz.model.modules.utils import chunk_indices_by_max_parallel_samples


def test_chunk_indices_by_max_parallel_samples_caps_chunk_size():
    chunks = chunk_indices_by_max_parallel_samples(7, 3)

    assert [chunk.tolist() for chunk in chunks] == [[0, 1, 2], [3, 4, 5], [6]]


def test_chunk_indices_by_max_parallel_samples_supports_sequential_mode():
    chunks = chunk_indices_by_max_parallel_samples(4, 1)

    assert [chunk.tolist() for chunk in chunks] == [[0], [1], [2], [3]]


def test_chunk_indices_by_max_parallel_samples_defaults_to_full_batch():
    chunks = chunk_indices_by_max_parallel_samples(4, None)

    assert [chunk.tolist() for chunk in chunks] == [[0, 1, 2, 3]]


def test_chunk_indices_by_max_parallel_samples_rejects_zero():
    with pytest.raises(ValueError, match="max_parallel_samples must be at least 1."):
        chunk_indices_by_max_parallel_samples(4, 0)


def test_boltz1_predict_step_threads_max_parallel_samples():
    captured = {}

    class FakeBoltz1:
        def __init__(self):
            self.predict_args = {
                "recycling_steps": 3,
                "sampling_steps": 5,
                "diffusion_samples": 4,
                "max_parallel_samples": 2,
                "write_confidence_summary": False,
                "write_full_pae": False,
                "write_full_pde": False,
            }

        def __call__(self, batch, **kwargs):
            captured["kwargs"] = kwargs
            return {
                "sample_atom_coords": torch.zeros((1, 1, 3)),
                "s": torch.zeros((1, 1)),
                "z": torch.zeros((1, 1)),
            }

    fake_model = FakeBoltz1()
    batch = {"atom_pad_mask": torch.ones((1, 1), dtype=torch.float32)}

    pred = Boltz1.predict_step(fake_model, batch, batch_idx=0)

    assert pred["exception"] is False
    assert captured["kwargs"]["max_parallel_samples"] == 2


def test_fk_resampling_chunks_base_samples_before_particle_expansion(monkeypatch):
    calls = []

    class DummyDiffusion:
        pass

    dummy = DummyDiffusion()

    def fake_single_pass(self, **kwargs):
        calls.append(kwargs["multiplicity"])
        multiplicity = kwargs["multiplicity"]
        return {
            "sample_atom_coords": torch.zeros((multiplicity, 1, 3)),
            "diff_token_repr": None,
        }

    dummy._sample_single_pass = fake_single_pass.__get__(dummy, DummyDiffusion)
    monkeypatch.setattr(
        diffusion_module,
        "get_runtime_steering_args",
        lambda steering_args, feats: {"resampling_enabled": True},
    )

    out = diffusion_module.AtomDiffusion.sample(
        dummy,
        atom_mask=torch.ones((1, 1), dtype=torch.float32),
        multiplicity=5,
        max_parallel_samples=2,
        steering_args={"fk_steering": True},
        feats={},
    )

    assert calls == [2, 2, 1]
    assert out["sample_atom_coords"].shape == (5, 1, 3)
    assert out["diff_token_repr"] is None


def test_fk_resampling_chunks_base_samples_before_particle_expansion_boltz2(
    monkeypatch,
):
    calls = []

    class DummyDiffusion:
        pass

    dummy = DummyDiffusion()

    def fake_single_pass(self, **kwargs):
        calls.append(kwargs["multiplicity"])
        multiplicity = kwargs["multiplicity"]
        return {
            "sample_atom_coords": torch.zeros((multiplicity, 1, 3)),
            "diff_token_repr": None,
        }

    dummy._sample_single_pass = fake_single_pass.__get__(dummy, DummyDiffusion)
    monkeypatch.setattr(
        diffusionv2_module,
        "get_runtime_steering_args",
        lambda steering_args, feats: {"resampling_enabled": True},
    )

    out = diffusionv2_module.AtomDiffusion.sample(
        dummy,
        atom_mask=torch.ones((1, 1), dtype=torch.float32),
        multiplicity=5,
        max_parallel_samples=2,
        steering_args={"fk_steering": True},
        feats={},
    )

    assert calls == [2, 2, 1]
    assert out["sample_atom_coords"].shape == (5, 1, 3)
    assert out["diff_token_repr"] is None


def _fk_runtime_args(num_particles=3):
    return {
        "resampling_enabled": True,
        "physical_guidance_update": False,
        "contact_guidance_update": False,
        "guided_distance_guidance_update": False,
        "num_particles": num_particles,
        "verbose": False,
        "fk_resampling_interval": 1,
        "fk_lambda": 1.0,
    }


def test_fk_particles_share_one_forward_when_base_sample_cap_allows_boltz1(
    monkeypatch,
):
    calls = []

    class DummyDiffusion:
        pass

    dummy = DummyDiffusion()
    dummy.device = torch.device("cpu")
    dummy.num_sampling_steps = 2
    dummy.gamma_min = 0.0
    dummy.gamma_0 = 0.1
    dummy.noise_scale = 1.0
    dummy.token_s = 2
    dummy.use_inference_model_cache = False
    dummy.accumulate_token_repr = False
    dummy.step_scale = 1.0
    dummy.alignment_reverse_diff = False
    dummy.sample_schedule = lambda num_sampling_steps: torch.tensor([1.0, 0.5, 0.0])

    def fake_forward(atom_coords_noisy, t_hat, training, network_condition_kwargs):
        calls.append(atom_coords_noisy.shape[0])
        token_a = torch.zeros((atom_coords_noisy.shape[0], 1, 4))
        return torch.zeros_like(atom_coords_noisy), token_a

    dummy.preconditioned_network_forward = fake_forward
    monkeypatch.setattr(
        diffusion_module,
        "get_runtime_steering_args",
        lambda steering_args, feats: _fk_runtime_args(),
    )
    monkeypatch.setattr(diffusion_module, "get_potentials", lambda *args, **kwargs: [])

    out = diffusion_module.AtomDiffusion._sample_single_pass(
        dummy,
        atom_mask=torch.ones((1, 2), dtype=torch.float32),
        num_sampling_steps=2,
        multiplicity=2,
        max_parallel_samples=2,
        steering_args={"fk_steering": True},
        feats={"token_index": torch.zeros((1, 1), dtype=torch.long)},
    )

    assert calls == [6, 6]
    assert out["sample_atom_coords"].shape == (2, 2, 3)


def test_fk_particles_share_one_forward_when_base_sample_cap_allows_boltz2(
    monkeypatch,
):
    calls = []

    class DummyDiffusion:
        pass

    dummy = DummyDiffusion()
    dummy.device = torch.device("cpu")
    dummy.num_sampling_steps = 2
    dummy.gamma_min = 0.0
    dummy.gamma_0 = 0.1
    dummy.noise_scale = 1.0
    dummy.step_scale = 1.0
    dummy.training = False
    dummy.step_scale_random = None
    dummy.alignment_reverse_diff = False
    dummy.sample_schedule = lambda num_sampling_steps: torch.tensor([1.0, 0.5, 0.0])

    def fake_forward(atom_coords_noisy, t_hat, network_condition_kwargs):
        calls.append(atom_coords_noisy.shape[0])
        return torch.zeros_like(atom_coords_noisy)

    dummy.preconditioned_network_forward = fake_forward
    monkeypatch.setattr(
        diffusionv2_module,
        "get_runtime_steering_args",
        lambda steering_args, feats: _fk_runtime_args(),
    )
    monkeypatch.setattr(
        diffusionv2_module,
        "get_potentials",
        lambda *args, **kwargs: [],
    )

    out = diffusionv2_module.AtomDiffusion._sample_single_pass(
        dummy,
        atom_mask=torch.ones((1, 2), dtype=torch.float32),
        num_sampling_steps=2,
        multiplicity=2,
        max_parallel_samples=2,
        steering_args={"fk_steering": True},
        feats={"token_index": torch.zeros((1, 1), dtype=torch.long)},
    )

    assert calls == [6, 6]
    assert out["sample_atom_coords"].shape == (2, 2, 3)
