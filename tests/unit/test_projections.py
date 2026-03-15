"""Tests for src/training/projections.py

Parametrized across Identity, Linear, MLP projections. Tests output shape,
gradient flow, factories, and create_projection_if_needed().

Run:
    python -m pytest tests/unit/test_projections.py -v
"""

import pytest
import torch
import torch.nn as nn

from src.training.projections import (
    IdentityProjection,
    LinearProjection,
    MLPProjection,
    PROJECTION_REGISTRY,
    create_projection_if_needed,
    get_projection,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INPUT_DIM, OUTPUT_DIM = 64, 32


@pytest.fixture(params=[
    pytest.param("identity", id="identity"),
    pytest.param("linear", id="linear"),
    pytest.param("mlp", id="mlp"),
])
def projection(request):
    """Create a projection of each type with consistent dims."""
    name = request.param
    if name == "identity":
        return IdentityProjection(dim=INPUT_DIM)
    elif name == "linear":
        return LinearProjection(INPUT_DIM, OUTPUT_DIM)
    elif name == "mlp":
        return MLPProjection(INPUT_DIM, OUTPUT_DIM)


@pytest.fixture(params=[
    pytest.param("linear", id="linear"),
    pytest.param("mlp", id="mlp"),
])
def trainable_projection(request):
    """Non-identity projections that have learnable parameters."""
    if request.param == "linear":
        return LinearProjection(INPUT_DIM, OUTPUT_DIM)
    return MLPProjection(INPUT_DIM, OUTPUT_DIM)


# ---------------------------------------------------------------------------
# Cross-projection property tests
# ---------------------------------------------------------------------------

class TestOutputShape:
    """Output shape must match expected dimensions for all projections."""

    def test_2d_input(self, projection):
        x = torch.randn(4, INPUT_DIM)
        out = projection(x)
        expected_last = INPUT_DIM if isinstance(projection, IdentityProjection) else OUTPUT_DIM
        assert out.shape == (4, expected_last)

    def test_3d_input(self, projection):
        x = torch.randn(2, 8, INPUT_DIM)
        out = projection(x)
        expected_last = INPUT_DIM if isinstance(projection, IdentityProjection) else OUTPUT_DIM
        assert out.shape == (2, 8, expected_last)


class TestGradientFlow:
    """Gradients must flow through trainable projections."""

    def test_gradient_reaches_input(self, trainable_projection):
        x = torch.randn(2, INPUT_DIM, requires_grad=True)
        out = trainable_projection(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_has_learnable_parameters(self, trainable_projection):
        params = list(trainable_projection.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


# ---------------------------------------------------------------------------
# Identity-specific
# ---------------------------------------------------------------------------

class TestIdentity:
    def test_returns_same_tensor_object(self):
        proj = IdentityProjection()
        x = torch.randn(2, 64)
        assert proj(x) is x

    def test_call_and_forward_consistent(self):
        proj = IdentityProjection()
        x = torch.randn(4, 64)
        assert proj(x) is proj.forward(x)


# ---------------------------------------------------------------------------
# MLP-specific
# ---------------------------------------------------------------------------

class TestMLP:
    def test_default_hidden_dim_is_average(self):
        proj = MLPProjection(input_dim=100, output_dim=50)
        assert proj.proj[0].out_features == (100 + 50) // 2

    def test_custom_hidden_dim(self):
        proj = MLPProjection(input_dim=64, output_dim=32, hidden_dim=16)
        assert proj.proj[0].out_features == 16

    @pytest.mark.parametrize("act,cls", [
        ("relu", nn.ReLU),
        ("gelu", nn.GELU),
        ("tanh", nn.Tanh),
    ])
    def test_activation_selection(self, act, cls):
        proj = MLPProjection(input_dim=64, output_dim=32, activation=act)
        assert isinstance(proj.proj[1], cls)

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            MLPProjection(input_dim=64, output_dim=32, activation="swish")


# ---------------------------------------------------------------------------
# create_projection_if_needed
# ---------------------------------------------------------------------------

class TestCreateProjectionIfNeeded:
    def test_same_dims_returns_identity(self):
        proj = create_projection_if_needed(768, 768)
        assert isinstance(proj, IdentityProjection)

    def test_different_dims_returns_linear_by_default(self):
        proj = create_projection_if_needed(1024, 768)
        assert isinstance(proj, LinearProjection)

    @pytest.mark.parametrize("proj_type,cls", [
        ("linear", LinearProjection),
        ("mlp", MLPProjection),
    ])
    def test_respects_projection_type(self, proj_type, cls):
        proj = create_projection_if_needed(1024, 768, projection_type=proj_type)
        assert isinstance(proj, cls)

    def test_correct_dims_on_created_projection(self):
        proj = create_projection_if_needed(1024, 768)
        assert proj.input_dim == 1024
        assert proj.output_dim == 768


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestGetProjection:
    @pytest.mark.parametrize("name,cls", [
        ("identity", IdentityProjection),
        ("linear", LinearProjection),
        ("mlp", MLPProjection),
    ])
    def test_returns_correct_type(self, name, cls):
        kwargs = {"input_dim": 64, "output_dim": 32} if name != "identity" else {}
        assert isinstance(get_projection(name, **kwargs), cls)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown projection"):
            get_projection("transformer")

    def test_registry_has_three_entries(self):
        assert set(PROJECTION_REGISTRY.keys()) == {"identity", "linear", "mlp"}

    @pytest.mark.parametrize("name", ["linear", "mlp"])
    def test_requires_dims(self, name):
        with pytest.raises(ValueError):
            get_projection(name)

    def test_fallback_for_new_registry_entry(self):
        """When a new projection type is registered without an explicit branch,
        the factory falls back to calling PROJECTION_REGISTRY[name]()."""
        PROJECTION_REGISTRY["_test_identity"] = IdentityProjection
        try:
            proj = get_projection("_test_identity")
            assert isinstance(proj, IdentityProjection)
        finally:
            del PROJECTION_REGISTRY["_test_identity"]
