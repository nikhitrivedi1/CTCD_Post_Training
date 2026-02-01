"""Tests for L1 KD Strategy implementation (Feature & Attention distillation)."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
import sys
sys.path.insert(0, '.')

from src.training import (
    # Loss strategies
    get_loss_strategy,
    KLDistillationLoss,
    FeatureLossStrategy,
    AttentionLossStrategy,
    LOSS_REGISTRY,
    # Layer mapping
    get_layer_mapping,
    UniformMappingStrategy,
    LastNMappingStrategy,
    FirstLastMappingStrategy,
    LAYER_MAPPING_REGISTRY,
    # Projections
    get_projection,
    create_projection_if_needed,
    IdentityProjection,
    LinearProjection,
    MLPProjection,
    PROJECTION_REGISTRY,
)


# =============================================================================
# Mock Model Outputs (simulating HuggingFace model outputs)
# =============================================================================

@dataclass
class MockModelOutput:
    """Simulates HuggingFace model output structure."""
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


def create_mock_outputs(
    batch_size: int = 4,
    seq_len: int = 32,
    num_classes: int = 3,
    num_layers: int = 12,
    hidden_dim: int = 768,
    num_heads: int = 12,
    include_hidden: bool = False,
    include_attention: bool = False,
) -> MockModelOutput:
    """Create mock model outputs for testing."""
    logits = torch.randn(batch_size, num_classes)
    loss = torch.tensor(1.5)  # Mock CE loss

    hidden_states = None
    if include_hidden:
        # num_layers + 1 for embedding layer
        hidden_states = tuple(
            torch.randn(batch_size, seq_len, hidden_dim)
            for _ in range(num_layers + 1)
        )

    attentions = None
    if include_attention:
        attentions = tuple(
            torch.randn(batch_size, num_heads, seq_len, seq_len)
            for _ in range(num_layers)
        )

    return MockModelOutput(
        logits=logits,
        loss=loss,
        hidden_states=hidden_states,
        attentions=attentions,
    )


# =============================================================================
# Test: Loss Registry
# =============================================================================

def test_loss_registry():
    """Test that all expected loss strategies are registered."""
    print("\n=== Test: Loss Registry ===")

    expected = [
        "kl_divergence",
        "mse_logits",
        "cosine_embedding",
        "soft_target",
        "feature_mse",
        "attention_kl",
    ]

    for name in expected:
        assert name in LOSS_REGISTRY, f"Missing: {name}"
        strategy = get_loss_strategy(name, alpha=0.5)
        print(f"  ✓ {name}: {strategy.name}")

    print("  All loss strategies registered correctly!")


# =============================================================================
# Test: Layer Mapping Strategies
# =============================================================================

def test_layer_mapping_strategies():
    """Test all layer mapping strategies."""
    print("\n=== Test: Layer Mapping Strategies ===")

    teacher_layers, student_layers = 12, 6

    # Uniform mapping
    uniform = get_layer_mapping("uniform")
    mapping = uniform.map(teacher_layers, student_layers)
    print(f"  Uniform (12→6): {mapping}")
    assert len(mapping) == student_layers
    assert mapping == [(1, 0), (3, 1), (5, 2), (7, 3), (9, 4), (11, 5)]
    print("  ✓ Uniform mapping correct")

    # Last-N mapping
    last_n = get_layer_mapping("last_n")
    mapping = last_n.map(teacher_layers, student_layers)
    print(f"  Last-N (12→6): {mapping}")
    assert mapping == [(6, 0), (7, 1), (8, 2), (9, 3), (10, 4), (11, 5)]
    print("  ✓ Last-N mapping correct")

    # First-Last mapping
    first_last = get_layer_mapping("first_last")
    mapping = first_last.map(teacher_layers, student_layers)
    print(f"  First-Last (12→6): {mapping}")
    assert mapping[0][0] == 0, "Should start with layer 0"
    assert mapping[-1][0] == 11, "Should end with last layer"
    print("  ✓ First-Last mapping correct")

    # Edge cases
    mapping = uniform.map(6, 6)  # Same size
    print(f"  Same size (6→6): {mapping}")
    assert len(mapping) == 6

    mapping = uniform.map(12, 1)  # Single student layer
    print(f"  Single layer (12→1): {mapping}")
    assert len(mapping) == 1

    print("  All layer mapping strategies work correctly!")


# =============================================================================
# Test: Projection Strategies
# =============================================================================

def test_projection_strategies():
    """Test projection strategies for dimension alignment."""
    print("\n=== Test: Projection Strategies ===")

    batch, seq, teacher_dim, student_dim = 4, 32, 1024, 768

    # Identity (same dimensions)
    proj = create_projection_if_needed(768, 768, "linear")
    assert isinstance(proj, IdentityProjection), "Same dims should use identity"
    x = torch.randn(batch, seq, 768)
    out = proj(x)
    assert out.shape == x.shape
    print("  ✓ Identity projection (same dims)")

    # Linear projection
    proj = create_projection_if_needed(student_dim, teacher_dim, "linear")
    assert isinstance(proj, LinearProjection)
    x = torch.randn(batch, seq, student_dim)
    out = proj(x)
    assert out.shape == (batch, seq, teacher_dim)
    print(f"  ✓ Linear projection ({student_dim}→{teacher_dim})")

    # MLP projection
    proj = get_projection("mlp", input_dim=student_dim, output_dim=teacher_dim)
    assert isinstance(proj, MLPProjection)
    out = proj(x)
    assert out.shape == (batch, seq, teacher_dim)
    print(f"  ✓ MLP projection ({student_dim}→{teacher_dim})")

    print("  All projection strategies work correctly!")


# =============================================================================
# Test: KL Divergence Loss (baseline)
# =============================================================================

def test_kl_loss():
    """Test KL divergence loss still works (backwards compatibility)."""
    print("\n=== Test: KL Divergence Loss (Backwards Compatibility) ===")

    strategy = get_loss_strategy("kl_divergence", temperature=2.0, alpha=0.5)

    assert strategy.requires_hidden_states == False
    assert strategy.requires_attentions == False

    student_out = create_mock_outputs()
    teacher_out = create_mock_outputs()

    result = strategy.compute(
        student_logits=student_out.logits,
        teacher_logits=teacher_out.logits,
        ce_loss=student_out.loss,
    )

    assert result.total_loss.shape == ()  # Scalar
    assert "ce_weighted" in result.components
    assert "kl_weighted" in result.components
    print(f"  Total loss: {result.total_loss.item():.4f}")
    print(f"  Components: {list(result.components.keys())}")
    print("  ✓ KL loss works correctly!")


# =============================================================================
# Test: Feature Loss Strategy
# =============================================================================

def test_feature_loss():
    """Test feature-based distillation loss."""
    print("\n=== Test: Feature Loss Strategy ===")

    strategy = get_loss_strategy("feature_mse", alpha=0.5)

    # Check requirements
    assert strategy.requires_hidden_states == True
    assert strategy.requires_attentions == False
    print("  ✓ Requires hidden_states=True")

    # Create mock outputs with hidden states
    teacher_out = create_mock_outputs(
        num_layers=12, hidden_dim=768, include_hidden=True
    )
    student_out = create_mock_outputs(
        num_layers=6, hidden_dim=768, include_hidden=True
    )

    # Create layer mapping
    mapping = get_layer_mapping("uniform").map(12, 6)
    print(f"  Layer mapping: {mapping}")

    result = strategy.compute(
        student_logits=student_out.logits,
        teacher_logits=teacher_out.logits,
        ce_loss=student_out.loss,
        student_outputs=student_out,
        teacher_outputs=teacher_out,
        layer_mapping=mapping,
    )

    assert result.total_loss.shape == ()
    assert "feature_weighted" in result.components
    assert "num_layers_matched" in result.components
    print(f"  Total loss: {result.total_loss.item():.4f}")
    print(f"  Feature loss (raw): {result.components['feature_raw'].item():.4f}")
    print(f"  Layers matched: {int(result.components['num_layers_matched'].item())}")
    print("  ✓ Feature loss works correctly!")


# =============================================================================
# Test: Feature Loss with Projection
# =============================================================================

def test_feature_loss_with_projection():
    """Test feature loss when dimensions don't match."""
    print("\n=== Test: Feature Loss with Projection ===")

    strategy = get_loss_strategy("feature_mse", alpha=0.5)

    # Teacher: 1024 dim, Student: 768 dim
    teacher_out = create_mock_outputs(
        num_layers=12, hidden_dim=1024, include_hidden=True
    )
    student_out = create_mock_outputs(
        num_layers=6, hidden_dim=768, include_hidden=True
    )

    # Create projection: student (768) → teacher (1024)
    projector = create_projection_if_needed(768, 1024, "linear")
    print(f"  Projector: {projector}")

    mapping = get_layer_mapping("uniform").map(12, 6)

    result = strategy.compute(
        student_logits=student_out.logits,
        teacher_logits=teacher_out.logits,
        ce_loss=student_out.loss,
        student_outputs=student_out,
        teacher_outputs=teacher_out,
        layer_mapping=mapping,
        projector=projector,
    )

    print(f"  Total loss: {result.total_loss.item():.4f}")
    print("  ✓ Feature loss with projection works correctly!")


# =============================================================================
# Test: Attention Loss Strategy
# =============================================================================

def test_attention_loss():
    """Test attention-based distillation loss."""
    print("\n=== Test: Attention Loss Strategy ===")

    strategy = get_loss_strategy("attention_kl", alpha=0.5, temperature=1.0)

    # Check requirements
    assert strategy.requires_hidden_states == False
    assert strategy.requires_attentions == True
    print("  ✓ Requires attentions=True")

    # Create mock outputs with attention weights
    teacher_out = create_mock_outputs(
        num_layers=12, num_heads=12, include_attention=True
    )
    student_out = create_mock_outputs(
        num_layers=6, num_heads=12, include_attention=True
    )

    mapping = get_layer_mapping("uniform").map(12, 6)
    print(f"  Layer mapping: {mapping}")

    result = strategy.compute(
        student_logits=student_out.logits,
        teacher_logits=teacher_out.logits,
        ce_loss=student_out.loss,
        student_outputs=student_out,
        teacher_outputs=teacher_out,
        layer_mapping=mapping,
    )

    assert result.total_loss.shape == ()
    assert "attention_weighted" in result.components
    print(f"  Total loss: {result.total_loss.item():.4f}")
    print(f"  Attention loss (raw): {result.components['attention_raw'].item():.4f}")
    print("  ✓ Attention loss works correctly!")


# =============================================================================
# Test: Attention Loss with Different Head Counts
# =============================================================================

def test_attention_loss_different_heads():
    """Test attention loss handles different numbers of attention heads."""
    print("\n=== Test: Attention Loss (Different Head Counts) ===")

    strategy = get_loss_strategy("attention_kl", alpha=0.5)

    # Teacher: 12 heads, Student: 6 heads
    teacher_out = create_mock_outputs(
        num_layers=12, num_heads=12, include_attention=True
    )
    student_out = create_mock_outputs(
        num_layers=6, num_heads=6, include_attention=True
    )

    mapping = get_layer_mapping("uniform").map(12, 6)

    # Should not raise error - heads are averaged
    result = strategy.compute(
        student_logits=student_out.logits,
        teacher_logits=teacher_out.logits,
        ce_loss=student_out.loss,
        student_outputs=student_out,
        teacher_outputs=teacher_out,
        layer_mapping=mapping,
    )

    print(f"  Teacher heads: 12, Student heads: 6")
    print(f"  Total loss: {result.total_loss.item():.4f}")
    print("  ✓ Handles different head counts correctly!")


# =============================================================================
# Test: Error Handling
# =============================================================================

def test_error_handling():
    """Test that appropriate errors are raised for missing data."""
    print("\n=== Test: Error Handling ===")

    # Feature loss without hidden states
    feature_strategy = get_loss_strategy("feature_mse", alpha=0.5)
    student_out = create_mock_outputs(include_hidden=False)
    teacher_out = create_mock_outputs(include_hidden=False)

    try:
        feature_strategy.compute(
            student_logits=student_out.logits,
            teacher_logits=teacher_out.logits,
            ce_loss=student_out.loss,
            student_outputs=student_out,
            teacher_outputs=teacher_out,
            layer_mapping=[(0, 0)],
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "hidden_states" in str(e)
        print(f"  ✓ Feature loss raises error without hidden_states")

    # Attention loss without attentions
    attn_strategy = get_loss_strategy("attention_kl", alpha=0.5)
    student_out = create_mock_outputs(include_attention=False)
    teacher_out = create_mock_outputs(include_attention=False)

    try:
        attn_strategy.compute(
            student_logits=student_out.logits,
            teacher_logits=teacher_out.logits,
            ce_loss=student_out.loss,
            student_outputs=student_out,
            teacher_outputs=teacher_out,
            layer_mapping=[(0, 0)],
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "attentions" in str(e)
        print(f"  ✓ Attention loss raises error without attentions")

    print("  Error handling works correctly!")


# =============================================================================
# Test: Alpha Edge Cases
# =============================================================================

def test_alpha_edge_cases():
    """Test loss behavior at alpha boundaries."""
    print("\n=== Test: Alpha Edge Cases ===")

    student_out = create_mock_outputs(include_hidden=True)
    teacher_out = create_mock_outputs(include_hidden=True)
    mapping = [(0, 0)]

    # Alpha = 0 (CE only)
    strategy = get_loss_strategy("feature_mse", alpha=0.0)
    result = strategy.compute(
        student_logits=student_out.logits,
        teacher_logits=teacher_out.logits,
        ce_loss=student_out.loss,
        student_outputs=student_out,
        teacher_outputs=teacher_out,
        layer_mapping=mapping,
    )
    assert result.components["feature_weighted"].item() == 0.0
    print(f"  Alpha=0: feature_weighted = {result.components['feature_weighted'].item()}")

    # Alpha = 1 (Feature only)
    strategy = get_loss_strategy("feature_mse", alpha=1.0)
    result = strategy.compute(
        student_logits=student_out.logits,
        teacher_logits=teacher_out.logits,
        ce_loss=student_out.loss,
        student_outputs=student_out,
        teacher_outputs=teacher_out,
        layer_mapping=mapping,
    )
    assert result.components["ce_weighted"].item() == 0.0
    print(f"  Alpha=1: ce_weighted = {result.components['ce_weighted'].item()}")

    print("  ✓ Alpha edge cases handled correctly!")


# =============================================================================
# Test: Empty Layer Mapping
# =============================================================================

def test_empty_layer_mapping():
    """Test behavior with empty layer mapping."""
    print("\n=== Test: Empty Layer Mapping ===")

    strategy = get_loss_strategy("feature_mse", alpha=0.5)
    student_out = create_mock_outputs(include_hidden=True)
    teacher_out = create_mock_outputs(include_hidden=True)

    result = strategy.compute(
        student_logits=student_out.logits,
        teacher_logits=teacher_out.logits,
        ce_loss=student_out.loss,
        student_outputs=student_out,
        teacher_outputs=teacher_out,
        layer_mapping=[],  # Empty mapping
    )

    # With empty mapping, feature loss should be 0
    assert result.components["feature_raw"].item() == 0.0
    assert result.components["num_layers_matched"].item() == 0
    print(f"  Feature loss with empty mapping: {result.components['feature_raw'].item()}")
    print("  ✓ Empty layer mapping handled correctly!")


# =============================================================================
# Test: Combined Loss Strategy (RFA Matrix)
# =============================================================================

def test_combined_loss_rfa():
    """Test combined loss with Feature + Attention (for RFA matrix experiments)."""
    print("\n=== Test: Combined Loss (Feature + Attention) ===")

    from src.training import CombinedDistillationLoss

    # Create individual strategies
    feature_strategy = get_loss_strategy("feature_mse", alpha=0.5)
    attention_strategy = get_loss_strategy("attention_kl", alpha=0.5)

    # Combine Feature + Attention
    combined = CombinedDistillationLoss(
        strategies=[feature_strategy, attention_strategy],
        weights=[1.0, 1.0],
        alpha=0.5,
    )

    # Verify aggregated flags
    assert combined.requires_hidden_states == True, "Should require hidden_states from Feature"
    assert combined.requires_attentions == True, "Should require attentions from Attention"
    print("  ✓ Aggregates requires_hidden_states=True from Feature")
    print("  ✓ Aggregates requires_attentions=True from Attention")

    # Create mock outputs with both hidden states and attentions
    teacher_out = create_mock_outputs(
        num_layers=12, hidden_dim=768, num_heads=12,
        include_hidden=True, include_attention=True
    )
    student_out = create_mock_outputs(
        num_layers=6, hidden_dim=768, num_heads=6,
        include_hidden=True, include_attention=True
    )

    mapping = get_layer_mapping("uniform").map(12, 6)

    result = combined.compute(
        student_logits=student_out.logits,
        teacher_logits=teacher_out.logits,
        ce_loss=student_out.loss,
        student_outputs=student_out,
        teacher_outputs=teacher_out,
        layer_mapping=mapping,
    )

    assert result.total_loss.shape == ()
    assert "ce_weighted" in result.components
    # Should have components from both strategies
    assert any("feature" in k for k in result.components), "Missing feature components"
    assert any("attention" in k for k in result.components), "Missing attention components"

    print(f"  Total loss: {result.total_loss.item():.4f}")
    print(f"  Components: {list(result.components.keys())}")
    print("  ✓ Combined Feature+Attention loss works correctly!")


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("L1 KD Strategy Implementation Tests")
    print("=" * 60)

    test_loss_registry()
    test_layer_mapping_strategies()
    test_projection_strategies()
    test_kl_loss()
    test_feature_loss()
    test_feature_loss_with_projection()
    test_attention_loss()
    test_attention_loss_different_heads()
    test_error_handling()
    test_alpha_edge_cases()
    test_empty_layer_mapping()
    test_combined_loss_rfa()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
