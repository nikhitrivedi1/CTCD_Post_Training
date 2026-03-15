"""Unit tests for src/training/losses.py.

Tests loss strategy properties, alpha weighting, feature/attention strategies,
combined/staged wrappers, and the factory function. No gradient-flow tests
(those live in test_trainer.py).

Run:
    python -m pytest tests/unit/test_losses.py -v
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from src.training.losses import (
    AttentionLossStrategy,
    CombinedDistillationLoss,
    CosineDistillationLoss,
    CosineFeatureLoss,
    FeatureLossStrategy,
    KLDistillationLoss,
    LossOutput,
    MSEDistillationLoss,
    SoftTargetDistillationLoss,
    StagedDistillationLoss,
    compute_kl_divergence,
    get_loss_strategy,
)


# ---------------------------------------------------------------------------
# 1. Universal properties (parametrized across ALL strategies)
# ---------------------------------------------------------------------------

class TestUniversalProperties:
    """Every strategy must satisfy these basic contracts."""

    @pytest.fixture(params=[
        "response_kl", "response_reverse-kl", "response_mse",
        "response_cosine", "response_ce",
        "feature_mse", "feature_cosine", "attention_kl",
    ], ids=lambda x: x)
    def strategy_output(
        self, request, model_logits, target_logits, ce_loss,
        mock_model_outputs, mock_target_outputs, bert_distilbert_mapping,
    ):
        """Build a strategy by registry name and compute its output."""
        name = request.param
        needs_outputs = name in ("feature_mse", "feature_cosine", "attention_kl")
        kwargs = {}
        if needs_outputs:
            kwargs["layer_mapping"] = bert_distilbert_mapping

        strategy = get_loss_strategy(name, alpha=0.5, temperature=2.0, **kwargs)
        compute_kwargs = {}
        if needs_outputs:
            compute_kwargs["model_outputs"] = mock_model_outputs
            compute_kwargs["teacher_outputs"] = mock_target_outputs

        return strategy, strategy.compute(
            model_logits, target_logits, ce_loss, **compute_kwargs,
        )

    def test_returns_loss_output(self, strategy_output):
        _, out = strategy_output
        assert isinstance(out, LossOutput)

    def test_total_loss_finite(self, strategy_output):
        _, out = strategy_output
        assert torch.isfinite(out.total_loss)

    def test_total_loss_scalar(self, strategy_output):
        _, out = strategy_output
        assert out.total_loss.dim() == 0

    def test_components_non_empty(self, strategy_output):
        _, out = strategy_output
        assert len(out.components) > 0

    def test_components_all_finite(self, strategy_output):
        _, out = strategy_output
        for k, v in out.components.items():
            assert torch.isfinite(v), f"Component '{k}' is not finite"

    def test_name_is_non_empty_string(self, strategy_output):
        strategy, _ = strategy_output
        assert isinstance(strategy.name, str) and len(strategy.name) > 0


# ---------------------------------------------------------------------------
# 2. KL-specific tests
# ---------------------------------------------------------------------------

class TestKLDivergence:
    """Properties of compute_kl_divergence and KLDistillationLoss."""

    def test_t1_degenerates_to_standard_kl(self, model_logits, target_logits):
        """T=1 should equal standard KL(teacher || model)."""
        kl_t1 = compute_kl_divergence(model_logits, target_logits, temperature=1.0)
        log_q = F.log_softmax(model_logits.float(), dim=-1)
        p = F.softmax(target_logits.float(), dim=-1)
        expected = F.kl_div(log_q, p, reduction="batchmean")
        torch.testing.assert_close(kl_t1, expected, rtol=1e-4, atol=1e-6)

    def test_t_squared_hinton_formula(self, model_logits, target_logits):
        """KL output should scale by T^2 (Hinton formula)."""
        T = 4.0
        kl_scaled = compute_kl_divergence(model_logits, target_logits, temperature=T)

        # Manually compute without T^2 factor
        log_q = F.log_softmax(model_logits.float() / T, dim=-1)
        p = F.softmax(target_logits.float() / T, dim=-1)
        kl_raw = F.kl_div(log_q, p, reduction="batchmean")

        torch.testing.assert_close(kl_scaled, kl_raw * T**2, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("T", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0])
    def test_stability_sweep(self, model_logits, target_logits, T):
        """KL should be finite and non-negative across a wide T range."""
        kl = compute_kl_divergence(model_logits, target_logits, temperature=T)
        assert torch.isfinite(kl)
        assert kl >= -1e-6, f"KL should be non-negative, got {kl.item()}"

    def test_reverse_kl_differs_from_forward(self, model_logits, target_logits):
        """Forward KL(teacher||model) != reverse KL(model||teacher) in general."""
        fwd = compute_kl_divergence(model_logits, target_logits, temperature=2.0, reverse=False)
        rev = compute_kl_divergence(model_logits, target_logits, temperature=2.0, reverse=True)
        assert not torch.allclose(fwd, rev), "Forward and reverse KL should differ"


# ---------------------------------------------------------------------------
# 3. Alpha weighting (parametrized across response strategies)
# ---------------------------------------------------------------------------

class TestAlphaWeighting:
    """Alpha=0 -> pure CE, alpha=1 -> no CE, alpha=0.5 -> between."""

    STRATEGIES = [
        ("response_kl", {"temperature": 2.0}),
        ("response_mse", {}),
        ("response_cosine", {}),
        ("response_ce", {"temperature": 2.0}),
    ]

    @pytest.fixture(params=STRATEGIES, ids=lambda x: x[0])
    def strategy_spec(self, request):
        return request.param

    def test_alpha_zero_equals_ce(self, strategy_spec, model_logits, target_logits, ce_loss):
        name, kw = strategy_spec
        out = get_loss_strategy(name, alpha=0.0, **kw).compute(
            model_logits, target_logits, ce_loss,
        )
        torch.testing.assert_close(out.total_loss, ce_loss, rtol=1e-5, atol=1e-7)

    def test_alpha_one_ce_contribution_zero(self, strategy_spec, model_logits, target_logits, ce_loss):
        name, kw = strategy_spec
        out = get_loss_strategy(name, alpha=1.0, **kw).compute(
            model_logits, target_logits, ce_loss,
        )
        ce_key = [k for k in out.components if "ce" in k and "weighted" in k][0]
        torch.testing.assert_close(
            out.components[ce_key], torch.tensor(0.0), atol=1e-7, rtol=1e-5,
        )

    def test_alpha_half_between_extremes(self, strategy_spec, model_logits, target_logits, ce_loss):
        name, kw = strategy_spec
        l0 = get_loss_strategy(name, alpha=0.0, **kw).compute(
            model_logits, target_logits, ce_loss,
        ).total_loss.item()
        l1 = get_loss_strategy(name, alpha=1.0, **kw).compute(
            model_logits, target_logits, ce_loss,
        ).total_loss.item()
        l_half = get_loss_strategy(name, alpha=0.5, **kw).compute(
            model_logits, target_logits, ce_loss,
        ).total_loss.item()

        lo, hi = min(l0, l1), max(l0, l1)
        assert lo - 1e-5 <= l_half <= hi + 1e-5, (
            f"alpha=0.5 loss {l_half:.4f} not between "
            f"alpha=0 ({l0:.4f}) and alpha=1 ({l1:.4f})"
        )


# ---------------------------------------------------------------------------
# 4. Feature / Attention strategy-specific tests
# ---------------------------------------------------------------------------

class TestFeatureStrategies:
    """Feature loss requires hidden_states and handles dtype casting."""

    def test_requires_hidden_states_flag(self):
        assert FeatureLossStrategy.requires_hidden_states is True
        assert CosineFeatureLoss.requires_hidden_states is True

    def test_raises_without_hidden_states(self, model_logits, target_logits, ce_loss):
        s = FeatureLossStrategy(alpha=0.5, layer_mapping=[(0, 0)])
        with pytest.raises(ValueError, match="hidden_states"):
            s.compute(model_logits, target_logits, ce_loss)

    def test_feature_mse_with_mapping(
        self, model_logits, target_logits, ce_loss,
        mock_model_outputs, mock_target_outputs, bert_distilbert_mapping,
    ):
        s = FeatureLossStrategy(alpha=0.5, layer_mapping=bert_distilbert_mapping)
        out = s.compute(
            model_logits, target_logits, ce_loss,
            model_outputs=mock_model_outputs,
            teacher_outputs=mock_target_outputs,
        )
        assert torch.isfinite(out.total_loss)
        assert out.components["num_layers_matched"].item() == len(bert_distilbert_mapping)

    def test_cosine_feature_with_mapping(
        self, model_logits, target_logits, ce_loss,
        mock_model_outputs, mock_target_outputs, bert_distilbert_mapping,
    ):
        s = CosineFeatureLoss(alpha=0.5, layer_mapping=bert_distilbert_mapping)
        out = s.compute(
            model_logits, target_logits, ce_loss,
            model_outputs=mock_model_outputs,
            teacher_outputs=mock_target_outputs,
        )
        assert torch.isfinite(out.total_loss)
        assert "feature_cosine_raw" in out.components

    def test_dtype_casting_fp16_teacher(
        self, batch_size, seq_len, hidden_size,
        model_logits, target_logits, ce_loss,
    ):
        """Feature loss handles mixed dtypes (fp16 teacher, fp32 model)."""
        model_hidden = (torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32),)
        target_hidden = (torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16),)

        model_out = MagicMock()
        model_out.hidden_states = model_hidden
        target_out = MagicMock()
        target_out.hidden_states = target_hidden

        s = FeatureLossStrategy(alpha=0.5, layer_mapping=[(0, 0)])
        out = s.compute(
            model_logits, target_logits, ce_loss,
            model_outputs=model_out, teacher_outputs=target_out,
        )
        assert torch.isfinite(out.total_loss)


class TestAttentionStrategy:
    """Attention loss requires attention weights and handles head mismatch."""

    def test_requires_attentions_flag(self):
        assert AttentionLossStrategy.requires_attentions is True

    def test_raises_without_attentions(self, model_logits, target_logits, ce_loss):
        s = AttentionLossStrategy(alpha=0.5, layer_mapping=[(0, 0)])
        with pytest.raises(ValueError, match="attentions"):
            s.compute(model_logits, target_logits, ce_loss)

    def test_attention_with_mapping(
        self, model_logits, target_logits, ce_loss,
        mock_model_outputs, mock_target_outputs, bert_distilbert_mapping,
    ):
        s = AttentionLossStrategy(alpha=0.5, layer_mapping=bert_distilbert_mapping)
        out = s.compute(
            model_logits, target_logits, ce_loss,
            model_outputs=mock_model_outputs,
            teacher_outputs=mock_target_outputs,
        )
        assert torch.isfinite(out.total_loss)
        assert out.components["num_layers_matched"].item() == len(bert_distilbert_mapping)

    def test_different_head_counts(
        self, batch_size, seq_len, model_logits, target_logits, ce_loss,
    ):
        """When head counts differ, attention loss averages heads."""
        m_attn = torch.rand(batch_size, 4, seq_len, seq_len)
        m_attn = m_attn / m_attn.sum(dim=-1, keepdim=True)
        t_attn = torch.rand(batch_size, 8, seq_len, seq_len)
        t_attn = t_attn / t_attn.sum(dim=-1, keepdim=True)

        model_out = MagicMock()
        model_out.attentions = (m_attn,)
        target_out = MagicMock()
        target_out.attentions = (t_attn,)

        s = AttentionLossStrategy(alpha=0.5, layer_mapping=[(0, 0)])
        out = s.compute(
            model_logits, target_logits, ce_loss,
            model_outputs=model_out, teacher_outputs=target_out,
        )
        assert torch.isfinite(out.total_loss)


# ---------------------------------------------------------------------------
# 5. CombinedDistillationLoss
# ---------------------------------------------------------------------------

class TestCombinedLoss:
    """Combined loss normalizes weights, tracks components, respects zero weight."""

    def test_weights_normalize(self):
        s1 = KLDistillationLoss(alpha=0.5)
        s2 = MSEDistillationLoss(alpha=0.5)
        combined = CombinedDistillationLoss([s1, s2], [2.0, 8.0], alpha=0.5)
        assert abs(sum(combined.weights) - 1.0) < 1e-7

    def test_components_present_from_all(self, model_logits, target_logits, ce_loss):
        s1 = KLDistillationLoss(alpha=0.5)
        s2 = MSEDistillationLoss(alpha=0.5)
        combined = CombinedDistillationLoss([s1, s2], [1.0, 1.0], alpha=0.5)
        out = combined.compute(model_logits, target_logits, ce_loss)
        assert "kl_raw" in out.components
        assert "mse_raw" in out.components

    def test_zero_weight_excludes_contribution(self, model_logits, target_logits, ce_loss):
        """A sub-strategy with weight=0 should not change the loss vs weight=1e-30."""
        s_kl = KLDistillationLoss(alpha=0.5)
        s_mse = MSEDistillationLoss(alpha=0.5)

        combined_zero = CombinedDistillationLoss([s_kl, s_mse], [1.0, 0.0], alpha=0.5)
        combined_only_kl = CombinedDistillationLoss([s_kl], [1.0], alpha=0.5)

        out_zero = combined_zero.compute(model_logits, target_logits, ce_loss)
        out_only = combined_only_kl.compute(model_logits, target_logits, ce_loss)

        torch.testing.assert_close(
            out_zero.total_loss, out_only.total_loss, rtol=1e-5, atol=1e-7,
        )

    def test_aggregates_requirement_flags(self):
        s1 = KLDistillationLoss(alpha=0.5)
        s2 = FeatureLossStrategy(alpha=0.5)
        combined = CombinedDistillationLoss([s1, s2], [1.0, 1.0], alpha=0.5)
        assert combined.requires_hidden_states is True
        assert combined.requires_attentions is False

    def test_name_contains_sub_strategy_names(self):
        s1 = KLDistillationLoss(alpha=0.5)
        s2 = MSEDistillationLoss(alpha=0.5)
        combined = CombinedDistillationLoss([s1, s2], [1.0, 1.0], alpha=0.5)
        assert "response_kl" in combined.name
        assert "response_mse" in combined.name


# ---------------------------------------------------------------------------
# 6. StagedDistillationLoss
# ---------------------------------------------------------------------------

class TestStagedLoss:
    """Staged loss selects strategy by training progress."""

    @pytest.fixture
    def staged(self):
        return StagedDistillationLoss(
            stages=[
                {"strategy": "response_mse", "until_progress": 0.5},
                {"strategy": "response_kl", "until_progress": 1.0},
            ],
            alpha=0.5,
            temperature=2.0,
        )

    def test_correct_strategy_early(self, staged):
        assert staged.get_current_strategy(0.2).name == "response_mse"

    def test_correct_strategy_late(self, staged):
        assert staged.get_current_strategy(0.7).name == "response_kl"

    def test_boundary_uses_current_stage(self, staged):
        """progress=0.5 is <= until_progress=0.5, so still first stage."""
        assert staged.get_current_strategy(0.5).name == "response_mse"

    def test_progress_beyond_one_uses_last(self, staged):
        assert staged.get_current_strategy(1.5).name == "response_kl"

    def test_output_has_stage_metadata(self, staged, model_logits, target_logits, ce_loss):
        out = staged.compute(
            model_logits, target_logits, ce_loss, training_progress=0.3,
        )
        assert "active_stage" in out.components
        assert "training_progress" in out.components
        assert out.components["active_stage"].item() == 0

    def test_transition_changes_active_stage(self, staged, model_logits, target_logits, ce_loss):
        out_early = staged.compute(
            model_logits, target_logits, ce_loss, training_progress=0.1,
        )
        out_late = staged.compute(
            model_logits, target_logits, ce_loss, training_progress=0.9,
        )
        assert out_early.components["active_stage"].item() == 0
        assert out_late.components["active_stage"].item() == 1

    def test_aggregates_requirement_flags(self):
        staged = StagedDistillationLoss(
            stages=[
                {"strategy": "feature_mse", "until_progress": 0.5},
                {"strategy": "response_kl", "until_progress": 1.0},
            ],
            alpha=0.5,
        )
        assert staged.requires_hidden_states is True
        assert staged.requires_attentions is False

    def test_name_reflects_stages(self, staged):
        assert "staged" in staged.name
        assert "response_mse" in staged.name
        assert "response_kl" in staged.name


# ---------------------------------------------------------------------------
# 7. Factory: get_loss_strategy
# ---------------------------------------------------------------------------

class TestFactory:
    """get_loss_strategy returns correct types and filters kwargs."""

    @pytest.mark.parametrize("name,expected_type", [
        ("response_kl", KLDistillationLoss),
        ("response_reverse-kl", KLDistillationLoss),
        ("response_mse", MSEDistillationLoss),
        ("response_cosine", CosineDistillationLoss),
        ("response_ce", SoftTargetDistillationLoss),
        ("feature_mse", FeatureLossStrategy),
        ("feature_cosine", CosineFeatureLoss),
        ("attention_kl", AttentionLossStrategy),
    ])
    def test_returns_correct_type(self, name, expected_type):
        assert isinstance(get_loss_strategy(name, alpha=0.5), expected_type)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown loss strategy"):
            get_loss_strategy("nonexistent_loss")

    def test_irrelevant_kwargs_filtered(self):
        """Temperature kwarg silently dropped for strategies that don't use it."""
        s = get_loss_strategy("response_mse", alpha=0.3, temperature=5.0)
        assert isinstance(s, MSEDistillationLoss)
        assert s.alpha == 0.3

    def test_backward_compat_aliases(self):
        assert isinstance(get_loss_strategy("kl_divergence"), KLDistillationLoss)
        assert isinstance(get_loss_strategy("mse_logits"), MSEDistillationLoss)
        assert isinstance(get_loss_strategy("cosine_embedding"), CosineDistillationLoss)
        assert isinstance(get_loss_strategy("soft_target"), SoftTargetDistillationLoss)

    def test_reverse_kl_flag_set(self):
        s = get_loss_strategy("response_reverse-kl")
        assert s.reverse is True

    def test_fallback_for_unlisted_class(self):
        """When a class isn't in ACCEPTED_PARAMS, factory passes all kwargs through."""
        # Temporarily register a dummy class to exercise the fallback branch
        from src.training.losses import LOSS_REGISTRY

        class _DummyLoss(KLDistillationLoss):
            pass

        LOSS_REGISTRY["_dummy_test"] = _DummyLoss
        try:
            s = get_loss_strategy("_dummy_test", alpha=0.7, temperature=3.0)
            assert isinstance(s, _DummyLoss)
            assert s.alpha == 0.7
        finally:
            del LOSS_REGISTRY["_dummy_test"]


# ---------------------------------------------------------------------------
# 8. Coverage gap tests
# ---------------------------------------------------------------------------

class TestMSENormalizeFalse:
    """Exercise the normalize=False path in MSEDistillationLoss."""

    def test_mse_no_normalize(self, model_logits, target_logits, ce_loss):
        s = MSEDistillationLoss(alpha=0.5, normalize=False)
        out = s.compute(model_logits, target_logits, ce_loss)
        expected_mse = F.mse_loss(model_logits, target_logits)
        torch.testing.assert_close(
            out.components["mse_raw"], expected_mse, rtol=1e-5, atol=1e-7,
        )


class TestFeatureEmptyMapping:
    """Feature loss with empty layer_mapping returns zero feature loss."""

    def test_feature_mse_empty_mapping(
        self, model_logits, target_logits, ce_loss,
        mock_model_outputs, mock_target_outputs,
    ):
        s = FeatureLossStrategy(alpha=0.5, layer_mapping=[])
        out = s.compute(
            model_logits, target_logits, ce_loss,
            model_outputs=mock_model_outputs,
            teacher_outputs=mock_target_outputs,
        )
        torch.testing.assert_close(
            out.components["feature_raw"],
            torch.tensor(0.0),
            atol=1e-7, rtol=1e-5,
        )


class TestFeatureWithProjector:
    """Feature loss with a projector module."""

    def test_feature_mse_with_projector(
        self, batch_size, seq_len, model_logits, target_logits, ce_loss,
    ):
        import torch.nn as nn
        model_hidden_dim = 32
        teacher_hidden_dim = 64
        projector = nn.Linear(model_hidden_dim, teacher_hidden_dim)

        model_out = MagicMock()
        model_out.hidden_states = (
            torch.randn(batch_size, seq_len, model_hidden_dim),
        )
        teacher_out = MagicMock()
        teacher_out.hidden_states = (
            torch.randn(batch_size, seq_len, teacher_hidden_dim),
        )

        s = FeatureLossStrategy(alpha=0.5, layer_mapping=[(0, 0)], projector=projector)
        out = s.compute(
            model_logits, target_logits, ce_loss,
            model_outputs=model_out, teacher_outputs=teacher_out,
        )
        assert torch.isfinite(out.total_loss)


class TestCosineFeatureCoverage:
    """Cover CosineFeatureLoss compute paths: projector, dtype cast, empty mapping."""

    def test_cosine_empty_mapping(
        self, model_logits, target_logits, ce_loss,
        mock_model_outputs, mock_target_outputs,
    ):
        s = CosineFeatureLoss(alpha=0.5, layer_mapping=[])
        out = s.compute(
            model_logits, target_logits, ce_loss,
            model_outputs=mock_model_outputs,
            teacher_outputs=mock_target_outputs,
        )
        torch.testing.assert_close(
            out.components["feature_cosine_raw"],
            torch.tensor(0.0),
            atol=1e-7, rtol=1e-5,
        )

    def test_cosine_with_projector(
        self, batch_size, seq_len, model_logits, target_logits, ce_loss,
    ):
        import torch.nn as nn
        model_dim, teacher_dim = 32, 64
        projector = nn.Linear(model_dim, teacher_dim)

        model_out = MagicMock()
        model_out.hidden_states = (torch.randn(batch_size, seq_len, model_dim),)
        teacher_out = MagicMock()
        teacher_out.hidden_states = (torch.randn(batch_size, seq_len, teacher_dim),)

        s = CosineFeatureLoss(alpha=0.5, layer_mapping=[(0, 0)], projector=projector)
        out = s.compute(
            model_logits, target_logits, ce_loss,
            model_outputs=model_out, teacher_outputs=teacher_out,
        )
        assert torch.isfinite(out.total_loss)

    def test_cosine_dtype_casting(
        self, batch_size, seq_len, hidden_size,
        model_logits, target_logits, ce_loss,
    ):
        """CosineFeatureLoss handles mixed dtypes (fp16 teacher, fp32 model)."""
        model_out = MagicMock()
        model_out.hidden_states = (
            torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32),
        )
        teacher_out = MagicMock()
        teacher_out.hidden_states = (
            torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16),
        )

        s = CosineFeatureLoss(alpha=0.5, layer_mapping=[(0, 0)])
        out = s.compute(
            model_logits, target_logits, ce_loss,
            model_outputs=model_out, teacher_outputs=teacher_out,
        )
        assert torch.isfinite(out.total_loss)

    def test_cosine_raises_without_hidden_states(self, model_logits, target_logits, ce_loss):
        s = CosineFeatureLoss(alpha=0.5, layer_mapping=[(0, 0)])
        with pytest.raises(ValueError, match="hidden_states"):
            s.compute(model_logits, target_logits, ce_loss)


class TestAttentionEmptyMapping:
    """Attention loss with empty layer_mapping returns zero attention loss."""

    def test_attention_empty_mapping(
        self, model_logits, target_logits, ce_loss,
        mock_model_outputs, mock_target_outputs,
    ):
        s = AttentionLossStrategy(alpha=0.5, layer_mapping=[])
        out = s.compute(
            model_logits, target_logits, ce_loss,
            model_outputs=mock_model_outputs,
            teacher_outputs=mock_target_outputs,
        )
        torch.testing.assert_close(
            out.components["attention_raw"],
            torch.tensor(0.0),
            atol=1e-7, rtol=1e-5,
        )


class TestStagedOutOfOrder:
    """Stages created out of order get sorted by until_progress."""

    def test_stages_sorted_by_progress(self):
        staged = StagedDistillationLoss(
            stages=[
                {"strategy": "response_kl", "until_progress": 1.0},
                {"strategy": "response_mse", "until_progress": 0.3},
            ],
            alpha=0.5,
            temperature=2.0,
        )
        # After sorting, first stage should be the one with until_progress=0.3
        assert staged.stages[0]["until_progress"] == 0.3
        assert staged.stages[0]["strategy"].name == "response_mse"

    def test_staged_with_strategy_instance(self, model_logits, target_logits, ce_loss):
        """Pass a pre-built strategy instance instead of a string name."""
        mse_instance = MSEDistillationLoss(alpha=0.5)
        staged = StagedDistillationLoss(
            stages=[
                {"strategy": mse_instance, "until_progress": 0.5},
                {"strategy": "response_kl", "until_progress": 1.0},
            ],
            alpha=0.5,
            temperature=2.0,
        )
        assert staged.stages[0]["strategy"] is mse_instance
        out = staged.compute(model_logits, target_logits, ce_loss, training_progress=0.2)
        assert torch.isfinite(out.total_loss)
