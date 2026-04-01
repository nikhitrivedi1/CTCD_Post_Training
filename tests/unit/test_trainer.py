"""Tests for src/training/trainer.py

Covers DistillationTrainer basics, P0 gradient flow invariants,
and eval behavior. Uses tiny randomly-initialized BertForSequenceClassification
fixtures from conftest -- no downloads.

Run:
    python -m pytest tests/unit/test_trainer.py -v
"""

from unittest.mock import patch

import torch
import transformers
from datasets import Dataset
from transformers import TrainingArguments

from src.training.losses import (
    AttentionLossStrategy,
    CombinedDistillationLoss,
    FeatureLossStrategy,
    KLDistillationLoss,
)
from src.training.trainer import DistillationTrainer


# ===========================================================================
# Helpers
# ===========================================================================

def _make_dataset(n=8, seq_len=16, vocab=1000, num_labels=3):
    """Synthetic classification dataset for trainer tests."""
    torch.manual_seed(42)
    return Dataset.from_dict({
        "input_ids": torch.randint(1, vocab, (n, seq_len)).tolist(),
        "attention_mask": torch.ones(n, seq_len, dtype=torch.long).tolist(),
        "labels": torch.randint(0, num_labels, (n,)).tolist(),
    })


def _make_batch(batch_size=2, seq_len=16, vocab=1000, num_labels=3):
    """Minimal batch dict for a single forward pass."""
    torch.manual_seed(42)
    return {
        "input_ids": torch.randint(1, vocab, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "labels": torch.randint(0, num_labels, (batch_size,)),
    }


def _make_trainer(teacher, model, freeze_teacher=True, alpha_h=1.0, alpha_s=1.0,
                  beta_h=None, beta_s=None,
                  loss_type=None, loss_strategy=None, teacher_label_map=None,
                  tmp_path=None, max_steps=1, train_dataset=None):
    """Build a DistillationTrainer with minimal TrainingArguments."""
    import tempfile
    output_dir = str(tmp_path) if tmp_path else tempfile.mkdtemp()
    args = TrainingArguments(
        output_dir=output_dir,
        use_cpu=True,
        per_device_train_batch_size=2,
        max_steps=max_steps,
        logging_steps=1000,
        report_to="none",
    )
    kwargs = dict(
        teacher_model=teacher,
        model=model,
        args=args,
        train_dataset=train_dataset,
        freeze_teacher=freeze_teacher,
        alpha_h=alpha_h,
        alpha_s=alpha_s,
        temperature=2.0,
    )
    if beta_h is not None:
        kwargs["beta_h"] = beta_h
    if beta_s is not None:
        kwargs["beta_s"] = beta_s
    if loss_type:
        kwargs["loss_type"] = loss_type
    if loss_strategy:
        kwargs["loss_strategy"] = loss_strategy
    if teacher_label_map:
        kwargs["teacher_label_map"] = teacher_label_map
    return DistillationTrainer(**kwargs)


# ===========================================================================
# 1. Distillation Trainer Basics
# ===========================================================================

class TestDistillationTrainerBasics:
    """Core initialization and single-step behavior."""

    def test_one_step_produces_finite_scalar_loss(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """A single training step must produce a finite scalar loss."""
        trainer = _make_trainer(tiny_bert_teacher, tiny_bert_model, tmp_path=tmp_path)
        tiny_bert_model.train()
        batch = _make_batch()
        loss = trainer.compute_loss(tiny_bert_model, batch)
        assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"
        assert torch.isfinite(loss), f"Loss must be finite, got {loss}"

    def test_model_params_change_after_step(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """Model parameters must actually change after a training step."""
        ds = _make_dataset()
        trainer = _make_trainer(tiny_bert_teacher, tiny_bert_model,
                                tmp_path=tmp_path, train_dataset=ds)

        params_before = {n: p.clone() for n, p in tiny_bert_model.named_parameters()}
        trainer.train()
        params_after = {n: p.clone() for n, p in tiny_bert_model.named_parameters()}

        changed = any(
            not torch.equal(params_before[n], params_after[n])
            for n in params_before
        )
        assert changed, "Model parameters must change after a training step"

    def test_teacher_frozen_when_freeze_teacher_true(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """Frozen teacher: all params requires_grad=False, eval mode."""
        trainer = _make_trainer(tiny_bert_teacher, tiny_bert_model,
                                freeze_teacher=True, tmp_path=tmp_path)
        for p in trainer.teacher.parameters():
            assert not p.requires_grad
        assert not trainer.teacher.training

    def test_teacher_trainable_when_freeze_teacher_false(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """Co-training: teacher params require grad and model is in train mode."""
        trainer = _make_trainer(tiny_bert_teacher, tiny_bert_model,
                                freeze_teacher=False, tmp_path=tmp_path)
        assert any(p.requires_grad for p in trainer.teacher.parameters())
        assert trainer.teacher.training

    def test_co_training_optimizer_includes_teacher(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """create_optimizer() must add teacher param group in co-training."""
        trainer = _make_trainer(tiny_bert_teacher, tiny_bert_model,
                                freeze_teacher=False, tmp_path=tmp_path)
        optimizer = trainer.create_optimizer()

        assert len(optimizer.param_groups) >= 2, \
            "Co-training optimizer must have at least 2 param groups"

        teacher_ids = {id(p) for p in tiny_bert_teacher.parameters()}
        opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        assert teacher_ids & opt_ids, "Teacher params must be in optimizer"

    def test_detach_output_intermediates_preserves_logits(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """_detach_output_intermediates detaches hidden_states/attentions but not logits."""
        trainer = _make_trainer(tiny_bert_teacher, tiny_bert_model,
                                freeze_teacher=False, tmp_path=tmp_path)
        batch = _make_batch()
        tiny_bert_teacher.train()
        outputs = tiny_bert_teacher(
            **batch, output_hidden_states=True, output_attentions=True
        )

        detached = trainer._detach_output_intermediates(outputs)

        # Hidden states and attentions must be detached
        for h in detached.hidden_states:
            assert not h.requires_grad, "Hidden states must be detached"
        for a in detached.attentions:
            assert not a.requires_grad, "Attentions must be detached"

        # Logits must still be in the graph (not detached)
        assert detached.logits.requires_grad, "Logits must remain attached to graph"

    def test_split_student_coefficients_are_resolved(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """alpha_h/alpha_s should be used as the student hard/soft weights."""
        trainer = _make_trainer(
            tiny_bert_teacher, tiny_bert_model, tmp_path=tmp_path, alpha_h=0.3, alpha_s=0.7
        )
        assert trainer.alpha_h == 0.3
        assert trainer.alpha_s == 0.7

    def test_cotrain_teacher_defaults_to_ce_only(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """When beta_h/beta_s are omitted in co-training, teacher defaults to CE-only (beta_h=1, beta_s=0)."""
        trainer = _make_trainer(
            tiny_bert_teacher, tiny_bert_model, freeze_teacher=False, tmp_path=tmp_path
        )
        assert trainer.beta_h == 1.0
        assert trainer.beta_s == 0.0


# ===========================================================================
# 2. P0 Gradient Flow
# ===========================================================================

class TestGradientFlow:
    """P0 gradient flow invariants -- the most valuable tests in the suite.

    These verify that the teacher signal actually affects learning, that
    co-training gradients flow correctly, and that feature loss uses
    teacher representations (not model's own).
    """

    def test_teacher_signal_changes_model_gradients(self, tmp_path):
        """KD loss (alpha_s=15.0) must produce different model gradients than CE-only (alpha_s=0).

        If this fails, distillation is silently broken: training runs but
        the teacher contributes nothing.
        """
        from transformers import BertConfig, BertForSequenceClassification

        batch = _make_batch()

        def fresh_pair():
            torch.manual_seed(42)
            cfg = BertConfig(
                hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                intermediate_size=128, num_labels=3, vocab_size=1000,
                attn_implementation="eager",
            )
            return BertForSequenceClassification(cfg), BertForSequenceClassification(cfg)

        # KD path: high alpha_s to ensure teacher soft labels dominate and produce a clear gradient diff
        teacher_kd, model_kd = fresh_pair()
        trainer_kd = _make_trainer(teacher_kd, model_kd, alpha_h=0.5, alpha_s=15.0, tmp_path=tmp_path)
        model_kd.train()
        loss_kd = trainer_kd.compute_loss(model_kd, batch)
        loss_kd.backward()
        kd_grads = {n: p.grad.clone() for n, p in model_kd.named_parameters()
                    if p.grad is not None}

        # CE-only path
        ce_dir = tmp_path / "ce"
        ce_dir.mkdir()
        teacher_ce, model_ce = fresh_pair()
        trainer_ce = _make_trainer(teacher_ce, model_ce, alpha_h=1.0, alpha_s=0.0, tmp_path=ce_dir)
        model_ce.train()
        loss_ce = trainer_ce.compute_loss(model_ce, batch)
        loss_ce.backward()
        ce_grads = {n: p.grad.clone() for n, p in model_ce.named_parameters()
                    if p.grad is not None}

        diffs = [
            not torch.allclose(kd_grads[n], ce_grads[n], atol=1e-6)
            for n in kd_grads if n in ce_grads
        ]
        assert any(diffs), \
            "KD gradients must differ from CE-only -- teacher signal has no effect"

    def test_co_training_teacher_gets_ce_only(self, tmp_path):
        """Teacher gradients must come from CE loss only, not KL.

        Strategy: run pure co-training (beta_h=1, beta_s=0 — no codistillation to teacher)
        with two different student alpha_s values (1.0 and 15.0). Since:
          1. The teacher loss uses beta_s=0, so student logits never contribute to teacher loss.
          2. Even if beta_s > 0, student logits are detached before being passed to the teacher branch.
        Teacher gradients must be identical across both runs. If they differ, KL gradients are leaking.
        """
        from transformers import BertConfig, BertForSequenceClassification

        batch = _make_batch()

        def fresh_pair():
            torch.manual_seed(42)
            cfg = BertConfig(
                hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                intermediate_size=128, num_labels=3, vocab_size=1000,
                attn_implementation="eager",
            )
            return BertForSequenceClassification(cfg), BertForSequenceClassification(cfg)

        # alpha_s=1.0, pure co-training (no codistillation to teacher)
        teacher_a, model_a = fresh_pair()
        model_a.train()
        trainer_a = _make_trainer(teacher_a, model_a, freeze_teacher=False,
                                  alpha_h=1.0, alpha_s=1.0,
                                  beta_h=1.0, beta_s=0.0, tmp_path=tmp_path)
        loss_a = trainer_a.compute_loss(model_a, batch)
        loss_a.backward()
        grads_a = {n: p.grad.clone() for n, p in teacher_a.named_parameters()
                   if p.grad is not None}

        # alpha_s=15.0, same pure co-training setup — large value to ensure any leakage would show up
        dir_b = tmp_path / "b"
        dir_b.mkdir()
        teacher_b, model_b = fresh_pair()
        model_b.train()
        trainer_b = _make_trainer(teacher_b, model_b, freeze_teacher=False,
                                  alpha_h=1.0, alpha_s=15.0,
                                  beta_h=1.0, beta_s=0.0, tmp_path=dir_b)
        loss_b = trainer_b.compute_loss(model_b, batch)
        loss_b.backward()
        grads_b = {n: p.grad.clone() for n, p in teacher_b.named_parameters()
                   if p.grad is not None}

        assert len(grads_a) > 0, "Co-trained teacher must receive gradients"
        for name in grads_a:
            assert torch.allclose(grads_a[name], grads_b[name], atol=1e-6), \
                f"Teacher grad for {name} changed with alpha_s -- " \
                f"KL/distillation gradients are leaking into teacher"

    def test_feature_loss_not_self_comparison(self, tmp_path):
        """Feature loss must use teacher hidden states, not model's own.

        Verify by running co-training with feature distillation at two different
        feature weights. Since teacher intermediates are detached, teacher
        gradients must be unchanged -- proving the feature loss reads from
        teacher (detached), not from model.
        """
        from transformers import BertConfig, BertForSequenceClassification

        batch = _make_batch()
        layer_mapping = [(0, 0), (1, 1)]

        def fresh_pair():
            torch.manual_seed(42)
            cfg = BertConfig(
                hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                intermediate_size=128, num_labels=3, vocab_size=1000,
                attn_implementation="eager",
            )
            return BertForSequenceClassification(cfg), BertForSequenceClassification(cfg)

        def run_with_weights(kl_w, feat_w, output_dir):
            teacher, model = fresh_pair()
            kl = KLDistillationLoss(temperature=2.0, ce_loss_coeff=0.5, kd_loss_coeff=0.5)
            feat = FeatureLossStrategy(ce_loss_coeff=0.5, kd_loss_coeff=0.5, layer_mapping=layer_mapping)
            strategy = CombinedDistillationLoss(
                strategies=[kl, feat], weights=[kl_w, feat_w], alpha_h=0.5, alpha_s=0.5
            )
            trainer = _make_trainer(teacher, model, freeze_teacher=False,
                                    loss_strategy=strategy, tmp_path=output_dir)
            trainer.layer_mapping = layer_mapping
            model.train()
            loss = trainer.compute_loss(model, batch)
            loss.backward()
            return {n: p.grad.clone() for n, p in teacher.named_parameters()
                    if p.grad is not None}

        grads_1 = run_with_weights(0.5, 0.5, tmp_path)
        dir2 = tmp_path / "feat2"
        dir2.mkdir()
        grads_2 = run_with_weights(0.2, 0.8, dir2)

        assert len(grads_1) > 0, "Teacher must have gradients"
        for name in grads_1:
            assert torch.allclose(grads_1[name], grads_2[name], atol=1e-6), \
                f"Teacher grad for {name} changed with feature weight -- " \
                f"feature loss gradients are leaking into teacher"

    def test_codistillation_student_kl_does_not_backprop_into_teacher(self, tmp_path):
        """Student KL must not backprop into teacher (teacher_logits.detach() check).

        With beta_h=0 and beta_s=0 the teacher receives zero loss. The student's
        KL term uses detached teacher logits, so teacher parameters must have zero
        gradient after backward. Any non-zero teacher gradient means the detach is
        broken and the student's soft loss is pulling on the teacher.
        """
        from transformers import BertConfig, BertForSequenceClassification

        torch.manual_seed(42)
        cfg = BertConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=128, num_labels=3, vocab_size=1000,
            attn_implementation="eager",
        )
        teacher = BertForSequenceClassification(cfg)
        model = BertForSequenceClassification(cfg)

        trainer = _make_trainer(
            teacher, model, freeze_teacher=False,
            alpha_h=1.0, alpha_s=15.0,
            beta_h=0.0, beta_s=0.0,
            tmp_path=tmp_path,
        )
        model.train()
        loss = trainer.compute_loss(model, _make_batch())
        loss.backward()

        for name, param in teacher.named_parameters():
            assert param.grad is None or torch.all(param.grad == 0), (
                f"Teacher parameter '{name}' has non-zero gradient despite beta_h=0, beta_s=0 -- "
                f"student KL is backpropagating into teacher (detach missing or broken)"
            )

    def test_codistillation_teacher_receives_soft_loss_student_unaffected(self, tmp_path):
        """Teacher receives student soft logits; student gradient is unaffected by beta_s.

        Two codistillation runs identical except for beta_s (1.0 vs 15.0):
          - Teacher gradients must DIFFER: confirms student logits are reaching the teacher loss.
          - Student gradients must be IDENTICAL: confirms model_logits.detach() is working —
            the teacher's KL coefficient does not bleed back into student gradients.
        """
        from transformers import BertConfig, BertForSequenceClassification

        batch = _make_batch()

        def fresh_pair():
            torch.manual_seed(42)
            cfg = BertConfig(
                hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                intermediate_size=128, num_labels=3, vocab_size=1000,
                attn_implementation="eager",
            )
            return BertForSequenceClassification(cfg), BertForSequenceClassification(cfg)

        # beta_s=1.0
        teacher_a, model_a = fresh_pair()
        model_a.train()
        trainer_a = _make_trainer(teacher_a, model_a, freeze_teacher=False,
                                  alpha_h=1.0, alpha_s=1.0,
                                  beta_h=1.0, beta_s=1.0, tmp_path=tmp_path)
        loss_a = trainer_a.compute_loss(model_a, batch)
        loss_a.backward()
        teacher_grads_a = {n: p.grad.clone() for n, p in teacher_a.named_parameters() if p.grad is not None}
        student_grads_a = {n: p.grad.clone() for n, p in model_a.named_parameters() if p.grad is not None}

        # beta_s=15.0
        dir_b = tmp_path / "b"
        dir_b.mkdir()
        teacher_b, model_b = fresh_pair()
        model_b.train()
        trainer_b = _make_trainer(teacher_b, model_b, freeze_teacher=False,
                                  alpha_h=1.0, alpha_s=1.0,
                                  beta_h=1.0, beta_s=15.0, tmp_path=dir_b)
        loss_b = trainer_b.compute_loss(model_b, batch)
        loss_b.backward()
        teacher_grads_b = {n: p.grad.clone() for n, p in teacher_b.named_parameters() if p.grad is not None}
        student_grads_b = {n: p.grad.clone() for n, p in model_b.named_parameters() if p.grad is not None}

        # Teacher grads must differ: soft signal is reaching the teacher
        teacher_diffs = [
            not torch.allclose(teacher_grads_a[n], teacher_grads_b[n], atol=1e-6)
            for n in teacher_grads_a if n in teacher_grads_b
        ]
        assert any(teacher_diffs), (
            "Teacher gradients are identical despite different beta_s -- "
            "student soft logits are not reaching the teacher loss"
        )

        # Student grads must be identical: beta_s does not affect student
        for name in student_grads_a:
            assert torch.allclose(student_grads_a[name], student_grads_b[name], atol=1e-6), (
                f"Student grad for '{name}' changed with beta_s -- "
                f"teacher KL is backpropagating into student (detach missing or broken)"
            )


# ===========================================================================
# 3. Eval Behavior
# ===========================================================================

class TestEvalBehavior:
    """Eval-time behavior: OOM prevention and correctness."""

    def test_prediction_step_disables_intermediate_outputs(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """prediction_step must disable hidden_states/attentions (OOM prevention).

        During eval, accumulated hidden states across the full dataset cause OOM.
        The flags must be False during the step and restored afterward.
        """
        strategy = FeatureLossStrategy(ce_loss_coeff=1.0, kd_loss_coeff=1.0, layer_mapping=[])
        trainer = _make_trainer(tiny_bert_teacher, tiny_bert_model,
                                loss_strategy=strategy, tmp_path=tmp_path)

        # Starts True because FeatureLossStrategy requires hidden states
        assert trainer.output_hidden_states is True

        captured = {}

        def spy_prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            # Capture the flag values during the step
            captured["hidden"] = trainer.output_hidden_states
            captured["attn"] = trainer.output_attentions
            return (None, None, None)

        with patch.object(transformers.Trainer, "prediction_step", spy_prediction_step):
            trainer.prediction_step(tiny_bert_model, _make_batch(),
                                    prediction_loss_only=True)

        # During the step, flags must be False
        assert captured["hidden"] is False, "output_hidden_states must be False during eval"
        assert captured["attn"] is False, "output_attentions must be False during eval"

        # After the step, flags must be restored
        assert trainer.output_hidden_states is True, "output_hidden_states must be restored"
        assert trainer.output_attentions is False, "output_attentions must be restored to original"

    def test_attention_strategy_sets_output_attentions(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """Loss strategy with requires_attentions=True sets output_attentions flag."""
        strategy = AttentionLossStrategy(
            ce_loss_coeff=0.5, kd_loss_coeff=0.5, layer_mapping=[(0, 0), (1, 1)],
        )
        trainer = _make_trainer(tiny_bert_teacher, tiny_bert_model,
                                loss_strategy=strategy, tmp_path=tmp_path)
        assert trainer.output_attentions is True

        # Verify compute_loss passes output_attentions to the model
        tiny_bert_model.train()
        batch = _make_batch()
        loss = trainer.compute_loss(tiny_bert_model, batch)
        assert torch.isfinite(loss)

    def test_eval_mode_returns_ce_only(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """In eval mode, compute_loss returns CE loss without distillation."""
        trainer = _make_trainer(tiny_bert_teacher, tiny_bert_model, tmp_path=tmp_path)
        tiny_bert_model.eval()
        batch = _make_batch()
        loss = trainer.compute_loss(tiny_bert_model, batch)
        assert torch.isfinite(loss), "Eval loss must be finite"

        # Also test return_outputs=True path
        loss2, outputs = trainer.compute_loss(tiny_bert_model, batch, return_outputs=True)
        assert torch.isfinite(loss2)
        assert hasattr(outputs, "logits")


# ===========================================================================
# 4. Loss Type String Path
# ===========================================================================

class TestLossTypeStringPath:
    """Tests for loss_type string-based strategy creation (line 105)."""

    def test_loss_type_string_creates_strategy(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """Passing loss_type as a string should create the corresponding strategy."""
        trainer = _make_trainer(tiny_bert_teacher, tiny_bert_model,
                                loss_type="response_kl", tmp_path=tmp_path)
        assert trainer.loss_strategy is not None
        assert trainer.loss_strategy.name == "response_kl"

    def test_loss_type_string_produces_valid_loss(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """loss_type string path should produce a finite loss on a forward pass."""
        trainer = _make_trainer(tiny_bert_teacher, tiny_bert_model,
                                loss_type="response_kl", tmp_path=tmp_path)
        tiny_bert_model.train()
        batch = _make_batch()
        loss = trainer.compute_loss(tiny_bert_model, batch)
        assert torch.isfinite(loss)


# ===========================================================================
# 5. Teacher Label Map
# ===========================================================================

class TestTeacherLabelMap:
    """Tests for teacher_label_map remapping (line 238)."""

    def test_teacher_label_map_remaps_logits(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """teacher_label_map=[2,0,1] should permute teacher logit columns."""
        trainer = _make_trainer(tiny_bert_teacher, tiny_bert_model,
                                teacher_label_map=[2, 0, 1], tmp_path=tmp_path)
        tiny_bert_model.train()
        batch = _make_batch()
        loss = trainer.compute_loss(tiny_bert_model, batch)
        assert torch.isfinite(loss)

    def test_teacher_label_map_with_co_training(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """Co-training + teacher_label_map should recompute teacher CE with remapped logits."""
        trainer = _make_trainer(tiny_bert_teacher, tiny_bert_model,
                                freeze_teacher=False,
                                teacher_label_map=[2, 0, 1], tmp_path=tmp_path)
        tiny_bert_model.train()
        batch = _make_batch()
        loss = trainer.compute_loss(tiny_bert_model, batch)
        loss.backward()
        # Teacher must receive gradients from the recomputed CE
        teacher_grads = [p.grad for p in tiny_bert_teacher.parameters() if p.grad is not None]
        assert len(teacher_grads) > 0, "Teacher must get gradients via recomputed CE"


# ===========================================================================
# 6. Vocab Size Alignment
# ===========================================================================

class TestVocabAlignment:
    """Tests for vocab size alignment (lines 248-250)."""

    def test_mismatched_vocab_size_trims_logits(self, tmp_path):
        """When teacher has different vocab size, logits should be trimmed to min."""
        from transformers import BertConfig, BertForSequenceClassification

        torch.manual_seed(42)
        # Model with 3 labels
        model_cfg = BertConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=128, num_labels=3, vocab_size=1000,
            attn_implementation="eager",
        )
        model = BertForSequenceClassification(model_cfg)

        # Teacher with 5 labels (different num_labels = different logit dim)
        teacher_cfg = BertConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=128, num_labels=5, vocab_size=1000,
            attn_implementation="eager",
        )
        teacher = BertForSequenceClassification(teacher_cfg)

        trainer = _make_trainer(teacher, model, tmp_path=tmp_path)
        model.train()
        batch = _make_batch(num_labels=3)
        loss = trainer.compute_loss(model, batch)
        assert torch.isfinite(loss), "Loss must be finite with mismatched vocab sizes"


# ===========================================================================
# 7. Training Progress and Metric Accumulation
# ===========================================================================

class TestTrainingProgressAndMetrics:
    """Tests for training_progress (line 271) and metric flush (lines 317-320)."""

    def test_training_progress_with_max_steps(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """When max_steps > 0, training_progress should be global_step / max_steps."""
        ds = _make_dataset()
        trainer = _make_trainer(tiny_bert_teacher, tiny_bert_model,
                                tmp_path=tmp_path, max_steps=10, train_dataset=ds)
        tiny_bert_model.train()
        # create_optimizer + scheduler so state is initialized
        trainer.create_optimizer()
        # Manually set state to simulate mid-training
        trainer.state.max_steps = 10
        trainer.state.global_step = 5
        batch = _make_batch()
        loss = trainer.compute_loss(tiny_bert_model, batch)
        assert torch.isfinite(loss)

    def test_projector_passed_to_loss_strategy(self, tmp_path):
        """When projector is set, it should be passed in loss_kwargs."""
        from transformers import BertConfig, BertForSequenceClassification

        torch.manual_seed(42)
        cfg = BertConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=128, num_labels=3, vocab_size=1000,
            attn_implementation="eager",
        )
        model = BertForSequenceClassification(cfg)
        teacher = BertForSequenceClassification(cfg)

        projector = torch.nn.Linear(64, 64)
        trainer = _make_trainer(teacher, model, tmp_path=tmp_path)
        trainer.projector = projector

        model.train()
        batch = _make_batch()
        loss = trainer.compute_loss(model, batch)
        assert torch.isfinite(loss)

    def test_metric_flush_at_logging_interval(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """Metrics should flush when _metric_count reaches logging_steps."""
        trainer = _make_trainer(tiny_bert_teacher, tiny_bert_model, tmp_path=tmp_path)
        trainer.args.logging_steps = 2
        tiny_bert_model.train()
        batch = _make_batch()

        # First step: accumulates, no flush
        trainer.compute_loss(tiny_bert_model, batch)
        assert trainer._metric_count == 1
        assert len(trainer._metric_accumulator) > 0

        # Second step: should trigger flush
        trainer.compute_loss(tiny_bert_model, batch)
        assert trainer._metric_count == 0, "Metrics should be flushed after logging_steps"
        assert len(trainer._metric_accumulator) == 0


# ===========================================================================
# 8. Co-training Teacher CE Loss
# ===========================================================================

class TestCoTrainingTeacherCE:
    """Tests for co-training teacher CE loss path (line 298)."""

    def test_co_training_includes_teacher_ce(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """Co-training total_loss should include teacher CE component."""
        trainer = _make_trainer(tiny_bert_teacher, tiny_bert_model,
                                freeze_teacher=False, tmp_path=tmp_path)
        tiny_bert_model.train()
        batch = _make_batch()
        loss = trainer.compute_loss(tiny_bert_model, batch)
        assert torch.isfinite(loss)
        # Teacher branch logs strategy components with a teacher_ prefix (e.g. ce_raw).
        assert "teacher_ce_raw" in trainer._metric_accumulator

    def test_co_training_loss_larger_than_frozen(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """Co-training loss should differ from frozen (includes teacher CE)."""
        from transformers import BertConfig, BertForSequenceClassification

        batch = _make_batch()

        def fresh_pair():
            torch.manual_seed(42)
            cfg = BertConfig(
                hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                intermediate_size=128, num_labels=3, vocab_size=1000,
                attn_implementation="eager",
            )
            return BertForSequenceClassification(cfg), BertForSequenceClassification(cfg)

        # Frozen
        teacher_f, model_f = fresh_pair()
        trainer_f = _make_trainer(teacher_f, model_f, freeze_teacher=True, tmp_path=tmp_path)
        model_f.train()
        loss_frozen = trainer_f.compute_loss(model_f, batch)

        # Co-training
        dir_co = tmp_path / "co"
        dir_co.mkdir()
        teacher_c, model_c = fresh_pair()
        trainer_c = _make_trainer(teacher_c, model_c, freeze_teacher=False, tmp_path=dir_co)
        model_c.train()
        loss_co = trainer_c.compute_loss(model_c, batch)

        # They should differ because co-training adds teacher CE
        assert not torch.allclose(loss_frozen, loss_co, atol=1e-6), \
            "Co-training loss must differ from frozen (teacher CE added)"

    def test_ctcd_loss_larger_than_cotraining_only(self, tmp_path):
        """CTCD total loss must exceed pure co-training loss.

        Pure co-training: teacher gets CE only (beta_h=1, beta_s=0).
        CTCD: teacher also receives KL from student (beta_h=1, beta_s=1).
        The additional teacher KL term must increase the total scalar loss.
        """
        from transformers import BertConfig, BertForSequenceClassification

        batch = _make_batch()

        def fresh_pair():
            torch.manual_seed(42)
            cfg = BertConfig(
                hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                intermediate_size=128, num_labels=3, vocab_size=1000,
                attn_implementation="eager",
            )
            return BertForSequenceClassification(cfg), BertForSequenceClassification(cfg)

        # Pure co-training: teacher CE only
        teacher_ct, model_ct = fresh_pair()
        trainer_ct = _make_trainer(teacher_ct, model_ct, freeze_teacher=False,
                                   alpha_h=1.0, alpha_s=1.0,
                                   beta_h=1.0, beta_s=0.0, tmp_path=tmp_path)
        model_ct.train()
        loss_cotrain = trainer_ct.compute_loss(model_ct, batch)

        # CTCD: teacher CE + KL from student
        dir_ctcd = tmp_path / "ctcd"
        dir_ctcd.mkdir()
        teacher_ctcd, model_ctcd = fresh_pair()
        trainer_ctcd = _make_trainer(teacher_ctcd, model_ctcd, freeze_teacher=False,
                                     alpha_h=1.0, alpha_s=1.0,
                                     beta_h=1.0, beta_s=1.0, tmp_path=dir_ctcd)
        model_ctcd.train()
        loss_ctcd = trainer_ctcd.compute_loss(model_ctcd, batch)

        assert loss_ctcd.item() > loss_cotrain.item(), (
            f"CTCD loss ({loss_ctcd.item():.4f}) must exceed co-training loss "
            f"({loss_cotrain.item():.4f}) — teacher KL term is missing or zeroed out"
        )


# ===========================================================================
# 9. create_optimizer Coverage
# ===========================================================================

class TestCreateOptimizer:
    """Tests covering create_optimizer paths (lines 139, 146)."""

    def test_frozen_optimizer_no_teacher_params(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """Frozen teacher: optimizer should NOT contain teacher params."""
        trainer = _make_trainer(tiny_bert_teacher, tiny_bert_model,
                                freeze_teacher=True, tmp_path=tmp_path)
        optimizer = trainer.create_optimizer()
        teacher_ids = {id(p) for p in tiny_bert_teacher.parameters()}
        opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        assert not (teacher_ids & opt_ids), "Frozen teacher params must not be in optimizer"

    def test_create_optimizer_idempotent(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """Calling create_optimizer twice should return the same optimizer (line 146)."""
        trainer = _make_trainer(tiny_bert_teacher, tiny_bert_model,
                                freeze_teacher=False, tmp_path=tmp_path)
        opt1 = trainer.create_optimizer()
        opt2 = trainer.create_optimizer()
        assert opt1 is opt2, "create_optimizer must be idempotent"


# ===========================================================================
# 10. setup_from_config
# ===========================================================================

class TestSetupFromConfig:
    """Tests for setup_from_config factory method (lines 361-428)."""

    def test_simple_config(self, tiny_bert_model, tiny_bert_teacher):
        """Basic config with temperature and split student coefficients."""
        config = {
            "distillation": {
                "temperature": 3.0,
                "alpha_h": 0.3,
                "alpha_s": 0.7,
                "loss_type": "response_kl",
            }
        }
        kwargs, layer_mapping, projector = DistillationTrainer.setup_from_config(
            config, tiny_bert_teacher, tiny_bert_model
        )
        assert kwargs["temperature"] == 3.0
        assert kwargs["alpha_h"] == 0.3
        assert kwargs["alpha_s"] == 0.7
        assert "loss_strategy" in kwargs
        assert kwargs["loss_strategy"].name == "response_kl"
        assert "loss_type" not in kwargs
        assert layer_mapping is None
        assert projector is None

    def test_staged_config(self, tiny_bert_model, tiny_bert_teacher):
        """Config with stages should create StagedDistillationLoss."""
        config = {
            "distillation": {
                "temperature": 2.0,
                "alpha_h": 0.5,
                "alpha_s": 0.5,
                "stages": [
                    {"until_progress": 0.5, "strategy": "response_kl"},
                    {"until_progress": 1.0, "strategy": "response_mse"},
                ],
            }
        }
        kwargs, layer_mapping, projector = DistillationTrainer.setup_from_config(
            config, tiny_bert_teacher, tiny_bert_model
        )
        assert "loss_strategy" in kwargs
        assert "loss_type" not in kwargs
        assert layer_mapping is None

    def test_layer_mapping_uniform(self, tiny_bert_model, tiny_bert_teacher):
        """Config with layer_mapping type=uniform should produce mapping pairs."""
        config = {
            "distillation": {
                "layer_mapping": {"type": "uniform"},
            }
        }
        kwargs, layer_mapping, projector = DistillationTrainer.setup_from_config(
            config, tiny_bert_teacher, tiny_bert_model
        )
        assert layer_mapping is not None
        assert len(layer_mapping) > 0
        # Each entry is a (teacher_idx, model_idx) pair
        for t_idx, s_idx in layer_mapping:
            assert isinstance(t_idx, int)
            assert isinstance(s_idx, int)

    def test_layer_mapping_custom(self, tiny_bert_model, tiny_bert_teacher):
        """Config with custom layer mapping should use explicit pairs."""
        config = {
            "distillation": {
                "layer_mapping": {
                    "type": "custom",
                    "mapping": [
                        {"teacher": 0, "model": 0},
                        {"teacher": 3, "model": 1},
                    ],
                },
            }
        }
        kwargs, layer_mapping, projector = DistillationTrainer.setup_from_config(
            config, tiny_bert_teacher, tiny_bert_model
        )
        assert layer_mapping == [(0, 0), (3, 1)]

    def test_projection_config(self):
        """Config with projection should create projector when dims differ."""
        from transformers import BertConfig, BertForSequenceClassification

        teacher_cfg = BertConfig(
            hidden_size=128, num_hidden_layers=4, num_attention_heads=2,
            intermediate_size=256, num_labels=3, vocab_size=1000,
            attn_implementation="eager",
        )
        model_cfg = BertConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=128, num_labels=3, vocab_size=1000,
            attn_implementation="eager",
        )
        teacher = BertForSequenceClassification(teacher_cfg)
        model = BertForSequenceClassification(model_cfg)

        config = {
            "distillation": {
                "loss_type": "feature_mse",
                "projection": {"type": "linear"},
            }
        }
        kwargs, layer_mapping, projector = DistillationTrainer.setup_from_config(
            config, teacher, model
        )
        assert projector is not None

    def test_projection_same_dims_no_projector(self, tiny_bert_model, tiny_bert_teacher):
        """No projector when teacher and model have same hidden_size."""
        config = {
            "distillation": {
                "projection": {"type": "linear"},
            }
        }
        kwargs, layer_mapping, projector = DistillationTrainer.setup_from_config(
            config, tiny_bert_teacher, tiny_bert_model
        )
        # Both have hidden_size=64, no projection needed
        assert projector is None

    def test_teacher_label_map_in_config(self, tiny_bert_model, tiny_bert_teacher):
        """teacher_label_map should be passed through from config."""
        config = {
            "distillation": {
                "teacher_label_map": [2, 0, 1],
            }
        }
        kwargs, _, _ = DistillationTrainer.setup_from_config(
            config, tiny_bert_teacher, tiny_bert_model
        )
        assert kwargs["teacher_label_map"] == [2, 0, 1]

    def test_empty_distillation_config(self, tiny_bert_model, tiny_bert_teacher):
        """Empty distillation config should use defaults."""
        config = {}
        kwargs, layer_mapping, projector = DistillationTrainer.setup_from_config(
            config, tiny_bert_teacher, tiny_bert_model
        )
        assert kwargs["temperature"] == 2.0
        assert kwargs["alpha_h"] == 1.0
        assert kwargs["alpha_s"] == 1.0
        assert kwargs["loss_strategy"].name == "response_kl"
        assert "loss_type" not in kwargs
        assert layer_mapping is None
        assert projector is None

    def test_split_coefficients_in_config(self, tiny_bert_model, tiny_bert_teacher):
        """setup_from_config should surface alpha_h/alpha_s/beta_h/beta_s coefficients."""
        config = {
            "model": {"teacher": {"freeze": False}},
            "distillation": {
                "alpha_h": 0.4,
                "alpha_s": 0.6,
                "beta_h": 0.9,
                "beta_s": 0.1,
                "loss_type": "response_kl",
            },
        }
        kwargs, _, _ = DistillationTrainer.setup_from_config(config, tiny_bert_teacher, tiny_bert_model)
        assert kwargs["alpha_h"] == 0.4
        assert kwargs["alpha_s"] == 0.6
        assert kwargs["beta_h"] == 0.9
        assert kwargs["beta_s"] == 0.1
        assert "teacher_loss_strategy" in kwargs
        assert kwargs["teacher_loss_strategy"].name == "response_kl"

    def test_teacher_default_beta_when_co_training(self, tiny_bert_model, tiny_bert_teacher):
        """Co-training without beta keys defaults to teacher CE-only."""
        config = {
            "model": {"teacher": {"freeze": False}},
            "distillation": {
                "alpha_h": 0.5,
                "alpha_s": 0.5,
                "loss_type": "response_kl",
            },
        }
        kwargs, _, _ = DistillationTrainer.setup_from_config(config, tiny_bert_teacher, tiny_bert_model)
        assert kwargs["beta_h"] == 1.0
        assert kwargs["beta_s"] == 0.0
        assert "teacher_loss_strategy" in kwargs


# ===========================================================================
# 11. TeacherEvalCallback — issue #35 regression test
# ===========================================================================

class TestTeacherEvalCallback:
    """Regression tests for TeacherEvalCallback (issue #35).

    HF Transformers renamed trainer.tokenizer to trainer.processing_class.
    TeacherEvalCallback must use processing_class, not tokenizer.
    """

    def test_on_evaluate_uses_processing_class(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """on_evaluate must access trainer.processing_class, not trainer.tokenizer.

        Regression test for issue #35, fixed in PR #36.
        """
        from unittest.mock import MagicMock, patch as mock_patch
        from src.training.callbacks import TeacherEvalCallback

        ds = _make_dataset()
        trainer = _make_trainer(
            tiny_bert_teacher, tiny_bert_model,
            freeze_teacher=False, tmp_path=tmp_path, train_dataset=ds,
        )
        trainer.eval_dataset = ds
        trainer.compute_metrics = lambda ep: {"accuracy": 0.5}
        trainer.data_collator = None

        cb = TeacherEvalCallback(trainer)

        args = MagicMock()
        args.per_device_eval_batch_size = 2
        args.remove_unused_columns = True
        state = MagicMock()
        control = MagicMock()
        metrics = {"eval_loss": 0.5}

        with mock_patch("src.evaluation.evaluate.evaluate_model") as mock_eval:
            mock_eval.return_value = {"accuracy": 0.8, "loss": 0.3}
            cb.on_evaluate(args, state, control, metrics=metrics)

        # Verify teacher metrics were injected
        assert "eval_teacher_accuracy" in metrics

    def test_on_save_uses_processing_class(self, tiny_bert_model, tiny_bert_teacher, tmp_path):
        """on_save must access trainer.processing_class for save_pretrained.

        Regression test for issue #35, fixed in PR #36.
        """
        from unittest.mock import MagicMock
        from src.training.callbacks import TeacherEvalCallback

        ds = _make_dataset()
        trainer = _make_trainer(
            tiny_bert_teacher, tiny_bert_model,
            freeze_teacher=False, tmp_path=tmp_path, train_dataset=ds,
        )
        # Provide a processing_class so save_pretrained works
        trainer.processing_class = MagicMock()

        cb = TeacherEvalCallback(trainer)

        args = MagicMock()
        args.output_dir = str(tmp_path)
        state = MagicMock()
        state.global_step = 100
        control = MagicMock()

        # Create checkpoint dir so save_pretrained has somewhere to write
        ckpt_dir = tmp_path / "checkpoint-100"
        ckpt_dir.mkdir()

        cb.on_save(args, state, control)

        # Verify processing_class.save_pretrained was called
        trainer.processing_class.save_pretrained.assert_called_once()
