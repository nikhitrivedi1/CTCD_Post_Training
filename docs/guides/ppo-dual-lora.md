# PPO with Dual LoRA Adapters

Design doc for adding PPO to Forge using LoRA adapter swapping instead of duplicating the full model.

---

## Background

Forge supports GRPO for RL training. GRPO computes advantages by comparing N generations of the same prompt against their group mean - no critic needed. PPO uses a learned critic (value network) to estimate expected reward, computing advantages as `actual - expected`. More sample-efficient (1 generation per prompt instead of N) but requires training a second network.

### GRPO vs PPO

| | GRPO | PPO |
|---|---|---|
| Advantage baseline | Group mean of N generations | Learned critic prediction |
| Generations per prompt | N (e.g. 4) | 1 |
| Extra memory | None | Critic network + value head |
| Extra hyperparams | None | clip_range, vf_coef, gae_lambda, critic_lr |
| Training stability | Simple, fewer knobs | Harder to tune, critic must track policy |

### The LoRA Insight

With LoRA, the base model is frozen. Two small adapters (policy + critic) can share the same backbone, with a value head (single linear layer) on top of the critic. Adapter swapping via PEFT's `set_adapter()` avoids PPO's usual 2x model memory cost. The overhead is ~100MB (two adapter parameter sets + value head + two optimizer states) instead of ~4GB (duplicate base model).

---

## Architecture

### Mode A: Shared Backbone (1 model, 2 adapters)

```
              Frozen Base Model (Qwen3-VL-2B)
                        |
          +-------------+-------------+
          |                           |
   "policy" LoRA              "critic" LoRA
   (generates, policy loss)    (value estimation)
                                      |
                                 ValueHead
                              nn.Linear(H, 1)

   Reference = base with disable_adapter()
```

- PEFT `set_adapter("policy")` / `set_adapter("critic")` swaps active adapter
- `disable_adapter()` context gives the reference model for KL penalty
- ~100MB overhead vs GRPO
- Both adapters share the same frozen backbone in memory

### Mode B: Separate Backbones (2 models, 2 adapters)

```
   Policy Model (e.g. 2B)          Critic Model (e.g. 0.5B)
         |                                |
   "policy" LoRA                   "critic" LoRA
   (generates, policy loss)               |
                                     ValueHead
                                   nn.Linear(H, 1)

   Reference = policy model with disable_adapter()
```

- No adapter swapping needed - each model owns its own adapter
- Critic can be a smaller, cheaper model (e.g. 0.5B critic for 2B policy)
- More memory but simpler forward pass routing
- Policy and critic have independent architectures, hidden sizes, tokenizers
- Useful when the critic doesn't need full model capacity (value estimation is simpler than generation)

### When to Use Each

- **Mode A** when memory is tight or models are the same architecture. Saves ~model_size of memory.
- **Mode B** when you want a lightweight critic, or when policy and critic benefit from different architectures (e.g. policy is a VLM, critic is text-only since value estimation doesn't need vision).

---

## Config

```yaml
rl:
  algorithm: ppo
  reward: [edit_similarity, inline_formatting]
  reward_weights: [1.0, 3.0]
  max_completion_length: 1024
  temperature: 0.8
  log_completions: true

  ppo:
    # Core PPO hyperparams
    clip_range: 0.2           # Clipping epsilon for policy ratio
    vf_coef: 0.5              # Value loss coefficient
    entropy_coef: 0.01        # Entropy bonus coefficient
    kl_coef: 0.05             # KL penalty against reference model
    gamma: 1.0                # Discount factor (1.0 for single-turn tasks)
    gae_lambda: 0.95          # GAE lambda for advantage smoothing
    ppo_epochs: 4             # Inner optimization epochs per batch
    generation_batch_size: 4  # Prompts per rollout batch
    whiten_advantages: true   # Normalize advantages before policy update
    max_grad_norm: 1.0        # Gradient clipping

    # Critic config
    critic_lr: 1e-5           # Separate learning rate for critic
    critic_model: null        # null = shared backbone (Mode A)
    # critic_model: Qwen/Qwen3-VL-0.5B  # set = separate backbone (Mode B)
    critic_lora:              # Override LoRA config for critic (optional)
      r: 8
      alpha: 8
```

`ppo.critic_model: null` triggers Mode A (shared backbone). Setting it to a model name triggers Mode B.

---

## Training Loop

```python
for batch in dataloader:
    # ==== ROLLOUT PHASE (no gradients) ====
    with torch.no_grad():
        # 1. Generate completions with policy adapter
        activate_policy()
        completions = model.generate(batch["prompts"])

        # 2. Score with reward functions (same callable API as GRPO)
        rewards = compute_rewards(completions, batch)

        # 3. Compute log-probs under current policy
        old_logprobs = compute_logprobs(policy_model, completions)

        # 4. Compute log-probs under reference (base model, no adapters)
        with disable_adapters():
            ref_logprobs = compute_logprobs(policy_model, completions)

        # 5. Compute value estimates with critic
        activate_critic()
        hidden = critic_model(completions, output_hidden_states=True)
        values = value_head(hidden.hidden_states[-1])

        # 6. Compute KL-penalized rewards and GAE advantages
        kl = old_logprobs - ref_logprobs
        token_rewards = -kl_coef * kl
        token_rewards[:, -1] += rewards      # scalar reward at sequence end
        advantages, returns = gae(token_rewards, values, gamma, gae_lambda)

    # ==== PPO UPDATE PHASE (multiple inner epochs) ====
    for epoch in range(ppo_epochs):
        # --- Policy update ---
        activate_policy()
        new_logprobs = compute_logprobs(policy_model, completions)
        ratio = (new_logprobs - old_logprobs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - clip_range, 1 + clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        policy_loss -= entropy_coef * entropy.mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()       # gradients flow only to policy adapter
        clip_grad_norm_(policy_params, max_grad_norm)
        policy_optimizer.step()

        # --- Critic update ---
        activate_critic()
        hidden = critic_model(completions, output_hidden_states=True)
        new_values = value_head(hidden.hidden_states[-1])
        critic_loss = F.mse_loss(new_values, returns)

        critic_optimizer.zero_grad()
        critic_loss.backward()       # gradients flow only to critic adapter + value head
        clip_grad_norm_(critic_params, max_grad_norm)
        critic_optimizer.step()
```

### Gradient Isolation

- **Mode A**: `set_adapter("policy")` sets critic LoRA params to `requires_grad=False` and vice versa. Each backward pass only accumulates gradients for the active adapter. PEFT handles this automatically.
- **Mode B**: Two separate models with separate optimizers. No isolation needed.
- **Value head**: Always in the critic optimizer group. Receives gradients during critic backward pass only.

### Key Difference from GRPO

The rollout phase generates only 1 completion per prompt (not N). The advantage comes from the critic's value estimate instead of group statistics. This means:
- Less inference compute per step (1 generation vs 4)
- More training compute per step (critic forward + backward)
- Net effect depends on generation length vs model size

---

## Implementation

### Why Custom Trainer

TRL 0.29.0 has an experimental `PPOTrainer` (`trl.experimental.ppo`), but it doesn't fit:
- Expects `reward_model: nn.Module`, not callable reward functions
- `PolicyAndValueWrapper` couples policy+value forward passes, defeating adapter swapping
- Manages its own Accelerator/DataLoader/optimizer, conflicting with Forge's callbacks
- Experimental, unstable API

The core PPO math (GAE, clipped surrogate, value loss) is ~50 lines. The rest reuses existing Forge infrastructure.

### New Files

1. **`src/training/ppo_trainer.py`** (~400 lines)
   - `ForgePPOTrainer` class
   - Manages dual adapters (Mode A) or dual models (Mode B)
   - Training loop: rollout + PPO update
   - Checkpoint save/load for both adapters + value head
   - Callback compatibility (fires on_step_end, on_save, on_log)

2. **`src/training/value_head.py`** (~30 lines)
   - `ValueHead(nn.Module)`: dropout + `nn.Linear(hidden_size, 1)`

### Modified Files

3. **`forge.py`**
   - Add `_train_ppo()` function (mirrors `_train_rl()` structure)
   - Add `elif rl_algorithm == "ppo":` routing
   - For Mode B: load critic model separately

4. **`src/training/callbacks.py`**
   - StepMetricsCallback handles PPO metric prefixes

### Reused As-Is

- `src/training/rewards.py` - `get_reward_funcs()`, `ParallelRewardGroup`
- `src/models/loader.py` - `_load_vlm()`, `_load_causal_lm()`
- `src/data/formatters.py` - `format_for_rl()`
- `src/training/callbacks.py` - `SLURMSignalCallback`, `CheckpointMarkerCallback`

---

## Checkpoint Structure

```
checkpoint-{step}/
    policy_adapter/
        adapter_config.json
        adapter_model.safetensors
    critic_adapter/
        adapter_config.json
        adapter_model.safetensors
    value_head.pt
    policy_optimizer.pt
    critic_optimizer.pt
    training_state.json
```

Final save (training complete): merge policy adapter into base, save full model. Critic is discarded (it's only useful during training).

---

## Memory Estimates (Qwen3-VL-2B, bf16, LoRA r=16)

### Mode A (Shared Backbone)

| Component | Memory |
|---|---|
| Base model | ~4 GB |
| Policy LoRA adapter | ~20 MB |
| Critic LoRA adapter | ~20 MB |
| Value head | ~8 KB |
| Two optimizer states | ~80 MB |
| KV cache (generation) | ~2 GB |
| Activations (forward + backward) | ~4-8 GB |
| **Total** | **~12 GB** |

### Mode B (2B policy + 0.5B critic)

| Component | Memory |
|---|---|
| Policy base model | ~4 GB |
| Critic base model | ~1 GB |
| Policy LoRA adapter | ~20 MB |
| Critic LoRA adapter | ~5 MB |
| Value head | ~2 KB |
| Two optimizer states | ~50 MB |
| KV cache (generation) | ~2 GB |
| Activations | ~6-10 GB |
| **Total** | **~15 GB** |

Both fit comfortably on H200 (140 GB) and A100 (40 GB).

---

## Verification Plan

1. **Unit test**: Tiny model (GPT2-small), attach dual LoRA, verify adapter swapping isolates gradients correctly
2. **Smoke test**: PPO config with `--max_steps 10`, verify losses decrease
3. **Cluster test**: Full PPO run on formatting task, compare reward trajectory vs GRPO baseline
4. **Resume test**: Kill job mid-training, verify both adapters + value head load correctly from checkpoint
