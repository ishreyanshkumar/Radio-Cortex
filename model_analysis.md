# Complete Model Analysis

This document provides a comprehensive analysis of the reinforcement learning models found in the `models/` directory. The metadata (parameter counts, training steps, layer architectures) is extracted **directly from the `.pt` checkpoint files** rather than relying on prior documentation.

## 📊 Summary Table

| Model | File Size (MB) | Total Params | Timesteps Trained | Hidden Dim | Architecture |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BDH** | 72.94 | 6,418,451 | 983,040 | 512 | Scale-Free Transformer (BDH) |
| **GPT2** | 36.96 | 3,234,323 | 1,000,000 | 256 | GPT-2 Style Transformer |
| **LINEAR** | 36.89 | 3,217,939 | 1,000,000 | 256 | Linear Attention Transformer |
| **REFORMER** | 37.05 | 3,232,019 | 1,000,000 | 256 | Reformer Transformer |
| **TRXL** | 36.62 | 3,195,155 | 1,000,000 | 256 | Transformer-XL |
| **UNIVERSAL** | 9.73 | 851,475 | 1,000,000 | 256 | Universal Transformer |
| **NN** | 1.62 | 140,563 | 1,000,000 | 256 | 2-layer MLP Baseline |



---

## 🔬 Per-Model Detailed Metadata

### 1. BDH (Scale-Free Transformer)
- **File Size:** 72.94 MB
- **Total Parameters:** 6,418,451
- **Timesteps Trained:** 983,040
- **Hyperparameters:**
  - `hidden_dim`: 512
  - `lr`: 3e-05 (stable learning rate config)
  - `batch_size`: 512, `rollout_steps`: 512, `n_envs`: 48
  - `clip_epsilon`: 0.1, `vf_coef`: 1.0
  - Entropy annealing: `ent_coef_start`: 0.03 → `ent_coef_end`: 0.005 (`ent_decay_fraction`: 0.8)
- **Key Architecture Characteristics:**
  - Uses an independent `logstd_head`: [1, 9] mapped to 9 action spaces.
  - Scale-free slot-based memory mapping inside the `encoder`: [4, 128, 4096] (4 heads, 128 dim, 4096 keys) and `decoder`: [16384, 128].

### 2. GPT2
- **File Size:** 36.96 MB
- **Total Parameters:** 3,234,323
- **Timesteps Trained:** 1,000,000 
- **Hyperparameters:** `hidden_dim`: 256, `lr`: 0.0005, `batch_size`: 512, `rollout_steps`: 512, `n_envs`: 48
- **Key Architecture Characteristics:**
  - `pos_embed`: [1, 64, 256] contextual sequence embedding for sequential state representations.
  - State projection block utilizes an initialized size of `state_embed.weight` [256, 144].

### 3. LINEAR (Linear Attention Transformer)
- **File Size:** 36.89 MB
- **Total Parameters:** 3,217,939
- **Timesteps Trained:** 1,000,000 
- **Hyperparameters:** `hidden_dim`: 256, `lr`: 0.0005, `batch_size`: 512, `rollout_steps`: 512, `n_envs`: 48
- **Key Architecture Characteristics:**
  - Standard autoregressive linear attention mechanisms. Identical input mapping block to the GPT2 backbone via `pos_embed`: [1, 64, 256] and `state_embed.weight`: [256, 144]. 

### 4. REFORMER
- **File Size:** 37.05 MB
- **Total Parameters:** 3,232,019
- **Timesteps Trained:** 1,000,000 
- **Hyperparameters:** `hidden_dim`: 256, `lr`: 0.0005, `batch_size`: 512, `rollout_steps`: 512, `n_envs`: 48
- **Key Architecture Characteristics:**
  - Designed for much longer context sequence history. 
  - Notable distinction in position embedding sequence length - `pos_embed`: [1, 128, 256] (128 context tokens compared to 64 for GPT2/Linear)
  - Also utilizes independent `actor_logstd`: [1, 9] for log-based standard deviation bounding.

### 5. TRXL (Transformer-XL)
- **File Size:** 36.62 MB
- **Total Parameters:** 3,195,155
- **Timesteps Trained:** 1,000,000 
- **Hyperparameters:** `hidden_dim`: 256, `lr`: 0.0005, `batch_size`: 512, `rollout_steps`: 512, `n_envs`: 48
- **Key Architecture Characteristics:**
  - Standard TrXL blocks without hardcoded positional embeddings natively printed, but features the standard projection mappings such as `state_embed.weight`: [256, 144] and independent `actor_logstd`: [1, 9]. 

### 6. UNIVERSAL (Universal Transformer)
- **File Size:** 9.73 MB
- **Total Parameters:** 851,475
- **Timesteps Trained:** 1,000,000 
- **Hyperparameters:** `hidden_dim`: 256, `lr`: 0.0005, `batch_size`: 512, `rollout_steps`: 512, `n_envs`: 48
- **Key Architecture Characteristics:**
  - Uniquely features a fraction of the parameter count relative to other Transformer variants due to **weight-tied blocks** across depth.
  - Implements recurrent depth embeddings (`step_embed.weight`: [4, 256]) for depth conditioning. Uses standard `pos_embed`: [1, 64, 256].

### 7. NN (MLP Baseline)
- **File Size:** 1.62 MB
- **Total Parameters:** 140,563
- **Timesteps Trained:** 1,000,000
- **Hyperparameters:** `hidden_dim`: 256, `lr`: 0.0005, `batch_size`: 512, `rollout_steps`: 512, `n_envs`: 48
- **Key Architecture Characteristics:**
  - The lightest model mapping observations using simple dense mappings (`feature_net.0.weight`: [256, 144] and `feature_net.2.weight`: [256, 256]) then linearly mapped independently to an `actor_mean` [9, 256] vector.
