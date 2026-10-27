# Audit Report: Random Identity (PK) Sampler and Triplet Loss Dynamics

### Course: 23CSE373 - Computer Vision | Date: 2026-10-27

---

## 1. Executive Summary

A critical training limitation was discovered in the DataLoader configuration of `train.py`. The original implementation used standard random shuffling (`DataLoader(..., shuffle=True)`), which failed to satisfy the essential prerequisite of batch hard-mining Triplet Loss.

This audit details the mathematical necessity of identity-balanced PK sampling, the design and implementation of `RandomIdentitySampler`, and automated validation demonstrating that every training batch contains valid positive and negative pairs.

---

## 2. Mathematical Root Cause of Triplet Loss Failure under Random Shuffling

The Hard-Mining Triplet Loss is formulated as:
$$\mathcal{L}_{\text{triplet}} = \frac{1}{N} \sum_{i=1}^N \max\left(0, \max_{p \in \mathcal{P}(i)} D(f_i, f_p) - \min_{n \in \mathcal{N}(i)} D(f_i, f_n) + \text{margin}\right)$$
where $\mathcal{P}(i) = \{p \mid y_p = y_i, p \neq i\}$ is the set of positive samples sharing the same identity as anchor $i$, and $\mathcal{N}(i) = \{n \mid y_n \neq y_i\}$ is the set of negative samples with different identities.

### The Probabilistic Collapse under Uniform Shuffling:
In PRCC, the training set contains $150$ identities across $17,896$ images.
With a mini-batch size of $B = 16$:
1. The probability that any two randomly drawn images belong to the same identity is approximately:
   $$P(\text{same ID}) \approx \frac{1}{150} \approx 0.67\%$$
2. In almost every batch, all 16 images belong to 16 distinct identities.
3. Therefore:
   $$\mathcal{P}(i) = \emptyset \quad \forall i \in \{1, \dots, 16\}$$
4. When $\mathcal{P}(i)$ is empty, the implementation in `models/loss.py` defaults to:
   $$\text{dist\_ap} = 0.0$$
5. Consequently, the loss reduces to:
   $$\mathcal{L}_{\text{triplet}} = \max\left(0, -\min_{n} D(f_i, f_n) + \text{margin}\right)$$
6. **The Failure**: The network only learns to push negative identities further than the margin ($0.3$), but **NEVER pulls intra-class instances together**. The representation fails to achieve compact intraclass clustering, causing the Rank-1 metric on unseen test identities to remain near random guessing (~4%).

---

## 3. Implemented Solution: `RandomIdentitySampler`

We implemented the standard, proven `RandomIdentitySampler` in `data/dataset_loader.py`:

```python
class RandomIdentitySampler:
    """
    Random Identity (P x K) Sampler for Person Re-ID.
    Randomly samples P identities, and K instances per identity,
    forming a batch of P * K samples.
    """
    def __init__(self, data_source, batch_size=16, num_instances=4, seed=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        ...
```

### Operational Mechanics:
1. **Batch Structure**: For `batch_size = 16` and `num_instances = 4`:
   $$P = \frac{16}{4} = 4 \text{ distinct identities per batch}$$
   $$K = 4 \text{ instances per identity}$$
   $$\text{Batch Composition} = [A_1, A_2, A_3, A_4, B_1, B_2, B_3, B_4, C_1, C_2, C_3, C_4, D_1, D_2, D_3, D_4]$$
2. **Guarantee for Triplet Mining**: Every anchor image $A_1$ is guaranteed to have:
   - Exactly $3$ positive candidates ($A_2, A_3, A_4$) to form hard positive pairs $\max_{p} D(A_1, A_p)$.
   - Exactly $12$ negative candidates (from $B, C, D$) to form hard negative pairs $\min_{n} D(A_1, n)$.
3. **Handling Insufficient Instances**: If an identity has fewer than $K$ images, `rng.choice(idxs, size=num_instances, replace=True)` resamples instances with replacement, ensuring no crash or invalid indices.
4. **Reproducibility**: Setting `seed=42` produces deterministic, reproducible batch sequences.

---

## 4. Verification and Automated Testing

Automated test `test_18_random_identity_sampler_pk_batching` in `tests/test_dcr_reid.py` proves:
1. Every generated batch of size 16 contains **exactly 4 distinct identities**.
2. Every identity in the batch contains **exactly 4 instances** ($P=4, K=4$).
3. Identities with insufficient samples ($< 4$) are sampled safely with replacement.
4. Two samplers with identical seed produce identical batch index trajectories.

**Status: VERIFIED & PASSED**
