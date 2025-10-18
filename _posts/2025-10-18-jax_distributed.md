---
layout: post
title: Distributed Training with JAX Simplified
date: 2025-10-18 21:00:00
description: Training a large model like GPT-3 with JAX.
tags: jax distributed-training deep-learning
categories: ai
thumbnail: assets/img/jax.png
---

# Distributed Training with JAX Simplified

Training large language models like GPT-3 (175B parameters) requires distributing computation across dozens or hundreds of GPUs. JAX makes this remarkably elegant through its functional programming paradigm and sharding primitives. However, the mental model required differs significantly from PyTorch's imperative style. This post demystifies JAX's distributed training by addressing the key conceptual hurdles that arise when learning the framework.

## Why JAX for Distributed Training?

JAX excels at distributed training for three fundamental reasons:

**1. Functional paradigm**: Parameters are data structures, not hidden object state. This makes sharding trivial—just split the data structure across devices.

**2. Explicit state management**: No global random state or hidden device placement. Everything is passed explicitly.

**3. Automatic communication**: Given sharding specifications, JAX's compiler (XLA) figures out optimal communication patterns.

For comparison:

**PyTorch (DDP):**
```python
# ~50+ lines of boilerplate
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
model = DDP(model, device_ids=[rank])
sampler = DistributedSampler(dataset, rank=rank)
# ... manual device management, rank checks, cleanup
```

**JAX:**
```python
@jax.pmap
def train_step(params, batch):
    return compute_grads(params, batch)
```

## The Functional Foundation

JAX's functional approach is the first conceptual hurdle. Unlike PyTorch where parameters live inside model objects, JAX separates computation from state.

**PyTorch:**
```python
model = Model()  # Parameters hidden inside
loss = model(x)  # Uses internal state
loss.backward()  # Modifies internal .grad
```

**JAX:**
```python
model = Model()  # Just defines computation
params = model.init(key, x)  # Parameters are separate data
logits = model.apply(params, x)  # Explicit parameter passing
grads = grad(loss_fn)(params, x)  # Explicit differentiation
```

Why does this matter? Because parameters being "just data" means you can trivially split them:

```python
sharded_params = jax.device_put(params, sharding_spec)
```

No DDP wrappers, no process groups, no manual device management.

## Understanding Device Mesh: The Core Abstraction

The device mesh is JAX's fundamental abstraction for organizing GPUs. Understanding this thoroughly is critical—most confusion in JAX distributed training stems from misunderstanding the mesh.

### Physical Layout vs. Logical Organization

A device mesh is a multi-dimensional array of devices with **named axes**:

```python
# Physical: 8x8 grid = 64 GPUs
devices = mesh_utils.create_device_mesh((8, 8))

# Logical: Give axes semantic names
mesh = Mesh(devices, axis_names=('data', 'model'))
```

Key insight: **axis names define how work is distributed**, not the physical layout.

```
        model axis (8 devices) →
data  ┌────┬────┬────┬────┬────┬────┬────┬────┐
axis  │ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │
(8)   ├────┼────┼────┼────┼────┼────┼────┼────┤
↓     │ 8  │ 9  │ 10 │ 11 │ 12 │ 13 │ 14 │ 15 │
      ├────┼────┼────┼────┼────┼────┼────┼────┤
      │ ... (64 GPUs total)                    │
      └────┴────┴────┴────┴────┴────┴────┴────┘
```

Semantics:
- **Same row**: Process different batch slices with same model piece
- **Same column**: Process same batch slice with different model pieces

This enables hybrid parallelism: 8-way data parallelism × 8-way model parallelism = 64-way total parallelism.

## PartitionSpec: Mapping Tensors to Mesh

`PartitionSpec` specifies tensor distribution across the mesh. The critical insight: **PartitionSpec dimensions match tensor dimensions, not mesh dimensions**.

```python
Tensor shape:      (batch=64, seq=2048, embed=12288)
PartitionSpec:     ('data',    None,    'model')
                     ↑          ↑         ↑
                     │          │         └─ Tensor dim 2: use 'model' axis
                     │          └─────────── Tensor dim 1: replicate
                     └────────────────────── Tensor dim 0: use 'data' axis
```

The mesh has 2 axes, but the tensor has 3 dimensions. **PartitionSpec provides 3 entries, each referencing a mesh axis name or None.**

### Example: Sharding a 3D Tensor

```python
input_batch = jnp.ones((64, 2048, 12288))
spec = PartitionSpec('data', None, 'model')

# What happens:
# - Dim 0 (batch=64): Split 8 ways along 'data' axis → 8 per device
# - Dim 1 (seq=2048): Replicate (all devices get full sequence)
# - Dim 2 (embed=12288): Split 8 ways along 'model' axis → 1536 per device

# Result per GPU: (8, 2048, 1536)
```

## Memory Layout: The Hidden Complexity

Understanding memory layout is crucial for two reasons: correctness and performance. This is where reshape vs. transpose becomes important.

### Why Reshape Then Transpose in Attention?

In multi-head attention, we perform:

```python
q = nn.Dense(num_heads * head_dim)(x)  # Shape: (batch, seq, 512)

q = q.reshape(batch, seq, num_heads, head_dim)    # Step 1
q = jnp.transpose(q, (0, 2, 1, 3))                # Step 2
```

Why not directly reshape to `(batch, num_heads, seq, head_dim)`?

**Answer: Memory layout**. After the Dense layer, the 512 dimensions are laid out in memory as:

```
[head0_dim0, head0_dim1, ..., head0_dim63,    # First 64: head 0
 head1_dim0, head1_dim1, ..., head1_dim63,    # Next 64: head 1
 ...
 head7_dim0, head7_dim1, ..., head7_dim63]    # Last 64: head 7
```

**Reshape** changes how we interpret the data without moving it. Reshaping to `(batch, seq, 8, 64)` naturally groups the 512 dimensions into 8 groups of 64, which matches the memory layout.

**Transpose** actually reorders data in memory. We need it to put `num_heads` before `seq_len` for efficient attention computation.

Attempting to directly reshape to `(batch, num_heads, seq, head_dim)` would create a view where the data interpretation doesn't match the underlying memory layout, resulting in incorrect groupings.

**Key principle**: Reshape operations must respect the underlying memory layout. You can only reshape in ways that maintain the contiguity of data in memory.

## Critical Mistake: Wrong Sharding for Weights

The most common error is applying data parallelism to model weights:

```python
# WRONG
weight = jnp.ones((12288, 49152))
spec = PartitionSpec('data', None)  # Split first dim on data axis
```

Let's trace what happens in memory:

```
Weight split along 'data' axis (8 ways):

Data position 0 (GPUs 0-7):   Rows 0-1535
Data position 1 (GPUs 8-15):  Rows 1536-3071
Data position 2 (GPUs 16-23): Rows 3072-4607
...

During training:
- Batch slice 0 → Data position 0 → Uses weight rows 0-1535
- Batch slice 1 → Data position 1 → Uses weight rows 1536-3071

Each data replica has DIFFERENT weights = different models!
Training is broken.
```

**Correct approach**:

```python
spec = PartitionSpec(None, 'model')  # Replicate rows, split columns

# All data replicas get all 12288 rows (same model)
# Columns split 8 ways: each device gets 6144 columns
```

## When Does Batch Splitting Actually Happen?

This reveals a critical insight: **sharding happens at device_put time, not during computation**.

```python
# Original batch in CPU/main memory
batch = jnp.ones((64, 2048, 12288))

# Apply sharding - data is NOW physically distributed
input_spec = PartitionSpec('data', None, None)
sharded_batch = jax.device_put(batch, NamedSharding(mesh, input_spec))

# At this moment, batch is split across devices:
# Data position 0 → batch[0:8] on GPUs 0-7
# Data position 1 → batch[8:16] on GPUs 8-15
# ...
```

The split happens before the forward pass. Each GPU already has its slice when computation begins. This is fundamentally different from PyTorch's DistributedSampler which creates different batches per process.

## Redundant Computation: A Subtle Pitfall

Consider this seemingly reasonable sharding:

```python
input_spec = PartitionSpec('model', None, None)   # Batch on model axis
weight_spec = PartitionSpec(None, 'model')        # Weights on model axis
```

This is mathematically correct but computationally wasteful:

```
GPU 0:  Batch 0-7  × Weight cols 0-6143    → Result₀
GPU 8:  Batch 0-7  × Weight cols 0-6143    → Result₀ (IDENTICAL!)
GPU 16: Batch 0-7  × Weight cols 0-6143    → Result₀ (IDENTICAL!)
...
```

**Why?** Both batch and weights are split along the model axis. GPUs in the same column (same model axis position) receive:
- Same batch slice (model position 0 → batch 0-7)
- Same weight slice (model position 0 → cols 0-6143)
- Therefore: Identical computation

All GPUs in each column duplicate work. Only 12.5% of compute power is utilized (8 unique computations across 64 GPUs).

**Solution**: Orthogonal splits:

```python
input_spec = PartitionSpec('data', None, None)    # Different batches per row
weight_spec = PartitionSpec(None, 'model')        # Different weights per column
```

Now every GPU does unique work: 8 data replicas × 8 model pieces = true 64-way parallelism.

## Complete Training Loop

Putting it together:

```python
# 1. Setup mesh
devices = mesh_utils.create_device_mesh((8, 8))
mesh = Mesh(devices, axis_names=('data', 'model'))

# 2. Initialize and shard parameters
params = model.init(key, dummy_input)

def shard_param(path, param):
    name = '/'.join(path)
    
    # Large embeddings: split vocab
    if 'embedding' in name and param.shape[0] > 10000:
        return jax.device_put(param, NamedSharding(mesh, PartitionSpec('model', None)))
    
    # Large matrices: split hidden dimension
    if 'kernel' in name and param.shape[1] > 1000:
        return jax.device_put(param, NamedSharding(mesh, PartitionSpec(None, 'model')))
    
    # Small params: replicate
    return jax.device_put(param, NamedSharding(mesh, PartitionSpec()))

sharded_params = jax.tree_util.tree_map_with_path(shard_param, params)

# 3. Training step
@jax.jit
def train_step(params, batch, opt_state):
    def loss_fn(params):
        logits = model.apply(params, batch['input_ids'])
        return cross_entropy(logits, batch['labels'])
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# 4. Main loop
for batch in dataloader:
    # Shard input
    batch = jax.device_put(batch, NamedSharding(mesh, PartitionSpec('data', None)))
    
    # Train (all communication automatic)
    sharded_params, opt_state, loss = train_step(sharded_params, batch, opt_state)
```

## Memory Calculation for GPT-3

With 8-way model parallelism on 64 A100 GPUs:

```
Total parameters: 175B
Per device: 175B / 8 = 22B params

Memory per GPU (FP16):
- Parameters:        22B × 2 bytes = 44 GB → 11 GB (with optimizations)
- Gradients:         Same as parameters = 11 GB
- Optimizer (Adam):  2× parameters = 22 GB
- Activations:       ~20 GB
─────────────────────────────────────────
Total: ~64 GB ✓ Fits on 80GB A100
```

Without sharding: 175B × 4 bytes = 700 GB for parameters alone. Impossible on single GPU.

## Key Principles

1. **JAX's functional paradigm** makes parameters explicit data structures that can be trivially split across devices.

2. **Device mesh with named axes** provides semantic organization. The 'data' axis represents different data batches, the 'model' axis represents different model pieces.

3. **PartitionSpec dimensions match tensor dimensions**, not mesh dimensions. Each entry references a named mesh axis or None.

4. **Memory layout matters**: Reshape operations must respect contiguous memory layout. This is why attention requires both reshape and transpose.

5. **Weights use model parallelism** (split along model axis), **inputs use data parallelism** (split along data axis). Mixing these causes either incorrect training (different models per replica) or redundant computation (wasted GPUs).

6. **Sharding happens at device_put time**, not during computation. Once sharded, JAX/XLA handles all communication automatically.

7. **Efficiency requires orthogonal splits**: batch along data axis, model along model axis. This achieves true N×M parallelism on an N×M mesh.

Understanding these principles, particularly the memory layout considerations and the distinction between physical device arrangement and logical axis semantics, demystifies JAX's sharding and reveals why it's particularly elegant for large-scale training.

## Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX Sharding Guide](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
- [Flax Documentation](https://flax.readthedocs.io/)
