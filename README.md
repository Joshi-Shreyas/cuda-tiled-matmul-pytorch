# Tiled CUDA Matmul — PyTorch Custom Extension

A hand-written, shared-memory-tiled CUDA matrix multiplication kernel, verified
and benchmarked against PyTorch's built-in `torch.matmul` (cuBLAS) — plus an
equivalent implementation in OpenAI Triton, including an autotuned variant.
Developed and tested on a SLURM-managed HPC cluster (V100 GPUs).

## What's here
- `matmul_ext.cu` — the CUDA kernel (16x16 shared-memory tiling), a C++ wrapper
  taking real `torch::Tensor` inputs, and a pybind11 binding exposing it to Python.
- `test_matmul_ext.py` — correctness check: exact match against `torch.matmul`
  on an all-ones input, and `torch.allclose` on random input.
- `benchmark.py` — wall-clock comparison of the CUDA kernel against `torch.matmul`.
- `triton_matmul.py` — the same matmul in OpenAI Triton: a fixed-block-size
  version and a `@triton.autotune`-decorated version.
- `three_way_benchmark.py` — head-to-head timing and correctness comparison
  across all four implementations (CUDA, Triton fixed, Triton autotuned, cuBLAS).

## Build & run
```bash
python test_matmul_ext.py       # CUDA extension correctness
python benchmark.py             # CUDA extension vs. cuBLAS
python three_way_benchmark.py   # CUDA vs. Triton (fixed & autotuned) vs. cuBLAS
```
Requires PyTorch with CUDA support and the CUDA toolkit (`nvcc`) available.

## Results (2048x2048x2048, V100)

| Implementation            | Time     | vs. cuBLAS         |
|----------------------------|----------|---------------------|
| Custom CUDA (hand-tuned)   | ~5.0 ms  | ~4.1-4.2x slower    |
| Triton (fixed BLOCK=32)    | ~1.63 ms | ~1.34-1.37x slower  |
| Triton (autotuned)         | ~1.55 ms | ~1.31x slower       |
| torch.matmul (cuBLAS)      | ~1.2 ms  | baseline            |

*(Exact numbers vary slightly run-to-run due to normal timing noise; ranges
reflect multiple benchmark runs across this project.)*

All three custom implementations verified correct against `torch.matmul` via
`torch.allclose` (exact match on deterministic all-ones input; `atol=1e-2` on
random input, since floating-point addition is not associative and accumulation
order differs from cuBLAS internally).

### Autotuning cost/benefit
The `@triton.autotune` search (5 block-size/warp/pipeline configurations) is a
real, one-time cost — about 20 seconds for the first call at a given shape —
cached and skipped on every later call with the same dimensions. Here it bought
only a ~5% improvement over a reasonably chosen fixed block size: a genuine
tradeoff, worth paying only when a shape is reused enough times (e.g. the same
layer shape across a real training loop) to earn back the search cost.

### Why cuBLAS still wins
Triton closes most of the gap to cuBLAS (no hand-placed `__syncthreads()`, no
manual accumulate loop — `tl.dot` handles the core matrix-multiply instruction
selection automatically). The raw CUDA kernel's larger gap reflects years of
architecture-specific hand-tuning in cuBLAS (register blocking, vectorized
loads, autotuned tile sizes) that a single from-scratch kernel doesn't replicate.

## Background
Built as part of a broader from-scratch study of CUDA: data parallelism vs.
reduction, race conditions and atomics, warp divergence, the full memory
hierarchy with measured latencies, shared-memory bank conflicts, and
host-device transfer optimization (pinned memory, measured via Nsight Systems
to cut a page-fault-inflated transfer time by ~3.4x). This extension is where
that theory met real ML framework integration, followed by a comparison
against OpenAI's Triton kernel language.
