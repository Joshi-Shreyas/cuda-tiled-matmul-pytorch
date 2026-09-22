# Tiled CUDA Matmul — PyTorch Custom Extension

A hand-written, shared-memory-tiled CUDA matrix multiplication kernel, verified
against and benchmarked against PyTorch's built-in `torch.matmul` (cuBLAS),
developed and tested on a SLURM-managed HPC cluster (V100 GPUs).

## Triton Comparison

The same tiled matmul was also implemented in OpenAI Triton (`triton_matmul.py`) and
benchmarked head-to-head against both the hand-written CUDA kernel and cuBLAS
(`three_way_benchmark.py`), at 2048x2048x2048:

| Implementation          | Time      | vs. cuBLAS |
|--------------------------|-----------|------------|
| Custom CUDA (hand-tuned) | 5.188 ms  | ~4.1x slower |
| Custom Triton            | 1.683 ms  | ~1.34x slower |
| torch.matmul (cuBLAS)    | 1.256 ms  | baseline |

Both custom kernels verified correct against `torch.matmul` via `torch.allclose`.

Triton closed most of the gap to cuBLAS with substantially less manual tuning than
the raw CUDA version required (no hand-placed `__syncthreads()`, no manual
accumulate loop -- `tl.dot` handles the core matrix-multiply instruction selection).
This mirrors Triton's well-known real-world value proposition: near-hand-tuned
performance with a fraction of the development effort.

### Autotuning

Added a `@triton.autotune`-decorated variant searching 5 block-size/warp/pipeline
configurations. Benchmarked at 2048x2048x2048:

| Implementation           | Time      | vs. cuBLAS   |
|---------------------------|-----------|--------------|
| Custom CUDA (hand-tuned)  | 4.999 ms  | ~4.2x slower |
| Triton (fixed BLOCK=32)   | 1.629 ms  | ~1.37x slower|
| Triton (autotuned)        | 1.550 ms  | ~1.31x slower|
| torch.matmul (cuBLAS)     | 1.187 ms  | baseline     |

The autotuning search itself is a real, one-time cost (~20 seconds for the first
call at a given shape) that gets cached and skipped on every later call with the
same dimensions. In this case it bought only a ~5% improvement over a reasonably
chosen fixed block size -- a genuine cost/benefit tradeoff, worth paying only
when a shape gets reused enough times (e.g. the same layer shape across a real
training loop) to earn back the search cost.

All three custom implementations verified correct against `torch.matmul` via
`torch.allclose`.

## What's here
- `matmul_ext.cu` — the CUDA kernel (16x16 shared-memory tiling), a C++ wrapper
  taking real `torch::Tensor` inputs, and a pybind11 binding exposing it to Python.
- `test_matmul_ext.py` — correctness check: exact match against `torch.matmul`
  on an all-ones input, and `torch.allclose` on random input.
- `benchmark.py` — wall-clock comparison against `torch.matmul` at 2048x2048x2048.

## Results
- Correctness: bit-identical on deterministic input; `torch.allclose` (atol=1e-2)
  on random input (floating-point accumulation order differs from cuBLAS, so
  exact equality isn't expected there).
- Performance: ~5.47 ms (this kernel) vs. ~1.37 ms (cuBLAS) — cuBLAS is ~4x
  faster, reflecting years of architecture-specific hand-tuning (register
  blocking, vectorized loads, autotuned tile sizes) that a single from-scratch
  kernel doesn't replicate.

## Background
This kernel was built as part of a broader from-scratch study of CUDA:
data parallelism vs. reduction, race conditions and atomics, warp divergence,
the full memory hierarchy with measured latencies, shared-memory bank
conflicts, and host-device transfer optimization (pinned memory, measured via
Nsight Systems to cut a page-fault-inflated transfer time by ~3.4x). This
extension is where that theory met a real ML framework integration.

## Build & run
```bash
python test_matmul_ext.py   # correctness
python benchmark.py         # performance vs. cuBLAS
```
Requires PyTorch with CUDA support and the CUDA toolkit (`nvcc`) available.
EOF
