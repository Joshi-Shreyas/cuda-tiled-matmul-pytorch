# Tiled CUDA Matmul — PyTorch Custom Extension

A hand-written, shared-memory-tiled CUDA matrix multiplication kernel, verified
against and benchmarked against PyTorch's built-in `torch.matmul` (cuBLAS),
developed and tested on a SLURM-managed HPC cluster (V100 GPUs).

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
