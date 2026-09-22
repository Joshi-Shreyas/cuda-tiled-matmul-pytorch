import torch
import time
from torch.utils.cpp_extension import load
from triton_matmul import triton_matmul, triton_matmul_autotuned

cuda_ext = load(name="tiled_matmul", sources=["matmul_ext.cu"], verbose=False)

M, N, K = 2048, 2048, 2048
A = torch.randn(M, K, device='cuda', dtype=torch.float32)
B = torch.randn(K, N, device='cuda', dtype=torch.float32)

# -- Measure the one-time autotuning search cost --
torch.cuda.synchronize()
t0 = time.time()
_ = triton_matmul_autotuned(A, B)
torch.cuda.synchronize()
t1 = time.time()
print(f"Autotune first call (includes search): {(t1-t0)*1000:.1f} ms")

torch.cuda.synchronize()
t0 = time.time()
_ = triton_matmul_autotuned(A, B)
torch.cuda.synchronize()
t1 = time.time()
print(f"Autotune second call (cached):          {(t1-t0)*1000:.1f} ms\n")

def benchmark(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) / iters * 1000

cuda_ms = benchmark(lambda: cuda_ext.tiled_matmul(A, B))
triton_ms = benchmark(lambda: triton_matmul(A, B))
triton_auto_ms = benchmark(lambda: triton_matmul_autotuned(A, B))
torch_ms = benchmark(lambda: torch.matmul(A, B))

print(f"Custom CUDA kernel:          {cuda_ms:.3f} ms")
print(f"Custom Triton (fixed=32):    {triton_ms:.3f} ms")
print(f"Custom Triton (autotuned):   {triton_auto_ms:.3f} ms")
print(f"torch.matmul (cuBLAS):       {torch_ms:.3f} ms")

c_ref = torch.matmul(A, B)
print("\nCUDA matches cuBLAS:", torch.allclose(cuda_ext.tiled_matmul(A, B), c_ref, atol=1e-2))
print("Triton (fixed) matches cuBLAS:", torch.allclose(triton_matmul(A, B), c_ref, atol=1e-2))
print("Triton (autotuned) matches cuBLAS:", torch.allclose(triton_matmul_autotuned(A, B), c_ref, atol=1e-2))
