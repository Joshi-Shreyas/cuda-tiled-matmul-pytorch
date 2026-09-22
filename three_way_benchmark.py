import torch
import time
from torch.utils.cpp_extension import load
from triton_matmul import triton_matmul

cuda_ext = load(name="tiled_matmul", sources=["matmul_ext.cu"], verbose=False)

M, N, K = 2048, 2048, 2048
A = torch.randn(M, K, device='cuda', dtype=torch.float32)
B = torch.randn(K, N, device='cuda', dtype=torch.float32)

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
torch_ms = benchmark(lambda: torch.matmul(A, B))

print(f"Custom CUDA kernel:    {cuda_ms:.3f} ms")
print(f"Custom Triton kernel:  {triton_ms:.3f} ms")
print(f"torch.matmul (cuBLAS): {torch_ms:.3f} ms")

c_cuda = cuda_ext.tiled_matmul(A, B)
c_triton = triton_matmul(A, B)
c_ref = torch.matmul(A, B)
print("CUDA matches cuBLAS:", torch.allclose(c_cuda, c_ref, atol=1e-2))
print("Triton matches cuBLAS:", torch.allclose(c_triton, c_ref, atol=1e-2))
