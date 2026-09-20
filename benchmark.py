import torch
from torch.utils.cpp_extension import load
import time

tiled_matmul = load(name="tiled_matmul", sources=["matmul_ext.cu"], verbose=False)

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
    return (end - start) / iters * 1000  # ms per call

custom_ms = benchmark(lambda: tiled_matmul.tiled_matmul(A, B))
torch_ms = benchmark(lambda: torch.matmul(A, B))

print(f"Custom tiled kernel:   {custom_ms:.3f} ms")
print(f"torch.matmul (cuBLAS): {torch_ms:.3f} ms")
print(f"cuBLAS is {custom_ms/torch_ms:.1f}x faster")

C_custom = tiled_matmul.tiled_matmul(A, B)
C_ref = torch.matmul(A, B)
print("Correctness (allclose):", torch.allclose(C_custom, C_ref, atol=1e-2))
