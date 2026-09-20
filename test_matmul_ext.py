import torch
from torch.utils.cpp_extension import load

tiled_matmul = load(name="tiled_matmul", sources=["matmul_ext.cu"], verbose=True)

M, N, K = 512, 512, 512
A = torch.ones(M, K, device='cuda', dtype=torch.float32)
B = torch.ones(K, N, device='cuda', dtype=torch.float32)

C = tiled_matmul.tiled_matmul(A, B)
print("Your kernel: C[0,0] =", C[0, 0].item(), "(expected", K, ")")

C_ref = torch.matmul(A, B)
print("Matches torch.matmul exactly:", torch.allclose(C, C_ref))
