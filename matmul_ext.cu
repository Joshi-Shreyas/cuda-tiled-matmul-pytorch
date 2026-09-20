#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void tiledMatmulKernel(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        int aCol = t * TILE_WIDTH + threadIdx.x;
        tileA[threadIdx.y][threadIdx.x] = (aCol < K) ? A[row * K + aCol] : 0;

        int bRow = t * TILE_WIDTH + threadIdx.y;
        tileB[threadIdx.y][threadIdx.x] = (bRow < K) ? B[bRow * N + col] : 0;

        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor tiled_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A and B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    tiledMatmulKernel<<<blocksPerGrid, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiled_matmul", &tiled_matmul_cuda, "Tiled matrix multiplication (CUDA)");
}
