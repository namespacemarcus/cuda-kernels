#include <cuda.h>
#include <iostream>

// grid: 1-dim
// block: 1-dim

__global__ void add(float *x, float *y, float *z, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        z[i] = x[i] + y[i];
    }
}

int main() {
    int N = 1 << 20;
    int nBytes = N * sizeof(float);

    float *x, *y, *z;
    cudaMallocManaged((void **)&x, nBytes);
    cudaMallocManaged((void **)&y, nBytes);
    cudaMallocManaged((void **)&z, nBytes);

    for (int i = 0; i < N; ++i) {
        x[i] = 10.0;
        y[i] = 20.0;
    }

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    add<<<gridSize, blockSize>>>(x, y, z, N);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    float maxError = 0.0;
    for (int i = 0; i < N; ++i) {
        maxError = fmax(maxError, fabs(z[i] - 30.0));
    }
    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    free(x);
    free(y);
    free(z);

    return 0;
}
