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

    // Allocate host memory
    float *x, *y, *z;
    x = (float *)malloc(nBytes);
    y = (float *)malloc(nBytes);
    z = (float *)malloc(nBytes);

    for (int i = 0; i < N; ++i) {
        x[i] = 10.0;
        y[i] = 20.0;
    }

    // Allocate device memory
    float *d_x, *d_y, *d_z;
    cudaMalloc((void **)&d_x, nBytes);
    cudaMalloc((void **)&d_y, nBytes);
    cudaMalloc((void **)&d_z, nBytes);

    // Copy data from host to device
    cudaMemcpy((void *)d_x, (void *)x, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_y, (void *)y, nBytes, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    add<<<gridSize, blockSize>>>(d_x, d_y, d_z, N);

    // Copy result from device to host
    cudaMemcpy((void *)z, (void *)d_z, nBytes, cudaMemcpyDeviceToHost);

    float maxError = 0.0;
    for (int i = 0; i < N; ++i) {
        maxError = fmax(maxError, fabs(z[i] - 30.0));
    }
    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    free(x);
    free(y);
    free(z);

    return 0;
}
