#include <cuda_runtime.h>
#include <random>

#define THREADS_PER_BLOCK 256

__global__ void reduce(float *d_input, float *d_output) {
    __shared__ float shared[THREADS_PER_BLOCK];
    float *input_begin = d_input + blockIdx.x * blockDim.x;
    shared[threadIdx.x] = input_begin[threadIdx.x];
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2) {
        if (threadIdx.x % (i * 2) == 0) {
            shared[threadIdx.x] += shared[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_output[blockIdx.x] = shared[0];
    }
}

bool check(float *out, float *res, int n) {
    for (int i = 0; i < n; ++i) {
        if (abs(out[i] - res[i]) > 1e-4) {
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 32 * 1024 * 1024;
    float *input = (float *)malloc(N * sizeof(float));
    float *d_input;
    cudaMalloc((void **)&d_input, N * sizeof(float));

    int block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // 为每个块分配一个float即可，每个块产生一个结果
    float *output = (float *)malloc(block_num * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, block_num * sizeof(float));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        input[i] = dist(gen);
    }
    // cpu calc
    float *result = (float *)malloc(block_num * sizeof(float));
    for (int i = 0; i < block_num; ++i) {
        float cur = 0;
        for (int j = 0; j < THREADS_PER_BLOCK; ++j) {
            cur += input[i * THREADS_PER_BLOCK + j];
        }
        result[i] = cur;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num);
    dim3 Block(THREADS_PER_BLOCK);

    reduce<<<Grid, Block>>>(d_input, d_output);
    cudaMemcpy(output, d_output, block_num * sizeof(float),
               cudaMemcpyDeviceToHost);
    if (check(output, result, block_num)) {
        printf("check pass\n");
    } else {
        printf("check fail\n");
        for (int i = 0; i < block_num; ++i) {
            printf("%f ", output[i]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
