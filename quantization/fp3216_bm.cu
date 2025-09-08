// addition of a+b, fp32 v fp16, time benchmarking, Tesla L4

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <iomanip>

__global__ void fp32_add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void fp16_add_kernel(half* a, half* b, half* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

float benchmark_fp32(int n, int num_runs) {
    float *d_a32, *d_b32, *d_c32;
    cudaMalloc(&d_a32, n * sizeof(float));
    cudaMalloc(&d_b32, n * sizeof(float));
    cudaMalloc(&d_c32, n * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        fp32_add_kernel<<<gridSize, blockSize>>>(d_a32, d_b32, d_c32, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float fp32_time;
    cudaEventElapsedTime(&fp32_time, start, stop);

    cudaFree(d_a32);
    cudaFree(d_b32);
    cudaFree(d_c32);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return fp32_time;
}

float benchmark_fp16(int n, int num_runs) {
    half *d_a16, *d_b16, *d_c16;
    cudaMalloc(&d_a16, n * sizeof(half));
    cudaMalloc(&d_b16, n * sizeof(half));
    cudaMalloc(&d_c16, n * sizeof(half));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        fp16_add_kernel<<<gridSize, blockSize>>>(d_a16, d_b16, d_c16, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float fp16_time;
    cudaEventElapsedTime(&fp16_time, start, stop);

    cudaFree(d_a16);
    cudaFree(d_b16);
    cudaFree(d_c16);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return fp16_time;
}

int main() {
    const int n = 1024 * 1024;
    const int num_runs = 100;

    float fp32_time = benchmark_fp32(n, num_runs);
    float fp16_time = benchmark_fp16(n, num_runs);

    std::cout << "FP32 time: " << fp32_time << " ms" << std::endl;
    std::cout << "FP16 time: " << fp16_time << " ms" << std::endl;
    std::cout << "Speedup: " << std::fixed << std::setprecision(2)
                << fp32_time / fp16_time << "x" << std::endl;

    return 0;
}

/*

nvidia-smi
nvcc --version
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
%%writefile fp3216_bm.cu
nvcc -O3 -arch=sm_75 -o fp3216_bm fp3216_bm.cu
./fp3216_bm

FP32 time: 5.27155 ms
FP16 time: 3.87587 ms
Speedup: 1.36x

*/
