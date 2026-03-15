//SPDX-License-Identifier: GPL-2.0

// nvcc test_tc_mma_sync.cu -O3 -gencode arch=compute_90a,code=sm_90a -o test_tc_mma_sync
// srun -n1 -p h01 --gres=gpu:1 ./test_tc_mma_sync
#include <iostream>
#include <cuda_bf16.h>
#include <ctime>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cfloat>
#include <cassert>

#define CP_VER 3

#define CUDART_CHECK(status) \
    do {\
        cudaError_t ss = (status); \
        if(ss != cudaSuccess) {\
            fprintf(stderr, "CUDA RUNTIME API Error occured in " __FILE__ " line %d (" #status ") with code %d\n", __LINE__, ss); \
            exit(ss); \
        }\
    } while(0)

void fill_rand(float *buf, size_t n) {
    for(size_t i = 0; i < n; i++) {
        buf[i] = rand() * 1. / RAND_MAX;
    }
}

__global__
void to_bf16_krnl(__nv_bfloat16 *dst, float *src, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    dst[i] = __float2bfloat16_rn(src[i]);
}

void to_bf16(__nv_bfloat16 *dst, float *src, size_t n) {
    to_bf16_krnl<<<(n+255)/256, 256>>>(dst, src, n);
}

bool check(size_t n, float *expected, float *actual) {
    bool unequal = false;
    auto max_err = 0.f;
    constexpr float atol = 1e-5;
    constexpr float rtol = 1.6e-2;
    for(size_t i = 0; i < n; i++) {
        auto err = fabs(expected[i] - actual[i]);
        auto failed = err > atol + rtol * fabs(expected[i]);
        // MPI_CHECK(failed);
        unequal |= failed;
        max_err = (max_err < err) ? err : max_err;
    }
    printf("Max error: %f\n", max_err);
    return unequal;
}

// do A@B^T
__global__
void gemm_naive_bf16_krnl(size_t m, size_t n, size_t k, float *C , const __nv_bfloat16 *A, __nv_bfloat16 *B) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if(m <= i || n <= j) return;
    float acc = 0;
    for(size_t _k = 0; _k < k; _k++) {
        acc += __bfloat162float(A[i * k + _k] * B[j * k + _k]);
    }
    C[i * n + j] = acc;
}

void gemm_naive_bf16(size_t m, size_t n, size_t k, float *C , const __nv_bfloat16 *A, __nv_bfloat16 *B) {
    gemm_naive_bf16_krnl<<< dim3((n + 15) / 16, (m + 15) / 16, 1), dim3(16,16, 1)>>>(m, n, k, C, A, B);
}

__global__
void gemm_cp_async_bf16_krnl(size_t m, size_t n, size_t k, float *C , const __nv_bfloat16 *A, __nv_bfloat16 *B) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    // Coordinate flattening
    size_t tid = blockDim.x * threadIdx.y + threadIdx.x;

    // to be consistent with main()
    __shared__ nv_bfloat16 sA[48 * 16];
    // __shared__ nv_bfloat16 sB[32 * 16];
    size_t threads_per_block = blockDim.x * blockDim.y;

#if CP_VER == 0
    // do nothing
#elif CP_VER == 1
    // version 1. load element by element

    for(auto it = tid; it < (m * k); it += threads_per_block) {
        size_t start_idx = it;
        size_t _i = start_idx / k; 
        size_t _k = start_idx % k; 
        sA[_i * k + _k] = A[_i * k + _k];
    }
#elif CP_VER == 2
    // version 2. vectorized load

    for(auto it = tid; it < (m * k) / 8; it += threads_per_block) {
        size_t start_idx = it;
        size_t _i = start_idx / k; 
        size_t _k = start_idx % k; 
        reinterpret_cast<int4 *>(sA)[_i * k + _k] = 
        reinterpret_cast<const int4 *>(A)[_i * k + _k];
    }

    // version 3. asynchronous load 
#elif CP_VER == 3
    for(auto it = tid; it < (m * k) / 8; it += threads_per_block) {
        size_t start_idx = it;
        size_t _i = start_idx / k; 
        size_t _k = start_idx % k; 
        auto sA_ptr = __cvta_generic_to_shared(&reinterpret_cast<int4 *>(sA)[_i * k + _k]);
        __asm__ __volatile__("cp.async.cg.shared.global [%0], [%1], 16;" ::
            "r"((uint32_t)sA_ptr), "l"(&reinterpret_cast<const int4 *>(A)[_i * k + _k]));
    }
    
    __asm__ __volatile__("cp.async.commit_group;");
    // set 1 (or more) if you want to do staged pipeline
    __asm__ __volatile__("cp.async.wait_group 0;");
#else
    assert(false, "VER Should be 0, 1, 2, or 3");
#endif
    __syncthreads();
    if(m <= i || n <= j) return;
    float acc = 0;
    for(size_t _k = 0; _k < k; _k++) {
#if CP_VER == 0
        acc += __bfloat162float(A[i * k + _k] * B[j * k + _k]);
#else
        acc += __bfloat162float(sA[i * 16 + _k] * B[j * k + _k]);
#endif
    }
    C[i * n + j] = acc;
}

void gemm_cp_async_bf16(size_t m, size_t n, size_t k, float *C , const __nv_bfloat16 *A, __nv_bfloat16 *B) {
    gemm_cp_async_bf16_krnl<<< dim3((n + 15) / 16, (m + 15) / 16, 1), dim3(16,16, 1)>>>(m, n, k, C, A, B);
}

int main(int argc, char **argv) {
    srand(time(0));

    float *A_h, *A_d, *B_h, *B_d, *C1_h, *C2_h, *C1_d, *C2_d;
    __nv_bfloat16 *A_d1, *B_d1;
    constexpr int M = 48, N = 32, K = 16;
    A_h = new float[M * K]; B_h = new float[N * K]; C1_h = new float[M * N]; C2_h = new float[M * N]; 
    fill_rand(A_h, M * K);
    fill_rand(B_h, N * K);
    CUDART_CHECK(cudaMalloc(&A_d, M * K * sizeof(float)));
    CUDART_CHECK(cudaMalloc(&B_d, N * K * sizeof(float)));
    CUDART_CHECK(cudaMalloc(&C1_d, M * N * sizeof(float)));
    CUDART_CHECK(cudaMalloc(&C2_d, M * N * sizeof(float)));
    CUDART_CHECK(cudaMalloc(&A_d1, M * K * sizeof(__nv_bfloat16)));
    CUDART_CHECK(cudaMalloc(&B_d1, N * K * sizeof(__nv_bfloat16)));

    CUDART_CHECK(cudaMemcpy(A_d, A_h, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDART_CHECK(cudaMemcpy(B_d, B_h, N * K * sizeof(float), cudaMemcpyHostToDevice));

    to_bf16(A_d1, A_d, M * K);
    to_bf16(B_d1, B_d, N * K);
    gemm_naive_bf16(M, N, K, C1_d, A_d1, B_d1);
    CUDART_CHECK(cudaGetLastError());

    gemm_cp_async_bf16(M, N, K, C2_d, A_d1, B_d1);
    CUDART_CHECK(cudaGetLastError());

    CUDART_CHECK(cudaMemcpy(C1_h, C1_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDART_CHECK(cudaMemcpy(C2_h, C2_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    bool failed = check(M * N, C1_h, C2_h);
    if(failed) {
        printf("CP ASYNC GeMM Check failed.\n");
    } else {
        printf("CP ASYNC GeMM Check succeed.\n");
    }
    CUDART_CHECK(cudaFree(A_d));
    CUDART_CHECK(cudaFree(B_d));
    CUDART_CHECK(cudaFree(C1_d));
    CUDART_CHECK(cudaFree(C2_d));
    CUDART_CHECK(cudaFree(A_d1));
    CUDART_CHECK(cudaFree(B_d1));
    delete [] A_h; delete [] B_h; delete [] C1_h; delete [] C2_h;
    return 0;
}
