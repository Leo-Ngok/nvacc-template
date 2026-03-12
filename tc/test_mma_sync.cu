//SPDX-License-Identifier: GPL-2.0

// nvcc test_tc_mma_sync.cu -O3 -gencode arch=compute_90a,code=sm_90a -o test_tc_mma_sync
// srun -n1 -p h01 --gres=gpu:1 ./test_tc_mma_sync
#include <iostream>
#include <cuda_bf16.h>
#include <ctime>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cfloat>

#define CUDART_CHECK(status) \
    do {\
        cudaError_t ss = (status); \
        if(ss != cudaSuccess) {\
            fprintf(stderr, "CUDA RUNTIME API Error occured in " __FILE__ " line %d (" #status ") with code %d\n", __LINE__, ss); \
            exit(ss); \
        }\
    } while(0)

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

void fill_rand(float *buf, size_t n) {
    for(size_t i = 0; i < n; i++) {
        buf[i] = rand() * 1. / RAND_MAX;
    }
}

__global__
void gemm_mma_sync_bf16_krnl(size_t m, size_t n, size_t k, float *C , const __nv_bfloat16 *A, __nv_bfloat16 *B) {
    // m16n8k16
    // assert m == k ==16, n == 8
    __nv_bfloat16 rA[16 * 16 / 32];
    __nv_bfloat16 rB[16 * 8 / 32];
    float rC[16 * 8 / 32] = {0};
    constexpr int stride_a = 16;
    constexpr int stride_b = 16;
    constexpr int stride_c = 8;
    int i1 = threadIdx.x / 4;
    int j1 = (threadIdx.x % 4) * 2;
    rA[0] = A[(i1    ) * stride_a + (j1    )];
    rA[1] = A[(i1    ) * stride_a + (j1 + 1)];
    rA[2] = A[(i1 + 8) * stride_a + (j1    )];
    rA[3] = A[(i1 + 8) * stride_a + (j1 + 1)];

    rA[4] = A[(i1    ) * stride_a + (j1 + 8)];
    rA[5] = A[(i1    ) * stride_a + (j1 + 9)];
    rA[6] = A[(i1 + 8) * stride_a + (j1 + 8)];
    rA[7] = A[(i1 + 8) * stride_a + (j1 + 9)];

    rB[0] = B[(i1    ) * stride_b + (j1    )];
    rB[1] = B[(i1    ) * stride_b + (j1 + 1)];
    rB[2] = B[(i1    ) * stride_b + (j1 + 8)];
    rB[3] = B[(i1    ) * stride_b + (j1 + 9)];


    auto rAu = reinterpret_cast<const uint32_t *>(rA);
    auto rBu = reinterpret_cast<const uint32_t *>(rB);

    auto D = rC;
    __asm__ __volatile__(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "    // D
        "{%4, %5, %6, %7}, "    // A
        "{%8, %9}, "            // B
        "{%10, %11, %12, %13};" // C
    : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
    : "r"(rAu[0]), "r"(rAu[1]), "r"(rAu[2]), "r"(rAu[3]),
    "r"(rBu[0]), "r"(rBu[1]),
    "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));

    C[(i1    ) * stride_c + (j1    )] = rC[0];
    C[(i1    ) * stride_c + (j1 + 1)] = rC[1];
    C[(i1 + 8) * stride_c + (j1    )] = rC[2];
    C[(i1 + 8) * stride_c + (j1 + 1)] = rC[3];
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
        auto failed = err > atol + rtol * fabs(expected[i]); //100 * FLT_EPSILON;
        // MPI_CHECK(failed);
        unequal |= failed;
        max_err = (max_err < err) ? err : max_err;
    }
    printf("Max error: %f\n", max_err);
    return unequal;
}

int main(int argc, char **argv) {
    // 1. Test mma.sync
    // test for m16n8k16
    // 2. Test wgmma.async.sync

    srand(time(0));

    float *A_h, *A_d, *B_h, *B_d, *C1_h, *C2_h, *C1_d, *C2_d;
    __nv_bfloat16 *A_d1, *B_d1;
    constexpr int M = 16, N = 8, K = 16;
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

    gemm_mma_sync_bf16_krnl<<<1, 32>>>(M, N, K, C2_d, A_d1, B_d1);
    CUDART_CHECK(cudaGetLastError());

    CUDART_CHECK(cudaMemcpy(C1_h, C1_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDART_CHECK(cudaMemcpy(C2_h, C2_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    bool failed = check(M * N, C1_h, C2_h);
    if(failed) {
        printf("MMA SYNC GeMM Check failed.\n");
    } else {
        printf("MMA SYNC GeMM Check succeed.\n");
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
