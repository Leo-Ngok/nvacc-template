//SPDX-License-Identifier: GPL-2.0

// vcc test_wgmma_async.cu -O3 -gencode arch=compute_90a,code=sm_90a -o test_wgmma_async
// srun -n1 -p h01 --gres=gpu:1 ./test_wgmma_async
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

void fill_seq(float *buf, size_t n) {
    for(size_t i = 0; i < n; i++) {
        buf[i] = ((float)i) * 0.5;
    }
}

// Before diving into krnl impl, read the following first:
// smem descriptor
// [0, 14): start smem addr, byte address / 16

// (K Major) For N: This is the stride from the first col to the second col of the 8x2 brick in INTERLEAVED
//   Unused for all SWIZZLE_* layouts (and assumed to be 1)
// [16, 30): LBO, / 16

// (K-Major) For N: This is the stride from the first 8 rows to the next 8 rows.
// [32, 46): SBO

// [49, 52): Base offset (swizzle 128b / swizzle 64b)
// [62, 64): layout type: 0 (no swizzle), 3 (32b), 2 (64b), 1 (128b)

// That is from PTX docs.

/*
However, it is confusing. Don't try to read PTX doc directly, when you have no idea on tc coordinates.
Consider the following first:
1. Assume doing A @ B^T, where A and B are both row-major, or in nv-word, k-major, i.e. C[i][j] += A[i][k] * B[j][k];
2. wgmma operands must be from (A: reg or smem) and (B: smem only)
3. wgmma instructions MUST be MxNxK = 64xNx"32B"
4. The "32B" part means 32 / sizeof(T) elements for dtype T.
5. A core matrix is a 8 x 16B "sub-matrix".
6. Since wgmma is 64 x N x 32B, the core matrix grid for A must be ((64 / 8) x (32 / 16)) = (8 x 2)
7. Core matrix grid for B is (N/8 x 2)
8. Adjacent core matrix in a column of the core matrix grid must be contiguous.
9. Elements inside a core matrix are also contiguous.
   - Run the code below and read the coordinate mapping of A, assume you do 64 x 64 x 16 with dtype as fp16/bf16:

def convert(i: int, k: int):
    idx = ((k // 8) * 512) + (i * 8) + (k % 8)
    return idx
for i in range(128):
    for k in range(16):
        print(f'{converti, k):03d} ', end='')
    print()

10. If you want to cascade multiple wgmma window, then you add to sA (sB) MMA_M * MMA_K (* sizeof(T))
11. You may want to consider using TMA to load to Smem. Then you may want to use 3D TMA descriptor
12. This version does NOT consider swizzling. If you want to do so, understand the non-swizzled version first.
*/
__global__
void gemm_wgmma_bf16_krnl(size_t m, size_t n, size_t k, float *C , const __nv_bfloat16 *A, __nv_bfloat16 *B) {
    // 64 X 64 X 16
    size_t tid = blockDim.x * threadIdx.y + threadIdx.x;
    constexpr int MMA_M = 64, MMA_N = 64, MMA_K = 16;
    constexpr int BLOCK_M = MMA_M, BLOCK_N = MMA_N, BLOCK_K = MMA_K;
    constexpr int CORE_MAT_COLS = 16 / sizeof(nv_bfloat16);
    alignas(128) __shared__ nv_bfloat16 sA[BLOCK_M * BLOCK_K];
    alignas(128) __shared__ nv_bfloat16 sB[BLOCK_M * BLOCK_K];
    float rC[MMA_M * MMA_N / 128] = {0};
    uint64_t desc_a = 0, desc_b = 0;
    // smem descriptor

    desc_a |= ((uint16_t) __cvta_generic_to_shared(sA)) >> 4;
    desc_a |= ((/*bM*/BLOCK_M * 16) >> 4) << 16; // LBO
    desc_a |= ((8ULL * 16) >> 4) << 32; // SBO Note that a core matrix must be size of 8 x 16B.
    // Base offset: 0 do nothing
    // layout type: 0 do nothing

    desc_b |= ((uint16_t) __cvta_generic_to_shared(sB)) >> 4;
    desc_b |= ((/*bN*/BLOCK_N * 16) >> 4) << 16; 
    desc_b |= ((8ULL * 16) >> 4) << 32; 

    size_t threads_per_block = blockDim.x * blockDim.y;

    for(auto it = tid; it < (m * k); it += threads_per_block) {
        size_t start_idx = it;
        size_t _i = start_idx / k; 
        size_t _k = start_idx % k; 
        sA[( (_k / CORE_MAT_COLS) * (BLOCK_M * CORE_MAT_COLS) ) + (_i * CORE_MAT_COLS) + (_k % CORE_MAT_COLS)] = A[_i * k + _k];
    }
    for(auto it = tid; it < (n * k); it += threads_per_block) {
        size_t start_idx = it;
        size_t _j = start_idx / k; 
        size_t _k = start_idx % k; 
        sB[( (_k / CORE_MAT_COLS) * (BLOCK_N * CORE_MAT_COLS) ) + (_j * CORE_MAT_COLS) + (_k % CORE_MAT_COLS)] = B[_j * k + _k];
    }
    __syncthreads();
    
    for(int t = 0; t < 16; t++)
        __asm__ __volatile__("" : "+f"(rC[t]) :: "memory");
    // wg arrive
    __asm__ __volatile__("wgmma.fence.sync.aligned;\n" ::: "memory");
    // m64n64k16 BF16 to F32
    __asm__ __volatile__ (
      ".reg .pred p;\n"
      "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},  "
        "%32, %33, p, 1, 1, 0, 0;" 
        /* scale-d = p (=1), imm-scale-a = imm-scale-b = 1 (can be -1), 
        imm-trans-a = imm-trans-b = 0 */
        // trans 0 = k-major
        // trans 1 = m/n-major
        // set imm-trans-b to be 1 if B is no transpose
        // default to row-major * coln-major, or A @ B^T for row major A, B
        : "+f"(rC[0]),  "+f"(rC[1]),  "+f"(rC[2]),  "+f"(rC[3]),
          "+f"(rC[4]),  "+f"(rC[5]),  "+f"(rC[6]),  "+f"(rC[7]),
          "+f"(rC[8]),  "+f"(rC[9]),  "+f"(rC[10]), "+f"(rC[11]),
          "+f"(rC[12]), "+f"(rC[13]), "+f"(rC[14]), "+f"(rC[15]),
          "+f"(rC[16]), "+f"(rC[17]), "+f"(rC[18]), "+f"(rC[19]),
          "+f"(rC[20]), "+f"(rC[21]), "+f"(rC[22]), "+f"(rC[23]),
          "+f"(rC[24]), "+f"(rC[25]), "+f"(rC[26]), "+f"(rC[27]),
          "+f"(rC[28]), "+f"(rC[29]), "+f"(rC[30]), "+f"(rC[31])
        : "l"(desc_a), "l"(desc_b), "r"(1));
    
    // wg commit
    __asm__ __volatile__("wgmma.commit_group.sync.aligned;\n" ::: "memory");

    for(int t = 0; t < 16; t++)
        __asm__ __volatile__("" : "+f"(rC[t]) :: "memory");
    // wg wait
    __asm__ __volatile__("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    // warning: You should add fence to rC when putting it into loop.
    size_t ci = (threadIdx.x / 32) * 16 + ((threadIdx.x % 32) / 4);
    size_t cj = (threadIdx.x % 4) * 2;
    for (int t = 0; t < 8; t++) {
        int _t = t * 4;
        int __t = t * 8;
        C[(ci    ) * n + (cj + __t + 0)] = rC[_t + 0];
        C[(ci    ) * n + (cj + __t + 1)] = rC[_t + 1];
        C[(ci + 8) * n + (cj + __t + 0)] = rC[_t + 2];
        C[(ci + 8) * n + (cj + __t + 1)] = rC[_t + 3];
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
        auto failed = err > atol + rtol * fabs(expected[i]); //100 * FLT_EPSILON;
        // MPI_CHECK(failed);
        unequal |= failed;
        max_err = (max_err < err) ? err : max_err;
    }
    printf("Max error: %f\n", max_err);
    return unequal;
}

int main(int argc, char **argv) {
    srand(time(0));

    float *A_h, *A_d, *B_h, *B_d, *C1_h, *C2_h, *C1_d, *C2_d;
    __nv_bfloat16 *A_d1, *B_d1;
    constexpr int M = 64, N = 64, K = 16;
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

    gemm_wgmma_bf16_krnl<<<1,128>>>(M, N, K, C2_d, A_d1, B_d1);
    CUDART_CHECK(cudaDeviceSynchronize());
    CUDART_CHECK(cudaGetLastError());
    // goto debug_out;
    CUDART_CHECK(cudaMemcpy(C1_h, C1_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDART_CHECK(cudaMemcpy(C2_h, C2_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    bool failed = check(M * N, C1_h, C2_h);
    if(failed) {
        printf("WGMMA ASYNC GeMM Check failed.\n");
    } else {
        printf("WGMMA ASYNC GeMM Check succeed.\n");
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
