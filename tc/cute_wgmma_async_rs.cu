//SPDX-License-Identifier: GPL-2.0

// nvcc cute_wgmma_async_rs.cu -O3 -std=c++17 -I/home/fit/zhaijdyzq/repos/DeepGEMM/third-party/cutlass/include -I/home/fit/zhaijdyzq/repos/DeepGEMM/third-party/cutlass/tools/util/include -gencode arch=compute_90a,code=sm_90a -o cute_wgmma_async_rs
// srun -n1 -p h01 --gres=gpu:1 ./cute_wgmma_async_rs
#include <cstdio>
#include <cuda_bf16.h>
#include <ctime>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cfloat>
#include <cute/tensor.hpp>
#include <cstdint>

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

template<int CORE_MAT_COLS, int STRIDE_BLOCK_DIM>
__forceinline__ __device__
int smem_pos(int i, int j) {
    return ( (j / CORE_MAT_COLS) * (STRIDE_BLOCK_DIM * CORE_MAT_COLS) ) + (i * CORE_MAT_COLS) + (j % CORE_MAT_COLS);
}

__global__
void gemm_wgmma_bf16_krnl(size_t m, size_t n, size_t k, float *C , const __nv_bfloat16 *A, __nv_bfloat16 *B) {
    // 64 X 64 X 16
    size_t tid = blockDim.x * threadIdx.y + threadIdx.x;
    constexpr int MMA_M = 64, MMA_N = 64, MMA_K = 16;
    constexpr int BLOCK_M = MMA_M, BLOCK_N = MMA_N, BLOCK_K = MMA_K;
    constexpr int CORE_MAT_COLS = 16 / sizeof(nv_bfloat16);
    alignas(128) __shared__ nv_bfloat16 sA[BLOCK_M * BLOCK_K];
    alignas(128) __shared__ nv_bfloat16 sB[BLOCK_M * BLOCK_K];

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

    nv_bfloat16 rA[MMA_M * MMA_K / 128];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int top_left_i = warp_id * 16 + (lane_id / 4);
    int top_left_k = (lane_id % 4) * 2;

    rA[0] = sA[smem_pos<CORE_MAT_COLS, BLOCK_M>(top_left_i,     top_left_k    )];
    rA[1] = sA[smem_pos<CORE_MAT_COLS, BLOCK_M>(top_left_i,     top_left_k + 1)];
    rA[2] = sA[smem_pos<CORE_MAT_COLS, BLOCK_M>(top_left_i + 8, top_left_k    )];
    rA[3] = sA[smem_pos<CORE_MAT_COLS, BLOCK_M>(top_left_i + 8, top_left_k + 1)];
    rA[4] = sA[smem_pos<CORE_MAT_COLS, BLOCK_M>(top_left_i,     top_left_k + 8)];
    rA[5] = sA[smem_pos<CORE_MAT_COLS, BLOCK_M>(top_left_i,     top_left_k + 9)];
    rA[6] = sA[smem_pos<CORE_MAT_COLS, BLOCK_M>(top_left_i + 8, top_left_k + 8)];
    rA[7] = sA[smem_pos<CORE_MAT_COLS, BLOCK_M>(top_left_i + 8, top_left_k + 9)];

    auto tiled_mma = cute::make_tiled_mma(
        cute::SM90_64x64x16_F32BF16BF16_RS<cute::SM90::GMMA::Major::K, cute::SM90::GMMA::Major::K>{}
    );
    using sBLayout = decltype(cute::tile_to_shape(
        cute::SM90::GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        cute::Shape<cute::Int<MMA_N>, cute::Int<MMA_K>>{}
    ));
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto _sB = cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<cute::bfloat16_t*>(sB)), sBLayout{}
    );
    auto tCrB = thr_mma.partition_fragment_B(_sB);
    auto tCrC = cute::partition_fragment_C(tiled_mma, 
        cute::Shape<cute::Int<MMA_M>, cute::Int<MMA_N>>{});
    
    auto tCrA_layout = thr_mma.partition_fragment_A(
        cute::make_tensor(cute::make_rmem_ptr<cute::bfloat16_t>((const void *)nullptr), 
        cute::Shape<cute::Int<MMA_M>, cute::Int<MMA_K>>{})
    ).layout();

    auto tCrA = cute::make_tensor(cute::make_rmem_ptr(reinterpret_cast<cute::bfloat16_t*>(rA)), tCrA_layout);
    // uint32_t *rAu = reinterpret_cast<uint32_t *>(rA);
    // auto tCrA = cute::make_tensor(
    //     cute::make_rmem_ptr(rAu),
    //     cute::make_layout(cute::make_shape(cute::Int<4>{}))
    // );
    cute::clear(tCrC);
    cute::warpgroup_fence_operand(tCrC);
    cute::warpgroup_arrive();
    // tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;
    cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
    cute::warpgroup_commit_batch();
    cute::warpgroup_wait<0>();
    cute::warpgroup_fence_operand(tCrC);
    size_t ci = (threadIdx.x / 32) * 16 + ((threadIdx.x % 32) / 4);
    size_t cj = (threadIdx.x % 4) * 2;
    for (int t = 0; t < 8; t++) {
        int _t = t * 4;
        int __t = t * 8;
        C[(ci    ) * n + (cj + __t + 0)] = tCrC(_t + 0);
        C[(ci    ) * n + (cj + __t + 1)] = tCrC(_t + 1);
        C[(ci + 8) * n + (cj + __t + 0)] = tCrC(_t + 2);
        C[(ci + 8) * n + (cj + __t + 1)] = tCrC(_t + 3);
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
        printf("WGMMA ASYNC (RS) CuTe GeMM Check failed.\n");
    } else {
        printf("WGMMA ASYNC (RS) CuTe GeMM Check succeed.\n");
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
