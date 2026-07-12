// SPDX-License-Identifier: GPL-2.0

// nvcc test_wgmma_async_v3.cu -O3 -gencode arch=compute_90a,code=sm_90a -o
// test_wgmma_async_v3 srun -n1 -p h01 --gres=gpu:1 ./test_wgmma_async_v3
// // Use the following in fuse0
// nvcc test_wgmma_async_v3.cu -O3 -U_GNU_SOURCE -gencode
// arch=compute_90a,code=sm_90a -o test_wgmma_async_v3 srun -n1
// --gres=gpu:H100:1 ./test_wgmma_async_v3
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define CUDART_CHECK(status)                                                   \
  do {                                                                         \
    cudaError_t ss = (status);                                                 \
    if (ss != cudaSuccess) {                                                   \
      fprintf(stderr,                                                          \
              "CUDA RUNTIME API Error occured in " __FILE__                    \
              " line %d (" #status ") with code %d\n",                         \
              __LINE__, ss);                                                   \
      exit(ss);                                                                \
    }                                                                          \
  } while (0)

// do A@B^T
__global__ void gemm_naive_bf16_krnl(size_t m, size_t n, size_t k, float *C,
                                     const __nv_bfloat16 *A, __nv_bfloat16 *B) {
  size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (m <= i || n <= j)
    return;
  float acc = 0;
  for (size_t _k = 0; _k < k; _k++) {
    acc += __bfloat162float(A[i * k + _k] * B[j * k + _k]);
  }
  C[i * n + j] = acc;
}

void gemm_naive_bf16(size_t m, size_t n, size_t k, float *C,
                     const __nv_bfloat16 *A, __nv_bfloat16 *B) {
  gemm_naive_bf16_krnl<<<dim3((n + 15) / 16, (m + 15) / 16, 1),
                         dim3(16, 16, 1)>>>(m, n, k, C, A, B);
}

void fill_rand(float *buf, size_t n) {
  for (size_t i = 0; i < n; i++) {
    buf[i] = rand() * 1. / RAND_MAX;
  }
}

__global__ void to_bf16_krnl(__nv_bfloat16 *dst, float *src, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  dst[i] = __float2bfloat16_rn(src[i]);
}

void to_bf16(__nv_bfloat16 *dst, float *src, size_t n) {
  to_bf16_krnl<<<(n + 255) / 256, 256>>>(dst, src, n);
}

bool check(size_t n, float *expected, float *actual) {
  bool unequal = false;
  auto max_err = 0.f;
  constexpr float atol = 1e-5;
  constexpr float rtol = 1.6e-2;
  for (size_t i = 0; i < n; i++) {
    auto err = fabs(expected[i] - actual[i]);
    auto failed = err > atol + rtol * fabs(expected[i]); // 100 * FLT_EPSILON;
    // MPI_CHECK(failed);
    unequal |= failed;
    max_err = (max_err < err) ? err : max_err;
  }
  printf("Max error: %f\n", max_err);
  return unequal;
}

enum class SwizzleMode {
  INVALID,
  SWIZZLE_16B,
  SWIZZLE_32B,
  SWIZZLE_64B,
  SWIZZLE_128B,
};

// constexpr size_t getRepeatK(SwizzleMode mode) {
// [MODIFIED]
__host__ __device__ constexpr size_t getRepeatK(SwizzleMode mode) {
  switch (mode) {
  case SwizzleMode::SWIZZLE_16B:
    return 1;
  case SwizzleMode::SWIZZLE_32B:
    return 2;
  case SwizzleMode::SWIZZLE_64B:
    return 4;
  case SwizzleMode::SWIZZLE_128B:
    return 8;
  case SwizzleMode::INVALID:
  default:
    assert(false);
  }
  return 0;
}

template <SwizzleMode mode>
__device__ size_t swizzled_col(size_t row, size_t col) {
  assert(row < 8);
  switch (mode) {
  case SwizzleMode::SWIZZLE_16B:
    return col;
  case SwizzleMode::SWIZZLE_32B:
    return ((row >> 2) & 1) ^ col;
    ;
  case SwizzleMode::SWIZZLE_64B:
    return ((row >> 1) & 3) ^ col;
  case SwizzleMode::SWIZZLE_128B:
    return row ^ col;
  case SwizzleMode::INVALID:
    return col;
  }
  return col;
}

template <size_t kBMN, size_t kMMA_MN, typename T, size_t kBK, size_t kNThreads,
          size_t kMMA_K = 32 / sizeof(T),
          SwizzleMode swzMode = SwizzleMode::SWIZZLE_128B>
__device__ void refillSmem(int4 *smemPtr, const T *gmemPtr, size_t M,
                           size_t K) {
  const int4 *gmemPtr_ = reinterpret_cast<const int4 *>(gmemPtr);
  constexpr size_t kBKLoad = kBK * sizeof(T) / sizeof(int4);
  constexpr size_t kRepeatMN = 8;
  constexpr size_t kRepeatK = getRepeatK(swzMode);
  // For operand loads, dissect into the following:

  // Fill up the smem block
  // 1. 0 <= i1 < bM / MMA_M
  // 2. 0 <= j1 < bK / MMA_K

  // Fill up repeating pattern to match M/N of MMA instruction
  // 3. 0 <= i2 < MMA_M / RepeatM

  // For each repeating pattern
  // 4. 0 <= i3 < RepeatM
  // 5. 0 <= j2 < RepeatK

  // Hence I is 3-dimensional: (bM / MMA_M, MMA_M / 8, 8)
  // K is 2-dimensional: (bK / RepeatK, RepeatK)
  constexpr size_t kChunksToLoad = kBMN * kBKLoad;

  size_t tid = threadIdx.x;
  size_t cta_size = blockDim.x;
  assert(cta_size == 128);
#pragma unroll
  for (size_t t = tid; t < kChunksToLoad; t += kNThreads) {
    auto i = t / kBKLoad;
    auto k = t % kBKLoad;

    // which mma atom it belongs to along m/n direction
    auto i1 = i / kMMA_MN;
    // inside mma atom, which repeating pattern it belongs to
    auto i2 = (i % kMMA_MN) / kRepeatMN;
    // inside repeating pattern, which row
    auto i3 = i % kRepeatMN;

    // which repeating pattern it belongs to
    auto k1 = k / kRepeatK;
    // inside the repeating pattern, which local (swizzled) column it belongs to
    auto k2 = swizzled_col<swzMode>(i3, k % kRepeatK);

    auto smem_off = i1 * kMMA_MN * kBKLoad + k1 * kMMA_MN * kRepeatK +
                    i2 * kRepeatMN * kRepeatK + i3 * kRepeatK + k2;
    smemPtr[smem_off] = gmemPtr_[i * (K * sizeof(T) / sizeof(int4)) + k];
  }
}

template <int kMMA_M, SwizzleMode mode = SwizzleMode::SWIZZLE_128B>
__device__ constexpr uint64_t make_smem_desc() {
  uint64_t desc = 0;
  // SBO: num bytes between 8 rows, i.e. Size of repeating pattern
  constexpr uint64_t sbo = 8 * getRepeatK(mode) * sizeof(int4);
  constexpr uint64_t lbo = (mode == SwizzleMode::SWIZZLE_16B)
                               ? kMMA_M * getRepeatK(mode) * sizeof(int4)
                               : 16;
  uint64_t base_offset = 0;
  uint64_t type = 0;
  // Specifies the swizzling mode to be used:
  // 0: No swizzle
  // 1: 128-Byte swizzle
  // 2: 64-Byte swizzle
  // 3: 32-Byte swizzle
  switch (mode) {
  case SwizzleMode::SWIZZLE_128B:
    type = 1;
    break;
  case SwizzleMode::SWIZZLE_64B:
    type = 2;
    break;
  case SwizzleMode::SWIZZLE_32B:
    type = 3;
    break;
  default:
    type = 0;
    break;
  }
  desc |= (lbo >> 4) << 16;
  desc |= (sbo >> 4) << 32;
  desc |= (base_offset) << 49;
  desc |= type << 62;
  return desc;
}

template <size_t kMMA_MN, SwizzleMode swzMode>
__device__ __forceinline__ size_t k_offset(size_t mma_k_off) {
  // 2x 16B is 32B, or k16 for bf16
  constexpr size_t kRepeatKBytes = getRepeatK(swzMode) * sizeof(int4);
  constexpr size_t LBORepeatPatternBytes = kMMA_MN * kRepeatKBytes;
  // num of leading byte elements counted per 16B
  return (mma_k_off / kRepeatKBytes) * LBORepeatPatternBytes +
         (mma_k_off % kRepeatKBytes);
}

// [MODIFIED]
template <int kMMA_MN, SwizzleMode swzMode>
__device__ __forceinline__ uint64_t
make_smem_desc_from_ptr(const void *smem_ptr) {
  return make_smem_desc<kMMA_MN, swzMode>() |
         (static_cast<uint16_t>(__cvta_generic_to_shared(smem_ptr)) >> 4);
}

__global__ void gemm_wgmma_bf16_krnl(size_t M, size_t N, size_t K, float *C,
                                     const __nv_bfloat16 *A, __nv_bfloat16 *B) {
  // May be modified for tuning
  constexpr SwizzleMode swzMode = SwizzleMode::SWIZZLE_128B;
  constexpr size_t kBM = 128;
  constexpr size_t kBN = 96;
  constexpr size_t kBK = 128;

  // Depend on the PTX instruction of wgmma
  constexpr size_t kMMA_M = 64;
  constexpr size_t kMMA_N = 48;
  constexpr size_t kMMA_K = 32 / sizeof(nv_bfloat16);

  // You should not modify the following
  static_assert(kBM % kMMA_M == 0,
                "Block Size (bM) should be divisible by MMA atom size (MMA_M)");
  static_assert(kBN % kMMA_N == 0,
                "Block Size (bN) should be divisible by MMA atom size (MMA_N)");
  static_assert(kBK % kMMA_K == 0,
                "Block Size (bK) should be divisible by MMA atom size (MMA_K)");

  constexpr size_t kBKLoad = kBK * sizeof(nv_bfloat16) / sizeof(int4);
  alignas(128) extern __shared__ char smem[];

  int4 *sALoad = reinterpret_cast<int4 *>(smem);
  int4 *sBLoad = reinterpret_cast<int4 *>(smem) + kBM * kBKLoad;
  refillSmem<kBM, kMMA_M, nv_bfloat16, kBK, 128, kMMA_K, swzMode>(sALoad, A, M,
                                                                  K);
  refillSmem<kBN, kMMA_N, nv_bfloat16, kBK, 128, kMMA_K, swzMode>(sBLoad, B, N,
                                                                  K);

  __syncthreads();
  float rC_[kBM / kMMA_M][kBN / kMMA_N][kMMA_M * kMMA_N / 128] = {0};

  for (int mm = 0; mm < kBM / kMMA_M; mm++) {
    for (int nn = 0; nn < kBN / kMMA_N; nn++) {
      float *rC = rC_[mm][nn];
      size_t smemDescABase = make_smem_desc_from_ptr<kMMA_M, swzMode>(sALoad) +
                             (mm * kMMA_M * kBK * sizeof(nv_bfloat16) >> 4);
      size_t smemDescBBase = make_smem_desc_from_ptr<kMMA_N, swzMode>(sBLoad) +
                             (nn * kMMA_N * kBK * sizeof(nv_bfloat16) >> 4);
      for (int t = 0; t < kMMA_M * kMMA_N / 128; t++)
        __asm__ __volatile__("" : "+f"(rC[t])::"memory");
      __asm__ __volatile__("wgmma.fence.sync.aligned;\n" ::: "memory");
      for (int kk = 0; kk < kBK * sizeof(nv_bfloat16);
           kk += kMMA_K * sizeof(nv_bfloat16)) {
        uint64_t smemDescA =
            smemDescABase + (k_offset<kMMA_M, swzMode>(kk) >> 4);
        uint64_t smemDescB =
            smemDescBBase + (k_offset<kMMA_N, swzMode>(kk) >> 4);
        __asm__ __volatile__(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %26, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "
            "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
            " %8,  %9,  %10, %11, %12, %13, %14, %15, "
            " %16, %17, %18, %19, %20, %21, %22, %23}, "
            "%24, %25, p, 1, 1, 0, 0;\n"
            "}\n"
            : "+f"(rC[0]), "+f"(rC[1]), "+f"(rC[2]), "+f"(rC[3]), "+f"(rC[4]),
              "+f"(rC[5]), "+f"(rC[6]), "+f"(rC[7]), "+f"(rC[8]), "+f"(rC[9]),
              "+f"(rC[10]), "+f"(rC[11]), "+f"(rC[12]), "+f"(rC[13]),
              "+f"(rC[14]), "+f"(rC[15]), "+f"(rC[16]), "+f"(rC[17]),
              "+f"(rC[18]), "+f"(rC[19]), "+f"(rC[20]), "+f"(rC[21]),
              "+f"(rC[22]), "+f"(rC[23])
            : "l"(smemDescA), "l"(smemDescB), "r"(1));
      }
      __asm__ __volatile__("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      for (int t = 0; t < kMMA_M * kMMA_N / 128; t++)
        __asm__ __volatile__("" : "+f"(rC[t])::"memory");
      // warpgroup wait
      __asm__ __volatile__("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    }
  }
  // epilog here ...
  // TODO
  // [MODIFIED]
  size_t tid = threadIdx.x;
  size_t lane_m = (tid / 32) * 16 + ((tid % 32) / 4);
  size_t lane_n = (tid % 4) * 2;
#pragma unroll
  for (int mm = 0; mm < kBM / kMMA_M; mm++) {
#pragma unroll
    for (int nn = 0; nn < kBN / kMMA_N; nn++) {
      float *rC = rC_[mm][nn];
      size_t row0 = mm * kMMA_M + lane_m;
      size_t row1 = row0 + 8;
      size_t col_base = nn * kMMA_N + lane_n;
#pragma unroll
      for (int t = 0; t < kMMA_N / 8; t++) {
        int reg = t * 4;
        size_t col = col_base + t * 8;
        C[row0 * N + col + 0] = rC[reg + 0];
        C[row0 * N + col + 1] = rC[reg + 1];
        C[row1 * N + col + 0] = rC[reg + 2];
        C[row1 * N + col + 1] = rC[reg + 3];
      }
    }
  }
}

int main(int argc, char **argv) {
  srand(time(0));

  float *A_h, *A_d, *B_h, *B_d, *C1_h, *C2_h, *C1_d, *C2_d;
  __nv_bfloat16 *A_d1, *B_d1;
  constexpr int M = 128, N = 96, K = 128;
  A_h = new float[M * K];
  B_h = new float[N * K];
  C1_h = new float[M * N];
  C2_h = new float[M * N];
  fill_rand(A_h, M * K);
  fill_rand(B_h, N * K);
  CUDART_CHECK(cudaMalloc(&A_d, M * K * sizeof(float)));
  CUDART_CHECK(cudaMalloc(&B_d, N * K * sizeof(float)));
  CUDART_CHECK(cudaMalloc(&C1_d, M * N * sizeof(float)));
  CUDART_CHECK(cudaMalloc(&C2_d, M * N * sizeof(float)));
  CUDART_CHECK(cudaMalloc(&A_d1, M * K * sizeof(__nv_bfloat16)));
  CUDART_CHECK(cudaMalloc(&B_d1, N * K * sizeof(__nv_bfloat16)));

  CUDART_CHECK(
      cudaMemcpy(A_d, A_h, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDART_CHECK(
      cudaMemcpy(B_d, B_h, N * K * sizeof(float), cudaMemcpyHostToDevice));

  to_bf16(A_d1, A_d, M * K);
  to_bf16(B_d1, B_d, N * K);
  gemm_naive_bf16(M, N, K, C1_d, A_d1, B_d1);
  CUDART_CHECK(cudaGetLastError());

  constexpr int smem_sz = (M + N) * K * sizeof(nv_bfloat16);
  CUDART_CHECK(cudaFuncSetAttribute(gemm_wgmma_bf16_krnl,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_sz));
  gemm_wgmma_bf16_krnl<<<1, 128, smem_sz>>>(M, N, K, C2_d, A_d1, B_d1);
  CUDART_CHECK(cudaDeviceSynchronize());
  CUDART_CHECK(cudaGetLastError());
  // goto debug_out;
  CUDART_CHECK(
      cudaMemcpy(C1_h, C1_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDART_CHECK(
      cudaMemcpy(C2_h, C2_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  bool failed = check(M * N, C1_h, C2_h);
  if (failed) {
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
  delete[] A_h;
  delete[] B_h;
  delete[] C1_h;
  delete[] C2_h;
  return 0;
}
