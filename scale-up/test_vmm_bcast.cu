//SPDX-License-Identifier: GPL-2.0

/*
nvcc test_vmm_bcast.cu mem_handle_vmm.cc -o test_vmm_bcast \
-gencode arch=compute_90,code=sm_90 -std=c++20 -O3 \
-I$(mpicxx -showme:incdirs) \
-L$(mpicxx -showme:libdirs) \
-L$(spack location -i /dqz)/lib \
-L$(spack location -i /wa3)/lib \
-lmpi -lpmix -lprrte -lcuda -lcudart
*/
// /dqz: pmix intel compiler
// /wa3: prrte intel compiler

// salloc -p h01 -N1 -n2 --gres=gpu:2 mpirun -np 2 ./test_vmm_bcast
// salloc -p h01 -N1 -n4 --gres=gpu:4 mpirun -np 4 ./test_vmm_bcast

#include "mem_handle.hh"
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <tuple>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cfloat>

constexpr size_t BUF_SIZE = 32768;

#define CHKPT() \
    do { \
        fprintf(stderr, "[Rank %d] Checkpoint at " __FILE__ " line %d\n", rank, __LINE__); \
    } while(0)

auto mpi_start(int argc, char **argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_Comm comm = MPI_COMM_WORLD;
    int rk, sz;
    MPI_CHECK(MPI_Comm_rank(comm, &rk));
    MPI_CHECK(MPI_Comm_size(comm, &sz));
    return std::make_tuple(rk, sz);
}


void fill_rand(float *buf, size_t n) {
    for(size_t i = 0; i < n; i++) {
        buf[i] = rand() * 1. / RAND_MAX;
    }
}

bool check(size_t n, float *expected, float *actual) {
    bool unequal = false;
    for(size_t i = 0; i < n; i++) {
        auto failed = fabs(expected[i] - actual[i]) > 100 * FLT_EPSILON;
        // MPI_CHECK(failed);
        unequal |= failed;
    }
    return unequal;
}


__global__
void bcast_krnl(float *bcast_mc_ptr, const float *payload, size_t nelem) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= nelem) return;
    float val = payload[i];
    __asm__ __volatile__ (
        "multimem.st.relaxed.sys.global.f32 [%0], %1;" :: "l"(&bcast_mc_ptr[i]), "f"(val) : "memory"
    );
}

int main(int argc, char **argv) {
    auto [rank, world_size] = mpi_start(argc, argv);
    int dev_cnt;
    CUDART_CHECK(cudaGetDeviceCount(&dev_cnt));
    assert(dev_cnt == world_size);
    CUDART_CHECK(cudaSetDevice(rank));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    auto vmm_handle = new VmmMemHandle();
    vmm_handle->malloc(BUF_SIZE);
    vmm_handle->enableMulticast();

    // TEST BEGIN
    auto aligned_size = vmm_handle->getSize();
    size_t nelem = vmm_handle->getSize() / sizeof(float);

    float *test_buf = new float[nelem];
    if(rank == 0) {
        srand(time(0) + rank);
        fill_rand(test_buf, nelem);
        float *test_buf_d_root;
        CUDART_CHECK(cudaMalloc(&test_buf_d_root, aligned_size));
        CUDART_CHECK(cudaMemcpy(test_buf_d_root, test_buf, aligned_size, cudaMemcpyHostToDevice));
        bcast_krnl<<< (nelem + 255) / 256, 256>>>(reinterpret_cast<float *>(vmm_handle->getAPIBaseMC()), test_buf_d_root, nelem);
        CUDART_CHECK(cudaDeviceSynchronize());
        CUDART_CHECK(cudaFree(test_buf_d_root));
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    float *bcast_mc = new float[nelem];
    CUDA_CHECK(cuMemcpyDtoH(bcast_mc, vmm_handle->getAPIBaseLocal(), aligned_size));
    // TEST END

    MPI_CHECK(MPI_Bcast(test_buf, nelem, MPI_FLOAT, 0, MPI_COMM_WORLD));
    bool failed = check(nelem, test_buf, bcast_mc);
    if(failed) {
        printf("[RANK %d] Bcast Check failed.\n", rank);
    } else {
        printf("[RANK %d] Bcast Check succeed.\n", rank);
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    vmm_handle->discard();
    MPI_CHECK(MPI_Finalize());
    delete [] test_buf;
    delete [] bcast_mc;
    return 0;
}
