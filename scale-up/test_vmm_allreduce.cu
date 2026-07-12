//SPDX-License-Identifier: GPL-2.0

/*
nvcc test_vmm_allreduce.cu mem_handle_vmm.cc -o test_vmm_allreduce \
-gencode arch=compute_90,code=sm_90 -std=c++20 -O3 \
-I/home/fit/zhaijdyzq/spack/opt/spack/linux-broadwell/openmpi-5.0.9-3bi2uxl42fzehc5tkenpzvkartxxgwc6/include \
-L/home/fit/zhaijdyzq/spack/opt/spack/linux-broadwell/openmpi-5.0.9-3bi2uxl42fzehc5tkenpzvkartxxgwc6/lib \
-L/home/fit/zhaijdyzq/spack/opt/spack/linux-broadwell/pmix-6.0.0-dqzkwte3b5sotgatuiiqn6au6nj5bjut/lib \
-L/home/fit/zhaijdyzq/spack/opt/spack/linux-broadwell/prrte-4.0.0-wa3gvjktvvcglz7tq6vbqyttcnlblks3/lib \
-lmpi -lpmix -lprrte -lcuda -lcudart
*/
// /dqz: pmix intel compiler
// /wa3: prrte intel compiler

// salloc -p h01 -N1 -n2 --gres=gpu:2 mpirun -np 2 ./test_vmm_allreduce
// salloc -p h01 -N1 -n4 --gres=gpu:4 mpirun -np 4 ./test_vmm_allreduce

#include "mem_handle.hh"
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <tuple>
#include <cassert>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
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
void allreduce_krnl(float *red_sto, const float *red_mc_ptr, size_t nelem) {
    float sum_val;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= nelem) return;
    __asm__ __volatile__ (
        "multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];" : 
        "=f"(sum_val) : "l"(&(red_mc_ptr[i])) : "memory");
    red_sto[i] = sum_val;
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
    auto d_buf_loc = vmm_handle->getAPIBaseLocal();
    auto d_buf_mc = vmm_handle->getAPIBaseMC();
    size_t nelem = aligned_size / sizeof(float);
    float *test_buf = new float[nelem];
    srand(time(0) + rank);
    fill_rand(test_buf, nelem);
    CUDA_CHECK(cuMemcpyHtoD(d_buf_loc, test_buf, aligned_size));
    // allreduce krnl here ...
    float *red_mc_d; CUDART_CHECK(cudaMalloc(&red_mc_d, aligned_size));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    allreduce_krnl<<< (nelem + 255) / 256, 256>>>(red_mc_d, reinterpret_cast<float *>(d_buf_mc), nelem);
    CUDART_CHECK(cudaDeviceSynchronize());
    float *red_mc  = new float[nelem];
    // CUDA_CHECK(cuMemcpyDtoH(red_mc, d_buf_loc, aligned_size));
    CUDART_CHECK(cudaMemcpy(red_mc, red_mc_d, aligned_size, cudaMemcpyDeviceToHost));
    CUDART_CHECK(cudaFree(red_mc_d));
    float *red_ref = new float[nelem];
    MPI_CHECK(MPI_Allreduce(test_buf, red_ref, nelem, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
    // TEST END


    bool failed = check(nelem, red_ref, red_mc);
    if(failed) {
        printf("[RANK %d] All Reduce Check failed.\n", rank);
    } else {
        printf("[RANK %d] All Reduce Check succeed.\n", rank);
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    vmm_handle->discard();
    MPI_CHECK(MPI_Finalize());
    delete [] test_buf;
    delete [] red_ref;
    delete [] red_mc;
    return 0;
}