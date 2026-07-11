//SPDX-License-Identifier: GPL-2.0

/*
nvcc test_vmm_handle.cu mem_handle_vmm.cc -o test_vmm_handle \
-std=c++20 -O3 \
-I$(mpicxx -showme:incdirs) \
-L$(mpicxx -showme:libdirs) \
-L$(spack location -i /dqz)/lib \
-L$(spack location -i /wa3)/lib \
-lmpi -lpmix -lprrte -lcuda -lcudart
*/
// /dqz: pmix intel compiler
// /wa3: prrte intel compiler

// salloc -p h01 -N1 -n2 --gres=gpu:2 mpirun -np 2 ./test_vmm_handle

#include "mem_handle.hh"
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <tuple>
#include <cassert>

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


int main(int argc, char **argv) {
    auto [rank, world_size] = mpi_start(argc, argv);
    int dev_cnt;
    CUDART_CHECK(cudaGetDeviceCount(&dev_cnt));
    assert(dev_cnt == world_size);
    CUDART_CHECK(cudaSetDevice(rank));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    auto remote_rank = (rank + 1) % world_size;

    auto vmm_handle = new VmmMemHandle();
    vmm_handle->malloc(BUF_SIZE);

    char buf_loc_h[50], buf_remote_h[50];
    snprintf(buf_loc_h, sizeof(buf_loc_h), "[2ND] VMM LOCAL WRITE (= %d)", rank);
    snprintf(buf_remote_h, sizeof(buf_remote_h), "[1 ST] VMM REMOTE WRITE (%d -> %d)", rank, remote_rank);

    CUDART_CHECK(cudaMemcpy((char *)vmm_handle->getBaseLocal() + 50, buf_loc_h, sizeof(buf_loc_h), cudaMemcpyHostToDevice));
    CUDART_CHECK(cudaMemcpy(vmm_handle->getBasePeer(remote_rank), buf_remote_h, sizeof(buf_remote_h), cudaMemcpyHostToDevice));
    
    CUDART_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    
    memset(buf_loc_h, 0, sizeof(buf_loc_h));
    memset(buf_remote_h, 0, sizeof(buf_remote_h));
    
    CUDART_CHECK(cudaMemcpy(buf_remote_h, (char *)vmm_handle->getBasePeer(remote_rank) + 50, sizeof(buf_remote_h), cudaMemcpyDeviceToHost));
    CUDART_CHECK(cudaMemcpy(buf_loc_h, vmm_handle->getBaseLocal(), sizeof(buf_loc_h), cudaMemcpyDeviceToHost));
    printf("[RANK %d] Buffer is now (1 SELF): %s, (2 NEXT): %s\n", rank, buf_loc_h, buf_remote_h);


    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    vmm_handle->discard();
    delete vmm_handle;
    MPI_CHECK(MPI_Finalize());
    return 0;
}
