//SPDX-License-Identifier: GPL-2.0

/*
nvcc test_ipc_handle.cu mem_handle_ipc.cc -o test_ipc_handle \
-std=c++20 -O3 \
-I$(mpicxx -showme:incdirs) \
-L$(mpicxx -showme:libdirs) \
-L$(spack location -i /dqz)/lib \
-L$(spack location -i /wa3)/lib \
-lmpi -lpmix -lprrte -lcuda -lcudart
*/

// salloc -p h01 -N1 -n2 --gres=gpu:2 mpirun -np 2 ./test_ipc_handle

#include <cuda_runtime.h>
#include <cuda.h>
#include <mpi.h>
#include <stdio.h>
#include <tuple>
#include <cassert>
#include "mem_handle.hh"

auto mpi_start(int argc, char **argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_Comm comm = MPI_COMM_WORLD;
    int rk, sz;
    MPI_CHECK(MPI_Comm_rank(comm, &rk));
    MPI_CHECK(MPI_Comm_size(comm, &sz));
    return std::make_tuple(rk, sz);
}

int main(int argc, char **argv) {
    auto [rank, size] = mpi_start(argc, argv);    
    int num_devices;
    CUDART_CHECK(cudaGetDeviceCount(&num_devices));
    assert(num_devices == size);
    auto peer = (rank + 1) % size;
    CUDART_CHECK(cudaSetDevice(rank));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));


    auto ipc_handle = new IpcMemHandle();
    ipc_handle->malloc(100);

    char buf_loc_h[50], buf_remote_h[50];
    snprintf(buf_loc_h, sizeof(buf_loc_h), "[2ND] IPC LOCAL WRITE (= %d)", rank);
    snprintf(buf_remote_h, sizeof(buf_remote_h), "[1 ST] IPC REMOTE WRITE (%d -> %d)", rank, peer);

    CUDART_CHECK(cudaMemcpy((char *)ipc_handle->getBaseLocal() + 50, buf_loc_h, sizeof(buf_loc_h), cudaMemcpyHostToDevice));
    CUDART_CHECK(cudaMemcpy(ipc_handle->getBasePeer(peer), buf_remote_h, sizeof(buf_remote_h), cudaMemcpyHostToDevice));
    
    CUDART_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    
    memset(buf_loc_h, 0, sizeof(buf_loc_h));
    memset(buf_remote_h, 0, sizeof(buf_remote_h));
    
    CUDART_CHECK(cudaMemcpy(buf_remote_h, (char *)ipc_handle->getBasePeer(peer) + 50, sizeof(buf_remote_h), cudaMemcpyDeviceToHost));
    CUDART_CHECK(cudaMemcpy(buf_loc_h, ipc_handle->getBaseLocal(), sizeof(buf_loc_h), cudaMemcpyDeviceToHost));
    printf("[RANK %d] Buffer is now (1 SELF): %s, (2 NEXT): %s\n", rank, buf_loc_h, buf_remote_h);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    ipc_handle->discard();
    delete ipc_handle;
    MPI_CHECK(MPI_Finalize());
    return 0;
}
