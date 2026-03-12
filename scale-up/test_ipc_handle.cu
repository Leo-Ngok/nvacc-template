//SPDX-License-Identifier: GPL-2.0

/*
nvcc test_ipc_handle.cu -o test_ipc_handle \
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

#define MPI_CHECK(status) \
    do {\
        int ss = (status); \
        if(ss != MPI_SUCCESS) { \
            fprintf(stderr, "MPI Error occured in " __FILE__ " line %d (" #status ") with code %d\n", __LINE__,  ss); \
            MPI_Abort(MPI_COMM_WORLD, ss); \
        } \
    } while(0)

#define CUDART_CHECK(status) \
    do {\
        cudaError_t ss = (status); \
        if(ss != cudaSuccess) {\
            fprintf(stderr, "CUDA RUNTIME API Error occured in " __FILE__ " line %d (" #status ") with code %d\n", __LINE__, ss); \
            MPI_Abort(MPI_COMM_WORLD, ss); \
        }\
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
    auto [rank, size] = mpi_start(argc, argv);
    char *buf_loc_d;
    char *buf_remote_d;
    
    int num_devices;
    CUDART_CHECK(cudaGetDeviceCount(&num_devices));
    assert(num_devices == size);
    auto peer = (rank + 1) % size;
    cudaSetDevice(rank);
    
    // CudaMallocHost(&buf_loc_h, 20);
    MPI_Barrier(MPI_COMM_WORLD);
    CUDART_CHECK(cudaMalloc(&buf_loc_d, 40));
    cudaIpcMemHandle_t buf_handle, buf_remote_handle;
    CUDART_CHECK(cudaIpcGetMemHandle(&buf_handle, buf_loc_d));
    MPI_CHECK(MPI_Sendrecv(
        buf_handle.reserved,        sizeof(buf_handle.reserved),        MPI_CHAR, peer, 0, 
        buf_remote_handle.reserved, sizeof(buf_remote_handle.reserved), MPI_CHAR, peer, 0, 
        MPI_COMM_WORLD, MPI_STATUS_IGNORE));

    CUDART_CHECK(cudaIpcOpenMemHandle(reinterpret_cast<void **>(&buf_remote_d), buf_remote_handle, 
    cudaIpcMemLazyEnablePeerAccess));


    

    char buf_loc_h[40], buf_remote_h[40];
    sprintf(buf_loc_h, "Shared buffer from rank %d\n", rank);
    CUDART_CHECK(cudaMemcpy(buf_loc_d, buf_loc_h, 40, cudaMemcpyHostToDevice));
    CUDART_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    CUDART_CHECK(cudaMemcpy(buf_remote_h, buf_remote_d, 40, cudaMemcpyDeviceToHost));
    CUDART_CHECK(cudaDeviceSynchronize());
    printf("Rank %d receives: %s", rank, buf_remote_h);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    CUDART_CHECK(cudaIpcCloseMemHandle(buf_remote_d));
    CUDART_CHECK(cudaFree(buf_loc_d));
    MPI_CHECK(MPI_Finalize());
    return 0;
}
