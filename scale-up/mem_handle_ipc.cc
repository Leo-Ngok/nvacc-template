#include "mem_handle.hh"
#include <algorithm>

void IpcMemHandle::malloc(size_t bufsz) {
    bufsz_ = bufsz;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size_));
    rbuf_ = new void*[world_size_];
    std::fill(rbuf_, rbuf_ + world_size_, nullptr);

    CUDART_CHECK(cudaMalloc(&buf_, bufsz_));
    
    cudaIpcMemHandle_t handle_loc;
    CUDART_CHECK(cudaIpcGetMemHandle(&handle_loc, buf_));
    auto rhandles_ = new cudaIpcMemHandle_t[world_size_];
    MPI_CHECK(MPI_Allgather(&handle_loc, sizeof(cudaIpcMemHandle_t), MPI_BYTE, rhandles_, sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int r = 0; r < world_size_; r++) {
        if (r == my_rank_) continue;
        CUDART_CHECK(cudaIpcOpenMemHandle(&rbuf_[r], rhandles_[r], cudaIpcMemLazyEnablePeerAccess));
    }
    delete [] rhandles_;
}

void IpcMemHandle::discard(void) {
    for(int r = 0; r < world_size_; r++) {
        if (r == my_rank_) continue;
        CUDART_CHECK(cudaIpcCloseMemHandle(rbuf_[r]));
    }
    delete [] rbuf_; rbuf_ = nullptr;
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDART_CHECK(cudaFree(buf_));
}





