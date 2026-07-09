#ifndef DOCA_GPUNETIO_COMMON_H
#define DOCA_GPUNETIO_COMMON_H

#include <mpi.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <infiniband/verbs.h>
#include <host/doca_verbs.h>
#include <doca_gpunetio_high_level.h>

#define MPI_CHECK(status) \
    do {\
        int ss = (status); \
        if(ss != MPI_SUCCESS) { \
            fprintf(stderr, "MPI Error occured in " __FILE__ " line %d (" #status ") with code %d\n", __LINE__,  ss); \
            cleanup(); \
            if(cleaned_up()) { MPI_Abort(MPI_COMM_WORLD, ss); } \
        } \
    } while(0)

#define CUDA_CHECK(status) \
    do {\
        CUresult ss = (status); \
        if(ss != CUDA_SUCCESS) {\
            fprintf(stderr, "CUDA API Error occured in " __FILE__ " line %d (" #status ") with code %d\n", __LINE__, ss); \
            cleanup(); \
            if(cleaned_up()) { MPI_Abort(MPI_COMM_WORLD, ss); } \
        }\
    } while(0)

#define CUDART_CHECK(status) \
    do {\
        cudaError_t ss = (status); \
        if(ss != cudaSuccess) {\
            fprintf(stderr, "CUDA RUNTIME API Error occured in " __FILE__ " line %d (" #status ") with code %d\n", __LINE__, ss); \
            cleanup(); \
            if(cleaned_up()) { MPI_Abort(MPI_COMM_WORLD, ss); } \
        }\
    } while(0)

#define DOCA_CHECK(status) \
    do { \
        auto res = (status); \
        if (res != DOCA_SUCCESS) { \
            fprintf(stderr, "DOCA API Error occured in " __FILE__ " line %d (" #status ") with code %d\n", __LINE__, res); \
            cleanup(); \
            if(cleaned_up()) { MPI_Abort(MPI_COMM_WORLD, res); } \
        }\
        \
    } while(0)

#define LOG_PRINT(fmt, ...) \
    do { \
    fprintf(stderr, fmt, ##__VA_ARGS__); \
    fflush(stderr); \
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD)); \
    } while(0)

void build(int argc, char **argv);
void cleanup(void);

size_t rank(void);
size_t world_size(void);

class buffer_dev_handle {
public:
    buffer_dev_handle(size_t _bufsz);
    ~buffer_dev_handle();
    // use with care !!
    void *buf;
    uint32_t lkey;
    void **rbuf;
    uint32_t *rkey;

    size_t bufsz;
private:
    ibv_mr *mr;
};

doca_gpu_verbs_qp_hl *get_peer_qp(size_t peer_rank);
bool cleaned_up(void);
#endif // DOCA_GPUNETIO_COMMON_H
