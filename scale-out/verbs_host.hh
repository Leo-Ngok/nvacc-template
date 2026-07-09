#ifndef VERBS_HOST_CTRL_HH
#define VERBS_HOST_CTRL_HH

#include <infiniband/verbs.h>

// #define USE_CUDART

#define MPI_CHECK(status) \
    do {\
        int ss = (status); \
        if(ss != MPI_SUCCESS) { \
            fprintf(stderr, "MPI Error occured in " __FILE__ " line %d (" #status ") with code %d\n", __LINE__,  ss); \
            cleanup(); \
            if(cleaned_up()) { MPI_Abort(MPI_COMM_WORLD, ss); } \
        } \
    } while(0)
#define IBV_CHECK(status) \
    do {\
        int ss = (status); \
        if(ss != 0) { \
            fprintf(stderr, "RDMA (verbs) Error occured in " __FILE__ " line %d (" #status ") with code %d\n", __LINE__,  ss); \
            cleanup(); \
            if(cleaned_up()) { MPI_Abort(MPI_COMM_WORLD, ss); } \
        } \
    } while(0)

#ifdef USE_CUDART
#define CUDART_CHECK(status) \
    do {\
        cudaError_t ss = (status); \
        if(ss != cudaSuccess) {\
            fprintf(stderr, "CUDA RUNTIME API Error occured in " __FILE__ " line %d (" #status ") with code %d\n", __LINE__, ss); \
            MPI_Abort(MPI_COMM_WORLD, ss); \
        }\
    } while(0)
#endif

void build(int argc, char **argv);
void cleanup(void);

size_t rank();
size_t world_size();

ibv_qp *get_peer_qp(size_t peer_rank);
ibv_cq *get_cq(void);

struct buffer_handle {
    void *buf;
    size_t size;

    uint32_t lkey;
    uint32_t *rkey;
    void **rbuf;
    ibv_mr *mr;

    int type;
};

int pair_malloc(size_t size, int type, buffer_handle &bufh);
void pair_free(buffer_handle &bufh);

// YOU ARE NOT SUPPOSED TO USE THESE APIs
// ibv_pd *get_pd(void);
bool cleaned_up(void);
#endif // VERBS_HOST_CTRL_HH