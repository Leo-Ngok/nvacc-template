/*
nvcc test_vmm_allreduce.cu -o test_vmm_allreduce \
-gencode arch=compute_90,code=sm_90 -std=c++20 -O3 \
-I$(mpicxx -showme:incdirs) \
-L$(mpicxx -showme:libdirs) \
-L$(spack location -i /dqz)/lib \
-L$(spack location -i /wa3)/lib \
-lmpi -lpmix -lprrte -lcuda -lcudart
*/
// /dqz: pmix intel compiler
// /wa3: prrte intel compiler

// salloc -p h01 -N1 -n2 --gres=gpu:2 mpirun -np 2 ./test_vmm_allreduce
// salloc -p h01 -N1 -n4 --gres=gpu:4 mpirun -np 4 ./test_vmm_allreduce

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

#define MPI_CHECK(status) \
    do {\
        int ss = (status); \
        if(ss != MPI_SUCCESS) { \
            fprintf(stderr, "MPI Error occured in " __FILE__ " line %d (" #status ") with code %d\n", __LINE__,  ss); \
            MPI_Abort(MPI_COMM_WORLD, ss); \
        } \
    } while(0)

#define CUDA_CHECK(status) \
    do {\
        CUresult ss = (status); \
        if(ss != CUDA_SUCCESS) {\
            fprintf(stderr, "CUDA API Error occured in " __FILE__ " line %d (" #status ") with code %d\n", __LINE__, ss); \
            MPI_Abort(MPI_COMM_WORLD, ss); \
        }\
    } while(0)

#define CUDART_CHECK(status) \
    do {\
        cudaError_t ss = (status); \
        if(ss != cudaSuccess) {\
            fprintf(stderr, "CUDA RUNTIME API Error occured in " __FILE__ " line %d (" #status ") with code %d\n", __LINE__, ss); \
            MPI_Abort(MPI_COMM_WORLD, ss); \
        }\
    } while(0)
#define CHKPT() \
    do { \
        fprintf(stderr, "[Rank %d] Checkpoint at " __FILE__ " line %d\n", rank, __LINE__); \
    } while(0)
__forceinline__
auto round_up(size_t sz, size_t gran) {
    return ((sz + gran - 1) / gran) * gran;
}

auto mpi_start(int argc, char **argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_Comm comm = MPI_COMM_WORLD;
    int rk, sz;
    MPI_CHECK(MPI_Comm_rank(comm, &rk));
    MPI_CHECK(MPI_Comm_size(comm, &sz));
    return std::make_tuple(rk, sz);
}

auto vmm_alloc_gran(int rank) {
    size_t gran;
    CUmemAllocationProp prop{};
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = rank;
    CUDA_CHECK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    return std::make_tuple(gran, prop);
}

/// 1. cuMemAddressReserve
/// 2. cuMemMap
/// 3. cuMemSetAccess
auto vmm_mem_with_handle(const CUmemGenericAllocationHandle &mem_handle, size_t sz, int dev_id) {
    CUdeviceptr ptr_d;
    CUmemAccessDesc desc;
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = dev_id;

    CUDA_CHECK(cuMemAddressReserve(&ptr_d, sz, /*alignment*/0, /*hint addr*/0, /*flags*/0));
    CUDA_CHECK(cuMemMap(ptr_d, sz, /*offset*/ 0, mem_handle, /*flags*/0));
    CUDA_CHECK(cuMemSetAccess(ptr_d, sz, &desc, /*desc count*/1));
    return ptr_d;
}

auto vmm_mem_cleanup(CUmemGenericAllocationHandle &mem_handle, CUdeviceptr &ptr_d, size_t sz) {
    CUDA_CHECK(cuMemUnmap      (ptr_d, sz));
    CUDA_CHECK(cuMemAddressFree(ptr_d, sz));
    CUDA_CHECK(cuMemRelease(mem_handle));
}

void send_fd(int sock, int fd) {
    struct msghdr msg = {0};
    struct cmsghdr *cmsg;
    char buf[CMSG_SPACE(sizeof(int))];
    char dummy = 'A';
    struct iovec io = { .iov_base = &dummy, .iov_len = 1 };

    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    msg.msg_control = buf;
    msg.msg_controllen = sizeof(buf);

    cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    *((int *)CMSG_DATA(cmsg)) = fd;

    if (sendmsg(sock, &msg, 0) < 0) perror("sendmsg");
}

// Helper to receive a File Descriptor via Unix Socket
int recv_fd(int sock) {
    struct msghdr msg = {0};
    struct cmsghdr *cmsg;
    char buf[CMSG_SPACE(sizeof(int))];
    char dummy;
    struct iovec io = { .iov_base = &dummy, .iov_len = 1 };

    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    msg.msg_control = buf;
    msg.msg_controllen = sizeof(buf);

    if (recvmsg(sock, &msg, 0) < 0) perror("recvmsg");

    cmsg = CMSG_FIRSTHDR(&msg);
    return *((int *)CMSG_DATA(cmsg));
}

auto fwd_mem_handle(
          CUmemGenericAllocationHandle &recv_mem_handle, 
    const CUmemGenericAllocationHandle &send_mem_handle,
    int send_rank, int recv_rank, int rank) {
    
    // // 1. Exit early if this rank is not part of the pair
    // if (rank != send_rank && rank != recv_rank) {
    //     return;
    // }
    char sock_path[128];
    sprintf(sock_path, "/tmp/cuda_vmm_socket_%d_to_%d", send_rank, recv_rank);
    int sync_tag = 1024; // Use a specific tag for VMM sync
    int dummy = 1;

    // --- RECEIVER LOGIC ---
    if (rank == recv_rank) {
        int serv_sock = socket(AF_UNIX, SOCK_STREAM, 0);
        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path)-1);

        // remove possible duplicates
        unlink(sock_path);

        if (bind(serv_sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) perror("bind");
        listen(serv_sock, 1);

        // Signal the sender that the socket is ready for connection
        MPI_CHECK(MPI_Send(&dummy, 1, MPI_INT, send_rank, sync_tag, MPI_COMM_WORLD));

        int conn_sock = accept(serv_sock, NULL, NULL);
        int fd = recv_fd(conn_sock);
        
        CUDA_CHECK(cuMemImportFromShareableHandle(&recv_mem_handle, (void*)(uintptr_t)fd, 
                                                 CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

        // Wait for sender to finish before cleaning up the socket file
        MPI_CHECK(MPI_Recv(&dummy, 1, MPI_INT, send_rank, sync_tag + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

        close(fd);
        close(conn_sock);
        close(serv_sock);
        // revoke the comm socket
        unlink(sock_path);
    } 
    
    // --- SENDER LOGIC ---
    if (rank == send_rank) {
        int fd;
        CUDA_CHECK(cuMemExportToShareableHandle((void*)&fd, send_mem_handle, 
                                               CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));

        // Wait for receiver to signal that the socket is bound and listening
        MPI_CHECK(MPI_Recv(&dummy, 1, MPI_INT, recv_rank, sync_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

        int client_sock = socket(AF_UNIX, SOCK_STREAM, 0);
        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path)-1);

        if (connect(client_sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("connect");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    
        send_fd(client_sock, fd);
        
        // Signal receiver that we are done with the socket
        MPI_CHECK(MPI_Send(&dummy, 1, MPI_INT, recv_rank, sync_tag + 1, MPI_COMM_WORLD));

        close(fd);
        close(client_sock);
    }
    
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
}

auto vmm_construct_multicast_handles(int rank, int world_size, size_t buf_size) {
    size_t gran;
    CUmulticastObjectProp prop {};
    prop.numDevices = world_size;
    prop.size = buf_size;
    prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    CUDA_CHECK(cuMulticastGetGranularity(&gran, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    // TODO: Make it more portable
    assert(buf_size % gran == 0);


    // Build and share Multicast handle.
    CUmemGenericAllocationHandle mc_handle, mc_recv_handle;
    if(rank == 0) {
        CUDA_CHECK(cuMulticastCreate(&mc_handle, &prop));
    }
    for(int r = 1; r < world_size; r++) {
        fwd_mem_handle(mc_recv_handle, mc_handle, 0, r, rank);
    }
    if(rank != 0) mc_handle = mc_recv_handle;

    CUdevice dev_handle;
    CUDA_CHECK(cuDeviceGet(&dev_handle, rank));
    CUDA_CHECK(cuMulticastAddDevice(mc_handle, dev_handle));

    return mc_handle;
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

    auto [gran, mem_prop] = vmm_alloc_gran(rank);
    auto aligned_size = round_up(BUF_SIZE, gran);

    CUmemGenericAllocationHandle local_handle, mc_handle;
    CUDA_CHECK(cuMemCreate(&local_handle, aligned_size, &mem_prop, /*flags*/0));
    mc_handle = vmm_construct_multicast_handles(rank, world_size, aligned_size);

    auto d_buf_loc = vmm_mem_with_handle(local_handle, aligned_size, rank);
    auto d_buf_mc  = vmm_mem_with_handle(mc_handle,    aligned_size, rank);

    CUDA_CHECK(cuMulticastBindMem(mc_handle, 0, local_handle, 0, aligned_size, 0));


    // TEST BEGIN
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
    CUdevice dev_handle;
    CUDA_CHECK(cuDeviceGet(&dev_handle, rank));
    CUDA_CHECK(cuMulticastUnbind(mc_handle, dev_handle, 0, aligned_size));
    vmm_mem_cleanup(local_handle, d_buf_loc, aligned_size);
    vmm_mem_cleanup(mc_handle, d_buf_mc, aligned_size);
    MPI_CHECK(MPI_Finalize());
    delete [] test_buf;
    delete [] red_ref;
    delete [] red_mc;
    return 0;
}