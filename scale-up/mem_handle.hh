#ifndef MEM_HANDLE_HH
#define MEM_HANDLE_HH

// #include <concepts>
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda.h>
#include <mpi.h>
#include <cstdio>
#include <thread>
#include <sys/socket.h>
#include <sys/un.h>
// template<typename ImplT>
// concept IMemHandle = requires(ImplT handle, size_t sz, size_t peer_rank_buf) {
//     { handle.malloc(sz) } -> std::same_as<void>;
//     { handle.free() } -> std::same_as<void>;
//     { handle.getBaseLocal() } -> std::same_as<void *>;
//     { handle.getBasePeer(peer_rank_buf) } -> std::same_as<void *>;
// };

// enum class HandleType {
//     IPC, VMM
// };

// template <HandleType Type>
// struct HandleFactory;

class IpcMemHandle {
public:
    void malloc(size_t sz);
    void discard(void);
    inline void *getBaseLocal(void) const {
        return buf_;
    }
    inline void *getBasePeer(size_t rank) const {
        return rbuf_[rank];
    }
    inline size_t getSize(void) const {return bufsz_;}
private:
    void *buf_ = nullptr;
    void **rbuf_ = nullptr;
    size_t bufsz_ = 0;
    int my_rank_; 
    int world_size_;
};

class VmmMemHandle {
public:
    void malloc(size_t sz);
    void discard(void);
    inline void *getBaseLocal(void) const {
        return buf_;
    }
    inline void *getBasePeer(size_t rank) const {
        return rbuf_[rank];
    }
    inline CUdeviceptr getAPIBaseLocal(void) const {
        return buf_alt_;
    }
    inline CUdeviceptr getAPIBasePeer(size_t rank) const {
        return rbuf_alt_[rank];
    }
    void enableMulticast(void);
    inline void *getBaseMC(void) const {
        if (!mc_enabled_) {
            fprintf(stderr, "Accessing Multicast Pointer without MC enabled.\n");
        }
        return mcbuf_;
    }
    inline CUdeviceptr getAPIBaseMC(void) const {
        if (!mc_enabled_) {
            fprintf(stderr, "Accessing Multicast Pointer without MC enabled.\n");
        }
        return mcbuf_alt_;
    }
    inline size_t getSize(void) const {return bufsz_;}
private:
    struct RecvHandlesCtx {
        std::thread t;
        int num_handles_;
        int *peer_ranks_;
        int *peer_fd_;

        int inboundsockfd;
        char buf[CMSG_SPACE(sizeof(int))];
        int sd_rank;
        iovec io;
        msghdr msg;
        sockaddr_un addr;
    };
    CUdeviceptr mapHandle(const CUmemGenericAllocationHandle mhandle) const;
    void discardHandle(CUmemGenericAllocationHandle mhandle, CUdeviceptr ptr_d) const;
    void populateAndSendHandle(CUmemGenericAllocationHandle mhandle) const;
    void recvHandles(RecvHandlesCtx &ctx) const;
    void recvHandlesResolvePeer(RecvHandlesCtx &ctx);
    void recvHandlesResolveMC(RecvHandlesCtx &ctx);
    void *buf_ = nullptr;
    CUdeviceptr buf_alt_ = 0;
    void **rbuf_ = nullptr;
    CUdeviceptr *rbuf_alt_ = nullptr;
    size_t bufsz_ = 0;
    int my_rank_; 
    int world_size_;
    CUmemGenericAllocationHandle mhandle_;
    CUmemGenericAllocationHandle *rmhandle_ = nullptr;

    bool mc_enabled_ = false;
    CUmemGenericAllocationHandle mc_handle_;
    void *mcbuf_ = nullptr;
    CUdeviceptr mcbuf_alt_;
};

// template<>
// struct HandleFactory<HandleType::IPC> {
//     static IpcMemHandle create(size_t sz) { 
//         auto handle = IpcMemHandle(); 
//         handle.malloc(sz);
//         return handle;
//     }
// };

// template<>
// struct HandleFactory<HandleType::VMM> {
//     static VmmMemHandle create(size_t sz) { 
//         auto handle = VmmMemHandle(); 
//         handle.malloc(sz);
//         return handle;
//     }
// };

// void foo(IMemHandle auto& handle) {
//     auto buf = handle.getBaseLocal();
// }

// void ttest() {
//     auto handle = HandleFactory<HandleType::VMM>::create(100);
// }

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
#define KRNL_CHECK(status) \
    do {\
        int ss = (status); \
        if(ss < 0) {\
            fprintf(stderr, "LIBC Call Error occured in " __FILE__ " line %d (" #status ") with code %d\n", __LINE__, ss); \
            MPI_Abort(MPI_COMM_WORLD, ss); \
        }\
    } while(0)
#endif // MEM_HANDLE_HH
