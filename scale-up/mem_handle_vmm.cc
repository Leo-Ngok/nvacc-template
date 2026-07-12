
#include "mem_handle.hh"
#include "mpi.h"
#include <algorithm>
#include <asm-generic/socket.h>
#include <cassert>
#include <cstddef>
#include <new>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>

constexpr bool kSupernodeSupported = false;

auto round_up(size_t sz, size_t gran) {
  return ((sz + gran - 1) / gran) * gran;
}

void VmmMemHandle::malloc(size_t bufsz) {
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size_));
  // cuMem:

  // Local Part
  // (0) Get (Mem) Granularity and align size with it.
  // ** Distinguish with cuMulticast Get Granularity
  // 1. Create
  size_t gran;
  CUmemAllocationProp prop{
      .type = CU_MEM_ALLOCATION_TYPE_PINNED,
      .requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
      .location = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = my_rank_}};
  CUDA_CHECK(cuMemGetAllocationGranularity(
      &gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  bufsz_ = round_up(bufsz, gran);
  CUDA_CHECK(cuMemCreate(&mhandle_, bufsz_, &prop, 0));
  // treat it as mmap ...
  buf_alt_ = mapHandle(mhandle_);

  // Allgather the handles ...
  rmhandle_ = new CUmemGenericAllocationHandle[world_size_];
  allGatherHandles(mhandle_, rmhandle_);

  // 4. Address Reserve, Map, Set Access
  rbuf_alt_ = new CUdeviceptr[world_size_];
  for (int r = 0; r < world_size_; r++) {
    if (r == my_rank_)
      continue;
    rbuf_alt_[r] = mapHandle(rmhandle_[r]);
  }
}

void VmmMemHandle::discard(void) {
  if (mc_enabled_) {
    CUdevice dev_handle;
    CUDA_CHECK(cuDeviceGet(&dev_handle, my_rank_));
    CUDA_CHECK(cuMulticastUnbind(mc_handle_, dev_handle, 0, bufsz_));
    discardHandle(mc_handle_, mcbuf_alt_);
  }
  for (int r = 0; r < world_size_; r++) {
    if (r == my_rank_)
      continue;
    discardHandle(rmhandle_[r], rbuf_alt_[r]);
  }
  delete[] rmhandle_;
  delete[] rbuf_alt_;
  discardHandle(mhandle_, buf_alt_);
}

void VmmMemHandle::populateAndSendHandle(
    CUmemGenericAllocationHandle mhandle) const {
  int fd;
  CUDA_CHECK(cuMemExportToShareableHandle(
      (void *)&fd, mhandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));

  // Payload Start
  char buf[CMSG_SPACE(sizeof(int))];
  auto src_rank = my_rank_;
  iovec io = {.iov_base = &src_rank, .iov_len = sizeof(src_rank)};
  msghdr msg = {.msg_iov = &io,
                .msg_iovlen = 1,
                .msg_control = buf,
                .msg_controllen = sizeof(buf)};
  cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  *((int *)CMSG_DATA(cmsg)) = fd;
  // Payload End

  sockaddr_un addr = {0};
  addr.sun_family = AF_UNIX;

  for (int r = 0; r < world_size_; r++) {
    if (r == my_rank_)
      continue;
    int sdsockfd = socket(AF_UNIX, SOCK_STREAM, 0);
    KRNL_CHECK(sdsockfd);
    snprintf(addr.sun_path, sizeof(addr.sun_path) - 1,
             "/tmp/vmm_mem_handle_acceptor_%d", r);
    KRNL_CHECK(
        connect(sdsockfd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)));
    KRNL_CHECK(sendmsg(sdsockfd, &msg, 0));
    KRNL_CHECK(close(sdsockfd));
  }
  KRNL_CHECK(close(fd));
}

void VmmMemHandle::recvHandles(RecvHandlesCtx &ctx) const {
  auto n = ctx.num_handles_;
  ctx.peer_fd_ = new int[n];
  ctx.peer_ranks_ = new int[n];
  std::fill(ctx.peer_ranks_, ctx.peer_ranks_ + n, 0);
  std::fill(ctx.peer_fd_, ctx.peer_fd_ + n, 0);

  sockaddr_un addr = {0};
  addr.sun_family = AF_UNIX;
  snprintf(addr.sun_path, sizeof(addr.sun_path) - 1,
           "/tmp/vmm_mem_handle_acceptor_%d", my_rank_);

  int inboundsockfd = socket(AF_UNIX, SOCK_STREAM, 0);
  KRNL_CHECK(inboundsockfd);
  unlink(addr.sun_path);
  KRNL_CHECK(
      bind(inboundsockfd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)));
  KRNL_CHECK(listen(inboundsockfd, n));

  // Capture arrays and parameters explicitly BY VALUE
  int *peer_ranks = ctx.peer_ranks_;
  int *peer_fd = ctx.peer_fd_;
  std::string socket_path = addr.sun_path;

  ctx.t =
      std::thread([inboundsockfd, n, peer_ranks, peer_fd, socket_path](void) {
        for (int t = 0; t < n; t++) {
          int recvsockfd = accept(inboundsockfd, nullptr, nullptr);
          KRNL_CHECK(recvsockfd);

          // Construct payload objects safely inside the worker thread stack
          // frame
          char buf[CMSG_SPACE(sizeof(int))] = {0};
          int sd_rank = 0;
          iovec io = {.iov_base = &sd_rank, .iov_len = sizeof(sd_rank)};
          msghdr msg = {.msg_iov = &io,
                        .msg_iovlen = 1,
                        .msg_control = buf,
                        .msg_controllen = sizeof(buf)};

          KRNL_CHECK(recvmsg(recvsockfd, &msg, 0));
          KRNL_CHECK(close(recvsockfd));

          peer_ranks[t] = sd_rank;
          auto src_fd_loc = CMSG_DATA(CMSG_FIRSTHDR(&msg));
          peer_fd[t] = *((int *)src_fd_loc);
        }
        KRNL_CHECK(close(inboundsockfd));
        unlink(socket_path.c_str());
      });
}

void VmmMemHandle::recvHandlesResolve(
    RecvHandlesCtx &ctx, CUmemGenericAllocationHandle *trg_handles) const {
  ctx.t.join();
  auto n = ctx.num_handles_;
  for (int it = 0; it < n; it++) {
    auto fd = ctx.peer_fd_[it];
    auto r = ctx.peer_ranks_[it];
    CUDA_CHECK(cuMemImportFromShareableHandle(
        &trg_handles[r], (void *)(uintptr_t)fd,
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    KRNL_CHECK(close(fd));
  }
  delete[] ctx.peer_fd_;
  delete[] ctx.peer_ranks_;
}

void VmmMemHandle::allGatherHandles(
    CUmemGenericAllocationHandle src_handle,
    CUmemGenericAllocationHandle *trg_handles) const {
  if constexpr (kSupernodeSupported) {
    CUmemFabricHandle fabricHandle;
    CUDA_CHECK(cuMemExportToShareableHandle(&fabricHandle, src_handle,
                                            CU_MEM_HANDLE_TYPE_FABRIC, 0));
    auto peerFabricHandles = new CUmemFabricHandle[world_size_];
    MPI_CHECK(MPI_Allgather(&fabricHandle, sizeof(CUmemFabricHandle), MPI_BYTE,
                            peerFabricHandles, sizeof(CUmemFabricHandle),
                            MPI_BYTE, MPI_COMM_WORLD));
    for (int r = 0; r < world_size_; r++) {
      if (r == my_rank_)
        continue;
      CUDA_CHECK(cuMemImportFromShareableHandle(
          &trg_handles[r], &peerFabricHandles[r], CU_MEM_HANDLE_TYPE_FABRIC));
    }
    trg_handles[my_rank_] = src_handle;
    delete[] peerFabricHandles;
    return;
  }
  RecvHandlesCtx exCtx;
  exCtx.num_handles_ = world_size_ - 1;
  recvHandles(exCtx);
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  // 1. Export to sharable handle (fd)
  // 2. Exchange fd via unix socket
  populateAndSendHandle(mhandle_);
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  // 3. Import from sharable handle to mem handle
  recvHandlesResolve(exCtx, trg_handles);
}

CUdeviceptr
VmmMemHandle::mapHandle(const CUmemGenericAllocationHandle mhandle) const {
  CUdeviceptr ptr_d;
  CUmemAccessDesc desc = {
      .location = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = my_rank_},
      .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
  CUDA_CHECK(cuMemAddressReserve(&ptr_d, bufsz_, 0, 0, 0));
  CUDA_CHECK(cuMemMap(ptr_d, bufsz_, 0, mhandle, 0));
  CUDA_CHECK(cuMemSetAccess(ptr_d, bufsz_, &desc, 1));
  return ptr_d;
}

void VmmMemHandle::discardHandle(CUmemGenericAllocationHandle mhandle,
                                 CUdeviceptr ptr_d) const {
  CUDA_CHECK(cuMemUnmap(ptr_d, bufsz_));
  CUDA_CHECK(cuMemAddressFree(ptr_d, bufsz_));
  CUDA_CHECK(cuMemRelease(mhandle));
}

void VmmMemHandle::enableMulticast(void) {
  if (my_rank_ == 0) {
    size_t gran;
    CUmulticastObjectProp prop{};
    prop.numDevices = world_size_;
    prop.size = bufsz_;
    prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    CUDA_CHECK(cuMulticastGetGranularity(&gran, &prop,
                                         CU_MULTICAST_GRANULARITY_RECOMMENDED));
    // TODO: Make it more portable
    assert(bufsz_ % gran == 0);
    CUDA_CHECK(cuMulticastCreate(&mc_handle_, &prop));
  }
  bcastHandles(mc_handle_, 0);
  CUdevice dev_handle;
  CUDA_CHECK(cuDeviceGet(&dev_handle, my_rank_));
  CUDA_CHECK(cuMulticastAddDevice(mc_handle_, dev_handle));
  CUDA_CHECK(cuMulticastBindMem(mc_handle_, 0, mhandle_, 0, bufsz_, 0));
  mcbuf_alt_ = mapHandle(mc_handle_);
  mc_enabled_ = true;
}

void VmmMemHandle::bcastHandles(CUmemGenericAllocationHandle &mhandle,
                                int src_rank) const {
  if constexpr (kSupernodeSupported) {
    CUmemFabricHandle fabricHandle;
    if (src_rank == my_rank_) {
      CUDA_CHECK(cuMemExportToShareableHandle(&fabricHandle, mhandle,
                                              CU_MEM_HANDLE_TYPE_FABRIC, 0));
    }
    MPI_CHECK(MPI_Bcast(&fabricHandle, sizeof(CUmemFabricHandle), MPI_BYTE,
                        src_rank, MPI_COMM_WORLD));
    if (src_rank != my_rank_) {
      CUDA_CHECK(cuMemImportFromShareableHandle(&mhandle, &fabricHandle,
                                                CU_MEM_HANDLE_TYPE_FABRIC));
    }
    return;
  }
  RecvHandlesCtx mcCtx;
  mcCtx.num_handles_ = 1;
  // TODO: Modify the implementation of recvHandlesResolve and remove this assertion.
  assert(src_rank == 0);
  if (my_rank_ != src_rank) {
    recvHandles(mcCtx);
  }
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  // 1. Export to sharable handle (fd)
  // 2. Exchange fd via unix socket
  if (my_rank_ == src_rank) {
    populateAndSendHandle(mhandle);
  }
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  if (my_rank_ != src_rank) {
    recvHandlesResolve(mcCtx, &mhandle);
  }
}