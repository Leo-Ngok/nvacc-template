#include "doca_gpunetio_verbs_def.h"
#include "verbs_dev.hh"
#include <cstdint>
#include <cstdio>
#include <mpi.h>
#include <unistd.h>
#include <vector>
// #include <unordered_map>
#include <doca_gpunetio.h>
#include <doca_gpunetio_high_level.h>
#include <doca_verbs.h>
#include <infiniband/verbs.h>

bool cleanup_pd = false;
bool cleanup_mr = false;

struct verbs_dev_context {
  size_t rank;
  size_t world_size;

  // Part 1. Context Management
  // Local GPU Context managed by DOCA
  doca_gpu *doca_ctx = nullptr;

  // HCA context from RDMA core (verbs)
  ibv_context *ib_ctx = nullptr;
  // Protection Domain bookkeeping MR and QP
  ibv_pd *ib_pd = nullptr;
  // // Local ID of peer HCA that the process acquires
  // // Currently RoCE not supported, assuming IB used.
  // std::vector<uint16_t> peer_lid;
  // // Queue Pair Number of Peer process instance
  // std::vector<uint32_t> peer_qpn;
  std::vector<doca_gpu_verbs_qp_hl *> peer_qp;

  size_t pgsz;

  // relate GPU VRAM with HCA control
  bool dmabuf_supported = false;
  bool peermem_supported = false;

  // cleanup related
  // prunes multiple triggering
  bool has_cleanup = false;
  // exit only context cleared
  bool cleanup_completed = false;
};

verbs_dev_context ctx;

static doca_gpu_verbs_qp_hl *create_qp(void) {
  // TODO: Replace with lower level interface to move CQ / SRQ out.
  doca_gpu_verbs_qp_hl *qp = nullptr;
  doca_gpu_verbs_qp_init_attr_hl qp_init = {
      .gpu_dev = ctx.doca_ctx,
      .ibpd = ctx.ib_pd,
      .sq_nwqe =
          2048, // TODO: BAD PRACTICE TO HARDWIRE IT, MAKE IT PORTABLE !!!
      .nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY,
      .mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT,
      .send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR,
      .cq_collapsed = false, // may have to set it to true for latency
  };
  DOCA_CHECK(doca_gpu_verbs_create_qp_hl(&qp_init, &qp));
  return qp;
}

static void connect_qp(doca_verbs_qp *qp, uint16_t dlid, uint32_t dst_qpn) {
  doca_verbs_qp_attr *verbs_qp_attr = nullptr;
  DOCA_CHECK(doca_verbs_qp_attr_create(&verbs_qp_attr));
  doca_verbs_ah_attr *verbs_ah_attr = nullptr;
  DOCA_CHECK(doca_verbs_ah_attr_create(ctx.ib_ctx, &verbs_ah_attr));

  // Construct the QP attributes for state transitions
  // Refer to nvshmem/src/modules/transport/ibgda/ibgda.cpp
  // Also refer to src/doca_verbs_qp.cpp line 123
  // Also include/host/doca_verbs.h
  // RST->INIT
  DOCA_CHECK(doca_verbs_qp_attr_set_next_state(verbs_qp_attr,
                                               DOCA_VERBS_QP_STATE_INIT));
  DOCA_CHECK(doca_verbs_qp_attr_set_pkey_index(verbs_qp_attr, 0));
  DOCA_CHECK(doca_verbs_qp_attr_set_port_num(verbs_qp_attr, 1));
  DOCA_CHECK(doca_verbs_qp_attr_set_allow_remote_write(verbs_qp_attr, 1));
  DOCA_CHECK(doca_verbs_qp_attr_set_allow_remote_read(verbs_qp_attr, 1));
  DOCA_CHECK(doca_verbs_qp_attr_set_allow_remote_atomic(
      verbs_qp_attr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC));
  DOCA_CHECK(doca_verbs_qp_modify(
      qp, verbs_qp_attr,
      DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_PKEY_INDEX |
          DOCA_VERBS_QP_ATTR_PORT_NUM | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
          DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ));

  // INIT->RTR
  DOCA_CHECK(doca_verbs_qp_attr_set_next_state(verbs_qp_attr,
                                               DOCA_VERBS_QP_STATE_RTR));
  DOCA_CHECK(doca_verbs_qp_attr_set_path_mtu(verbs_qp_attr,
                                             DOCA_VERBS_MTU_SIZE_4K_BYTES));
  DOCA_CHECK(doca_verbs_qp_attr_set_rq_psn(verbs_qp_attr, 0));
  DOCA_CHECK(doca_verbs_qp_attr_set_dest_qp_num(verbs_qp_attr, dst_qpn));
  // pack infiniband lid here.
  // Wrong: attach the AH object only after all AH fields are populated. If the
  // setter snapshots the object, attaching it here captures the default AH.
  // DOCA_CHECK(doca_verbs_qp_attr_set_ah_attr(verbs_qp_attr, verbs_ah_attr));
  /*-->*/ DOCA_CHECK(doca_verbs_ah_attr_set_dlid(verbs_ah_attr, dlid));
  /*-->*/ DOCA_CHECK(doca_verbs_ah_attr_set_addr_type(
      verbs_ah_attr, DOCA_VERBS_ADDR_TYPE_IB_NO_GRH));
  /*-->*/ DOCA_CHECK(doca_verbs_ah_attr_set_sl(verbs_ah_attr, 0));
  DOCA_CHECK(doca_verbs_qp_attr_set_ah_attr(verbs_qp_attr, verbs_ah_attr));
  // Max Dest RD Atomic ??
  // doca_verbs_qp_attr_set_max_dest_rd_atomic(verbs_qp_attr, 1);
  DOCA_CHECK(doca_verbs_qp_attr_set_min_rnr_timer(verbs_qp_attr, 1));
  DOCA_CHECK(doca_verbs_qp_modify(
      qp, verbs_qp_attr,
      DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
          DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU |
          DOCA_VERBS_QP_ATTR_AH_ATTR | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER /*|
DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC This is optional in DOCA GPUNetIO, but
mandatory in RDMA Core (verbs) */
      ));

  // RTR->RTS
  DOCA_CHECK(doca_verbs_qp_attr_set_next_state(verbs_qp_attr,
                                               DOCA_VERBS_QP_STATE_RTS));
  DOCA_CHECK(doca_verbs_qp_attr_set_sq_psn(verbs_qp_attr, 0));
  // Likewise, Max Read atomic is not needed
  // DOCA_CHECK(doca_verbs_qp_attr_set_max_rd_atomic (verbs_qp_attr, 1));
  DOCA_CHECK(doca_verbs_qp_attr_set_ack_timeout(verbs_qp_attr, 14));
  DOCA_CHECK(doca_verbs_qp_attr_set_retry_cnt(verbs_qp_attr, 7));
  DOCA_CHECK(doca_verbs_qp_attr_set_rnr_retry(verbs_qp_attr, 1));
  DOCA_CHECK(doca_verbs_qp_modify(
      qp, verbs_qp_attr,
      DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
          DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
          DOCA_VERBS_QP_ATTR_RNR_RETRY /*|
DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC*/
      ));
  DOCA_CHECK(doca_verbs_ah_attr_destroy(verbs_ah_attr));
  DOCA_CHECK(doca_verbs_qp_attr_destroy(verbs_qp_attr));
}

// you are supposed to create one app instance per node only!!
void build(int argc, char **argv) {
  MPI_CHECK(MPI_Init(&argc, &argv));
  int rank, size;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  ctx.rank = rank;
  ctx.world_size = size;
  CUDART_CHECK(cudaFree(0));
  LOG_PRINT(
      "WARNING: Currently assuming one instance per Node is created, i.e. -N k "
      "-n k when you srun / salloc, so cudaDevice hardwired to 0\n");

  CUDART_CHECK(cudaSetDevice(0));
  char dev_pcie_bus[13];
  CUDART_CHECK(cudaDeviceGetPCIBusId(dev_pcie_bus, sizeof(dev_pcie_bus), 0));

  // 1. Setup DOCA context
  DOCA_CHECK(doca_gpu_create(dev_pcie_bus, &ctx.doca_ctx));

  // 2. Set Infiniband Context from RDMA verbs
  struct ibv_device **dev_list = ibv_get_device_list(nullptr);
  ctx.ib_ctx = ibv_open_device(dev_list[0]); // Pick first HCA (mlx5_0)
  ibv_free_device_list(dev_list);
  ctx.ib_pd = ibv_alloc_pd(ctx.ib_ctx);
  // Put Queue Pair and Completion Queue on GPU VRAM buffer, not on CPU RAM
  // 3. Create Queue Pair
  doca_gpu_verbs_qp_init_attr_hl qp_init = {
      .gpu_dev = ctx.doca_ctx,
      .ibpd = ctx.ib_pd,
      .sq_nwqe =
          2048, // TODO: BAD PRACTICE TO HARDWIRE IT, MAKE IT PORTABLE !!!
      .nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY,
      .mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT,
      .send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR,
      .cq_collapsed = false, // may have to set it to true for latency
  };

  // 1. Get LID (from port attr), will be sent to peer to choose qp for remote
  // do work on.
  // 2. Get Queue Pair number for remote to choose.
  uint32_t *lids = new uint32_t[size];
  uint32_t *local_qpn = new uint32_t[size];
  uint32_t *remote_qpn = new uint32_t[size];
  ibv_port_attr port_attr;
  ibv_query_port(ctx.ib_ctx, 1, &port_attr);
  ctx.peer_qp.resize(size);
  for (size_t r = 0; r < size; r++) {
    if (r == rank)
      continue;
    DOCA_CHECK(doca_gpu_verbs_create_qp_hl(&qp_init, &ctx.peer_qp[r]));
    local_qpn[r] = doca_verbs_qp_get_qpn(ctx.peer_qp[r]->qp);
  }

  MPI_CHECK(MPI_Allgather(&port_attr.lid, 1, MPI_UINT32_T, lids, 1,
                          MPI_UINT32_T, MPI_COMM_WORLD));
  MPI_CHECK(MPI_Alltoall(local_qpn, 1, MPI_UINT32_T, remote_qpn, 1,
                         MPI_UINT32_T, MPI_COMM_WORLD));
  // for its configuration for INIT->RTR

  for (size_t r = 0; r < size; r++) {
    if (r == rank)
      continue;
    connect_qp(ctx.peer_qp[r]->qp, lids[r], remote_qpn[r]);
  }

  delete[] lids;
  delete[] remote_qpn;
  delete[] local_qpn;

  ctx.pgsz = sysconf(_SC_PAGESIZE);
  CUdevice devh;
  CUDA_CHECK(cuDeviceGet(&devh, 0));
  int has_support;
  CUDA_CHECK(cuDeviceGetAttribute(&has_support,
                                  CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, devh));
  ctx.dmabuf_supported = (has_support == 1);
}
void cleanup(void) {
  if (ctx.has_cleanup)
    return;
  ctx.has_cleanup = true;

  for (int r = 0; r < ctx.world_size; r++) {
    if (r == ctx.rank)
      continue;
    DOCA_CHECK(doca_gpu_verbs_destroy_qp_hl(ctx.peer_qp[r]));
  }
  if (ctx.ib_pd) {
    ibv_dealloc_pd(ctx.ib_pd);
  }
  if (ctx.ib_ctx) {
    ibv_close_device(ctx.ib_ctx);
  }
  if (ctx.doca_ctx) {
    DOCA_CHECK(doca_gpu_destroy(ctx.doca_ctx));
  }
  ctx.cleanup_completed = true;
}

buffer_dev_handle::buffer_dev_handle(size_t _bufsz) {
  bufsz = _bufsz;
  DOCA_CHECK(doca_gpu_mem_alloc(ctx.doca_ctx, _bufsz, ctx.pgsz,
                                DOCA_GPU_MEM_TYPE_GPU, &buf, nullptr));
  CUDART_CHECK(cudaMemset(buf, 0, bufsz));
  // DO NOT Use the following, if DMABUF not supported
  if (ctx.dmabuf_supported) {
    int buf_dma_fd = -1;
    DOCA_CHECK(doca_gpu_dmabuf_fd(ctx.doca_ctx, buf, bufsz, &buf_dma_fd));
    mr = ibv_reg_dmabuf_mr(ctx.ib_pd, 0, bufsz, (uint64_t)buf, buf_dma_fd,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                               IBV_ACCESS_REMOTE_ATOMIC |
                               IBV_ACCESS_RELAXED_ORDERING);
  } else {
    // if DMABUF not supported, use NVIDIA-Peermem instead.
    // assume nvidia-peermem supported
    mr = ibv_reg_mr(ctx.ib_pd, buf, bufsz,
                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                        IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_RELAXED_ORDERING);
  }

  lkey = mr->lkey;
  rbuf = new void *[ctx.world_size];
  rkey = new uint32_t[ctx.world_size];
  MPI_CHECK(MPI_Allgather(&buf, 1, MPI_UINT64_T, rbuf, 1, MPI_UINT64_T,
                          MPI_COMM_WORLD));
  MPI_CHECK(MPI_Allgather(&mr->rkey, 1, MPI_UINT32_T, rkey, 1, MPI_UINT32_T,
                          MPI_COMM_WORLD));
}

buffer_dev_handle::~buffer_dev_handle() {
  delete[] rkey;
  delete[] rbuf;
  ibv_dereg_mr(mr);
  doca_gpu_mem_free(ctx.doca_ctx, buf);
}

size_t rank() { return ctx.rank; }
size_t world_size() { return ctx.world_size; }

doca_gpu_verbs_qp_hl *get_peer_qp(size_t peer_rank) { 
  auto ret = ctx.peer_qp[peer_rank]; 
  if (ret == nullptr) {
    fprintf(stderr, "GetPeerQP got nullptr for peer rank %lu, my rank is %lu\n", peer_rank, ctx.rank);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return ret;
}

bool cleaned_up(void) {
    return ctx.cleanup_completed;
}
