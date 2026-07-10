#include "verbs_host.hh"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <infiniband/verbs.h>
#include <mpi.h>
#include <vector>
#ifdef USE_CUDART
#include <cuda_runtime.h>
#else
#include <cassert>
#endif
struct verbs_host_context {
  size_t rank;
  size_t world_size;

  ibv_context *ib_ctx = nullptr;
  ibv_pd *ib_pd = nullptr;

  ibv_srq *srq;
  ibv_cq *cq;
  std::vector<ibv_qp *> peer_qp;

  // cleanup related
  // prunes multiple triggering
  bool has_cleanup = false;
  // exit only context cleared
  bool cleanup_completed = false;
};

verbs_host_context ctx;

static ibv_qp *create_qp(void) {
  ibv_qp_init_attr qp_attr = {.send_cq = ctx.cq,
                              .recv_cq = ctx.cq,
                              .cap = {.max_send_wr = 10,
                                      .max_recv_wr = 10,
                                      .max_send_sge = 1,
                                      .max_recv_sge = 1},
                              .qp_type = IBV_QPT_RC};
  return ibv_create_qp(ctx.ib_pd, &qp_attr);
}

/// Think of
/// QP as the pair socket
/// dlid as the peer's mac address
/// dst_qpn as the peer's port
static void connect_qp(ibv_qp *qp, uint16_t dlid, uint32_t dst_qpn) {
  ibv_qp_attr attr;
  // RESET -> INIT -> RTR -> RTS

  // 1. RST -> INIT
  attr = {.qp_state = IBV_QPS_INIT,
          .qp_access_flags =
              IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_ATOMIC,
          .pkey_index = 0,
          .port_num = 1};
  IBV_CHECK(ibv_modify_qp(qp, &attr,
                          IBV_QP_STATE | IBV_QP_ACCESS_FLAGS |
                              IBV_QP_PKEY_INDEX | IBV_QP_PORT));

  // 2. INIT -> RTR
  attr = {.qp_state = IBV_QPS_RTR,
          .path_mtu = IBV_MTU_1024,
          .rq_psn = 0,
          .dest_qp_num = dst_qpn,
          .ah_attr = {.dlid = dlid, .is_global = 0, .port_num = 1},
          .max_dest_rd_atomic = 1, // I don't see it in doca gpunetio ...
          .min_rnr_timer = 12};
  IBV_CHECK(ibv_modify_qp(
      qp, &attr,
      IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_RQ_PSN | IBV_QP_DEST_QPN |
          IBV_QP_AV /*address handle, or address vector*/ |
          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));

  // 3. RTR -> RTS
  attr = {.qp_state = IBV_QPS_RTS,
          .sq_psn = 0,
          .max_rd_atomic = 1,
          .timeout = 14,
          .retry_cnt = 7,
          .rnr_retry = 7};
  IBV_CHECK(ibv_modify_qp(qp, &attr,
                          IBV_QP_STATE | IBV_QP_SQ_PSN |
                              IBV_QP_MAX_QP_RD_ATOMIC | IBV_QP_TIMEOUT |
                              IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY));
}

void build(int argc, char **argv) {
  // 1. MPI Bootstrap
  MPI_CHECK(MPI_Init(&argc, &argv));
  int rank, size;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  ctx.rank = rank;
  ctx.world_size = size;

  // 2. RDMA bootstrap
  ibv_device **dev_list = ibv_get_device_list(nullptr);
  ctx.ib_ctx = ibv_open_device(dev_list[0]); // Pick first HCA (mlx5_0)
  ibv_free_device_list(dev_list);
  ctx.ib_pd = ibv_alloc_pd(ctx.ib_ctx);

  ctx.cq = ibv_create_cq(ctx.ib_ctx, 20, nullptr, nullptr, 0);

  ctx.peer_qp.resize(size);
  memset(ctx.peer_qp.data(), 0, size * sizeof(uintptr_t));
  // 3. Setup QP (and get QPN)
  uint32_t *local_qpns = new uint32_t[size];
  uint32_t *remote_qpns = new uint32_t[size];
  uint16_t *lids = new uint16_t[size];
  for (size_t r = 0; r < size; r++) {
    if (r == rank)
      continue;
    ctx.peer_qp[r] = create_qp();
    local_qpns[r] = ctx.peer_qp[r]->qp_num;
  }

  // 4. Exchange LID and QPN
  ibv_port_attr port_attr;
  IBV_CHECK(ibv_query_port(ctx.ib_ctx, 1, &port_attr));
  MPI_CHECK(MPI_Allgather(&port_attr.lid, 1, MPI_UINT16_T, lids, 1,
                          MPI_UINT16_T, MPI_COMM_WORLD));
  MPI_CHECK(MPI_Alltoall(local_qpns, 1, MPI_UINT32_T, remote_qpns, 1,
                         MPI_UINT32_T, MPI_COMM_WORLD));

  delete[] local_qpns;
  for (size_t r = 0; r < size; r++) {
    if (r == rank)
      continue;
    connect_qp(ctx.peer_qp[r], lids[r], remote_qpns[r]);
  }
  delete[] lids;
  delete[] remote_qpns;
}

void cleanup(void) {
  if (ctx.has_cleanup)
    return;
  ctx.has_cleanup = true;

  for (size_t r = 0; r < ctx.world_size; r++) {
    if (r == ctx.rank)
      continue;

    if (ctx.peer_qp[r]) {
      ibv_destroy_qp(ctx.peer_qp[r]);
      ctx.peer_qp[r] = nullptr;
    }
  }
  if (ctx.cq) {
    ibv_destroy_cq(ctx.cq);
    ctx.cq = nullptr;
  }
  if (ctx.ib_pd) {
    ibv_dealloc_pd(ctx.ib_pd);
    ctx.ib_pd = nullptr;
  }
  if (ctx.ib_ctx) {
    ibv_close_device(ctx.ib_ctx);
    ctx.ib_ctx = nullptr;
  }
  ctx.cleanup_completed = true;
}

ibv_pd *get_pd(void) { return ctx.ib_pd; }

size_t rank() { return ctx.rank; }
size_t world_size() { return ctx.world_size; }

ibv_qp *get_peer_qp(size_t peer_rank) { 
  auto ret = ctx.peer_qp[peer_rank]; 
  if (ret == nullptr) {
    fprintf(stderr, "GetPeerQP got nullptr for peer rank %lu, my rank is %lu\n", peer_rank, ctx.rank);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return ret;
}

ibv_cq *get_cq(void) { return ctx.cq; }

bool cleaned_up(void) { return ctx.cleanup_completed; }

#define ALIGNMENT 4096

int pair_malloc(size_t size, int type, buffer_handle &bufh) {
  if (type) {
    
#ifdef USE_CUDART
    CUDART_CHECK(cudaMalloc(&bufh.buf, size));
    CUDART_CHECK(cudaMemset(bufh.buf, 0, size));
    printf("A GPU buffer allocated for rank %lu with size %lu\n", ctx.rank, size);
#else
    assert(false);
#endif
  } else
  {
    posix_memalign(&bufh.buf, ALIGNMENT, size);
    memset(bufh.buf, 0, size);
  }
  bufh.type = type;
  bufh.size = size;

  bufh.mr = ibv_reg_mr(get_pd(), bufh.buf, size,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
  bufh.lkey = bufh.mr->lkey;
  bufh.rbuf = new void *[world_size()];
  bufh.rkey = new uint32_t[world_size()];
  MPI_CHECK(MPI_Allgather(&bufh.buf, 1, MPI_LONG_LONG, bufh.rbuf, 1,
                          MPI_LONG_LONG, MPI_COMM_WORLD));
  MPI_CHECK(MPI_Allgather(&bufh.mr->rkey, 1, MPI_UINT32_T, bufh.rkey, 1,
                          MPI_UINT32_T, MPI_COMM_WORLD));
  return 0;
}

void pair_free(buffer_handle &bufh) {
  delete[] bufh.rbuf;
  bufh.rbuf = nullptr;
  delete[] bufh.rkey;
  bufh.rkey = nullptr;
  ibv_dereg_mr(bufh.mr);
  bufh.mr = nullptr;

#ifdef USE_CUDART
  if (bufh.type) {
    cudaFree(bufh.buf);
  } else
#endif
    free(bufh.buf);
}