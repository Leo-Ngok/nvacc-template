//
/*
mpicxx -O3 test_verbs_gdr_v2.cc verbs_host.cc -DUSE_CUDART -o test_verbs_gdr_v2 \
-libverbs -lcudart \
-I/home/fit/zhaijdyzq/spack/opt/spack/linux-broadwell/cuda-12.8.1-b7vhzqhmuyhdd22isdijtflnkazpnjqi/include \
-L/home/fit/zhaijdyzq/spack/opt/spack/linux-broadwell/cuda-12.8.1-b7vhzqhmuyhdd22isdijtflnkazpnjqi/lib64
*/
// salloc -p a01 -N4 -n4 --gres=gpu:1 mpirun --mca btl_tcp_if_include ens14f0 -np 4 ./test_verbs_gdr_v2

#include "mpi.h"
#include "verbs_host.hh"
#include <cstdio>
#include <cuda_runtime.h>

int main(int argc, char **argv) {
  build(argc, argv);

  auto rk = rank();
  auto sz = world_size();
  buffer_handle bufh;
  pair_malloc(1024, 1, bufh);

  // Now test connectivity here ...
  auto peer = (rk + 1) % sz;
  char buf_src_h[50] = {}, buf_dst_h[50] = {};
  snprintf(buf_src_h, 50, "Native GPUDirect RDMA payload from rank %lu.", rk);
  CUDART_CHECK(cudaMemcpy((char *)bufh.buf + 50 * peer, buf_src_h, 50,
                          cudaMemcpyHostToDevice));

  ibv_sge sge = {(uintptr_t)bufh.buf + 50 * peer, 50, bufh.lkey};
  ibv_send_wr wr = {.wr_id = 0,
                    .next = nullptr,
                    .sg_list = &sge,
                    .num_sge = 1,
                    .opcode = IBV_WR_RDMA_WRITE,
                    .send_flags = IBV_SEND_SIGNALED,
                    .wr = {.rdma = {(uintptr_t)bufh.rbuf[peer] + 50 * peer,
                                    bufh.rkey[peer]}}};
  ibv_send_wr *bad_wr;
  ibv_post_send(get_peer_qp(peer), &wr, &bad_wr);
  ibv_wc wc;
  while (ibv_poll_cq(get_cq(), 1, &wc) < 1)
    ;
  printf("Rank %lu: write completed.\n", rk);
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  CUDART_CHECK(cudaMemcpy(buf_dst_h, (char *)bufh.buf + 50 * rk, 50,
                          cudaMemcpyDeviceToHost));
  printf("Rank %lu receives write payload: %s\n", rk, buf_dst_h);
  // Test end here.

  pair_free(bufh);
  cleanup();
  MPI_Finalize();
  return 0;
}