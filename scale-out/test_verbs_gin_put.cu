#include "mpi.h"
#include "verbs_dev.hh"
#include <cstdint>
#include <device/doca_gpunetio_dev_verbs_qp.cuh>
#include <doca_gpunetio.h>
#include <doca_gpunetio_dev_verbs_onesided.cuh>
#include <doca_gpunetio_verbs_def.h>
#include <endian.h>
#include <immintrin.h>
#include <infiniband/mlx5dv.h>
#include <stddef.h>
/*
nvcc test_verbs_gin_put.cu verbs_dev.cc -o test_verbs_gin_put \
-gencode=arch=compute_80,code=sm_80 -Xcompiler -msse4.2 \
-std=c++20 -O3 \
-I$(mpicxx -showme:incdirs) -L$(mpicxx -showme:libdirs) \
-L/home/fit/zhaijdyzq/spack/opt/spack/linux-broadwell/pmix-6.0.0-dqzkwte3b5sotgatuiiqn6au6nj5bjut/lib
\
-L/home/fit/zhaijdyzq/spack/opt/spack/linux-broadwell/prrte-4.0.0-wa3gvjktvvcglz7tq6vbqyttcnlblks3/lib
\
-I/home/fit/zhaijdyzq/repos/gpunetio/include \
-I/home/fit/zhaijdyzq/repos/gpunetio/include/host \
-I/home/fit/zhaijdyzq/repos/gpunetio/include/device \
-I/home/fit/zhaijdyzq/repos/gpunetio/include/common \
-L/home/fit/zhaijdyzq/repos/gpunetio/lib \
-lmpi -lpmix -lprrte -lcuda -lcudart -libverbs -lmlx5 -ldoca_gpunetio_host
*/

//  export
//  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/fit/zhaijdyzq/repos/gpunetio/lib
/*
salloc -p a01 -N4 -n4 --gres=gpu:1 mpirun --mca btl_tcp_if_include ens14f0 -np 4
./test_verbs_gin_put
*/
uint32_t auto_crc32(const void *__restrict buf, size_t len) {
  const uint8_t *current = (const uint8_t *)buf;
  uint32_t crc = 0xFFFFFFFF;

  // Process 8 bytes at a time using amd64 SSE4.2 hardware instructions
  while (len >= 8) {
    crc = _mm_crc32_u64(crc, *(const uint64_t *)current);
    current += 8;
    len -= 8;
  }

  // Process remaining 1 to 7 bytes one-by-one
  const uint8_t *end = current + len;
  while (current < end) {
    crc = _mm_crc32_u8(crc, *current);
    current++;
  }

  return ~crc;
}
template <enum doca_gpu_dev_verbs_exec_scope scope>
__global__ void put_bw(struct doca_gpu_dev_verbs_qp *qp, uint32_t num_iters,
                       uint32_t data_size, uint8_t *src_buf,
                       uint32_t src_buf_mkey, uint8_t *dst_buf,
                       uint32_t dst_buf_mkey) {
  doca_gpu_dev_verbs_ticket_t out_ticket;
  uint32_t lane_idx = doca_gpu_dev_verbs_get_lane_id();
  uint32_t tidx = threadIdx.x + (blockIdx.x * blockDim.x);

  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_iters;
       idx += (blockDim.x * gridDim.x)) {

    doca_gpu_dev_verbs_put<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                           DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO, scope>(
        qp,
        doca_gpu_dev_verbs_addr{.addr =
                                    (uint64_t)(dst_buf + (data_size * tidx)),
                                .key = (uint32_t)dst_buf_mkey},
        doca_gpu_dev_verbs_addr{.addr =
                                    (uint64_t)(src_buf + (data_size * tidx)),
                                .key = (uint32_t)src_buf_mkey},
        data_size, &out_ticket);

    if (scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD) {
      if (doca_gpu_dev_verbs_poll_cq_at<
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
              doca_gpu_dev_verbs_qp_get_cq_sq(qp), out_ticket) != 0)
        printf("Error CQE!\n");
    }

    if (scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP) {
      if (lane_idx == 0) {
        if (doca_gpu_dev_verbs_poll_cq_at(doca_gpu_dev_verbs_qp_get_cq_sq(qp),
                                          out_ticket) != 0)
          printf("Error CQE!\n");
      }
    }

    __syncthreads();
  }
}

void launch_put(buffer_dev_handle *bufh, size_t my_rank, size_t world_size,
                doca_gpu_dev_verbs_exec_scope scope) {
  auto peer_rank = (my_rank + 1) % world_size;
  cudaStream_t stream = 0; // default stream
  auto qp = get_peer_qp(peer_rank);
  constexpr uint32_t cuda_blocks = 2;
  constexpr uint32_t cuda_threads = 512 / cuda_blocks;
  constexpr uint32_t data_size = 8192;
  doca_gpu_dev_verbs_qp *qp_gpu;
  doca_gpu_verbs_get_qp_dev(qp->qp_gverbs, &qp_gpu);
  constexpr uint32_t num_iters = 2048; // 4096;
  uint8_t *src_buf = (uint8_t *)bufh->buf;
  uint8_t *dst_buf = (uint8_t *)bufh->rbuf[peer_rank] + 8192 * 512;
  // [FIXME] WARNING: GPUNetIO compiled with DOCA_GPUNETIO_VERBS_MKEY_SWAPPED ==
  // 1
  auto src_buf_mkey = htobe32(bufh->lkey);
  auto dst_buf_mkey = htobe32(bufh->rkey[peer_rank]);
  void (*fn)(doca_gpu_dev_verbs_qp *, uint32_t, uint32_t, uint8_t *, uint32_t,
             uint8_t *, uint32_t) = nullptr;
  switch (scope) {
  case DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD:
    fn = put_bw<DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>;
    break;
  case DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP:
    fn = put_bw<DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP>;
    break;
  }
  printf("[RANK %lu] now launch kernel with scope %s\n", my_rank,
         (scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD) ? "thread" : "warp");
  fn<<<cuda_blocks, cuda_threads, 0, stream>>>(qp_gpu, num_iters, data_size,
                                               src_buf, src_buf_mkey, dst_buf,
                                               dst_buf_mkey);
  CUDART_CHECK(cudaGetLastError());
  CUDART_CHECK(cudaStreamSynchronize(stream));
}

int main(int argc, char **argv) {
  constexpr size_t nbuf_bytes = 8192 * 512;
  constexpr size_t nbuf = nbuf_bytes / sizeof(int);
  build(argc, argv);
  buffer_dev_handle *bufh = new buffer_dev_handle(2 * nbuf_bytes);

  auto rk = rank();
  auto sz = world_size();
  srand((unsigned int)(time(0)) ^ (rk * 104729) ^ getpid());
  auto tbuf_h = new int[nbuf];
  for (int i = 0; i < nbuf; i++) {
    tbuf_h[i] = rand();
  }
  CUDART_CHECK(
      cudaMemcpy(bufh->buf, tbuf_h, nbuf_bytes, cudaMemcpyHostToDevice));
  launch_put(bufh, rk, sz,
             (doca_gpu_dev_verbs_exec_scope)(argc > 1 ? atoi(argv[1]) : 0));
  auto local_crc = auto_crc32(tbuf_h, nbuf_bytes);
  uint32_t *ring_crc = new uint32_t[sz];
  MPI_CHECK(MPI_Allgather(&local_crc, 1, MPI_UINT32_T, ring_crc, 1,
                          MPI_UINT32_T, MPI_COMM_WORLD));
  size_t writer_rank = (rk + sz - 1) % sz;
  auto expected_remote_crc = ring_crc[writer_rank];
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  CUDART_CHECK(cudaMemcpy(tbuf_h, (uint8_t *)bufh->buf + nbuf_bytes, nbuf_bytes,
                          cudaMemcpyDeviceToHost));
  auto remote_crc = auto_crc32(tbuf_h, nbuf_bytes);
  printf("[RANK %lu] Local CRC = 0x%08X, Remote CRC = 0x%08X, Expected Remote "
         "CRC from rank %lu = 0x%08X%s\n",
         rk, local_crc, remote_crc, writer_rank, expected_remote_crc,
         (remote_crc == expected_remote_crc) ? " [OK]" : " [MISMATCH]");
  delete[] ring_crc;
  delete[] tbuf_h;
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  delete bufh;
  cleanup();
  MPI_Finalize();
  return 0;
}
