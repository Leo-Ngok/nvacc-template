#include <doca_gpunetio.h>
#include <doca_gpunetio_verbs_def.h>
#include <cstdint>
#include <device/doca_gpunetio_dev_verbs_qp.cuh>
#include <infiniband/mlx5dv.h>
#include "mpi.h"
#include "verbs_dev.hh"
#include <endian.h>
#include <immintrin.h>
#include <stddef.h>

/*
nvcc test_verbs_gin.cu verbs_dev.cc -o test_verbs_gin \
-gencode=arch=compute_80,code=sm_80 -Xcompiler -msse4.2 \
-std=c++20 -O3 \
-I$(mpicxx -showme:incdirs) -L$(mpicxx -showme:libdirs) \
-L/home/fit/zhaijdyzq/spack/opt/spack/linux-broadwell/pmix-6.0.0-dqzkwte3b5sotgatuiiqn6au6nj5bjut/lib \
-L/home/fit/zhaijdyzq/spack/opt/spack/linux-broadwell/prrte-4.0.0-wa3gvjktvvcglz7tq6vbqyttcnlblks3/lib \
-I/home/fit/zhaijdyzq/repos/gpunetio/include \
-I/home/fit/zhaijdyzq/repos/gpunetio/include/host \
-I/home/fit/zhaijdyzq/repos/gpunetio/include/device \
-I/home/fit/zhaijdyzq/repos/gpunetio/include/common \
-L/home/fit/zhaijdyzq/repos/gpunetio/lib \
-lmpi -lpmix -lprrte -lcuda -lcudart -libverbs -lmlx5 -ldoca_gpunetio_host
*/

//  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/fit/zhaijdyzq/repos/gpunetio/lib
//  salloc -p a01 -N4 -n4 --gres=gpu:1 mpirun --mca btl_tcp_if_include ens14f0 -np 4 ./test_verbs_gin

uint32_t auto_crc32(const void* __restrict buf, size_t len) {
    const uint8_t* current = (const uint8_t*)buf;
    uint32_t crc = 0xFFFFFFFF;

    // Process 8 bytes at a time using amd64 SSE4.2 hardware instructions
    while (len >= 8) {
        crc = _mm_crc32_u64(crc, *(const uint64_t*)current);
        current += 8;
        len -= 8;
    }

    // Process remaining 1 to 7 bytes one-by-one
    const uint8_t* end = current + len;
    while (current < end) {
        crc = _mm_crc32_u8(crc, *current);
        current++;
    }

    return ~crc;
}

__global__ void write_bw(struct doca_gpu_dev_verbs_qp *qp, uint32_t num_iters, uint32_t size,
                         uint8_t *src_buf, uint32_t src_buf_mkey, uint8_t *dst_buf,
                         uint32_t dst_buf_mkey) {
    uint64_t base_wqe_idx = 0;
    uint64_t wqe_idx = 0;
    struct doca_gpu_dev_verbs_wqe *wqe_ptr;

    // Wrong: sq_wqe_pi is the submitted producer index, not a reservation cursor. Writing WQEs
    // from it races the queue state and leaves sq_ready_index unchanged, so the CPU proxy may
    // never see these WQEs as ready.
    // wqe_idx = (doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&qp->sq_wqe_pi) + threadIdx.x);
    if (threadIdx.x == 0) {
        base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
            qp, num_iters);
    }
    __shared__ uint64_t shared_base_wqe_idx;
    if (threadIdx.x == 0)
        shared_base_wqe_idx = base_wqe_idx;
    __syncthreads();
    base_wqe_idx = shared_base_wqe_idx;

    for (uint32_t idx = threadIdx.x; idx < num_iters; idx += blockDim.x) {
        wqe_idx = base_wqe_idx + idx;
        wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

        doca_gpu_dev_verbs_wqe_prepare_write(
            qp, wqe_ptr, wqe_idx, MLX5_OPCODE_RDMA_WRITE, DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
            0, 
            // Wrong: this used threadIdx.x, which only works when num_iters == blockDim.x and
            // repeats the same offsets if each thread posts more than one WQE.
            // (uint64_t)(dst_buf + (size * threadIdx.x)),  dst_buf_mkey,
            // (uint64_t)(src_buf + (size * threadIdx.x)), src_buf_mkey, size);
            (uint64_t)(dst_buf + (size * idx)),  dst_buf_mkey,
            (uint64_t)(src_buf + (size * idx)), src_buf_mkey, size);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        // Wrong: submitting without marking the reserved WQEs ready leaves the CPU proxy with no
        // ordered ready range to process.
        // doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
        //     qp, (wqe_idx + 1), DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_CPU_PROXY_UPDATE_PI);
        doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
            qp, base_wqe_idx, base_wqe_idx + num_iters - 1);
        doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
            qp, (base_wqe_idx + num_iters), DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_CPU_PROXY_UPDATE_PI);
    }

    // Assumption: QP is long enough to hold all the WQEs posted in the loop.
    // Application needs to poll only the last CQE corresponding to the last posted WQE.
    __syncthreads();
    if (threadIdx.x == 0) {
        if (doca_gpu_dev_verbs_poll_cq_at(doca_gpu_dev_verbs_qp_get_cq_sq(qp),
                                          (base_wqe_idx + num_iters - 1)) != 0) {
#if ENABLE_DEBUG == 1
            printf("Error CQE!\n");
#endif
        }
    }
    __syncthreads();
}

// __global__ void write_bw(struct doca_gpu_dev_verbs_qp *qp, uint32_t num_iters, uint32_t size,
//                          uint8_t *src_buf, uint32_t src_buf_mkey, uint8_t *dst_buf,
//                          uint32_t dst_buf_mkey) {
//     uint64_t base_wqe_idx = 0;
//     uint64_t wqe_idx = 0;
//     struct doca_gpu_dev_verbs_wqe *wqe_ptr;

//     // Wrong: sq_wqe_pi is the submitted producer index, not a reservation cursor. Writing WQEs
//     // from it races the queue state and leaves sq_ready_index unchanged, so the CPU proxy may
//     // never see these WQEs as ready.
//     // wqe_idx = (doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&qp->sq_wqe_pi) + threadIdx.x);
//     if (threadIdx.x == 0) {
//         base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
//             qp, num_iters);
//     }
//     __shared__ uint64_t shared_base_wqe_idx;
//     if (threadIdx.x == 0)
//         shared_base_wqe_idx = base_wqe_idx;
//     __syncthreads();
//     base_wqe_idx = shared_base_wqe_idx;

//     for (uint32_t idx = threadIdx.x; idx < num_iters; idx += blockDim.x) {
//         // wqe_idx = base_wqe_idx + idx;
//         wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

//         doca_gpu_dev_verbs_wqe_prepare_write(
//             qp, wqe_ptr, wqe_idx, MLX5_OPCODE_RDMA_WRITE, DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
//             0, 
//             // Wrong: this used threadIdx.x, which only works when num_iters == blockDim.x and
//             // repeats the same offsets if each thread posts more than one WQE.
//             // (uint64_t)(dst_buf + (size * threadIdx.x)),  dst_buf_mkey,
//             // (uint64_t)(src_buf + (size * threadIdx.x)), src_buf_mkey, size);
//             (uint64_t)(dst_buf + (size * idx)),  dst_buf_mkey,
//             (uint64_t)(src_buf + (size * idx)), src_buf_mkey, size);
//     }

//     __syncthreads();
//     if (threadIdx.x == 0) {
//         // Wrong: submitting without marking the reserved WQEs ready leaves the CPU proxy with no
//         // ordered ready range to process.
//         // doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
//         //     qp, (wqe_idx + 1), DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_CPU_PROXY_UPDATE_PI);
//         doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
//             qp, base_wqe_idx, base_wqe_idx + num_iters - 1);
//         doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
//             qp, (base_wqe_idx + num_iters), DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_CPU_PROXY_UPDATE_PI);
//     }

//     // Assumption: QP is long enough to hold all the WQEs posted in the loop.
//     // Application needs to poll only the last CQE corresponding to the last posted WQE.
//     __syncthreads();
//     if (threadIdx.x == 0) {
//         if (doca_gpu_dev_verbs_poll_cq_at(doca_gpu_dev_verbs_qp_get_cq_sq(qp),
//                                           (base_wqe_idx + num_iters - 1/*wqe_idx - blockDim.x*/)) != 0) {
// #if ENABLE_DEBUG == 1
//             printf("Error CQE!\n");
// #endif
//         }
//     }
//     __syncthreads();
// }

void gpunetio_verbs_write_bw(cudaStream_t stream, struct doca_gpu_dev_verbs_qp *qp,
                                     uint32_t cuda_threads_iters, uint32_t cuda_blocks,
                                     uint32_t cuda_threads, uint32_t size, uint8_t *src_buf,
                                     uint32_t src_buf_mkey, uint8_t *dst_buf,
                                     uint32_t dst_buf_mkey) {
    if (cuda_blocks > 1) {
        fprintf(stderr, "[WARN] The kernel supports only one CUDA block.\n");
        return;
    }

    write_bw<<<cuda_blocks, cuda_threads, 0, stream>>>(qp, cuda_threads_iters, size, src_buf,
                                                       src_buf_mkey, dst_buf, dst_buf_mkey);
    CUDART_CHECK(cudaGetLastError());
}

struct cpu_proxy_args {
    struct doca_gpu_verbs_qp *qp_cpu;
    uint64_t *exit_flag;
};
void *progress_cpu_proxy(void *args_) {
    struct cpu_proxy_args *args = (struct cpu_proxy_args *)args_;

    printf("Thread CPU proxy progress is running... %ld\n",
           *((volatile uint64_t *)args->exit_flag));

    while (*((volatile uint64_t *)args->exit_flag) == 0) {
        doca_gpu_verbs_cpu_proxy_progress(args->qp_cpu, nullptr);
    }

    return nullptr;
}


void launch_write(buffer_dev_handle *bufh, size_t my_rank, size_t world_size) {
    auto peer_rank = (my_rank + 1) % world_size;
    cudaStream_t stream = 0; // default stream
    auto qp = get_peer_qp(peer_rank);
    uint64_t *local_exit_flag = new uint64_t{0};
    cpu_proxy_args proxy_args = {
        .qp_cpu = qp->qp_gverbs,
        .exit_flag = local_exit_flag
    };
    pthread_t proxy_tid;
    pthread_create(
        &proxy_tid, nullptr, progress_cpu_proxy, &proxy_args
    );
    uint32_t cuda_blocks = 1;
    uint32_t cuda_threads = 512;
    uint32_t size = 8192;
    doca_gpu_dev_verbs_qp *qp_gpu;
    doca_gpu_verbs_get_qp_dev(qp->qp_gverbs, &qp_gpu);
    uint32_t cuda_threads_iters = 512 ; //4096;
    uint8_t *src_buf = (uint8_t *) bufh->buf;
    uint8_t *dst_buf = (uint8_t *) bufh->rbuf[peer_rank] + 8192 * 512;
    // Wrong: the GPUNetIO example passes htobe32(lkey/rkey). This build has
    // DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1, so the device helper stores the key
    // as provided instead of byte-swapping it.
    // auto src_buf_mkey = bufh->lkey;
    // auto dst_buf_mkey = bufh->rkey[peer_rank];
    auto src_buf_mkey = htobe32(bufh->lkey);
    auto dst_buf_mkey = htobe32(bufh->rkey[peer_rank]);

    gpunetio_verbs_write_bw(
        stream, qp_gpu, cuda_threads_iters, cuda_blocks, cuda_threads, size, src_buf, src_buf_mkey,
        dst_buf, dst_buf_mkey
    );
    cudaStreamSynchronize(stream);
    *((volatile uint64_t *)proxy_args.exit_flag) = 1;
    pthread_join(proxy_tid, nullptr);
    delete local_exit_flag;
}

int main(int argc, char **argv) {
  constexpr size_t  nbuf_bytes = 8192 * 512;
  constexpr size_t nbuf = nbuf_bytes / sizeof(int);
  build(argc, argv);
  buffer_dev_handle *bufh = new buffer_dev_handle(2 * nbuf_bytes);

  auto rk = rank();
  auto sz = world_size();
  srand((unsigned int)(time(0)) ^ (rk * 104729) ^ getpid());
  auto tbuf_h = new int[nbuf];
  for(int i = 0; i < nbuf; i++) {
    tbuf_h[i] = rand();
  }
  CUDART_CHECK(cudaMemcpy(bufh->buf, tbuf_h, nbuf_bytes, cudaMemcpyHostToDevice));
  // Kick off proxy thread here ...
  launch_write(bufh, rk, sz);
  auto local_crc = auto_crc32(tbuf_h, nbuf_bytes);
  uint32_t *ring_crc = new uint32_t[sz];
  MPI_CHECK(MPI_Allgather(&local_crc, 1, MPI_UINT32_T, ring_crc, 1,
                          MPI_UINT32_T, MPI_COMM_WORLD));
  size_t writer_rank = (rk + sz - 1) % sz;
  auto expected_remote_crc = ring_crc[writer_rank];
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  CUDART_CHECK(cudaMemcpy(tbuf_h, (uint8_t*) bufh->buf + nbuf_bytes, nbuf_bytes, cudaMemcpyDeviceToHost));
  auto remote_crc =  auto_crc32(tbuf_h, nbuf_bytes);
  printf("[RANK %lu] Local CRC = 0x%08X, Remote CRC = 0x%08X, Expected Remote CRC from rank %lu = 0x%08X%s\n",
         rk, local_crc, remote_crc, writer_rank, expected_remote_crc,
         (remote_crc == expected_remote_crc) ? " [OK]" : " [MISMATCH]");
  delete [] ring_crc;
  delete [] tbuf_h;
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  delete bufh;
  cleanup();
  MPI_Finalize();
  return 0;
}
