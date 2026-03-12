// salloc -p a01 -N2 -n2 mpirun -np 2 ./test_verbs_device

/* 
nvcc test_verbs_device.cu -o test_verbs_device \ 
-std=c++20 \
-I$(mpicxx -showme:incdirs) \
-L$(mpicxx -showme:libdirs) \
-L$(spack location -i /dqz)/lib \
-L$(spack location -i /wa3)/lib \
-lmpi -lpmix -lprrte -lcuda -lcudart -libverbs
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <mpi.h>
#include <infiniband/verbs.h>
#include <cuda_runtime.h>
#include <cuda.h>

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

#define BUF_SIZE 1024

// Structure to exchange via MPI
struct exchange_data {
    uint32_t qpn;
    uint16_t lid;
    uint32_t rkey;
    uint64_t vaddr;
};

int main(int argc, char** argv) {
    // 1. MPI Layer Initialization
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank, size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    if (size < 2) {
        if (rank == 0) printf("Need at least 2 nodes.\n");
        MPI_Finalize(); return 0;
    }

    // 2. Verbs Layer Setup
    struct ibv_device **dev_list = ibv_get_device_list(NULL);
    struct ibv_context *context = ibv_open_device(dev_list[0]); // Pick first HCA (mlx5_0)
    struct ibv_pd *pd = ibv_alloc_pd(context);
    struct ibv_cq *cq = ibv_create_cq(context, 10, NULL, NULL, 0);

    // Create Queue Pair (RC - Reliable Connection)
    ibv_srq_init_attr srq_attr = {
        .attr = {
            .max_sge = 1, .max_wr = 1, .srq_limit = 1
        }
    };
    auto srq = ibv_create_srq(pd, &srq_attr);

    struct ibv_qp_init_attr qp_attr = {
        .send_cq = cq, .recv_cq = cq,
        .srq = srq,
        .cap = { .max_send_wr = 10, .max_recv_wr = 10, .max_send_sge = 1, .max_recv_sge = 1 },
        .qp_type = IBV_QPT_RC
    };
    struct ibv_qp *qp = ibv_create_qp(pd, &qp_attr);

    // 3. Register Memory (CPU Buffer)
    char *buf = (char *)calloc(1, BUF_SIZE);
    char *buf_d;
    CUDART_CHECK(cudaMalloc(&buf_d, BUF_SIZE));
    // GPU Direct RDMA
    struct ibv_mr *mr = ibv_reg_mr(pd, buf_d, BUF_SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);

    // 4. Transition QP to INIT
    struct ibv_port_attr port_attr;
    ibv_query_port(context, 1, &port_attr);
    struct ibv_qp_attr attr = {
        .qp_state = IBV_QPS_INIT, 
        .qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE,
        .pkey_index = 0,
        .port_num = 1, 
    };
    ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);

    // 5. MPI Handshake: Exchange QPN, LID, RKEY, VADDR
    struct exchange_data local_info = { qp->qp_num, port_attr.lid, mr->rkey, (uintptr_t)buf_d };
    struct exchange_data remote_info;
    
    // Rank 0 talks to Rank 1
    int dest = (rank == 0) ? 1 : 0;
    MPI_CHECK(MPI_Sendrecv(&local_info,  sizeof(local_info),  MPI_BYTE, dest, 0,
                 &remote_info, sizeof(remote_info), MPI_BYTE, dest, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE));

    // 6. Transition QP to RTR (Ready to Receive) then RTS (Ready to Send)
    struct ibv_qp_attr rtr_attr = {
        .qp_state = IBV_QPS_RTR, 
        .path_mtu = IBV_MTU_1024, 
        .rq_psn = 0, 
        .dest_qp_num = remote_info.qpn,
        .ah_attr = { .dlid = remote_info.lid, .port_num = 1 },
        .max_dest_rd_atomic = 1, 
        .min_rnr_timer = 12,
    };
    ibv_modify_qp(qp, &rtr_attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);

    struct ibv_qp_attr rts_attr = { 
        .qp_state = IBV_QPS_RTS, 
        .sq_psn = 0 ,
        .timeout = 14, 
        .retry_cnt = 7, 
        .rnr_retry = 7, 
    };
    ibv_modify_qp(qp, &rts_attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    // if(rank == 1) {
    //     ibv_post_srq_recv(srq, );
    // }
    // 7. Perform RDMA Write (Rank 0 -> Rank 1)
    if (rank == 0) {
        strcpy(buf, "Native Verbs GPU Direct RDMA Success! (from zero)");
        CUDART_CHECK(cudaMemcpy(buf_d, buf, 55, cudaMemcpyHostToDevice));
        struct ibv_sge sge = { (uintptr_t)(buf_d), BUF_SIZE, mr->lkey };
        struct ibv_send_wr wr = {
            .wr_id = 0, .next = NULL, .sg_list = &sge, .num_sge = 1,
            .opcode = IBV_WR_RDMA_WRITE, .send_flags = IBV_SEND_SIGNALED,
            .wr = { .rdma = { remote_info.vaddr, remote_info.rkey } }
        };
        struct ibv_send_wr *bad_wr;
        ibv_post_send(qp, &wr, &bad_wr);

        struct ibv_wc wc;
        while (ibv_poll_cq(cq, 1, &wc) < 1); // Wait for completion
        printf("Rank 0: RDMA Write completed.\n");
    }

    // Synchronize to ensure Rank 1 reads after Rank 0 writes
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (rank == 1) {
        CUDART_CHECK(cudaMemcpy(buf, buf_d, 55, cudaMemcpyDeviceToHost));
        printf("Rank 1: Buffer content after RDMA: %s\n", buf);
    }

    // Cleanup
    ibv_dereg_mr(mr); ibv_destroy_qp(qp); ibv_destroy_cq(cq);
    ibv_dealloc_pd(pd); ibv_close_device(context);
    MPI_CHECK(MPI_Finalize());
    free(buf);
    CUDART_CHECK(cudaFree(buf_d));
    return 0;
}

// /rmprog/slurm/v24.05.1/bin/salloc -p a01 -N2 -n2 mpirun -np 2 --mca pml ob1 --mca btl tcp,self ./mpi_rdma
// mpicc -O3 mpi-rdma.c -o mpi_rdma -libverbs