//SPDX-License-Identifier: GPL-2.0

// mpicc -O3 test_verbs.c -o test_verbs -libverbs
// salloc -p h01 -N2 -n2 mpirun -np 2 ./test_verbs
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <mpi.h>
#include <infiniband/verbs.h>

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
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
    struct ibv_qp_init_attr qp_attr = {
        .send_cq = cq, .recv_cq = cq,
        .cap = { .max_send_wr = 10, .max_recv_wr = 10, .max_send_sge = 1, .max_recv_sge = 1 },
        .qp_type = IBV_QPT_RC
    };
    struct ibv_qp *qp = ibv_create_qp(pd, &qp_attr);

    // 3. Register Memory (CPU Buffer)
    char *buf = calloc(1, BUF_SIZE);
    struct ibv_mr *mr = ibv_reg_mr(pd, buf, BUF_SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);

    // 4. Transition QP to INIT
    struct ibv_port_attr port_attr;
    ibv_query_port(context, 1, &port_attr);
    struct ibv_qp_attr attr = {
        .qp_state = IBV_QPS_INIT, .port_num = 1, .pkey_index = 0,
        .qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE
    };
    ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);

    // 5. MPI Handshake: Exchange QPN, LID, RKEY, VADDR
    struct exchange_data local_info = { qp->qp_num, port_attr.lid, mr->rkey, (uintptr_t)buf };
    struct exchange_data remote_info;
    
    // Rank 0 talks to Rank 1
    int dest = (rank == 0) ? 1 : 0;
    MPI_Sendrecv(&local_info, sizeof(local_info), MPI_BYTE, dest, 0,
                 &remote_info, sizeof(remote_info), MPI_BYTE, dest, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 6. Transition QP to RTR (Ready to Receive) then RTS (Ready to Send)
    struct ibv_qp_attr rtr_attr = {
        .qp_state = IBV_QPS_RTR, 
        .path_mtu = IBV_MTU_1024, 
        .dest_qp_num = remote_info.qpn,
        .rq_psn = 0, 
        .max_dest_rd_atomic = 1, 
        .min_rnr_timer = 12,
        .ah_attr = { .dlid = remote_info.lid, .port_num = 1 }
    };
    ibv_modify_qp(qp, &rtr_attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);

    struct ibv_qp_attr rts_attr = { 
        .qp_state = IBV_QPS_RTS, 
        .timeout = 14, 
        .retry_cnt = 7, 
        .rnr_retry = 7, 
        .sq_psn = 0 
    };
    ibv_modify_qp(qp, &rts_attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);

    // 7. Perform RDMA Write (Rank 0 -> Rank 1)
    if (rank == 0) {
        strcpy(buf, "Native Verbs RDMA Success!");
        struct ibv_sge sge = { (uintptr_t)buf, BUF_SIZE, mr->lkey };
        struct ibv_send_wr wr = {
            .wr_id = 0, .next = NULL, .sg_list = &sge, .num_sge = 1,
            .opcode = IBV_WR_RDMA_WRITE, .send_flags = IBV_SEND_SIGNALED,
            .wr.rdma = { remote_info.vaddr, remote_info.rkey }
        };
        struct ibv_send_wr *bad_wr;
        ibv_post_send(qp, &wr, &bad_wr);

        struct ibv_wc wc;
        while (ibv_poll_cq(cq, 1, &wc) < 1); // Wait for completion
        printf("Rank 0: RDMA Write completed.\n");
    }

    // Synchronize to ensure Rank 1 reads after Rank 0 writes
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) {
        printf("Rank 1: Buffer content after RDMA: %s\n", buf);
    }

    // Cleanup
    ibv_dereg_mr(mr); ibv_destroy_qp(qp); ibv_destroy_cq(cq);
    ibv_dealloc_pd(pd); ibv_close_device(context);
    MPI_Finalize();
    return 0;
}
