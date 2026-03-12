# nvacc-template

This repository contains boilerplates to understand and implement features on accelerators.

Each file in the code base is independent, so you can compile and use each alone.

## copy

[TODO]

This part is under construction. This illustrates asynchronous copy and the use of tensor memory accelerators.

## tc

This part contains examples of tensor core matrix multiply add (mma) instructions. One is advised to read through the straightforward implementations via inline PTX instructions first, before using high level cutlass / CuTe abstractions.

This part contains examples of the following mma instructions:

- `mma.sync`. This requires SM 8.0 (Ampere) or later.
- [TODO] `wgmma.mma_async.sync.aligned`. This requires SM 9.0 (Hopper) 
- [TODO] `tcgen05`. This requires SM 10.0 (Blackwell)

One can evaluate the indexing logic illustrated by the kernel implementations first. Quickly nesting the logics into tiles and blocks and loops, and sometimes consider swizzling can be confusing for starters.

## scale-up

This part contains gpu memory management in the same compute node via the following mechanisms:

- Inter-Process Communication (IPC) handles. Forward static (pinned) memory across different processes via IPC Memory handles.
- Virtual Memory Management (VMM) handles. The examples shares the handles via **POSIX file descriptors** (by SCM control with UNIX Socket). You may also try sharing via Fabric mode, if you wish and you have IMEX manager installed.
- VMM Multicast (MC) handles. Examples use the MC handles to do all reduce and broadcast operations via `multimem` instructions, which leverages NVLink SHARP (Scalable Hierarchical Aggrregation and Reduction Protocol) mechanisms.

For the VMM way, one can think of a VMM handle as the physical *page frame number*, and one writes the PTE page number and access level.

- To get a free physical page locally, do `cuMemCreate`.
- To get a range of (virtual) addresses to be used later, do `cuMemAddressReserve`
- To set the PTE, do `cuMemMap` and `cuMemSetAccess`

One can also access the peer GPU memory via GPU Direct P2P.
- To get the physical page of the remote, do `cuMemImportFromSharableHandle`, which is a fd / fabric handle from the `ExportTo` counterpart.

To do NVLink multicast operations (multicast / reduce), one utilize a Multicast object.
- To create a multicast object, do `cuMulticastCreate`
- To bind the MC object to device, do `cuMulticastAddDevice`
- To bind a local vmm memory handle with the MC object, do `cuMulticastBindMem`

## scale-out

### RDMA mechanisms takeaway

This part contains RDMA access across compute nodes. It leverages infiniband host channel adapters (HCAs) and the verbs API to achieve communications.

One can think of it as the low level socket mechanism. For a socket, you do the following for communication (TCP / IP):

For a server, you do:

- setup `socket()` file descriptor
- `bind()` to a (random or fixed) port (and an optional address)
- `listen()` to a port
- When a client connects, `accept()` the connection and returns the communicating fd.

For a client, you do:

- setup `socket()` file descriptor
- `connect()` to the remote socket instance, by specifying address and port.

With RDMA verbs API, you exchange the lids of infiniband HCAs, base memory address and rkey of remote process, and then:

- Create Queue Pairs.
- Switch Queue Pair states and inject the lids to it.
- With the remote key and pointer received, you post a send request along with local key and pointers **for the sender**.
- One then poll the completion queue **for the sender**.

### RDMA objects

One can understand the detailed mechanisms for the zero copy RDMA behaviors in the examples. After reading through them, one should have a better understanding of:

- Protection Domain (PD)
- Memory Region (MR)
- Completion Queue (CQ)
- Work Queue, as Send Queues (SQ) and Receive Queues (RQ), forming a Queue Pair (QP).

One do **NOT** actually write work queue element (WQE) directly. Instead, one delievers a work request (WR) with `ibv_post_send()`, then `post_send` do it for you. Once the WQE is written, `rdma-core` writes the memory-mapped PCIe IO register **doorbell**.

You use the HCA context to create PD and CQ. Inside the PD, you create QP. When a new heap memory is allocated, and you want that part to be transmitted, you do `ibv_reg_mr` with the PD to register that buffer as a MR in the PD. A MR actually boils down to `lkey` and `rkey`. Local key is used when a HCA access local memory, and remote key for transmitting to remote HCA to access remote memory.

### GPU Direct RDMA

The above is CPU Memory Direct RDMA. If instead, GPU Direct RDMA is needed, then just replace the CPU pointer as GPU device pointer when doing `ibv_reg_mr`.

But before doing that, make sure that IOMMU or VT-d is disabled, or else the traffic isolation mechanism may stop TLP forwarding across PCIe switch, and might not work or forward to the root complex instead. 

### Direct Verbs

One may refer to the DevX API from Mellanox.

Before diving into this part, let's think of the possible implementation of the ibv objects.

This part is essential, as it permits one to allocate and RDMA objects to custom memory. If one maps the objects to GPU memory, one can initiate RDMA operations on device. This is known as GPU Direct Async kernel initiated.

For RDMA objects, especially WQ and CQ, the HCA should able to access it by DMA. `ibv_post_send` parses the work request and writes to the WQE. 

After a WQE is written, ibv_post_send "rings" the doorbell. For `mlx5`, "doorbell" is actually the blueflame register, in which shares the same address that one may write the WQE to the doorbell to achieve low latency for small messages.

The blueflame register is contained inside user accessible regions (UAR). The base MMIO address of a UAR is actually the PCIe BAR register.

To achieve DMA to the HCA, one can get a DMABUF file descriptor of the buffer, and sends it to the driver for registration. Then you will get a umem object. With the ID of the umem object, and one can set up a `mlx5dv` Queue Pair object.

If one instead export the DMABUF fd of a GPU buffer, one can fill the WQE on GPU. Then, CPU only needs to write to doorbell register. If the doorbell is also mapped to GPU virtual memory, one can fully bypass the host memory and invoke RDMA operations on GPU entirely. 

### Examples

IB verbs:

- One sided write
- osu_bw like bandwidth test

GPU Direct RDMA
- One sided write to / from gpu memory

Direct Verbs [TODO]
- Export DMABUF on CPU memory and allocate `mlx5dv` objects.
- Export DMABUF on GPU memory.
