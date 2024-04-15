inline void h_tilePtr_symbolic_binning(compressed_bin *compressed_bin, NHC_CSR *C)
{
    int BS = 1024;
    int GS = (C->M + 1023) >> 10;
    cudaMemcpy((compressed_bin->max_row_nnz), C->d_ptr + C->M, sizeof(index_t), cudaMemcpyDeviceToHost);
#if checek_kernel
    {
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("XXXXXXXXXXXXXXXXXXXXXXXXX cudaMemcpy((compressed_bin->max_row_nnz), is failed\n");
        }
        else
        {
            printf("//////////// cudaMemcpy((compressed_bin->max_row_nnz), is cudaSuccess\n");
        }
    }
#endif
    if (*(compressed_bin->max_row_nnz) <= 26)
    {
        k_binning_small<<<GS, BS>>>(compressed_bin->d_bins, C->M);
        compressed_bin->bin_size[0] = C->M;
        for (int i = 1; i < NUM_BIN; i++)
        {
            compressed_bin->bin_size[i] = 0;
        }
        compressed_bin->bin_offset[0] = 0;
        for (int i = 1; i < NUM_BIN; i++)
        {
            compressed_bin->bin_offset[i] = C->M;
        }
    }
    else
    {
        k_symbolic_binning<<<GS, BS, 0, compressed_bin->streams[0]>>>(
            C->d_ptr, C->M, compressed_bin->d_bin_size);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXXXXXXXXXXXXXX k_symbolic_binning in h_tilePtr_symbolic_binning, is failed\n");
            }
            else
            {
                printf("//////////// k_symbolic_binning in h_tilePtr_symbolic_binning, is cudaSuccess\n");
            }
        }
#endif
        cudaMemcpy(compressed_bin->bin_size, compressed_bin->d_bin_size, NUM_BIN * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemset(compressed_bin->d_bin_size, 0, NUM_BIN * sizeof(int));
        compressed_bin->bin_offset[0] = 0;
        for (int i = 0; i < NUM_BIN - 1; i++)
        {
            compressed_bin->bin_offset[i + 1] = compressed_bin->bin_offset[i] + compressed_bin->bin_size[i];
        }
        cudaMemcpy(compressed_bin->d_bin_offset, compressed_bin->bin_offset, NUM_BIN * sizeof(int), cudaMemcpyHostToDevice);

        k_symbolic_binning2<<<GS, BS, 0, compressed_bin->streams[0]>>>(
            C->d_ptr, C->M,
            compressed_bin->d_bins, compressed_bin->d_bin_size, compressed_bin->d_bin_offset);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXXXXXXXXXXXXXX k_symbolic_binning2 in h_tilePtr_symbolic_binning, is failed\n");
            }
            else
            {
                printf("//////////// k_symbolic_binning2 in h_tilePtr_symbolic_binning, is cudaSuccess\n");
            }
        }
#endif
    }
}
inline void h_tileOR_binning(compressed_bin *compressed_bin, NHC_CSR *C)
{
    CHECK_ERROR(cudaMemsetAsync(compressed_bin->d_bin_size, 0, (NUM_BIN_FOR_Ccol + 1) * sizeof(int), compressed_bin->streams[0]));
#if checek_kernel
    {
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("XXXXXXXXXXXXXXXXXXXXXXXXX cudaMemsetAsync(compressed_bin->d_bin_size, 0, (NUM_BIN_FOR_Ccol+1) in h_tileOR_binning, is failed\n");
        }
        else
        {
            printf("//////////// cudaMemsetAsync(compressed_bin->d_bin_size, 0, (NUM_BIN_FOR_Ccol+1) in h_tileOR_binning, is cudaSuccess\n");
        }
    }
#endif
    int BS = 1024;
    int GS = div_up(C->M, BS);
    k_tileOR_binning<<<GS, BS, 0, compressed_bin->streams[0]>>>(C->d_tile_ptr, C->M,
                                                                compressed_bin->d_bin_size, compressed_bin->d_total_nnz, compressed_bin->d_max_row_nnz);
#if checek_kernel
    {
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("XXXXXXXXXXXXXXXXXXXXXXXXX k_tileOR_binning in h_tileOR_binning, is failed\n");
        }
        else
        {
            printf("//////////// k_tileOR_binning in h_tileOR_binning, is cudaSuccess\n");
        }
    }
#endif
    CHECK_ERROR(cudaMemcpyAsync(compressed_bin->bin_size, compressed_bin->d_bin_size, (NUM_BIN_FOR_Ccol + 1) * sizeof(int), cudaMemcpyDeviceToHost, compressed_bin->streams[0]));
#if checek_kernel
    {
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("XXXXXXXXXXXXXXXXXXXXXXXXX cudaMemcpyAsync in h_tileOR_binning, is failed\n");
        }
        else
        {
            printf("//////////// cudaMemcpyAsync in h_tileOR_binning, is cudaSuccess\n");
        }
    }
#endif
    CHECK_ERROR(cudaStreamSynchronize(compressed_bin->streams[0]));
    if (*(compressed_bin->max_row_nnz) <= 16)
    {
        k_binning_small<<<GS, BS>>>(compressed_bin->d_bins, C->M);
        compressed_bin->bin_size[0] = C->M;
        for (int i = 1; i < NUM_BIN; i++)
        {
            compressed_bin->bin_size[i] = 0;
        }
        compressed_bin->bin_offset[0] = 0;
        for (int i = 1; i < NUM_BIN; i++)
        {
            compressed_bin->bin_offset[i] = C->M;
        }
    }
    else
    {
        CHECK_ERROR(cudaMemsetAsync(compressed_bin->d_bin_size, 0, NUM_BIN * sizeof(int), compressed_bin->streams[0]));
        compressed_bin->bin_offset[0] = 0;
        for (int i = 0; i < NUM_BIN - 1; i++)
        {
            compressed_bin->bin_offset[i + 1] = compressed_bin->bin_offset[i] + compressed_bin->bin_size[i];
        }
        CHECK_ERROR(cudaMemcpyAsync(compressed_bin->d_bin_offset, compressed_bin->bin_offset, NUM_BIN * sizeof(int), cudaMemcpyHostToDevice, compressed_bin->streams[0]));

        k_tileOR_binning2<<<GS, BS, 0, compressed_bin->streams[0]>>>(C->d_tile_ptr, C->M,
                                                                     compressed_bin->d_bins, compressed_bin->d_bin_size, compressed_bin->d_bin_offset);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXXXXXXXXXXXXXX k_tileOR_binning2 in h_tileOR_binning, is failed\n");
            }
            else
            {
                printf("//////////// k_tileOR_binning2 in h_tileOR_binning, is cudaSuccess\n");
            }
        }
#endif
    }
}

template <int SH_ROW>
__global__ void k_tilePtr_symbolic_shared_hash_tb(
    const int *__restrict__ d_arpt, const int *__restrict__ d_acol,
    const int *__restrict__ d_brpt, const int *__restrict__ d_bcol,
    int *__restrict__ d_bins,
    int *__restrict__ d_row_nnz)
{
    int tid = threadIdx.x & (WSIZE - 1);
    int wid = threadIdx.x / WSIZE;
    int wnum = blockDim.x / WSIZE;
    int j, k;
    __shared__ int shared_col_table[SH_ROW];
    __shared__ int shared_nnz[1];
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x)
    {
        shared_col_table[j] = -1;
    }
    if (threadIdx.x == 0)
    {
        shared_nnz[0] = 0;
    }
    __syncthreads();
    int acol, bcol, hash, old;
    int rid = d_bins[blockIdx.x];
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum)
    {
        acol = d_acol[j];
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WSIZE)
        {
            bcol = d_bcol[k];
            hash = (bcol * HASH_SCALE) & (SH_ROW - 1);
            while (1)
            {
#ifdef HASH_SINGLE
                old = atomicCAS(shared_col_table + hash, -1, bcol);
                if (old == -1)
                {
                    atomicAdd(shared_nnz, 1);
                    break;
                }
                else if (old == bcol)
                {
                    break;
                }
                else
                {
                    hash = (hash + 1) & (SH_ROW - 1);
                }
#endif
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        d_row_nnz[rid] = shared_nnz[0];
    }
}

__global__ void k_tilePtr_symbolic_shared_hash_pwarp(
    const int *__restrict__ d_arpt, const int *__restrict__ d_acol,
    const int *__restrict__ d_brpt, const int *__restrict__ d_bcol,
    int *__restrict__ d_bins,
    int bin_size,
    int *__restrict__ d_row_nnz)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x & (PWARP_FOR_CtilePtr - 1);
    int rid = i / PWARP_FOR_CtilePtr;
    int block_rid = rid & (PWARP_ROWS_FOR_CtilePtr - 1);

    __shared__ int shared_mem[PWARP_ROWS_FOR_CtilePtr * PWARP_TSIZE_FOR_CtilePtr + PWARP_ROWS_FOR_CtilePtr];
    int *shared_col = shared_mem;
    int *shared_nnz = shared_mem + PWARP_ROWS_FOR_CtilePtr * PWARP_TSIZE_FOR_CtilePtr;
    int j, k;
    for (j = threadIdx.x; j < PWARP_ROWS_FOR_CtilePtr * PWARP_TSIZE_FOR_CtilePtr; j += blockDim.x)
    {
        shared_col[j] = -1;
    }
    if (threadIdx.x < PWARP_ROWS_FOR_CtilePtr)
    {
        shared_nnz[threadIdx.x] = 0;
    }
    if (rid >= bin_size)
    {
        return;
    }
    __syncthreads();
    int *col_table = shared_col + block_rid * PWARP_TSIZE_FOR_CtilePtr;
    rid = d_bins[rid];
    int acol, bcol;
    int hash, old;
    for (j = d_arpt[rid] + tid; j < d_arpt[rid + 1]; j += PWARP_FOR_CtilePtr)
    { // pwarp per row, thread per a item, thread per b row
        acol = d_acol[j];
        for (k = d_brpt[acol]; k < d_brpt[acol + 1]; k++)
        { // thread per b row
            bcol = d_bcol[k];
            hash = (bcol * HASH_SCALE) & (PWARP_TSIZE_FOR_CtilePtr - 1);
            while (1)
            {
                old = atomicCAS(col_table + hash, -1, bcol);
                if (old == -1)
                {
                    atomicAdd(shared_nnz + block_rid, 1);
                    break;
                }
                else if (old == bcol)
                {
                    break;
                }
                else
                {
                    hash = (hash + 1) & (PWARP_TSIZE_FOR_CtilePtr - 1);
                }
            }
        }
    }
    __syncthreads();
    if (tid == 0)
    {
        d_row_nnz[rid] = shared_nnz[block_rid];
    }
}
__global__ void k_tilePtr_symbolic_large_shared_hash_tb(
    const int *__restrict__ d_arpt, const int *__restrict__ d_acol,
    const int *__restrict__ d_brpt, const int *__restrict__ d_bcol,
    int *__restrict__ d_bins,
    int *__restrict__ d_row_nnz)
{
    int tid = threadIdx.x & (WSIZE - 1);
    int wid = threadIdx.x / WSIZE;
    int wnum = blockDim.x / WSIZE;
    int j, k;
    __shared__ int shared_col_table[12287];
    __shared__ int shared_nnz[1];
    const int tsize = 12287;

    for (j = threadIdx.x; j < tsize; j += blockDim.x)
    {
        shared_col_table[j] = -1;
    }
    if (threadIdx.x == 0)
    {
        shared_nnz[0] = 0;
    }
    __syncthreads();
    int rid = d_bins[blockIdx.x];
    int acol, bcol, hash, old;
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum)
    {
        acol = d_acol[j];
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WSIZE)
        {
            bcol = d_bcol[k];
            hash = (bcol * HASH_SCALE) % tsize;
            while (1)
            {
#ifdef HASH_SINGLE
                old = atomicCAS(shared_col_table + hash, -1, bcol);
                if (old == bcol)
                {
                    break;
                }
                else if (old == -1)
                {
                    atomicAdd(shared_nnz, 1);
                    break;
                }
                else
                {
                    hash = hash + 1 < tsize ? hash + 1 : 0;
                }
#endif
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        d_row_nnz[rid] = shared_nnz[0];
    }
}
__global__ void k_tilePtr_symbolic_max_shared_hash_tb_with_fail(
    const int *__restrict__ d_arpt, const int *__restrict__ d_acol,
    const int *__restrict__ d_brpt, const int *__restrict__ d_bcol,
    int *__restrict__ d_bins,
    int *__restrict__ d_fail_bins,
    int *__restrict__ d_fail_bin_size,
    int *__restrict__ d_row_nnz)
{
    int tid = threadIdx.x & (WSIZE - 1);
    int wid = threadIdx.x / WSIZE;
    int wnum = blockDim.x / WSIZE;
    int j, k;
    extern __shared__ int shared_mem[]; // size 24576
    const int tsize = 24575;
    int *shared_col_table = shared_mem;
    int *shared_nnz = shared_mem + tsize;

    int thresh_nnz = tsize * THRESH_SCALE;
    for (j = threadIdx.x; j < tsize; j += blockDim.x)
    {
        shared_col_table[j] = -1;
    }
    if (threadIdx.x == 0)
    {
        shared_nnz[0] = 0;
    }
    __syncthreads();

    int rid = d_bins[blockIdx.x];
    int acol, bcol, hash, old;
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum)
    {
        acol = d_acol[j];
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WSIZE)
        {
            bcol = d_bcol[k];
            hash = (bcol * HASH_SCALE) % tsize;
            while (shared_nnz[0] <= thresh_nnz)
            {
                old = atomicCAS(shared_col_table + hash, -1, bcol);
                if (old == bcol)
                {
                    break;
                }
                else if (old == -1)
                {
                    atomicAdd(shared_nnz, 1);
                    break;
                }
                else
                {
                    hash = hash + 1 < tsize ? hash + 1 : 0;
                }
            }
        }
    }
    __syncthreads();

    int row_nnz;
    int fail_index;
    if (threadIdx.x == 0)
    {
        row_nnz = shared_nnz[0];
        if (row_nnz <= thresh_nnz)
        { // success
            d_row_nnz[rid] = row_nnz;
        }
        else
        { // fail case
            fail_index = atomicAdd(d_fail_bin_size, 1);
            d_fail_bins[fail_index] = rid;
        }
    }
}
__global__ void k_tilePtr_symbolic_global_hash_tb(
    const int *__restrict__ d_arpt, const int *__restrict__ d_acol,
    const int *__restrict__ d_brpt, const int *__restrict__ d_bcol,
    int *__restrict__ d_bins,
    int *__restrict__ d_row_nnz,
    int *__restrict__ d_col_table,
    int max_tsize)
{
    int tid = threadIdx.x & (WSIZE - 1);
    int wid = threadIdx.x / WSIZE;
    int wnum = blockDim.x / WSIZE;
    int j, k;
    __shared__ int shared_nnz[1];

    int rid = d_bins[blockIdx.x];
    int *col_table = d_col_table + blockIdx.x * max_tsize;
    int tsize = d_row_nnz[rid] * SYMBOLIC_SCALE_LARGE;
    int acol, bcol, hash, old;
    for (j = threadIdx.x; j < tsize; j += blockDim.x)
    {
        col_table[j] = -1;
    }
    if (threadIdx.x == 0)
    {
        shared_nnz[0] = 0;
    }
    __syncthreads();

    int nnz = 0;
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum)
    {
        acol = d_acol[j];
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WSIZE)
        {
            bcol = d_bcol[k];
            hash = (bcol * HASH_SCALE) % tsize;
            while (1)
            {
#ifdef HASH_SINGLE
                old = atomicCAS(col_table + hash, -1, bcol);
                if (old == -1)
                {
                    nnz++;
                    break;
                }
                else if (old == bcol)
                {
                    break;
                }
                else
                {
                    hash = hash + 1 < tsize ? hash + 1 : 0;
                }
#endif
            }
        }
    }
    __syncthreads();
    atomicAdd(shared_nnz, nnz);

    __syncthreads();
    if (threadIdx.x == 0)
    {
        d_row_nnz[rid] = shared_nnz[0];
    }
}

void h_compress_symbolic_for_tilePtr(compressed_bin *compressed_bin, NHC_CSR *A, NHC_CSR *B, NHC_CSR *C)
{
    if (compressed_bin->bin_size[5])
    {
        k_tilePtr_symbolic_shared_hash_tb<8192><<<compressed_bin->bin_size[5], 1024, 0, compressed_bin->streams[5]>>>(
            A->d_ptr, A->d_col, B->d_tile_ptr, B->d_tile_col,
            compressed_bin->d_bins + compressed_bin->bin_offset[5],
            C->d_tile_ptr);
    }
    int *d_fail_bins, *d_fail_bin_size;
    int fail_bin_size = 0;
    if (compressed_bin->bin_size[7])
    {
        CHECK_ERROR(cudaMalloc(&d_fail_bins, (compressed_bin->bin_size[7] + 1) * sizeof(int)));
        d_fail_bin_size = d_fail_bins + compressed_bin->bin_size[7];

        CHECK_ERROR(cudaMemsetAsync(d_fail_bin_size, 0, sizeof(int), compressed_bin->streams[7]));
        CHECK_ERROR(cudaFuncSetAttribute(k_tilePtr_symbolic_max_shared_hash_tb_with_fail,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
        k_tilePtr_symbolic_max_shared_hash_tb_with_fail<<<compressed_bin->bin_size[7], 1024, 98304, compressed_bin->streams[7]>>>(
            A->d_ptr, A->d_col, B->d_tile_ptr, B->d_tile_col,
            compressed_bin->d_bins + compressed_bin->bin_offset[7],
            d_fail_bins, d_fail_bin_size,
            C->d_tile_ptr);
    }
    if (compressed_bin->bin_size[6])
    {
        k_tilePtr_symbolic_large_shared_hash_tb<<<compressed_bin->bin_size[6], 1024, 0, compressed_bin->streams[6]>>>(
            A->d_ptr, A->d_col, B->d_tile_ptr, B->d_tile_col,
            compressed_bin->d_bins + compressed_bin->bin_offset[6],
            C->d_tile_ptr);
    }
    if (compressed_bin->bin_size[0])
    {
        int BS = PWARP_ROWS_FOR_CtilePtr * PWARP_FOR_CtilePtr;
        int GS = div_up(compressed_bin->bin_size[0], PWARP_ROWS_FOR_CtilePtr);
        k_tilePtr_symbolic_shared_hash_pwarp<<<GS, BS, 0, compressed_bin->streams[0]>>>(
            A->d_ptr, A->d_col, B->d_tile_ptr, B->d_tile_col,
            compressed_bin->d_bins + compressed_bin->bin_offset[0],
            compressed_bin->bin_size[0],
            C->d_tile_ptr);
    }
    if (compressed_bin->bin_size[7])
    {
        CHECK_ERROR(cudaMemcpyAsync(&fail_bin_size, d_fail_bin_size, sizeof(int), cudaMemcpyDeviceToHost, compressed_bin->streams[7]));
        CHECK_ERROR(cudaStreamSynchronize(compressed_bin->streams[7]));
        if (fail_bin_size)
        { // global hash
            int max_tsize = *(compressed_bin->max_row_nnz) * SYMBOLIC_SCALE_LARGE;
            compressed_bin->global_mem_pool_size = fail_bin_size * max_tsize * sizeof(int);
            CHECK_ERROR(cudaMalloc(&compressed_bin->d_global_mem_pool, compressed_bin->global_mem_pool_size));
            compressed_bin->global_mem_pool_malloced = true;
            k_tilePtr_symbolic_global_hash_tb<<<fail_bin_size, 1024, 0, compressed_bin->streams[7]>>>(
                A->d_ptr, A->d_col, B->d_tile_ptr, B->d_tile_col,
                d_fail_bins,
                C->d_tile_ptr,
                compressed_bin->d_global_mem_pool, max_tsize);
        }
    }
    if (compressed_bin->bin_size[4])
    {
        k_tilePtr_symbolic_shared_hash_tb<4096><<<compressed_bin->bin_size[4], 512, 0, compressed_bin->streams[4]>>>(
            A->d_ptr, A->d_col, B->d_tile_ptr, B->d_tile_col,
            compressed_bin->d_bins + compressed_bin->bin_offset[4],
            C->d_tile_ptr);
    }
    if (compressed_bin->bin_size[3])
    {
        k_tilePtr_symbolic_shared_hash_tb<2048><<<compressed_bin->bin_size[3], 256, 0, compressed_bin->streams[3]>>>(
            A->d_ptr, A->d_col, B->d_tile_ptr, B->d_tile_col,
            compressed_bin->d_bins + compressed_bin->bin_offset[3],
            C->d_tile_ptr);
    }
    if (compressed_bin->bin_size[2])
    {
        k_tilePtr_symbolic_shared_hash_tb<1024><<<compressed_bin->bin_size[2], 128, 0, compressed_bin->streams[2]>>>(
            A->d_ptr, A->d_col, B->d_tile_ptr, B->d_tile_col,
            compressed_bin->d_bins + compressed_bin->bin_offset[2],
            C->d_tile_ptr);
    }
    if (compressed_bin->bin_size[1])
    {
        k_tilePtr_symbolic_shared_hash_tb<512><<<compressed_bin->bin_size[1], 64, 0, compressed_bin->streams[1]>>>(
            A->d_ptr, A->d_col, B->d_tile_ptr, B->d_tile_col,
            compressed_bin->d_bins + compressed_bin->bin_offset[1],
            C->d_tile_ptr);
    }
    if (compressed_bin->bin_size[7])
    {
        CHECK_ERROR(cudaFree(d_fail_bins));
    }
}
void compressed_Form_tileCptr(compressed_bin *compressed_bin, NHC_CSR *A, NHC_CSR *B, NHC_CSR *C)
{
    h_tilePtr_symbolic_binning(compressed_bin, C);
#if checek_kernel
    {
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("XXXXXXXXXXXXX h_tilePtr_symbolic_binning is failed\n");
        }
        else
        {
            printf("/////////// h_tilePtr_symbolic_binning is cudaSuccess\n");
        }
    }
#endif
    cudaDeviceSynchronize();
    h_compress_symbolic_for_tilePtr(compressed_bin, A, B, C);
#if checek_kernel
    {
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("XXXXXXXXXXXXX h_compress_symbolic_for_tilePtr is failed\n");
        }
        else
        {
            printf("/////////// h_compress_symbolic_for_tilePtr is cudaSuccess\n");
        }
    }
#endif
    cudaDeviceSynchronize();

    h_tileOR_binning(compressed_bin, C);
#if checek_kernel
    {
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("XXXXXXXXXXXXX h_tileOR_binning is failed\n");
        }
        else
        {
            printf("/////////// h_tileOR_binning is cudaSuccess\n");
        }
    }
#endif
    cudaDeviceSynchronize();
    int nums_tile = *(compressed_bin->total_nnz);
    cudaMalloc((void **)&(C->d_mask_num), (nums_tile) * sizeof(DateTypeStoreCompressMask));
    cudaMalloc((void **)&(C->d_tile_col), (nums_tile) * sizeof(index_t));
    // Prefix sum
    cub::DeviceScan::ExclusiveSum(compressed_bin->d_temp_storage, compressed_bin->temp_storage_bytes, C->d_tile_ptr, C->d_tile_ptr, A->M + 1, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(C->tile_ptr, C->d_tile_ptr, (A->M + 1) * sizeof(index_t), cudaMemcpyDeviceToHost);
}