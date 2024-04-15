__inline__ __device__ int warpReduce(int localSum)
{
    localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, 16);
    localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, 8);
    localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, 4);
    localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, 2);
    localSum += __shfl_xor_sync(0xFFFFFFFF, localSum, 1);
    return localSum;
}
__inline__ __device__ int warpReduceMax(int localMax)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        localMax = max(localMax, __shfl_down_sync(0xffffffff, localMax, offset));
    }
    return localMax;
}

template <int SH_ROW>
__global__ void k_tileOR_shared_hash_tb(
    const int *__restrict__ d_arpt, const int *__restrict__ d_acol,
    const int *__restrict__ d_brpt, const int *__restrict__ d_bcol,
    const DateTypeStoreCompressMask *__restrict__ d_bmask_num,
    const int *__restrict__ d_ctile_ptr,
    DateTypeStoreCompressMask *__restrict__ d_cmask_num,
    int *__restrict__ d_ctile_col,
    int *__restrict__ d_bins,
    int *d_Crow_nnz)
{
    int tid = threadIdx.x & (WSIZE - 1);
    int wid = threadIdx.x / WSIZE;
    int wnum = blockDim.x / WSIZE;
    int j, k;
    extern __shared__ int shared_mem[];
    int *shared_col_table = shared_mem;
    int *shared_nnz = shared_col_table + SH_ROW;
    int *shared_offset = shared_nnz + 1;
    DateTypeStoreCompressMask *shared_mask = (DateTypeStoreCompressMask *)(shared_offset + 1);
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x)
    {
        shared_col_table[j] = -1;
        shared_mask[j] = 0;
    }
    if (threadIdx.x == 0)
    {
        shared_nnz[0] = 0;
        shared_offset[0] = 0;
    }
    __syncthreads();
    int acol, bcol, hash, old;
    DateTypeStoreCompressMask bmask_num;
    int rid = d_bins[blockIdx.x];
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum)
    {
        acol = d_acol[j];
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WSIZE)
        {
            bcol = d_bcol[k];
            bmask_num = d_bmask_num[k];
            hash = (bcol * HASH_SCALE) & (SH_ROW - 1);
            while (1)
            {
                old = atomicCAS(shared_col_table + hash, -1, bcol);
                if (old == -1 || old == bcol)
                {
                    atomicOr(shared_mask + hash, bmask_num);
                    break;
                }
                else
                {
                    hash = (hash + 1) & (SH_ROW - 1);
                }
            }
        }
    }
    __syncthreads();
    // Count the number of ones
    int loc_offset, tmp_sum = 0;
    int C_offset = d_ctile_ptr[rid];
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x)
    {
        if (shared_mask[j])
        {
            loc_offset = atomicAdd(shared_offset, 1);
            d_cmask_num[C_offset + loc_offset] = shared_mask[j];
            tmp_sum += __popcll(shared_mask[j]);
            d_ctile_col[C_offset + loc_offset] = shared_col_table[j];
        }
    }
    __syncthreads();
    // use __shfl_xor do reduce
    tmp_sum = warpReduce(tmp_sum);
    if (tid == 0)
        shared_col_table[wid] = tmp_sum;
    __syncthreads();
    tmp_sum = (threadIdx.x < wnum) ? shared_col_table[tid] : 0;
    if (wid == 0)
        tmp_sum = warpReduce(tmp_sum);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        d_Crow_nnz[rid] = tmp_sum;
    }
}

__global__ void k_tileOR_shared_hash_pwarp(
    const int *__restrict__ d_arpt, const int *__restrict__ d_acol,
    const int *__restrict__ d_brpt, const int *__restrict__ d_bcol,
    const DateTypeStoreCompressMask *__restrict__ d_bmask_num,
    const int *__restrict__ d_ctile_ptr,
    DateTypeStoreCompressMask *__restrict__ d_cmask_num,
    int *__restrict__ d_ctile_col,
    int *__restrict__ d_bins,
    int bin_size,
    int *__restrict__ d_row_nnz,
    int *d_offset)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x & (PWARP_FOR_TileOR - 1);
    int rid = i / PWARP_FOR_TileOR;
    int warpid = i / PWARP_FOR_TileOR;
    int block_rid = rid & (PWARP_ROWS__FOR_TileOR - 1);
    __shared__ int shared_mem[PWARP_ROWS__FOR_TileOR * PWARP_TSIZE]; // PWARP_TSIZE is 32
    __shared__ DateTypeStoreCompressMask shared_mask[PWARP_ROWS__FOR_TileOR * PWARP_TSIZE];
    int *shared_col = shared_mem;
    int j, k;
    for (j = threadIdx.x; j < PWARP_ROWS__FOR_TileOR * PWARP_TSIZE; j += blockDim.x)
    {
        shared_col[j] = -1;
        shared_mask[j] = 0;
    }
    if (rid >= bin_size)
    {
        return;
    }
    __syncthreads();
    int *col_table = shared_col + block_rid * PWARP_TSIZE;
    DateTypeStoreCompressMask *mask_table = shared_mask + block_rid * PWARP_TSIZE;
    rid = d_bins[rid];
    int acol, bcol;
    int hash, old;
    DateTypeStoreCompressMask bmask_num;
    for (j = d_arpt[rid] + tid; j < d_arpt[rid + 1]; j += PWARP_FOR_TileOR)
    { // pwarp per row, thread per a item, thread per b row
        acol = d_acol[j];
        for (k = d_brpt[acol]; k < d_brpt[acol + 1]; k++)
        { // thread per b row
            bcol = d_bcol[k];
            bmask_num = d_bmask_num[k];
            hash = (bcol * HASH_SCALE) & (PWARP_TSIZE - 1);
            while (1)
            {
                old = atomicCAS(col_table + hash, -1, bcol);
                if (old == -1 || old == bcol)
                {
                    atomicOr(mask_table + hash, bmask_num);
                    break;
                }
                else
                {
                    hash = (hash + 1) & (PWARP_TSIZE - 1);
                }
            }
        }
    }
    __syncthreads();
    int loc_offset;
    int C_offset = d_ctile_ptr[rid];
    int tmp_sum;
    for (j = tid; j < PWARP_TSIZE; j += PWARP_FOR_TileOR)
    {
        if (mask_table[j])
        {
            loc_offset = atomicAdd(d_offset + warpid, 1);
            d_cmask_num[C_offset + loc_offset] = mask_table[j];
            tmp_sum = __popcll(mask_table[j]);
            d_ctile_col[C_offset + loc_offset] = col_table[j];
            atomicAdd(d_row_nnz + rid, tmp_sum);
        }
    }
}
__global__ void k_tileOR_max_shared_hash(
    const int *__restrict__ d_arpt, const int *__restrict__ d_acol,
    const int *__restrict__ d_brpt, const int *__restrict__ d_bcol,
    const DateTypeStoreCompressMask *__restrict__ d_bmask_num,
    const int *__restrict__ d_ctile_ptr,
    DateTypeStoreCompressMask *__restrict__ d_cmask_num,
    int *__restrict__ d_ctile_col,
    int *__restrict__ d_bins,
    int *__restrict__ d_Crow_nnz)
{
    int tid = threadIdx.x & (WSIZE - 1);
    int wid = threadIdx.x / WSIZE;
    int wnum = blockDim.x / WSIZE;
    int j, k;
    extern __shared__ int shared_mem[];
    const int tsize = 8192;
    int *shared_col_table = shared_mem;
    int *shared_offset = shared_mem + tsize;
    int *shared_nnz = shared_offset + 1;
    DateTypeStoreCompressMask *shared_mask = (DateTypeStoreCompressMask *)(shared_nnz + 1);
    for (j = threadIdx.x; j < tsize; j += blockDim.x)
    {
        shared_col_table[j] = -1;
        shared_mask[j] = 0;
    }
    if (threadIdx.x == 0)
    {
        shared_nnz[0] = 0;
        shared_offset[0] = 0;
    }
    __syncthreads();

    int rid = d_bins[blockIdx.x];
    int acol, bcol, hash, old;
    DateTypeStoreCompressMask bmask_num;
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum)
    {
        acol = d_acol[j];
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WSIZE)
        {
            bcol = d_bcol[k];
            bmask_num = d_bmask_num[k];
            hash = (bcol * HASH_SCALE) & (tsize - 1);
            while (1)
            {
                old = atomicCAS(shared_col_table + hash, -1, bcol);
                if (old == bcol || old == -1)
                {
                    atomicOr(shared_mask + hash, bmask_num);
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
    int loc_offset;
    int C_offset = d_ctile_ptr[rid];
    for (j = threadIdx.x; j < tsize; j += blockDim.x)
    {
        if (shared_mask[j])
        {
            loc_offset = atomicAdd(shared_offset, 1);
            d_cmask_num[C_offset + loc_offset] = shared_mask[j];
            atomicAdd(shared_nnz, __popcll(shared_mask[j]));
            d_ctile_col[C_offset + loc_offset] = shared_col_table[j];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        d_Crow_nnz[rid] = shared_nnz[0];
    }
}
__global__ void k_tileOR_global_hash_tb(
    const int *__restrict__ d_arpt, const int *__restrict__ d_acol,
    const int *__restrict__ d_brpt, const int *__restrict__ d_bcol,
    const DateTypeStoreCompressMask *__restrict__ d_bmask_num,
    const int *__restrict__ d_ctile_ptr,
    DateTypeStoreCompressMask *__restrict__ d_cmask_num,
    int *__restrict__ d_ctile_col,
    int *__restrict__ d_bins,
    int *__restrict__ d_Crow_nnz,
    int max_tsize,
    int *d_tables)
{
    int tid = threadIdx.x & (WSIZE - 1);
    int wid = threadIdx.x / WSIZE;
    int wnum = blockDim.x / WSIZE;
    int j, k;
    __shared__ int shared_offset[1];
    __shared__ int shared_nnz[1];
    int *table_col = d_tables + blockIdx.x * max_tsize * ((sizeof(int) + sizeof(DateTypeStoreCompressMask)) / sizeof(int));
    DateTypeStoreCompressMask *table_mask_num = (DateTypeStoreCompressMask *)(table_col + max_tsize);
    int rid = d_bins[blockIdx.x];
    int row_tiles = d_ctile_ptr[rid + 1] - d_ctile_ptr[rid];
    int tsize = row_tiles * NUMERIC_SCALE_LARGE;
    for (j = threadIdx.x; j < tsize; j += blockDim.x)
    {
        table_col[j] = -1;
        table_mask_num[j] = 0;
    }
    if (threadIdx.x == 0)
    {
        shared_offset[0] = 0;
        shared_nnz[0] = 0;
    }
    __syncthreads();
    int acol, bcol, hash, old;
    DateTypeStoreCompressMask bmask_num;
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum)
    {
        acol = d_acol[j];
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WSIZE)
        {
            bcol = d_bcol[k];
            bmask_num = d_bmask_num[k];
            hash = (bcol * HASH_SCALE) % tsize;
            while (1)
            {
                old = atomicCAS(table_col + hash, -1, bcol);
                if (old == -1 || old == bcol)
                {
                    atomicOr(table_mask_num + hash, bmask_num);
                    break;
                }
                else
                {
                    hash = (hash + 1) < tsize ? hash + 1 : 0;
                }
            }
        }
    }

    __syncthreads();
    int loc_offset;
    int C_offset = d_ctile_ptr[rid];
    for (j = threadIdx.x; j < tsize; j += blockDim.x)
    {
        if (table_mask_num[j])
        {
            loc_offset = atomicAdd(shared_offset, 1);
            d_cmask_num[C_offset + loc_offset] = table_mask_num[j];
            atomicAdd(shared_nnz, __popcll(table_mask_num[j]));
            d_ctile_col[C_offset + loc_offset] = table_col[j];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        d_Crow_nnz[rid] = shared_nnz[0];
    }
}

__global__ void kernel_compute_nnzs_per_tile(int *d_bins, int bin_size, index_t *cnnz, index_t *C_d_tile_ptr, index_t *sum_of_nnz, index_t *sum_of_tilenum, index_t *max_nnz_tilenum_rate)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int shared_sum_of_nnz[1];
    __shared__ int shared_sum_of_tilenum[1];
    __shared__ int shared_max_nnz_tilenum_rate[1];
    if (threadIdx.x == 0)
    {
        shared_sum_of_nnz[0] = 0;
        shared_sum_of_tilenum[0] = 0;
        shared_max_nnz_tilenum_rate[0] = 0;
    }
    __syncthreads();

    if (id < bin_size)
    {
        int local_row_id = d_bins[id];
        int local_cnnz = cnnz[local_row_id];
        int local_Cnum_tile = C_d_tile_ptr[local_row_id + 1] - C_d_tile_ptr[local_row_id];
        int local_max_nnz_tilenum_rate = local_cnnz / local_Cnum_tile;
        atomicAdd(shared_sum_of_nnz, local_cnnz);
        atomicAdd(shared_sum_of_tilenum, local_Cnum_tile);
        atomicMax(shared_max_nnz_tilenum_rate, local_max_nnz_tilenum_rate);
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicAdd(sum_of_nnz, shared_sum_of_nnz[0]);
        atomicAdd(sum_of_tilenum, shared_sum_of_tilenum[0]);
        atomicMax(max_nnz_tilenum_rate, shared_max_nnz_tilenum_rate[0]);
    }
}

void compute_nnzs_per_tile(compressed_bin *compressed_bin, NHC_CSR *C)
{
    index_t *d_sum_of_nnz;
    index_t *d_sum_of_tilenum;
    index_t *d_max_nnz_tilenum_rate;
    cudaMalloc((void **)&(d_sum_of_nnz), 3 * sizeof(index_t));
    cudaMemset(d_sum_of_nnz, 0, 3 * sizeof(index_t));
    d_sum_of_tilenum = d_sum_of_nnz + 1;
    d_max_nnz_tilenum_rate = d_sum_of_tilenum + 1;
    int block_size = 128;
    int grid_size = (compressed_bin->bin_size[8] + 127) / 128;
    kernel_compute_nnzs_per_tile<<<grid_size, block_size>>>(compressed_bin->d_bins + compressed_bin->bin_offset[8],
                                                            compressed_bin->bin_size[8], C->d_ptr, C->d_tile_ptr, d_sum_of_nnz, d_sum_of_tilenum, d_max_nnz_tilenum_rate);
    cudaDeviceSynchronize();
    cudaFree(d_sum_of_nnz);
}

void h_compress_tileOR_for_Cptr(compressed_bin *compressed_bin, NHC_CSR *A, NHC_CSR *B, NHC_CSR *C)
{

    cudaMemset(C->d_ptr, 0, (A->M + 1) * sizeof(index_t));
    if (compressed_bin->bin_size[6])
    {
        CHECK_ERROR(cudaFuncSetAttribute(k_tileOR_max_shared_hash,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, 98312));
        k_tileOR_max_shared_hash<<<compressed_bin->bin_size[6], 1024, 98312, compressed_bin->streams[6]>>>(
            A->d_ptr, A->d_col, B->d_tile_ptr, B->d_tile_col, B->d_mask_num,
            C->d_tile_ptr, C->d_mask_num, C->d_tile_col,
            compressed_bin->d_bins + compressed_bin->bin_offset[6],
            C->d_ptr);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX k_tileOR_shared_hash_tb bin_size[6] is failed\n");
            }
            else
            {
                printf("/////////// k_tileOR_shared_hash_tb bin_size[6] is cudaSuccess\n");
            }
        }
#endif
    }
    if (compressed_bin->bin_size[7])
    {
        int max_tsize = (*compressed_bin->max_row_nnz) * NUMERIC_SCALE_LARGE;
        size_t global_size = compressed_bin->bin_size[7] * max_tsize * (sizeof(int) + sizeof(DateTypeStoreCompressMask));
        CHECK_ERROR(cudaMalloc(&compressed_bin->d_global_mem_pool, global_size));
        compressed_bin->global_mem_pool_size = global_size;
        compressed_bin->global_mem_pool_malloced = true;
        k_tileOR_global_hash_tb<<<compressed_bin->bin_size[7], 1024, 0, compressed_bin->streams[7]>>>(
            A->d_ptr, A->d_col, B->d_tile_ptr, B->d_tile_col, B->d_mask_num,
            C->d_tile_ptr, C->d_mask_num, C->d_tile_col,
            compressed_bin->d_bins + compressed_bin->bin_offset[7],
            C->d_ptr, max_tsize,
            compressed_bin->d_global_mem_pool);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX k_tileOR_shared_hash_tb bin_size[7] is failed\n");
            }
            else
            {
                printf("/////////// k_tileOR_shared_hash_tb bin_size[7] is cudaSuccess\n");
            }
        }
#endif
    }

    if (compressed_bin->bin_size[5])
    {
        // shared mem size is 4096*12+8
        cudaFuncSetAttribute(k_tileOR_shared_hash_tb<4096>, cudaFuncAttributeMaxDynamicSharedMemorySize, 49160);
        k_tileOR_shared_hash_tb<4096><<<compressed_bin->bin_size[5], 1024, 49160, compressed_bin->streams[5]>>>(
            A->d_ptr, A->d_col, B->d_tile_ptr, B->d_tile_col, B->d_mask_num,
            C->d_tile_ptr, C->d_mask_num, C->d_tile_col,
            compressed_bin->d_bins + compressed_bin->bin_offset[5],
            C->d_ptr);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX k_tileOR_shared_hash_tb bin_size[5] is failed\n");
            }
            else
            {
                printf("/////////// k_tileOR_shared_hash_tb bin_size[5] is cudaSuccess\n");
            }
        }
#endif
    }
    cudaDeviceSynchronize();
    if (compressed_bin->bin_size[0])
    {
        int BS = PWARP_ROWS__FOR_TileOR * PWARP_FOR_TileOR;
        int GS = div_up(compressed_bin->bin_size[0], PWARP_ROWS__FOR_TileOR);
        int *d_offset;
        cudaMalloc((void **)&(d_offset), (compressed_bin->bin_size[0]) * sizeof(int));
        cudaMemset(d_offset, 0, (compressed_bin->bin_size[0]) * sizeof(int));
        k_tileOR_shared_hash_pwarp<<<GS, BS, 0, compressed_bin->streams[0]>>>(
            A->d_ptr, A->d_col, B->d_tile_ptr, B->d_tile_col, B->d_mask_num,
            C->d_tile_ptr, C->d_mask_num, C->d_tile_col,
            compressed_bin->d_bins + compressed_bin->bin_offset[0],
            compressed_bin->bin_size[0],
            C->d_ptr,
            d_offset);
        cudaFree(d_offset);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX k_tileOR_shared_hash_tb bin_size[0] is failed\n");
            }
            else
            {
                printf("/////////// k_tileOR_shared_hash_tb bin_size[0] is cudaSuccess\n");
            }
        }
#endif
    }
    if (compressed_bin->bin_size[4])
    {
        k_tileOR_shared_hash_tb<2048><<<compressed_bin->bin_size[4], 512, 24584, compressed_bin->streams[4]>>>(
            A->d_ptr, A->d_col, B->d_tile_ptr, B->d_tile_col, B->d_mask_num,
            C->d_tile_ptr, C->d_mask_num, C->d_tile_col,
            compressed_bin->d_bins + compressed_bin->bin_offset[4],
            C->d_ptr);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX k_tileOR_shared_hash_tb bin_size[4] is failed\n");
            }
            else
            {
                printf("/////////// k_tileOR_shared_hash_tb bin_size[4] is cudaSuccess\n");
            }
        }
#endif
    }
    if (compressed_bin->bin_size[3])
    {
        k_tileOR_shared_hash_tb<1024><<<compressed_bin->bin_size[3], 256, 12296, compressed_bin->streams[3]>>>(
            A->d_ptr, A->d_col, B->d_tile_ptr, B->d_tile_col, B->d_mask_num,
            C->d_tile_ptr, C->d_mask_num, C->d_tile_col,
            compressed_bin->d_bins + compressed_bin->bin_offset[3],
            C->d_ptr);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX k_tileOR_shared_hash_tb bin_size[3] is failed\n");
            }
            else
            {
                printf("/////////// k_tileOR_shared_hash_tb bin_size[3] is cudaSuccess\n");
            }
        }
#endif
    }
    if (compressed_bin->bin_size[2])
    {
        k_tileOR_shared_hash_tb<512><<<compressed_bin->bin_size[2], 128, 6152, compressed_bin->streams[2]>>>(
            A->d_ptr, A->d_col, B->d_tile_ptr, B->d_tile_col, B->d_mask_num,
            C->d_tile_ptr, C->d_mask_num, C->d_tile_col,
            compressed_bin->d_bins + compressed_bin->bin_offset[2],
            C->d_ptr);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX k_tileOR_shared_hash_tb bin_size[2] is failed\n");
            }
            else
            {
                printf("/////////// k_tileOR_shared_hash_tb bin_size[2] is cudaSuccess\n");
            }
        }
#endif
    }
    if (compressed_bin->bin_size[1])
    {
        k_tileOR_shared_hash_tb<256><<<compressed_bin->bin_size[1], 64, 3080, compressed_bin->streams[1]>>>(
            A->d_ptr, A->d_col, B->d_tile_ptr, B->d_tile_col, B->d_mask_num,
            C->d_tile_ptr, C->d_mask_num, C->d_tile_col,
            compressed_bin->d_bins + compressed_bin->bin_offset[1],
            C->d_ptr);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX k_tileOR_shared_hash_tb bin_size[1] is failed\n");
            }
            else
            {
                printf("/////////// k_tileOR_shared_hash_tb bin_size[1]is cudaSuccess\n");
            }
        }
#endif
    }
    cudaDeviceSynchronize();
    h_formcol_binning(compressed_bin, C);
#if checek_kernel
    {
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("XXXXXXXXXXXXX h_formcol_binning is failed\n");
        }
        else
        {
            printf("/////////// h_formcol_binning is cudaSuccess\n");
        }
    }
#endif
    cudaDeviceSynchronize();
#if checek_kernel
    {
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("XXXXXXXXXXXXX compute_nnzs_per_tile is failed\n");
        }
        else
        {
            printf("/////////// compute_nnzs_per_tile is cudaSuccess\n");
        }
    }
#endif
    cub::DeviceScan::ExclusiveSum(compressed_bin->d_temp_storage, compressed_bin->temp_storage_bytes, C->d_ptr, C->d_ptr, C->M + 1, 0);
    cudaMemcpy(C->rowPtr, C->d_ptr, (C->M + 1) * sizeof(index_t), cudaMemcpyDeviceToHost);
    C->nnz = C->rowPtr[C->M];
}