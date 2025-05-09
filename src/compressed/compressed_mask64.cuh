using namespace std;

int Partion_B_rely_tile64ptr(int *tile_ptr, int *bin, int nums_row, int *nums_over_size)
{
    int num_partition = 0;
    int Upper_limit = Numtile_per_partion_for_initial_compress_mask;
    int i = 0;
    int before = 0;
    int over_size_index = nums_row - 1;
    while (i <= nums_row)
    {
        if (tile_ptr[i] > Upper_limit)
        {
            if ((tile_ptr[i] - tile_ptr[i - 1]) <= Numtile_per_partion_for_initial_compress_mask)
            {
                bin[num_partition << 1] = before;
                bin[(num_partition << 1) + 1] = i - 1;
                Upper_limit = tile_ptr[i - 1] + Numtile_per_partion_for_initial_compress_mask;
                before = i - 1;
                num_partition++;
            }
            else
            {
                if (i == (before + 1))
                {
                    bin[over_size_index] = i - 1;
                    over_size_index--;
                    (*nums_over_size)++;
                    before = i;
                    Upper_limit = tile_ptr[i] + Numtile_per_partion_for_initial_compress_mask;
                }
                else
                {
                    bin[num_partition << 1] = before;
                    bin[(num_partition << 1) + 1] = i - 1;
                    bin[over_size_index] = i - 1;
                    over_size_index--;
                    (*nums_over_size)++;
                    before = i;
                    num_partition++;
                    Upper_limit = tile_ptr[i] + Numtile_per_partion_for_initial_compress_mask;
                }
            }
        }
        i++;
    }
    num_partition++;
    bin[num_partition * 2 - 2] = before;
    bin[num_partition * 2 - 1] = nums_row;
    return num_partition;
}
// form B mask matrix
void matrixB_partion_for_compute_tile64_mask_num(int *tile_ptr, NHC_bin *c_bin, int nums_row, int *nums_over_size)
{
    c_bin->bin = new int[nums_row](); 
    c_bin->num_of_partion = Partion_B_rely_tile64ptr(tile_ptr, c_bin->bin, nums_row, nums_over_size);
    cudaMalloc((void **)&c_bin->d_bin, (nums_row) * sizeof(int));
    cudaMemcpy(c_bin->d_bin, c_bin->bin, (nums_row) * sizeof(int), cudaMemcpyHostToDevice);
}

__global__ void compute_tile64_nums(int *d_tile_ptr, int h_brows, int h_rows_per_partion, int *d_bptr, int *d_brow, int *d_bcol, int h_32units_one_row)
{
    int start_row = h_rows_per_partion * blockIdx.x;
    int start_nnz = d_bptr[start_row];
    int end_nnz;
    if (blockIdx.x == gridDim.x - 1)
    {
        end_nnz = d_bptr[h_brows];
    }
    else
    {
        end_nnz = d_bptr[start_row + h_rows_per_partion];
    }
    int rowid, colid, i, k, row_migration;
    unsigned int tmp;
    extern __shared__ int shared_mem[];
    int wnum = blockDim.x / Wsize_for_compute_tile_nums;
    int tsize = 12288 - wnum;
    unsigned int *target = (unsigned int *)shared_mem;
    int *tmp_num_ones = shared_mem + tsize;
    for (int index = threadIdx.x; index < tsize; index += blockDim.x)
    {
        target[index] = 0;
    }
    __syncthreads();
    for (int index = start_nnz + threadIdx.x; index < end_nnz; index += blockDim.x)
    {
        rowid = d_brow[index];
        row_migration = rowid - start_row;
        colid = d_bcol[index];
        i = colid >> 11;
        k = colid & 2047;
        k = k >> 6;
        tmp = 1;
        tmp = tmp << k;
        atomicOr(target + row_migration * h_32units_one_row + i, tmp);
    }
    __syncthreads();
    int tid = threadIdx.x & (Wsize_for_compute_tile_nums - 1);
    int wid = threadIdx.x / Wsize_for_compute_tile_nums;
    int shared_offset;
    if ((blockIdx.x == gridDim.x - 1))
    {
        for (int offset = wid; offset < (h_brows - start_row); offset += wnum)
        {
            rowid = start_row + offset;
            shared_offset = offset * h_32units_one_row;
            if (tid == 0)
            {
                tmp_num_ones[wid] = 0;
            }
            __syncthreads();
            for (int index = shared_offset + tid; index < shared_offset + h_32units_one_row; index += Wsize_for_compute_tile_nums)
            {
                if (target[index])
                {
                    atomicAdd(tmp_num_ones + wid, __popc(target[index]));
                }
            }
            __syncthreads();
            if (tid == 0)
            {
                d_tile_ptr[rowid] = tmp_num_ones[wid];
            }
        }
    }
    else
    {
        for (int offset = wid; offset < h_rows_per_partion; offset += wnum)
        {
            rowid = start_row + offset;
            shared_offset = offset * h_32units_one_row;
            if (tid == 0)
            {
                tmp_num_ones[wid] = 0;
            }
            __syncthreads();
            for (int index = shared_offset + tid; index < shared_offset + h_32units_one_row; index += Wsize_for_compute_tile_nums)
            {
                if (target[index])
                {
                    atomicAdd(tmp_num_ones + wid, __popc(target[index]));
                }
            }
            __syncthreads();
            if (tid == 0)
            {
                d_tile_ptr[rowid] = tmp_num_ones[wid];
            }
        }
    }
}
__global__ void compute_tile64_col(int *d_tile_ptr, int *d_tile_col, int h_brows, int h_rows_per_partion, int *d_bptr, int *d_brow, int *d_bcol, int h_32units_one_row)
{
    int start_row = h_rows_per_partion * blockIdx.x;
    int start_nnz = d_bptr[start_row];
    int end_nnz;
    if (blockIdx.x == gridDim.x - 1)
    {
        end_nnz = d_bptr[h_brows];
    }
    else
    {
        end_nnz = d_bptr[start_row + h_rows_per_partion];
    }
    int rowid, colid, i, k, row_migration;
    unsigned int tmp;
    extern __shared__ int shared_mem[];
    int wnum = blockDim.x / Wsize_for_compute_tile_nums;
    int tsize = 12288 - wnum;
    unsigned int *target = (unsigned int *)shared_mem;
    int *shared_offset_col = shared_mem + tsize;
    for (int index = threadIdx.x; index < tsize; index += blockDim.x)
    {
        target[index] = 0;
    }
    __syncthreads();
    for (int index = start_nnz + threadIdx.x; index < end_nnz; index += blockDim.x)
    {
        rowid = d_brow[index];
        row_migration = rowid - start_row;
        colid = d_bcol[index];
        i = colid >> 11;
        k = colid & 2047;
        k = k >> 6;
        tmp = 1;
        tmp = tmp << k;
        atomicOr(target + row_migration * h_32units_one_row + i, tmp);
    }
    int tid = threadIdx.x & (Wsize_for_compute_tile_nums - 1);
    int wid = threadIdx.x / Wsize_for_compute_tile_nums;
    int loc_offset_col;
    int shared_offset;
    int row_tile_ptr;
    unsigned int target_num;
    if ((blockIdx.x == gridDim.x - 1))
    {
        for (int offset = wid; offset < (h_brows - start_row); offset += wnum)
        {
            rowid = start_row + offset;
            shared_offset = offset * h_32units_one_row;
            row_tile_ptr = d_tile_ptr[rowid];
            if (tid == 0)
            {
                shared_offset_col[wid] = 0;
            }
            __syncthreads();
            for (int index = shared_offset + tid; index < shared_offset + h_32units_one_row; index += Wsize_for_compute_tile_nums)
            {
                target_num = target[index];
                while (target_num)
                {
                    loc_offset_col = atomicAdd(shared_offset_col + wid, 1);
                    d_tile_col[row_tile_ptr + loc_offset_col] = __ffs(target_num) - 1 + ((index - shared_offset) << 5);
                    target_num &= (target_num - 1);
                }
            }
            __syncthreads();
        }
    }
    else
    {
        for (int offset = wid; offset < h_rows_per_partion; offset += wnum)
        {
            rowid = start_row + offset;
            shared_offset = offset * h_32units_one_row;
            row_tile_ptr = d_tile_ptr[rowid];
            if (tid == 0)
            {
                shared_offset_col[wid] = 0;
            }
            __syncthreads();
            for (int index = shared_offset + tid; index < shared_offset + h_32units_one_row; index += Wsize_for_compute_tile_nums)
            {
                target_num = target[index];
                while (target_num)
                {
                    loc_offset_col = atomicAdd(shared_offset_col + wid, 1);
                    d_tile_col[row_tile_ptr + loc_offset_col] = __ffs(target_num) - 1 + ((index - shared_offset) << 5);
                    target_num &= (target_num - 1);
                }
            }
            __syncthreads();
        }
    }
}
// nums of tile in one row may greater than Numtile_per_partion_for_initial_compress_mask
__global__ void compute_tile64_mask_num(DateTypeStoreCompressMask *d_mask_num,
                                        const int *__restrict__ d_tile_ptr, const int *__restrict__ d_tile_col, const int *__restrict__ d_bin,
                                        const int *__restrict__ B_bptr, const int *__restrict__ B_drow, const int *__restrict__ B_dcol)
{
    int index;
    int loc_par_id = blockIdx.x;
    int off_row_id = d_bin[loc_par_id << 1];
    int start_Bnnz = B_bptr[off_row_id];
    int end_Bnnz = B_bptr[d_bin[(loc_par_id << 1) + 1]];

    int Bcol, Brow;
    int tile_col, k;
    int tile_start = d_tile_ptr[off_row_id];
    int tile_end = d_tile_ptr[d_bin[(loc_par_id << 1) + 1]];
    extern __shared__ int shared_mem[];
    int tsize = Numtile_per_partion_for_initial_compress_mask;
    int *shared_tile_col = shared_mem;
    DateTypeStoreCompressMask *shared_tile_mask_num = (DateTypeStoreCompressMask *)(shared_mem + tsize);
    // initial shared mem
    for (index = threadIdx.x; index < tsize; index += blockDim.x)
    {
        shared_tile_mask_num[index] = 0;
    }

    for (index = tile_start + threadIdx.x; index < tile_end; index += blockDim.x)
    {
        shared_tile_col[index - tile_start] = d_tile_col[index];
    }
    __syncthreads();
    int loc_tile_start;
    DateTypeStoreCompressMask tmp;
    for (index = threadIdx.x + start_Bnnz; index < end_Bnnz; index += blockDim.x)
    {
        Brow = B_drow[index];
        Bcol = B_dcol[index];
        tmp = 1;
        tile_col = Bcol >> 6;
        k = (Bcol) & (63);
        tmp = tmp << k;
        loc_tile_start = d_tile_ptr[Brow] - tile_start;
        while (1)
        {
            if (tile_col == shared_tile_col[loc_tile_start])
            {
                atomicOr(shared_tile_mask_num + loc_tile_start, tmp);
                break;
            }
            loc_tile_start++;
        }
    }
    __syncthreads();
    for (index = threadIdx.x; index + tile_start < tile_end; index += blockDim.x)
    {
        d_mask_num[tile_start + index] = shared_tile_mask_num[index];
    }
}
__global__ void compute_tile64_mask_num_for_over_size(DateTypeStoreCompressMask *d_mask_num,
                                                      const int *__restrict__ d_tile_ptr, const int *__restrict__ d_tile_col, const int *__restrict__ d_bin,
                                                      const int *__restrict__ B_bptr, const int *__restrict__ B_dcol)
{
    int index;
    int loc_par_id = blockIdx.x;
    int off_row_id = d_bin[loc_par_id];
    int start_Bnnz = B_bptr[off_row_id];
    int end_Bnnz = B_bptr[off_row_id + 1];
    int Bcol;
    int tile_col, k;
    int tile_start = d_tile_ptr[off_row_id];
    int tile_end = d_tile_ptr[off_row_id + 1];
    for (index = tile_start + threadIdx.x; index < tile_end; index += blockDim.x)
    {
        d_mask_num[index] = 0;
    }
    __syncthreads();
    int loc_tile_start;
    DateTypeStoreCompressMask tmp;
    for (index = threadIdx.x + start_Bnnz; index < end_Bnnz; index += blockDim.x)
    {
        Bcol = B_dcol[index];
        loc_tile_start = tile_start;
        tmp = 1;
        tile_col = Bcol >> 6;
        k = (Bcol) & (63);
        tmp = tmp << k;
        while (1)
        {
            if (tile_col == d_tile_col[loc_tile_start])
            {
                atomicOr(d_mask_num + loc_tile_start, tmp);
                break;
            }
            loc_tile_start++;
        }
    }
}
void Initialize_compressed_mask64_matrix(NHC_CSR *A, NHC_CSR *B, NHC_CSR *C, NHC_bin *c_bin)
{
    int *d_Brow_id;
    CHECK_ERROR(cudaMalloc((void **)&(d_Brow_id), (B->nnz) * sizeof(index_t)));
    B->tile_ptr = new int[B->M + 1]();
    int block_size = 1024;
    int grid_size = ((B->M) + block_size - 1) / block_size;
    csr_to_row_indices<<<grid_size, block_size>>>(B->d_ptr, d_Brow_id, B->M);
    int h_32units_one_row = (B->N + 2047) >> 11;
    int nwarp = 32; // 1024/32
    int h_rows_per_partion = (12288 - nwarp) / h_32units_one_row;
    h_rows_per_partion = min(h_rows_per_partion, B->M);
    grid_size = ((B->M) + h_rows_per_partion - 1) / h_rows_per_partion;
    compute_tile64_nums<<<grid_size, block_size, 49152>>>(B->d_tile_ptr, B->M, h_rows_per_partion, B->d_ptr, d_Brow_id, B->d_col, h_32units_one_row);
#if checek_kernel
    {
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("XXXXXXXXXXXXX compute_tile64_nums is failed\n");
        }
        else
        {
            printf("/////////// compute_tile64_nums is cudaSuccess\n");
        }
    }
#endif
    grid_size = (A->M + 1023) >> 10;
    k_compute_flop<<<grid_size, block_size>>>(A->d_ptr, A->d_col, B->d_tile_ptr, A->M, C->d_ptr, C->d_ptr + C->M);
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cudaDeviceSynchronize();
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, B->d_tile_ptr, B->d_tile_ptr, B->M + 1, 0); 
    cudaMalloc(&d_temp_storage, temp_storage_bytes);                                                             
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, B->d_tile_ptr, B->d_tile_ptr, B->M + 1, 0); 
    cudaFree(d_temp_storage);
    cudaMemcpy(B->tile_ptr, B->d_tile_ptr, (B->M + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    int nums_tile = B->tile_ptr[B->M];
    CHECK_ERROR(cudaMalloc((void **)&(B->d_mask_num), nums_tile * sizeof(DateTypeStoreCompressMask)));
    CHECK_ERROR(cudaMalloc((void **)&(B->d_tile_col), nums_tile * sizeof(index_t)));
    grid_size = ((B->M) + h_rows_per_partion - 1) / h_rows_per_partion;
    compute_tile64_col<<<grid_size, block_size, 49152>>>(B->d_tile_ptr, B->d_tile_col, B->M, h_rows_per_partion, B->d_ptr, d_Brow_id, B->d_col, h_32units_one_row);
#if checek_kernel
    {
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("XXXXXXXXXXXXX compute_tile64_col is failed\n");
        }
        else
        {
            printf("/////////// compute_tile64_col is cudaSuccess\n");
        }
    }
#endif
    int nums_over_size = 0;
    matrixB_partion_for_compute_tile64_mask_num(B->tile_ptr, c_bin, B->M, &nums_over_size);
    if (nums_over_size)
    {
        printf("the nums_over_size is %d\n", nums_over_size);
        compute_tile64_mask_num_for_over_size<<<nums_over_size, 1024>>>(B->d_mask_num, B->d_tile_ptr, B->d_tile_col, c_bin->d_bin + B->M - nums_over_size, B->d_ptr, B->d_col);
    }
    grid_size = c_bin->num_of_partion;
    compute_tile64_mask_num<<<grid_size, block_size, 49152>>>(B->d_mask_num, B->d_tile_ptr, B->d_tile_col, c_bin->d_bin, B->d_ptr, d_Brow_id, B->d_col);
#if checek_kernel
    {
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("XXXXXXXXXXXXX compute_tile64_mask_num is failed\n");
        }
        else
        {
            printf("/////////// compute_tile64_mask_num is cudaSuccess\n");
        }
    }
#endif
    cudaFree(d_Brow_id);
}
