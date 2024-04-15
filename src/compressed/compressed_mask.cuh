using namespace std;
int partion_B_rely_tileptr(int *tile_ptr, int *bin, int nums_row)
{
    int num_partition = 0;
    int Upper_limit = Numtile_per_partion_for_initial_compress_mask;
    int i = 0;
    while (i <= nums_row)
    {
        if (tile_ptr[i] > Upper_limit)
        {
            bin[num_partition] = i - 1;
            Upper_limit = tile_ptr[i - 1] + Numtile_per_partion_for_initial_compress_mask;
            num_partition++;
        }
        i++;
    }
    bin[num_partition] = nums_row;
    return num_partition + 1;
}
// form B mask matrix
void matrixB_partion_for_compute_tile_mask_num(int *tile_ptr, NHC_bin *c_bin, int nums_row)
{
    c_bin->bin = new int[nums_row]();
    c_bin->bin[0] = 0;
    c_bin->num_of_partion = partion_B_rely_tileptr(tile_ptr, c_bin->bin + 1, nums_row);
    cudaMalloc((void **)&c_bin->d_bin, (c_bin->num_of_partion + 1) * sizeof(int)); //
    cudaMemcpy(c_bin->d_bin, c_bin->bin, (c_bin->num_of_partion + 1) * sizeof(int), cudaMemcpyHostToDevice);
}

__global__ void compute_tile_nums(int *d_tile_ptr, int h_brows, int h_rows_per_partion, int *d_bptr, int *d_brow, int *d_bcol, int h_32units_one_row)
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
    DateTypeStoreCompressMask tmp;
    extern __shared__ int shared_mem[];
    int wnum = blockDim.x / Wsize_for_compute_tile_nums;
    int tsize = 6144 - wnum;
    DateTypeStoreCompressMask *target = (DateTypeStoreCompressMask *)shared_mem;
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
        i = colid >> 10;
        k = colid & 1023;
        k = k >> 5;
        tmp = 1;
        tmp = tmp << k;
        atomicOr(target + row_migration * h_32units_one_row + i, tmp);
    }
    __syncthreads();
    int tid = threadIdx.x & (Wsize_for_compute_tile_nums - 1);
    int wid = threadIdx.x / Wsize_for_compute_tile_nums;
    int shared_offset;
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
__global__ void compute_tile_col(int *d_tile_ptr, int *d_tile_col, int h_brows, int h_rows_per_partion, int *d_bptr, int *d_brow, int *d_bcol, int h_32units_one_row)
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
    DateTypeStoreCompressMask tmp;
    extern __shared__ int shared_mem[];
    int wnum = blockDim.x / Wsize_for_compute_tile_nums;
    int tsize = 6144 - wnum;
    DateTypeStoreCompressMask *target = (DateTypeStoreCompressMask *)shared_mem;
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
        i = colid >> 10;
        k = colid & 1023;
        k = k >> 5;
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
// nums of tile in one row may greater than Numtile_per_partion_for_initial_compress_mask
__global__ void compute_tile_mask_num(DateTypeStoreCompressMask *d_mask_num,
                                      const int *__restrict__ d_tile_ptr, const int *__restrict__ d_tile_col, const int *__restrict__ d_bin,
                                      const int *__restrict__ B_bptr, const int *__restrict__ B_drow, const int *__restrict__ B_dcol)
{
    int index;
    int loc_par_id = blockIdx.x;
    int off_row_id = d_bin[loc_par_id];
    int start_Bnnz = B_bptr[off_row_id];
    int end_Bnnz = B_bptr[d_bin[loc_par_id + 1]];
    int Bcol, Brow;
    int tile_col, k;
    int tile_start = d_tile_ptr[off_row_id];
    int tile_end = d_tile_ptr[d_bin[loc_par_id + 1]];
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
        tile_col = Bcol >> 5;
        k = (Bcol) & (31);
        tmp = tmp << k;
        loc_tile_start = d_tile_ptr[Brow] - tile_start;
        while (1)
        {
            if (tile_col == shared_tile_col[loc_tile_start])
            {
                atomicOr(shared_tile_mask_num + loc_tile_start, tmp);
                break;
            }
            if (loc_tile_start + tile_start >= d_tile_ptr[Brow + 1])
            { 
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

void Initialize_compressed_mask_matrix(NHC_CSR *A, NHC_CSR *B, NHC_CSR *C, NHC_bin *c_bin)
{
    int *d_Brow_id;
    cudaMalloc((void **)&(d_Brow_id), (B->nnz) * sizeof(int)); //
    cudaMalloc((void **)&(B->d_tile_ptr), (B->M + 1) * sizeof(int));
    cudaMalloc((void **)&(B->d_tem_tile_ptr), (B->M + 1) * sizeof(int));
    B->tile_ptr = new int[B->M + 1]();
    int block_size = 256;
    int grid_size = ((B->nnz) + block_size - 1) / block_size;
    csr2coo_kernel<<<grid_size, block_size>>>(B->d_ptr, d_Brow_id, B->nnz, B->M);
    int h_32units_one_row = (B->N + 1023) >> 10;
    int nwarp = 16;
    int h_rows_per_partion = (6144 - nwarp) / h_32units_one_row;
    h_rows_per_partion = min(h_rows_per_partion, B->M);
    block_size = 512;
    grid_size = ((B->M) + h_rows_per_partion - 1) / h_rows_per_partion;
    compute_tile_nums<<<grid_size, block_size, 24576>>>(B->d_tem_tile_ptr, B->M, h_rows_per_partion, B->d_ptr, d_Brow_id, B->d_col, h_32units_one_row);
    cudaDeviceSynchronize();
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, B->d_tem_tile_ptr, B->d_tile_ptr, B->M + 1, 0); 
    cudaMalloc(&d_temp_storage, temp_storage_bytes);                                                                  
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, B->d_tem_tile_ptr, B->d_tile_ptr, B->M + 1, 0);
    cudaFree(d_temp_storage);
    cudaMemcpy(B->tile_ptr, B->d_tile_ptr, (B->M + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    int nums_tile = B->tile_ptr[B->M];
    printf("the num_tile is %d\n", nums_tile);
    CHECK_ERROR(cudaMalloc((void **)&(B->d_mask_num), nums_tile * sizeof(DateTypeStoreCompressMask)));
    CHECK_ERROR(cudaMalloc((void **)&(B->d_tile_col), nums_tile * sizeof(index_t)));
    compute_tile_col<<<grid_size, block_size, 24576>>>(B->d_tile_ptr, B->d_tile_col, B->M, h_rows_per_partion, B->d_ptr, d_Brow_id, B->d_col, h_32units_one_row);
    matrixB_partion_for_compute_tile_mask_num(B->tile_ptr, c_bin, B->M);

    block_size = 128;
    grid_size = c_bin->num_of_partion;
    compute_tile_mask_num<<<grid_size, block_size, 6144>>>(B->d_mask_num, B->d_tile_ptr, B->d_tile_col, c_bin->d_bin, B->d_ptr, d_Brow_id, B->d_col);
    block_size = 1024;
    grid_size = (A->M + 1023) >> 10;
    k_compute_flop<<<grid_size, block_size>>>(A->d_ptr, A->d_col, B->d_tem_tile_ptr, A->M, C->d_ptr, C->d_ptr + A->M);
    cudaFree(d_Brow_id);
}
