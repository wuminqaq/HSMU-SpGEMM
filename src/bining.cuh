__global__ void k_symbolic_binning(index_t *d_row_flop, int M, int *d_bin_size)
{
    __shared__ int shared_bin_size[NUM_BIN];
    if (threadIdx.x < NUM_BIN)
    {
        shared_bin_size[threadIdx.x] = 0;
    }
    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int row_nnz, j;
    int range[NUM_BIN] = {26, 426, 853, 1706, 3413, 6826, 10240, INT_MAX};
    if (i < M)
    {
        row_nnz = d_row_flop[i];
        for (j = 0; j < NUM_BIN; j++)
        {
            if (row_nnz <= range[j])
            {
                atomicAdd(shared_bin_size + j, 1);
                goto before_end;
            }
        }
    }
before_end:
    __syncthreads();
    if (threadIdx.x < NUM_BIN)
    {
        atomicAdd(d_bin_size + threadIdx.x, shared_bin_size[threadIdx.x]);
    }
}

__global__ void k_symbolic_binning2(
    index_t *__restrict__ d_row_flop,
    int M,
    int *__restrict__ d_bins,
    int *__restrict__ d_bin_size,
    int *__restrict__ d_bin_offset)
{

    __shared__ int shared_bin_size[NUM_BIN];
    __shared__ int shared_bin_offset[NUM_BIN];
    if (threadIdx.x < NUM_BIN)
    {
        shared_bin_size[threadIdx.x] = 0;
    }
    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int row_nnz, j;
    int range[NUM_BIN] = {26, 426, 853, 1706, 3413, 6826, 10240, INT_MAX};
    if (i < M)
    {
        row_nnz = d_row_flop[i];
        for (j = 0; j < NUM_BIN; j++)
        {
            if (row_nnz <= range[j])
            {
                atomicAdd(shared_bin_size + j, 1);
                goto before_end;
            }
        }
    }
before_end:

    __syncthreads();
    if (threadIdx.x < NUM_BIN)
    {
        shared_bin_offset[threadIdx.x] = atomicAdd(d_bin_size + threadIdx.x, shared_bin_size[threadIdx.x]);
        shared_bin_offset[threadIdx.x] += d_bin_offset[threadIdx.x];
        shared_bin_size[threadIdx.x] = 0;
    }
    __syncthreads();

    int index;
    if (i < M)
    {
        for (j = 0; j < NUM_BIN; j++)
        {
            if (row_nnz <= range[j])
            {
                index = atomicAdd(shared_bin_size + j, 1);
                d_bins[shared_bin_offset[j] + index] = i;
                return;
            }
        }
    }
}

__global__ void k_binning_small(
    int *d_bins, int M)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= M)
    {
        return;
    }
    d_bins[i] = i;
}

__global__ void k_tileOR_binning(
    int *__restrict__ d_row_nnz,
    int M,
    int *__restrict__ d_bin_size,
    int *__restrict__ d_total_nnz,
    int *__restrict__ d_max_row_nnz)
{
    __shared__ int shared_bin_size[NUM_BIN];
    __shared__ int shared_local_nnz[1];
    __shared__ int shared_max_row_nnz[1];
    if (threadIdx.x < NUM_BIN)
    {
        shared_bin_size[threadIdx.x] = 0;
    }
    if (threadIdx.x == 32)
    {
        shared_local_nnz[0] = 0;
        shared_max_row_nnz[0] = 0;
    }
    __syncthreads();
    int range[NUM_BIN] = {16, 128, 256, 512, 1024, 2048, 4095, INT_MAX}; // 2x
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int row_nnz, j;
    if (i < M)
    {
        row_nnz = d_row_nnz[i];
        atomicAdd(shared_local_nnz, row_nnz);
        atomicMax(shared_max_row_nnz, row_nnz);
        for (j = 0; j < NUM_BIN; j++)
        {
            if (row_nnz <= range[j])
            {
                atomicAdd(shared_bin_size + j, 1);
                goto before_end;
            }
        }
    }
before_end:

    __syncthreads();
    if (threadIdx.x < NUM_BIN)
    {
        atomicAdd(d_bin_size + threadIdx.x, shared_bin_size[threadIdx.x]);
    }
    if (threadIdx.x == 32)
    {
        atomicAdd(d_total_nnz, shared_local_nnz[0]);
    }
    if (threadIdx.x == 64)
    {
        atomicMax(d_max_row_nnz, shared_max_row_nnz[0]);
    }
}

__global__ void k_tileOR_binning2(
    int *__restrict__ d_row_nnz,
    int M,
    int *__restrict__ d_bins,
    int *__restrict__ d_bin_size,
    int *__restrict__ d_bin_offset)
{
    __shared__ int shared_bin_size[NUM_BIN];
    __shared__ int shared_bin_offset[NUM_BIN];
    if (threadIdx.x < NUM_BIN)
    {
        shared_bin_size[threadIdx.x] = 0;
    }
    __syncthreads();
    int range[NUM_BIN] = {16, 128, 256, 512, 1024, 2048, 4095, INT_MAX}; // 2x
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int row_nnz, j;
    if (i < M)
    {
        row_nnz = d_row_nnz[i];
        for (j = 0; j < NUM_BIN; j++)
        {
            if (row_nnz <= range[j])
            {
                atomicAdd(shared_bin_size + j, 1);
                goto before_end;
            }
        }
    }
before_end:

    __syncthreads();
    if (threadIdx.x < NUM_BIN)
    {
        shared_bin_offset[threadIdx.x] = atomicAdd(d_bin_size + threadIdx.x, shared_bin_size[threadIdx.x]);
        shared_bin_offset[threadIdx.x] += d_bin_offset[threadIdx.x];
        shared_bin_size[threadIdx.x] = 0;
    }
    __syncthreads();
    int index;
    if (i < M)
    {
        for (j = 0; j < NUM_BIN; j++)
        {
            if (row_nnz <= range[j])
            {
                index = atomicAdd(shared_bin_size + j, 1);
                d_bins[shared_bin_offset[j] + index] = i;
                return;
            }
        }
    }
}

__global__ void k_formCcol_binning(
    int *__restrict__ d_ctile,
    int *__restrict__ d_row_nnz,
    int M,
    int *__restrict__ d_bin_size,
    int *__restrict__ d_total_nnz)
{
    __shared__ int shared_bin_size[NUM_BIN_FOR_Ccol];
    __shared__ int shared_local_nnz[1];
    if (threadIdx.x < NUM_BIN_FOR_Ccol)
    {
        shared_bin_size[threadIdx.x] = 0;
    }
    if (threadIdx.x == 32)
    {
        shared_local_nnz[0] = 0;
    }
    __syncthreads();
    // int range[NUM_BIN_FOR_Ccol] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 12287, 24575, INT_MAX};
    int range[NUM_BIN_FOR_Ccol] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 24576, INT_MAX}; // for 14bin
    // int range[NUM_BIN_FOR_Ccol] = {8, 16, 32, 64, 128, 256, 512, 1024, INT_MAX}; //for 9bin
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int row_nnz, j;
    if (i < M)
    {
        row_nnz = d_row_nnz[i];
        atomicAdd(shared_local_nnz, row_nnz);
        for (j = 0; j < Critical_bin_id; j++)
        {
            if (row_nnz <= range[j])
            {
                atomicAdd(shared_bin_size + j, 1);
                goto before_end;
            }
        }
        for (; j < (NUM_BIN_FOR_Ccol); j++)
        {
            if (row_nnz <= range[j])
            {
                if (row_nnz / (d_ctile[i + 1] - d_ctile[i]) < Cnnz_ctile_rate_Threshold)
                {
                    atomicAdd(shared_bin_size + j, 1);
                    goto before_end;
                }
                else
                {
                    atomicAdd(shared_bin_size + NUM_BIN_FOR_Ccol - 1, 1);
                    goto before_end;
                }
            }
        }
    }
before_end:

    __syncthreads();
    if (threadIdx.x < NUM_BIN_FOR_Ccol)
    {
        atomicAdd(d_bin_size + threadIdx.x, shared_bin_size[threadIdx.x]);
    }
    if (threadIdx.x == 32)
    {
        atomicAdd(d_total_nnz, shared_local_nnz[0]);
    }
}

__global__ void k_formCcol_binning2(
    int *__restrict__ d_ctile,
    int *__restrict__ d_row_nnz,
    int M,
    int *__restrict__ d_bins,
    int *__restrict__ d_bin_size,
    int *__restrict__ d_bin_offset)
{
    __shared__ int shared_bin_size[NUM_BIN_FOR_Ccol];
    __shared__ int shared_bin_offset[NUM_BIN_FOR_Ccol];
    if (threadIdx.x < NUM_BIN_FOR_Ccol)
    {
        shared_bin_size[threadIdx.x] = 0;
    }
    __syncthreads();
    // int range[NUM_BIN_FOR_Ccol] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 12287, 24575, INT_MAX};
    int range[NUM_BIN_FOR_Ccol] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 24576, INT_MAX}; // for 14bin
    // int range[NUM_BIN_FOR_Ccol] = {8, 16, 32, 64, 128, 256, 512, 1024, INT_MAX}; //for 9bin
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int row_nnz, j;
    if (i < M)
    {
        row_nnz = d_row_nnz[i];
        for (j = 0; j < (Critical_bin_id); j++)
        {
            if (row_nnz <= range[j])
            {
                atomicAdd(shared_bin_size + j, 1);
                goto before_end;
            }
        }
        for (; j < (NUM_BIN_FOR_Ccol); j++)
        {
            if (row_nnz <= range[j])
            {
                if (row_nnz / (d_ctile[i + 1] - d_ctile[i]) < Cnnz_ctile_rate_Threshold)
                {
                    atomicAdd(shared_bin_size + j, 1);
                    goto before_end;
                }
                else
                {
                    atomicAdd(shared_bin_size + NUM_BIN_FOR_Ccol - 1, 1);
                    goto before_end;
                }
            }
        }
    }
before_end:

    __syncthreads();
    if (threadIdx.x < NUM_BIN_FOR_Ccol)
    {
        shared_bin_offset[threadIdx.x] = atomicAdd(d_bin_size + threadIdx.x, shared_bin_size[threadIdx.x]);
        shared_bin_offset[threadIdx.x] += d_bin_offset[threadIdx.x];
        shared_bin_size[threadIdx.x] = 0;
    }
    __syncthreads();
    int index;
    if (i < M)
    {
        for (j = 0; j < (Critical_bin_id); j++)
        {
            if (row_nnz <= range[j])
            {
                index = atomicAdd(shared_bin_size + j, 1);
                d_bins[shared_bin_offset[j] + index] = i;
                return;
            }
        }
        for (; j < (NUM_BIN_FOR_Ccol); j++)
        {
            if (row_nnz <= range[j])
            {
                if (row_nnz / (d_ctile[i + 1] - d_ctile[i]) < Cnnz_ctile_rate_Threshold)
                {
                    index = atomicAdd(shared_bin_size + j, 1);
                    d_bins[shared_bin_offset[j] + index] = i;
                    return;
                }
                else
                {
                    index = atomicAdd(shared_bin_size + NUM_BIN_FOR_Ccol - 1, 1);
                    d_bins[shared_bin_offset[NUM_BIN_FOR_Ccol - 1] + index] = i;
                    return;
                }
            }
        }
    }
}

__global__ void look_for_which_bin(int rowid, int *d_bins, int numsrow)
{

    for (int i = threadIdx.x; i < numsrow; i += blockDim.x)
    {
        if (d_bins[i] == rowid)
        {
            printf("the %dth row's order in d_bins is %d\n", rowid, i);
        }
    }
}

inline void h_formcol_binning(compressed_bin *compressed_bin, NHC_CSR *C)
{
    CHECK_ERROR(cudaMemsetAsync(compressed_bin->d_bin_size, 0, (NUM_BIN_FOR_Ccol + 1) * sizeof(int), compressed_bin->streams[0]));
    int BS = 1024;
    int GS = div_up(C->M, BS);
    k_formCcol_binning<<<GS, BS, 0, compressed_bin->streams[0]>>>(C->d_tile_ptr, C->d_ptr, C->M,
                                                                  compressed_bin->d_bin_size, compressed_bin->d_total_nnz);
    CHECK_ERROR(cudaMemcpyAsync(compressed_bin->bin_size, compressed_bin->d_bin_size, (NUM_BIN_FOR_Ccol + 1) * sizeof(int), cudaMemcpyDeviceToHost, compressed_bin->streams[0]));
    CHECK_ERROR(cudaStreamSynchronize(compressed_bin->streams[0]));
    CHECK_ERROR(cudaMemsetAsync(compressed_bin->d_bin_size, 0, NUM_BIN_FOR_Ccol * sizeof(int), compressed_bin->streams[0]));
    compressed_bin->bin_offset[0] = 0;
    for (int i = 0; i < NUM_BIN_FOR_Ccol - 1; i++)
    {
        compressed_bin->bin_offset[i + 1] = compressed_bin->bin_offset[i] + compressed_bin->bin_size[i];
    }
    CHECK_ERROR(cudaMemcpyAsync(compressed_bin->d_bin_offset, compressed_bin->bin_offset, NUM_BIN_FOR_Ccol * sizeof(int), cudaMemcpyHostToDevice, compressed_bin->streams[0]));
    k_formCcol_binning2<<<GS, BS, 0, compressed_bin->streams[0]>>>(C->d_tile_ptr, C->d_ptr, C->M,
                                                                   compressed_bin->d_bins, compressed_bin->d_bin_size, compressed_bin->d_bin_offset);
}

void partition_for_compressed_NHC(compressed_bin *compressed_bin, int *c_ptr, int nums_row)
{
    compressed_bin->bins[0] = 0;
    compressed_bin->num_of_partion = partion_C(c_ptr, compressed_bin->bins + 1, nums_row);
    cudaMemcpy(compressed_bin->d_bins, compressed_bin->bins, (compressed_bin->num_of_partion + 1) * sizeof(int), cudaMemcpyHostToDevice);
}
