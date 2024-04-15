void partitionMatrix_count_function(int start_row, int end_row, int *rowptr, int *numPartitions)
{
    int sum_nnz = rowptr[end_row] - rowptr[start_row];
    int limit_array[6] = {4096, 2048, 1024, 512, 256, 0};
    // get interval
    int interval;
    for (interval = 0; interval < 6; interval++)
    {
        if (sum_nnz > limit_array[interval])
        {
            break;
        }
    }
    int i = end_row;
    ;
    switch (interval)
    {
    case 0:
        while (sum_nnz > limit_array[0])
        {
            i--;
            if (rowptr[i] - rowptr[start_row] <= 4096)
            {
                start_row = i;
                i = end_row;
                numPartitions[0]++;
                sum_nnz = rowptr[end_row] - rowptr[start_row];
            }
        }
        if (sum_nnz > 2048)
        {
            numPartitions[0]++;
        }
        else if (sum_nnz > 1024)
        {
            numPartitions[1]++;
        }
        else if (sum_nnz > 512)
        {
            numPartitions[2]++;
        }
        else if (sum_nnz > 256)
        {
            numPartitions[3]++;
        }
        else
        {
            numPartitions[4]++;
        }

        break;
    case 1:
        numPartitions[0]++;
        break;
    case 2:
        numPartitions[1]++;
        break;
    case 3:
        numPartitions[2]++;
        break;
    case 4:
        numPartitions[3]++;
        break;
    case 5:
        numPartitions[4]++;
        break;
    }
}

int partiton_cout_with_special(int *rowptr, int *special_row, int num_special, int *numPartitions)
{
    int start_row;
    int end_row;
    // get the num of each partion
    for (int i = 1; i <= num_special + 1; i++)
    {
        start_row = special_row[i - 1] + 1;
        end_row = special_row[i];
        if (end_row > start_row)
        {
            partitionMatrix_count_function(start_row, end_row, rowptr, numPartitions);
        }
    }
    // compute all num of partion
    int all_num_of_partion = 0;
    for (int i = 0; i < 5; i++)
    {
        all_num_of_partion += numPartitions[i];
    }
    return all_num_of_partion;
}

void decide_partition_function(int start_row, int end_row, int *rowptr, int *numPartitions,
                               int *Partition0, int *Partition1, int *Partition2,
                               int *Partition3, int *Partition4, int *Partition5)
{

    int sum_nnz = rowptr[end_row] - rowptr[start_row];
    int limit_array[6] = {4096, 2048, 1024, 512, 256, 0};
    // get interval
    int interval;
    for (interval = 0; interval < 6; interval++)
    {
        if (sum_nnz > limit_array[interval])
        {
            break;
        }
    }
    int i = end_row;
    ;
    switch (interval)
    {
    case 0:
        while (sum_nnz > limit_array[0])
        {
            i--;
            if (rowptr[i] - rowptr[start_row] <= 4096)
            {
                start_row = i;
                i = end_row;
                numPartitions[0]++;
                sum_nnz = rowptr[end_row] - rowptr[start_row];
            }
        }
        if (sum_nnz > 2048)
        {
            Partition0[numPartitions[0] << 1] = start_row;
            Partition0[(numPartitions[0] << 1) + 1] = end_row;
            numPartitions[0]++;
        }
        else if (sum_nnz > 1024)
        {
            Partition1[numPartitions[1] << 1] = start_row;
            Partition1[(numPartitions[1] << 1) + 1] = end_row;
            numPartitions[1]++;
        }
        else if (sum_nnz > 512)
        {
            Partition2[numPartitions[2] << 1] = start_row;
            Partition2[(numPartitions[2] << 1) + 1] = end_row;
            numPartitions[2]++;
        }
        else if (sum_nnz > 256)
        {
            Partition3[numPartitions[3] << 1] = start_row;
            Partition3[(numPartitions[3] << 1) + 1] = end_row;
            numPartitions[3]++;
        }
        else
        {
            Partition4[numPartitions[4] << 1] = start_row;
            Partition4[(numPartitions[4] << 1) + 1] = end_row;
            numPartitions[4]++;
        }
        break;
    case 1:
        Partition0[numPartitions[0] << 1] = start_row;
        Partition0[(numPartitions[0] << 1) + 1] = end_row;
        numPartitions[0]++;
        break;
    case 2:
        Partition1[numPartitions[1] << 1] = start_row;
        Partition1[(numPartitions[1] << 1) + 1] = end_row;
        numPartitions[1]++;
        break;
    case 3:
        Partition2[numPartitions[2] << 1] = start_row;
        Partition2[(numPartitions[2] << 1) + 1] = end_row;
        numPartitions[2]++;
        break;
    case 4:
        Partition3[numPartitions[3] << 1] = start_row;
        Partition3[(numPartitions[3] << 1) + 1] = end_row;
        numPartitions[3]++;
        break;
    case 5:
        Partition4[numPartitions[4] << 1] = start_row;
        Partition4[(numPartitions[4] << 1) + 1] = end_row;
        numPartitions[4]++;
        break;
    }
}

void decide_partition(int *rowptr, int *special_row, int num_special, int *numPartitions,
                      int *Partition0, int *Partition1, int *Partition2,
                      int *Partition3, int *Partition4, int *Partition5)
{
    int start_row;
    int end_row;
    for (int i = 0; i < 6; i++)
    {
        numPartitions[i] = 0;
    }
    // get the num of each partion
    for (int i = 1; i <= num_special + 1; i++)
    {
        start_row = special_row[i - 1] + 1;
        end_row = special_row[i];
        if (end_row > start_row)
        {
            decide_partition_function(start_row, end_row, rowptr, numPartitions, Partition0, Partition1, Partition2, Partition3, Partition4, Partition5);
        }
    }
    // compute all num of partion
}

int partion_C(int *c_ptr, int *bin, int nums_row)
{
    int num_partition = 0;
    int Upper_limit = 4096;
    int i = 0;
    while (i <= nums_row)
    {
        if (c_ptr[i] > Upper_limit)
        {
            bin[num_partition] = i - 1;
            Upper_limit = c_ptr[i - 1] + 4096;
            num_partition++;
        }
        i++;
    }
    bin[num_partition] = nums_row;
    return num_partition + 1;
}

void matrix_partion_for_numeric_compute(int *c_ptr, NHC_bin *c_bin, int nums_row)
{
    c_bin->bin[0] = 0;
    c_bin->num_of_partion = partion_C(c_ptr, c_bin->bin + 1, nums_row);
    cudaMemcpy(c_bin->d_bin, c_bin->bin, (c_bin->num_of_partion + 1) * sizeof(int), cudaMemcpyHostToDevice);
}

__device__ __forceinline__ int Binary_search_for_hash_loction(const int *__restrict__ shared_col, int left, int right, int bcol)
{
    while (left <= right)
    {
        int mid = left + ((right - left) >> 1);
        if (shared_col[mid] == bcol)
        {
            return mid;
        }
        else if (shared_col[mid] < bcol)
        {
            left = mid + 1;
        }
        else
        {
            right = mid - 1;
        }
    }
    return -1;
}

template <int hash_size>
__global__ void old_kernel_compute_numeric(int *d_arow, int *d_aptr, int *d_acol, value_t *d_aval, int *d_brpt, int *d_bcol, value_t *d_bval, int *bin, int *d_ccol, int *d_cptr, value_t *d_cval, int num_of_partion)
{
    int loc_par_id;
    int j, k;
    int tid = threadIdx.x & (WSIZE_FOR_small_num - 1);
    int wid = threadIdx.x / WSIZE_FOR_small_num;
    int wnum = blockDim.x / WSIZE_FOR_small_num;

    extern __shared__ int shared_mem[];
    const int tsize = 4096;
    index_t *shared_col = shared_mem;
    value_t *shared_val;
    for (loc_par_id = blockIdx.x; loc_par_id < num_of_partion; loc_par_id += gridDim.x)
    {
        int off_row_id = bin[loc_par_id];
        int end_row_id = bin[loc_par_id + 1];
        int start_Annz = d_aptr[off_row_id];
        int start_Cnnz = d_cptr[off_row_id];
        int end_Annz = d_aptr[end_row_id];
        int end_Cnnz = d_cptr[end_row_id];
        int size_c = end_Cnnz - start_Cnnz;

        int arow, acol, bcol, hash, start, end;
        value_t aval, bval;
        if (size_c <= hash_size)
        {
            shared_val = (value_t *)(shared_mem + tsize);
            for (j = threadIdx.x; j < size_c; j += blockDim.x)
            {
                shared_col[j] = d_ccol[start_Cnnz + j];
                shared_val[j] = 0;
            }
            __syncthreads();
            for (j = start_Annz + wid; j < end_Annz; j += wnum)
            {
                aval = d_aval[j];
                acol = d_acol[j];
                arow = d_arow[j];
                start = d_cptr[arow] - start_Cnnz;
                end = d_cptr[arow + 1] - start_Cnnz - 1;
                for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WSIZE_FOR_small_num)
                {
                    bcol = d_bcol[k];
                    bval = d_bval[k];
                    hash = Binary_search_for_hash_loction(shared_col, start, end, bcol);
                    atomicAdd(shared_val + hash, aval * bval);
                }
            }
            __syncthreads();
            for (j = threadIdx.x; j < size_c; j += blockDim.x)
            {
                d_cval[start_Cnnz + j] = shared_val[j];
            }
        }
        else if (size_c <= (hash_size * 3))
        {
            for (j = threadIdx.x; j < size_c; j += blockDim.x)
            {
                shared_col[j] = d_ccol[start_Cnnz + j];
            }
            __syncthreads();
            for (j = start_Annz + wid; j < end_Annz; j += wnum)
            {
                aval = d_aval[j];
                acol = d_acol[j];
                arow = d_arow[j];
                start = d_cptr[arow] - start_Cnnz;
                end = d_cptr[arow + 1] - start_Cnnz - 1;
                for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WSIZE_FOR_small_num)
                {
                    bcol = d_bcol[k];
                    bval = d_bval[k];
                    hash = Binary_search_for_hash_loction(shared_col, start, end, bcol);
                    atomicAdd(d_cval + hash + start_Cnnz, aval * bval);
                }
            }
        }
        else
        {
            for (j = start_Annz + wid; j < end_Annz; j += wnum)
            {
                aval = d_aval[j];
                acol = d_acol[j];
                arow = d_arow[j];
                start = d_cptr[arow];
                end = d_cptr[arow + 1] - 1;
                for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WSIZE_FOR_small_num)
                {
                    bcol = d_bcol[k];
                    bval = d_bval[k];
                    hash = Binary_search_for_hash_loction(d_ccol, start, end, bcol);
                    atomicAdd(d_cval + hash, aval * bval);
                }
            }
        }
    }
}

template <int hash_size>
__global__ void kernel_compute_numeric(int *d_arow, int *d_aptr, int *d_acol, value_t *d_aval, int *d_brpt, int *d_bcol, value_t *d_bval, int *bin, int *d_ccol, int *d_cptr, value_t *d_cval, int num_of_partion)
{
    int loc_par_id;
    int j, k;
    int tid = threadIdx.x & (WSIZE_FOR_small_num - 1);
    int wid = threadIdx.x / WSIZE_FOR_small_num;
    int wnum = blockDim.x / WSIZE_FOR_small_num;

    extern __shared__ int shared_mem[];
    const int tsize = 4096;
    index_t *shared_col = shared_mem;
    value_t *shared_val = (value_t *)(shared_mem + tsize);
    value_t aval, bval;
    for (loc_par_id = blockIdx.x; loc_par_id < num_of_partion; loc_par_id += gridDim.x)
    {
        int off_row_id = bin[loc_par_id];
        int end_row_id = bin[loc_par_id + 1];
        int start_Annz = d_aptr[off_row_id];
        int start_Cnnz = d_cptr[off_row_id];
        int end_Annz = d_aptr[end_row_id];
        int end_Cnnz = d_cptr[end_row_id];
        int size_c = end_Cnnz - start_Cnnz;
        int arow, acol, bcol, hash, start, end;
        // initial ccol and cval
        for (j = threadIdx.x; j < size_c; j += blockDim.x)
        {
            shared_col[j] = d_ccol[start_Cnnz + j];
            shared_val[j] = 0;
        }
        __syncthreads();
        for (j = start_Annz + wid; j < end_Annz; j += wnum)
        {
            aval = d_aval[j];
            acol = d_acol[j];
            arow = d_arow[j];
            start = d_cptr[arow] - start_Cnnz;
            end = d_cptr[arow + 1] - start_Cnnz - 1;
            for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += WSIZE_FOR_small_num)
            {
                bcol = d_bcol[k];
                bval = d_bval[k];
                hash = Binary_search_for_hash_loction(shared_col, start, end, bcol);
                atomicAdd(shared_val + hash, aval * bval);
            }
        }
        __syncthreads();
        // write in globel memory
        for (j = threadIdx.x; j < size_c; j += blockDim.x)
        {
            d_cval[start_Cnnz + j] = shared_val[j];
        }
    }
}

void numeric_compute(NHC_bin *c_bin, NHC_CSR *A, NHC_CSR *B, NHC_CSR *C)
{
    int *d_Arow_id;
    cudaMalloc((void **)&(d_Arow_id), (A->nnz) * sizeof(int));
    int block_size = 1024;
    int grid_size = ((A->M) + block_size - 1) / block_size;
    csr_to_row_indices<<<grid_size, block_size>>>(A->d_ptr, d_Arow_id, A->M);
    kernel_compute_numeric<4096><<<c_bin->num_of_partion, 1024, 49152>>>(d_Arow_id, A->d_ptr, A->d_col, A->d_val, B->d_ptr, B->d_col, B->d_val, c_bin->d_bin, C->d_col, C->d_ptr, C->d_val, c_bin->num_of_partion);
    cudaDeviceSynchronize();
    cudaFree(d_Arow_id);
}
