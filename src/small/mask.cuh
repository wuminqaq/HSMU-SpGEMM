__global__ void Form_mask_array_for_CSR_kernel(int begin, int *colidx, DateTypeStoreMask *mask_array)
{
    int ti = blockDim.x * blockIdx.x + threadIdx.x;
    int serial_num_of_nnz = ti + begin;
    DateTypeStoreMask tmp = 1;
    int col_num = colidx[serial_num_of_nnz];
    int i, k;
    i = col_num >> 6;
    k = (col_num) & (63);
    tmp = tmp << k;
    atomicOr(&mask_array[i], tmp);
}

__global__ void k_compute_flop(
    const index_t *__restrict__ d_arpt,
    const index_t *__restrict__ d_acol,
    const index_t *__restrict__ d_brpt,
    int M,
    index_t *d_row_flop,
    index_t *d_max_row_flop)
{
    __shared__ index_t shared_max_row_flop[1];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M)
    {
        return;
    }
    if (threadIdx.x == 0)
    {
        shared_max_row_flop[0] = 0;
    }
    __syncthreads();
    int row_flop = 0;
    int j;
    int acol;
    int arow_start, arow_end;
    arow_start = d_arpt[i];
    arow_end = d_arpt[i + 1];
    for (j = arow_start; j < arow_end; j++)
    {
        acol = d_acol[j];
        row_flop += d_brpt[acol];
    }
    d_row_flop[i] = row_flop;
    atomicMax(shared_max_row_flop, row_flop);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicMax(d_max_row_flop, shared_max_row_flop[0]);
    }
}
__global__ void csr_to_row_indices(const int *csr_row_ptr, int *row_indices, int num_rows)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    for (int row = thread_id; row < num_rows; row += total_threads)
    {
        int start = csr_row_ptr[row];
        int end = csr_row_ptr[row + 1];
        for (int i = start; i < end; ++i)
        {
            row_indices[i] = row;
        }
    }
}

__global__ void csr2coo_kernel(const int *row_ptr, int *coo_row, int Annz, int num_rows)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < Annz)
    {
        int row = 0;
        for (int i = 1; i <= num_rows; i++)
        {
            if (tid < row_ptr[i])
            {
                row = i - 1;
                break;
            }
        }
        coo_row[tid] = row;
    }
}

int partion_B_for_mask(int *b_ptr, int *bin, int nums_row)
{
    int num_partition = 0;
    int Upper_limit = NnzB_per_partion_for_initial_mask;
    int i = 0;
    while (i <= nums_row)
    {
        if (b_ptr[i] > Upper_limit)
        {
            bin[num_partition] = i - 1;
            Upper_limit = b_ptr[i - 1] + NnzB_per_partion_for_initial_mask;
            num_partition++;
        }
        i++;
    }
    bin[num_partition] = nums_row;
    return num_partition + 1;
}
// form B mask matrix
void matrixB_partion_for_mask(int *b_ptr, NHC_bin *c_bin, int nums_row)
{
    c_bin->bin = new int[nums_row](); //
    c_bin->bin[0] = 0;
    c_bin->num_of_partion = partion_B_for_mask(b_ptr, c_bin->bin + 1, nums_row);
    cudaMalloc((void **)&c_bin->d_bin, (nums_row) * sizeof(int)); //
    c_bin->d_max_nnz_in_onerow = &(c_bin->d_bin[nums_row - 1]);
    cudaMemset(c_bin->d_max_nnz_in_onerow, 0, sizeof(int));
    cudaMemcpy(c_bin->d_bin, c_bin->bin, (c_bin->num_of_partion + 1) * sizeof(int), cudaMemcpyHostToDevice);
}
__global__ void Form_mask_matrix_for_CSR_kernel(int *d_bin, int *B_bptr, int *B_drow, int *B_dcol, DateTypeStoreMask *d_mask_matrix, int pitch)
{
    int index;
    int loc_par_id = blockIdx.x;
    int off_row_id = d_bin[loc_par_id];
    int end_row_id = d_bin[loc_par_id + 1];
    int start_Bnnz = B_bptr[off_row_id];
    int end_Bnnz = B_bptr[end_row_id];
    int Bcol, Brow;
    int i, k;
    DateTypeStoreMask tmp;
    DateTypeStoreMask *adress_of_row;
    for (index = threadIdx.x + start_Bnnz; index < end_Bnnz; index += blockDim.x)
    {
        Brow = B_drow[index];
        Bcol = B_dcol[index];
        tmp = 1;
        adress_of_row = (DateTypeStoreMask *)((char *)d_mask_matrix + Brow * pitch);
        i = Bcol >> 6;
        k = (Bcol) & (63);
        tmp = tmp << k;
        atomicOr(adress_of_row + i, tmp);
    }
}
void Initialize_mask_matrix(NHC_mask_matrix *mask_mat, NHC_CSR *B, NHC_bin *c_bin)
{                                                                       
    cudaMalloc((void **)&(mask_mat->d_Brow_id), (B->nnz) * sizeof(int)); 
    int block_size = 256;
    int grid_size = ((B->nnz) + block_size - 1) / block_size;
    csr2coo_kernel<<<grid_size, block_size>>>(B->d_ptr, mask_mat->d_Brow_id, B->nnz, B->M);
    mask_mat->num_unit = (B->N + 8 * sizeof(DateTypeStoreMask) - 1) / (8 * sizeof(DateTypeStoreMask));
    size_t pitch;
    cudaMallocPitch((void **)&mask_mat->mask_matrix, &pitch, mask_mat->num_unit * sizeof(DateTypeStoreMask), B->M);
    cudaMemset2D(mask_mat->mask_matrix, pitch, 0, mask_mat->num_unit * sizeof(DateTypeStoreMask), B->M);
    mask_mat->pitch = pitch;
    grid_size = c_bin->num_of_partion;
    Form_mask_matrix_for_CSR_kernel<<<grid_size, block_size>>>(c_bin->d_bin, B->d_ptr, mask_mat->d_Brow_id, B->d_col, mask_mat->mask_matrix, mask_mat->pitch);
    cudaFree(mask_mat->d_Brow_id);
}

__global__ void check_mask_matrix(int row_num, NHC_mask_matrix mask_mat)
{
    DateTypeStoreMask *adress_of_row = (DateTypeStoreMask *)((char *)mask_mat.mask_matrix + (row_num * mask_mat.pitch));
    for (int i = 0; i < mask_mat.num_unit; i++)
    {
        if (adress_of_row[i])
        {
            printf("one nnz in the %d row's the %dth longlong is %llu\n", row_num, i, adress_of_row[i]);
        }
    }
}
__global__ void check_after_or_mask_matrix(int row_num, NHC_mask_matrix mask_mat)
{
    DateTypeStoreMask *adress_of_row = (DateTypeStoreMask *)((char *)mask_mat.after_or_mask_matrix + (row_num * mask_mat.pitch));
    for (int i = 0; i < mask_mat.num_unit; i++)
    {
        if (adress_of_row[i])
        {
            printf("one nnz in the %d row's the %dth longlong is %llu\n", row_num, i, adress_of_row[i]);
        }
    }
}