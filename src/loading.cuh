using namespace std;
void NHC_CSR_load(NHC_CSR *mat, index_t &row, index_t &col, index_t &nnz, std::ifstream &fin)
{
    fin.clear();
    fin.seekg(0);
    while (fin.peek() == '%')
        fin.ignore(2048, '\n');
    fin >> row >> col >> nnz;
    int *row_occurances = new int[row + 1]();
    int row_id, col_id, i;
    value_t val;
    for (i = 0; i < nnz; i++)
    {
        fin >> row_id >> col_id >> val;
        row_occurances[row_id - 1]++;
    }
    mat->rowPtr = new index_t[row + 1]();
    mat->rowPtr[0] = 0;
    mat->col = new index_t[nnz]();
    mat->val = new value_t[nnz]();
    index_t j = 0;
    for (i = 0; i < row + 1; ++i)
        mat->rowPtr[i] = j, j += row_occurances[i];

    delete[] row_occurances;
    fin.clear();
    fin.seekg(0);
    while (fin.peek() == '%')
        fin.ignore(2048, '\n');
    fin >> row_id >> col_id >> val;
    for (i = 0; i < nnz; ++i)
    {
        mat->col[i] = -1;
    }
    for (i = 0; i < nnz; i++)
    {
        int k = 0;
        fin >> row_id >> col_id >> val;
        row_id--;
        col_id--;
        while (mat->col[k + mat->rowPtr[row_id]] != -1)
        {
            ++k;
        }
        k = k + mat->rowPtr[row_id];
        mat->col[k] = col_id;
        mat->val[k] = val;
    }
}

void setup(NHC_CSR *A, NHC_CSR *B, NHC_CSR *C, compressed_bin *compressed_bin)
{
    C->M = A->M;
    C->N = B->N;
    compressed_bin->streams = new cudaStream_t[NUM_BIN_FOR_Ccol];
    for (int i = 0; i < NUM_BIN_FOR_Ccol; ++i)
    {
        cudaStreamCreate(&(compressed_bin->streams[i]));
    }
    cudaMalloc((void **)&(B->d_tile_ptr), (B->M + 1) * sizeof(int));
    cudaMalloc((void **)&C->d_tile_ptr, (A->M + 1) * sizeof(index_t));
    cudaMalloc((void **)&C->d_tem_ptr, (A->M + 1) * sizeof(index_t));
    cudaMalloc((void **)&C->d_ptr, (A->M + 1) * sizeof(index_t));
    cudaMalloc((void **)&compressed_bin->d_bin_offset, (NUM_BIN_FOR_Ccol) * sizeof(int));
    cudaMalloc((void **)&compressed_bin->d_bin_size, (NUM_BIN_FOR_Ccol + 1) * sizeof(int));
    cudaMalloc((void **)&(compressed_bin->d_bins), (C->M) * sizeof(int));
    cudaMemset(compressed_bin->d_bin_size, 0, (NUM_BIN_FOR_Ccol + 1) * sizeof(int));
    compressed_bin->d_max_row_nnz = compressed_bin->d_bin_size + NUM_BIN; // size 1
    compressed_bin->d_total_nnz = compressed_bin->d_bin_size + NUM_BIN_FOR_Ccol;

    compressed_bin->bins = new index_t[A->M]();
    C->rowPtr = new index_t[A->M + 1]();
    C->tile_ptr = new index_t[A->M + 1]();
    compressed_bin->bin_offset = new int[NUM_BIN_FOR_Ccol]();
    compressed_bin->bin_size = new int[NUM_BIN_FOR_Ccol + 1]();
    compressed_bin->max_row_nnz = compressed_bin->bin_size + NUM_BIN; // size 1
    compressed_bin->total_nnz = compressed_bin->bin_size + NUM_BIN_FOR_Ccol;
    compressed_bin->global_mem_pool_malloced = false;

    compressed_bin->d_temp_storage = nullptr;
    uint32_t Max = max(A->M, B->M);
    cub::DeviceScan::ExclusiveSum(compressed_bin->d_temp_storage, compressed_bin->temp_storage_bytes, C->d_ptr, C->d_ptr, Max + 1, 0);
    cudaMalloc(&compressed_bin->d_temp_storage, compressed_bin->temp_storage_bytes);
}

void HtD_value_col_ptr(NHC_CSR *A)
{
    cudaMalloc((void **)&A->d_ptr, (A->M + 1) * sizeof(index_t));
    cudaMalloc((void **)&A->d_col, (A->nnz) * sizeof(index_t));
    cudaMalloc((void **)&A->d_val, (A->nnz) * sizeof(value_t));
    cudaMemcpy(A->d_ptr, A->rowPtr, (A->M + 1) * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(A->d_col, A->col, (A->nnz) * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(A->d_val, A->val, (A->nnz) * sizeof(value_t), cudaMemcpyHostToDevice);
}
void realse_NHC_tile(NHC_CSR *tile)
{
    cudaFree(tile->d_tile_ptr);
    delete[] tile->tile_ptr;
    cudaFree(tile->d_mask_num);
    cudaFree(tile->d_tile_col);
#if checek_release
    printf("get down realse_NHC_tile\n");
#endif
}
void realse_compressed_bin(compressed_bin *compressed_bin)
{
    cudaFree(compressed_bin->d_bin_size);
    cudaFree(compressed_bin->d_bins);
    cudaFree(compressed_bin->d_bin_offset);
    delete[] compressed_bin->bin_offset;
    delete[] compressed_bin->bin_size;
    delete[] compressed_bin->bins;
    for (int i = 0; i < NUM_BIN_FOR_Ccol; ++i)
    {
        cudaStreamDestroy(compressed_bin->streams[i]);
    }
    delete[] compressed_bin->streams;
    if (compressed_bin->global_mem_pool_malloced)
    {
        cudaFree(compressed_bin->d_global_mem_pool);
    }
    cudaFree(compressed_bin->d_temp_storage);

#if checek_release
    printf("get down realse_compressed_bin\n");
#endif
}
void realse_NHCcsr(NHC_CSR *A)
{
    cudaFree(A->d_ptr);
    cudaFree(A->d_col);
    cudaFree(A->d_val);
    delete[] A->rowPtr;
    delete[] A->col;
    delete[] A->val;
#if checek_release
    printf("get down realse_NHCcsr\n");
#endif
}

void free_mask_matrix(NHC_mask_matrix *mask_matrixB)
{

    cudaFree(mask_matrixB->mask_matrix);
    cudaFree(mask_matrixB->after_or_mask_matrix);
    cudaFree(mask_matrixB->d_nums_over_limit_hashtable);
#if checek_release
    printf("get down free_mask_matrix\n");
#endif
}

void free_C_matrix(NHC_CSR *C)
{
    cudaFree(C->d_ptr);
    cudaFree(C->d_val);
    cudaFree(C->d_col);
    delete[] C->rowPtr;
#if CHECK_RESULT
    delete[] C->col;
    delete[] C->val;
#endif
#if checek_release
    printf("get down free_C_matrix\n");
#endif
}

void free_bin(NHC_bin *c_bin)
{
    delete[] c_bin->bin;
    cudaFree(c_bin->d_bin);
#if checek_release
    printf("get down free_bin\n");
#endif
}
