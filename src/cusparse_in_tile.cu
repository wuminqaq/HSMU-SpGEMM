#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdint>
int tile_spgemm_cusparse_executor(cusparseHandle_t handle, cusparseSpMatDescr_t matA,
                                  const int mA,
                                  const int nA,
                                  const int nnzA,
                                  const int *d_csrRowPtrA,
                                  const int *d_csrColIdxA,
                                  const value_t *d_csrValA,
                                  cusparseSpMatDescr_t matB,
                                  const int mB,
                                  const int nB,
                                  const int nnzB,
                                  const int *d_csrRowPtrB,
                                  const int *d_csrColIdxB,
                                  const value_t *d_csrValB,
                                  cusparseSpMatDescr_t matC,
                                  const int mC,
                                  const int nC,
                                  unsigned long long int *nnzC,
                                  int **d_csrRowPtrC,
                                  int **d_csrColIdxC,
                                  value_t **d_csrValC)
{
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_64F;
    void *dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;

    double alpha = 1.0f;
    double beta = 0.0f;

    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void **)d_csrRowPtrC, (mC + 1) * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaMalloc d_csrRowPtrC is failed, mC + 1 is %d\n", mC + 1);
    }
    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseSpGEMM_createDescr(&spgemmDesc);

    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, NULL);
    cudaMalloc((void **)&dBuffer1, bufferSize1);
    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, dBuffer1);

    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, NULL);
    cudaMalloc((void **)&dBuffer2, bufferSize2);
    // compute the intermediate product of A * B
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, dBuffer2);
    // get matrix C non-zero entries C_num_nnz1
    int64_t C_num_rows1, C_num_cols1, C_num_nnz1;
    cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_num_nnz1);
    // allocate matrix C
    cudaStatus = cudaMalloc((void **)d_csrColIdxC, C_num_nnz1 * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaMalloc d_csrColIdxC is failed, C_num_nnz1 is %d\n", C_num_nnz1);
    }
    cudaStatus = cudaMalloc((void **)d_csrValC, C_num_nnz1 * sizeof(value_t));
    cudaMemset(*d_csrValC, 0, C_num_nnz1 * sizeof(value_t));
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaMalloc d_csrValC is failed, C_num_nnz1 is %d\n", C_num_nnz1);
    }
    // update matC with the new pointers
    cusparseCsrSetPointers(matC, *d_csrRowPtrC, *d_csrColIdxC, *d_csrValC);
    cudaDeviceSynchronize();
    cusparseSpGEMM_copy(handle, opA, opB,
                        &alpha, matA, matB, &beta, matC,
                        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);

    *nnzC = C_num_nnz1;
    cusparseSpGEMM_destroyDescr(spgemmDesc);
    return 0;
}

void tile_cusparse_spgemm(NHC_CSR *a, NHC_CSR *b, NHC_CSR *test_C, NHC_CSR *c)
{
    int tmp_nnz;
    double msec;
    struct timeval t1, t2;
    bool check_result = true;
    int error_times = 0;

    int *d_csrRowPtrC;
    int *d_csrColIdxC;
    value_t *d_csrValC;
    // CUSPARSE API
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    unsigned long long int nnzC;
    const int BENCH_REPEAT = 1;

    cusparseCreate(&handle);
    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, a->M, a->N, a->nnz,
                      a->d_ptr, a->d_col, a->d_val,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateCsr(&matB, b->M, b->N, b->nnz,
                      b->d_ptr, b->d_col, b->d_val,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateCsr(&matC, a->M, b->N, 0,
                      NULL, NULL, NULL,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    gettimeofday(&t1, NULL);
    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        tile_spgemm_cusparse_executor(handle, matA, a->M, a->N, a->nnz, a->d_ptr, a->d_col, a->d_val,
                                      matB, b->M, b->N, b->nnz, b->d_ptr, b->d_col, b->d_val,
                                      matC, a->M, b->N, &nnzC, &d_csrRowPtrC, &d_csrColIdxC, &d_csrValC);
        if (i != BENCH_REPEAT - 1)
        {
            cudaFree(d_csrRowPtrC);
            cudaFree(d_csrColIdxC);
            cudaFree(d_csrValC);
        }
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    msec = 0;
    msec = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    msec /= BENCH_REPEAT;

    printf("the cusparse C_nnz is %lld\n", nnzC);
    c->val = new value_t[c->nnz]();
    c->col = new index_t[c->nnz]();
    cudaMemcpy(c->val, c->d_val, (c->nnz) * sizeof(value_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(c->col, c->d_col, (c->nnz) * sizeof(index_t), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("cudaMemcpy(c->col, c->d_col) is failed\n");
    }
    else
    {
        printf("cudaMemcpy(c->col, c->d_col) is cudaSuccess\n");
    }
    int numsofzer = 0;
    int numsofidzer = 0;
    for (int i = 0; i < (c->nnz); i++)
    {
        if (c->val[i] == 0.0)
        {
            numsofzer++;
        }
        if (c->col[i] == 0)
        {
            numsofidzer++;
        }
    }
    printf("numsofzer is %d\n", numsofzer);
    printf("numsofidzer is %d\n", numsofidzer);

    if (nnzC <= 0)
    {
        printf("cuSPARSE failed!\n");
        return;
    }
    else
    {

        printf("the run time using cusparse for the %s is:%f[ms]\n", a->matrix_name, msec);
        int *h_csrRowPtrC = new int[a->M + 1]();
        int *h_csrColIdxC = new int[nnzC]();
        value_t *h_csrValC = new value_t[nnzC]();

        cudaError_t cudaStatus;
        cudaStatus = cudaMemcpy(h_csrRowPtrC, d_csrRowPtrC, (a->M + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMemcpy h_csrRowPtrC is failed\n");
        }
        cudaMemcpy(h_csrColIdxC, d_csrColIdxC, nnzC * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csrValC, d_csrValC, nnzC * sizeof(value_t), cudaMemcpyDeviceToHost);

#ifdef CHECK_RESULT
        for (int i = 0; i < (a->M); i++)
        {
            if (h_csrRowPtrC[i] != c->rowPtr[i])
            {
                check_result = false;
                if (error_times < 10)
                {
                    printf("the cusparse ptr[%d] is %d,the NHC_spgemm ptr[%d] result is %d\n", i, h_csrRowPtrC[i], i, c->rowPtr[i]);
                }
                error_times++;
            }
        }
        if (error_times)
        {
            printf("the cptr is wrong! the wrong times is %d\n", error_times);
        }
        else
        {
            printf("compute the Cptr is right!\n");
            unsigned long long int nums_Intermediate_product = 0;
            for (int i = 0; i < a->nnz; i++)
            {
                int rowidx = a->col[i];
                nums_Intermediate_product += b->rowPtr[rowidx + 1] - b->rowPtr[rowidx];
            }
            double gflops = 2.0 * (double)nums_Intermediate_product / (msec * 1e6);
            printf("for %s cusparse gflops = %lf\n", a->matrix_name, gflops);
        }
        error_times = 0;
        if (error_times)
        {
            printf("the cval is wrong! the wrong times is %d\n", error_times);
        }
        else
        {
            printf("compute the Cval is right!\n");
        }
        // check c_col
        error_times = 0;
        int nnz_order;
        for (int i = 0; i < (nnzC); i++)
        {
            if (h_csrColIdxC[i] != c->col[i])
            {
                if (error_times < 10)
                {
                    printf("the cusparse col[%d] result is %d,the NHC_spgemm col[%d] result is %d\n", i, h_csrColIdxC[i], i, c->col[i]);
                    if (error_times == 0)
                    {
                        nnz_order = i;
                    }
                }
                check_result = false;
                error_times++;
            }
        }
        if (error_times)
        {
            printf("the ccol is wrong! the wrong times is %d\n", error_times);
        }
        else
        {
            printf("compute the Ccol is right!\n");
        }
        if (check_result)
        {
            printf("the result is all right! Congratulations!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        }
#endif
        delete[] h_csrRowPtrC;
        delete[] h_csrColIdxC;
        delete[] h_csrValC;
    }
    cudaFree(d_csrRowPtrC);
    cudaFree(d_csrColIdxC);
    cudaFree(d_csrValC);
}
