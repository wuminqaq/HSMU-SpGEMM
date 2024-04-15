#include "NHC_spgemm.cuh"
#include "mmio_highlevel.h"
#include <iostream>
// #include "cusparse_spgemm.h"
#include "cusparse_in_tile.cu"
// #include "my_cusparse.cu"

using namespace std;

int main(int argc, char **argv)
{
    NHC_CSR A, B, C, test_C;
    NHC_mask_matrix mask_matrixB;
    NHC_bin c_bin;
    compressed_bin compressed_bin;
    if (argc < 2)
    {
        printf("Run the code by './test matrix.mtx'.\n");
        return 0;
    }
    char *filename;
    filename = argv[1];
    printf("the matrix is: -------------- %s --------------\n", filename);

    std::ifstream fs;
    fs.open(filename);
    if (!fs.is_open())
    {
        printf("Error opening file %s\n", filename);
        return 1;
    }
    if (fs.is_open())
    {
        printf("success open file\n");
    }
    A.matrix_name = filename;
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    mmio_allinone(&A.M, &A.N, &A.nnz, &A.isSymmetric, &A.rowPtr, &A.col, &A.val, filename);
    if (AAT)
    {
        index_t *cscColPtrA;
        index_t *cscRowIdxA;
        value_t *cscValA;
        B.M = A.N;
        B.N = A.M;
        B.nnz = A.nnz;
        cscColPtrA = new index_t[A.N + 1]();
        cscRowIdxA = new index_t[A.nnz]();
        cscValA = new value_t[A.nnz]();
        // transpose A from csr to csc
        csr2csc(A.M, A.N, A.nnz, A.rowPtr, A.col, A.val, cscRowIdxA, cscColPtrA, cscValA);
        B.rowPtr = cscColPtrA;
        B.col = cscRowIdxA;
        B.val = cscValA;
    }
    else
    {
        mmio_allinone(&B.M, &B.N, &B.nnz, &B.isSymmetric, &B.rowPtr, &B.col, &B.val, filename);
    }
    gettimeofday(&t2, NULL);
    double time_loadmat = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("input matrix A: ( %i, %i ) nnz = %lld\nloadfile time    = %4.5f sec\n", A.M, A.N, A.nnz, time_loadmat / 1000.0);
    // printf("isSymmetric of the matrix is %d\n", A.isSymmetric);
    C.M = A.M;
    C.N = B.N;
    HtD_value_col_ptr(&A);
    HtD_value_col_ptr(&B);

#if test_NHC
    NHC_spgemm(&compressed_bin, &c_bin, &A, &B, &C, &mask_matrixB);
#endif

#if CHECK_RESULT
    tile_cusparse_spgemm(&A, &B, &test_C, &C);
#endif
    realse_NHCcsr(&A);
    realse_NHCcsr(&B);
#if test_NHC
    free_C_matrix(&C);
    if (A.N < threshold_value_selected)
    {
        free_mask_matrix(&mask_matrixB);
    }
    else
    {
        realse_NHC_tile(&B);
        realse_NHC_tile(&C);
        realse_compressed_bin(&compressed_bin);
    }
    free_bin(&c_bin);
#endif
    return 0;
}
