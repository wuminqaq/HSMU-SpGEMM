#include "../inc/NHC_spgemm.h"
#include "../inc/cuda_comm.h"
#include "loading.cuh"
#include "external/spECK/dense_acc.cu"
#include "./small/small_scale_NHC_spgemm.cu"
#include "./compressed/compressed_NHC_spgemm.cu"
void NHC_spgemm(compressed_bin *compressed_bin, NHC_bin *c_bin, NHC_CSR *A, NHC_CSR *B, NHC_CSR *C, NHC_mask_matrix *mask_matrixB)
{
    double time1 = 0;
    double time2 = 0;
    double time_array[SPGEMM_TRI_NUM] = {0};

    for (int iter = 0; iter < SPGEMM_TRI_NUM; iter++)
    {
        if (B->N < threshold_value_selected)
        {
            small_scale_NHC_spgemm(compressed_bin, c_bin, A, B, C, mask_matrixB, time2);
        }
        else
        {
            compressed_NHC_spgemm(iter, time_array, compressed_bin, c_bin, A, B, C, time1, time2);
            if (iter < (SPGEMM_TRI_NUM - 1))
            {
                realse_NHC_tile(B);
                realse_NHC_tile(C);
                free_C_matrix(C);
                realse_compressed_bin(compressed_bin);
            }
        }
        if (iter < (SPGEMM_TRI_NUM - 1))
        {
            free_bin(c_bin);
        }
    }
    // output Performance information
    printf("the total run time of NHC_spgemm is %.2f ms\n", time2);
    unsigned long long int nums_Intermediate_product = 0;
    for (int i = 0; i < A->nnz; i++)
    {
        int rowidx = A->col[i];
        nums_Intermediate_product += B->rowPtr[rowidx + 1] - B->rowPtr[rowidx];
    }
}