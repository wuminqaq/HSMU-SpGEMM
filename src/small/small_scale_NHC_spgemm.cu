#include "mask.cuh"
#include "symbolic.cuh"
#include "numeric.cuh"
void small_scale_NHC_spgemm(compressed_bin *compressed_bin, NHC_bin *c_bin, NHC_CSR *A, NHC_CSR *B, NHC_CSR *C, NHC_mask_matrix *mask_matrixB, double &time2)
{
    double time_conversion = 0;
    double time_malloc = 0;
    double time_form_C_mask = 0;
    double time_form_ccol = 0;
    double time_for_matrix_partion = 0;
    double time_form_cval = 0;
    double space_mask; 
    double space_csr;
    double CSR_bytes = (A->M + 1) * sizeof(index_t) + (A->nnz) * sizeof(index_t) + A->nnz * sizeof(value_t);
    double csr_mem = CSR_bytes / 1024;
    space_mask = (A->M * A->N) / 8 / 1024;
    printf("mask matrix cost %lf kb\n", space_mask);
    printf("CSR matrix cost %lf kb\n", csr_mem);
    double time0;
    time2 = 0;
    time0 = fast_clock_time();
    matrixB_partion_for_mask(B->rowPtr, c_bin, B->M); 
    Initialize_mask_matrix(mask_matrixB, B, c_bin);   
    cudaDeviceSynchronize();
    time_conversion = fast_clock_time() - time0;
    printf("the run time for Initialize_mask_matrix is %.4lf\n", time_conversion * 1000);
    time0 = fast_clock_time();
    Form_Cptr(A, mask_matrixB, C, c_bin->d_max_nnz_in_onerow);
    cudaDeviceSynchronize();
    time_form_C_mask = fast_clock_time() - time0;
    time2 += time_form_C_mask;
    printf("the run time for Form_Cptr is %.4lf\n", time_form_C_mask * 1000);
    time0 = fast_clock_time();
    C->rowPtr = new index_t[A->M + 1]();
    cudaMemcpy(C->rowPtr, C->d_ptr, (A->M + 1) * sizeof(index_t), cudaMemcpyDeviceToHost);
    C->nnz = C->rowPtr[A->M];
    cudaMalloc((void **)&(C->d_col), (C->nnz) * sizeof(int));
    cudaMalloc((void **)&(C->d_val), (C->nnz) * sizeof(value_t));
    time_malloc = fast_clock_time() - time0;
    time2 += time_malloc;
    printf("the run time for cudaMalloc col and val is %.4lf\n", time_malloc * 1000);

    time0 = fast_clock_time();
    Form_Ccol(C, mask_matrixB, A, c_bin);
    cudaDeviceSynchronize();
    time_form_ccol = fast_clock_time() - time0;
    time2 += time_form_ccol;
    printf("the run time for Form_Ccol is %.4lf\n", time_form_ccol * 1000);

    time0 = fast_clock_time();
    matrix_partion_for_numeric_compute(C->rowPtr, c_bin, A->M); //
    cudaDeviceSynchronize();
    time_for_matrix_partion = fast_clock_time() - time0;
    time2 += time_for_matrix_partion;
    printf("the run time for matrix_partion_for_numeric_compute is %.4lf\n", time_for_matrix_partion * 1000);

    time0 = fast_clock_time();
    numeric_compute(c_bin, A, B, C); //
    cudaDeviceSynchronize();
    time_form_cval = fast_clock_time() - time0;
    time2 += time_form_cval;
    printf("the run time for numeric_compute is %.4lf\n", time_form_cval * 1000);

    unsigned long long int nums_Intermediate_product = 0;
    for (int i = 0; i < A->nnz; i++)
    {
        int rowidx = A->col[i];
        nums_Intermediate_product += B->rowPtr[rowidx + 1] - B->rowPtr[rowidx];
    }
    printf("nums_Intermediate_product = %lld\n", nums_Intermediate_product);
    printf("cnnz = %lld\n", C->nnz);
    double compress_rate = (double)(nums_Intermediate_product / C->nnz);
    printf("compress_rate = %lf\n", compress_rate);
    double flops = 2.0 * (double)nums_Intermediate_product;
    double gflops = 2.0 * (double)nums_Intermediate_product / (time2 * 1000 * 1e6);
    double numerical_gflops = 2.0 * (double)nums_Intermediate_product / ((time_form_cval + time_form_ccol) * 1000 * 1e6);
    printf("the numerical_gflops is %lf\n", numerical_gflops);
    printf("gflops = %lf\n", gflops);

    char filename[100];
    char *lastSlash = strrchr(A->matrix_name, '/');
    char *lastDot = strrchr(A->matrix_name, '.');

    if (lastSlash != NULL && lastDot != NULL && lastDot > lastSlash)
    {
        size_t length = lastDot - (lastSlash + 1);
        strncpy(filename, lastSlash + 1, length);
        filename[length] = '\0';
    }
    else
    {
        strcpy(filename, A->matrix_name);
    }

    FILE *fout = fopen("~/NHC_SPGEMM/data/NHC_4080S_result.csv", "a");
    if (fout == NULL)
        printf("Writing results fails.\n");
    fprintf(fout, "%s,%i,%i,%lld,%d,%f,%f,%f,%lf\n",
            filename, A->M, A->nnz, nums_Intermediate_product, C->nnz, compress_rate, time2 * 1000, gflops, numerical_gflops);
    fclose(fout);

    FILE *fout_mem = fopen("~/NHC_SPGEMM/data/small_mem_cost.csv", "a");
    if (fout_mem == NULL)
        printf("Writing results fails.\n");
    fprintf(fout_mem, "%s,%i,%lld,%f,%f\n",
            filename, A->M, A->nnz, csr_mem, space_mask);
    fclose(fout_mem);

    FILE *fout_conversion = fopen("~/NHC_SPGEMM/data/small_time_conversion.csv", "a");
    if (fout_conversion == NULL)
        printf("Writing results fails.\n");
    fprintf(fout_conversion, "%s,%i,%lld,%f,%f\n",
            filename, A->M, A->nnz, time_conversion * 1000, time2 * 1000);
    fclose(fout_conversion);

    FILE *fout_step_time = fopen("~/NHC_SPGEMM/data/small_step_runtime.csv", "a");
    if (fout_step_time == NULL)
        printf("Writing results fails.\n");
    fprintf(fout_step_time, "%s,%i,%lld,%f,%f,%f,%f,%f,%f\n",
            filename, A->M, A->nnz, time_form_C_mask * 1000, time_malloc * 1000, time_form_ccol * 1000, time_for_matrix_partion * 1000, time_form_cval * 1000, time2 * 1000);
    fclose(fout_step_time);
}