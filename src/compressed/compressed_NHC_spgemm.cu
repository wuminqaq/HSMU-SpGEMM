#include "compressed_mask.cuh"
#include "compressed_mask64.cuh"
#include "../bining.cuh"
#include "compressed_tilePtr_symbolic.cuh"
#include "compressed_tileOR.cuh"
#include "compressed_Ccol_symbolic.cuh"
#include "compressed_Cval.cuh"
using namespace std;
void startTimerVar(cudaEvent_t &start, cudaStream_t stream = 0)
{
    cudaEventRecord(start, stream);
    cudaEventSynchronize(start);
}
float recordTimerVar(cudaEvent_t &start, cudaEvent_t &end, cudaStream_t stream = 0)
{
    float time;
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    return time;
}
void compressed_NHC_spgemm(int iter, double *time_array, compressed_bin *compressed_bin, NHC_bin *c_bin, NHC_CSR *A, NHC_CSR *B, NHC_CSR *C, double &time1, double &time2)
{
    cudaEvent_t event_start, event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    cudaStream_t stream = 0;
    time2 = 0;
    cudaError_t error;
    float time_conversion;
    float time_Form_tileCptr;
    float time_tileOR;
    float time_malloc;
    float time_get_Ccol;
    float time_form_cval;

    setup(A, B, C, compressed_bin);
    cudaDeviceSynchronize();
    startTimerVar(event_start, stream);
    Initialize_compressed_mask64_matrix(A, B, C, c_bin);
    time_conversion = recordTimerVar(event_start, event_stop, stream);
#if checek_kernel
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("Initialize_compressed_mask64_matrix is failed\n");
    }
    else
    {
        printf("Initialize_compressed_mask64_matrix is cudaSuccess\n");
    }
#endif
    if (iter == 0)
    {
        printf("the run time for Initialize_compressed_mask64_matrix is %.4lf\n", time_conversion);
    }
    double space_compressed_mask;
    double csr_mem = (B->M + 1) * sizeof(index_t) + (B->nnz) * sizeof(index_t) + B->nnz * sizeof(value_t);
    csr_mem = csr_mem / (1024 * 1024);
    space_compressed_mask = (B->M + 1) * sizeof(index_t) + (B->tile_ptr[B->M]) * sizeof(index_t) + B->tile_ptr[B->M] * sizeof(DateTypeStoreCompressMask);
    space_compressed_mask /= (1024 * 1024);
    printf("space_compressed_mask = %.2lf MB\n",space_compressed_mask);
    startTimerVar(event_start, stream);
    compressed_Form_tileCptr(compressed_bin, A, B, C);
    time_Form_tileCptr = recordTimerVar(event_start, event_stop, stream);
    cudaDeviceSynchronize();
#if checek_kernel
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("compressed_Form_tileCptr is failed\n");
    }
    else
    {
        printf("compressed_Form_tileCptr is cudaSuccess\n");
    }
#endif
    time2 += time_Form_tileCptr;
    if (iter == 0)
    {
        printf("the run time for compressed_Form_tileCptr is %.4lf\n", time_Form_tileCptr);
    }
    startTimerVar(event_start, stream);
    h_compress_tileOR_for_Cptr(compressed_bin, A, B, C);
    time_tileOR = recordTimerVar(event_start, event_stop, stream);
    cudaDeviceSynchronize();
#if checek_kernel
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("h_compress_tileOR_for_Cptr is failed\n");
    }
    else
    {
        printf("h_compress_tileOR_for_Cptr is cudaSuccess\n");
    }
#endif
    time2 += time_tileOR;
    if (iter == 0)
    {
        printf("the run time for h_compress_tileOR_for_Cptr is %.4lf\n", time_tileOR);
    }
    // if (iter == 0)
    // {
    //     for (int i = 0; i < NUM_BIN_FOR_Ccol; i++)
    //     {
    //         printf("for h_get_Ccol_from_Ctile,the compressed_bin->bin_size[%d] is %d\n", i, compressed_bin->bin_size[i]);
    //     }
    // }
    startTimerVar(event_start, stream);
    cudaMalloc((void **)&(C->d_col), (C->nnz) * sizeof(int));
    cudaMalloc((void **)&(C->d_val), (C->nnz) * sizeof(value_t));
    time_malloc = recordTimerVar(event_start, event_stop, stream);
#if checek_kernel
    {
        cudaDeviceSynchronize();
        error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("XXXXXXXXXXXXX cudaMalloc is failed\n");
        }
        else
        {
            printf("/////////// cudaMalloc is cudaSuccess\n");
        }
    }
#endif

    time2 += time_malloc;
    if (iter == 0)
    {
        printf("the run time for cudaMalloc col and val is %.4lf\n", time_malloc);
    }
    typedef HashMapNoValue<1> GlobalMapRowOffsets;
    GlobalMapRowOffsets *rowOffsetMaps = nullptr;

    index_t rowOffsetMapCount = 0;

    startTimerVar(event_start, stream);
    h_get_Ccol_from_Ctile(compressed_bin, C);
    time_get_Ccol = recordTimerVar(event_start, event_stop, stream);
    cudaDeviceSynchronize();
#if checek_kernel
    {
        cudaDeviceSynchronize();
        error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("XXXXXXXXXXXXX h_get_Ccol_from_Ctile is failed\n");
        }
        else
        {
            printf("/////////// h_get_Ccol_from_Ctile is cudaSuccess\n");
        }
    }
#endif
    time2 += time_get_Ccol;
    if (iter == 0)
    {
        printf("the run time for h_get_Ccol_from_Ctile is %.4lf\n", time_get_Ccol);
    }

    startTimerVar(event_start, stream);
    h_form_cval_with_dense_accumulator<GlobalMapRowOffsets>(compressed_bin, A, B, C, rowOffsetMaps, rowOffsetMapCount);
    time_form_cval = recordTimerVar(event_start, event_stop, stream);
    cudaDeviceSynchronize();
    time2 += time_form_cval;
    if (iter == 0)
    {
        printf("the run time for h_form_cval is %.4lf\n", time_form_cval);
    }
    time_array[iter] = time2;
    // write result
    if (iter == (SPGEMM_TRI_NUM - 1))
    {
        time2 = 0;
        for (int i = 0; i <= iter; i++)
        {
            time2 += time_array[i];
        }
        time2 = (time2) / SPGEMM_TRI_NUM;

#if compute_variance
        double variance = 0.0;
        for (int j = 0; j <= iter; j++)
        {
            variance += pow(time_array[j] - time2, 2) / SPGEMM_TRI_NUM;
        }
        double standard = pow(variance, 0.5);
#endif
        unsigned long long int nums_Intermediate_product = 0;
        for (int i = 0; i < A->nnz; i++)
        {
            int rowidx = A->col[i];
            nums_Intermediate_product += B->rowPtr[rowidx + 1] - B->rowPtr[rowidx];
        }
        printf("nums_Intermediate_product = %lld\n", nums_Intermediate_product);
        printf("cnnz = %lld\n", C->nnz);
        float compress_rate = (float)(nums_Intermediate_product / C->nnz);
        printf("compress_rate = %lf\n", compress_rate);
        float flops = 2.0 * (float)nums_Intermediate_product;
        float gflops = 2.0 * (float)nums_Intermediate_product / (time2 * 1e6);
        float numerical_gflops = 2.0 * (float)nums_Intermediate_product / ((time_form_cval + time_get_Ccol) * 1e6);

        printf("the gflops is %lf\n", gflops);
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
        FILE *fout;
#if compute_total_time
        fout = fopen("../../data/NHC_4080S_result.csv", "a");
        if (fout == NULL)
            printf("Writing results fails1.\n");
        fprintf(fout, "%s,%i,%i,%lld,%d,%f,%f,%f,%lf\n",
                filename, A->M, A->nnz, nums_Intermediate_product, C->nnz, compress_rate, time2, gflops, numerical_gflops);
        fclose(fout); // the unit of time2 is ms
#endif
#if compute_conversion_time_and_space
        time1 = (time1) / SPGEMM_TRI_NUM;
        fout = fopen("../../data/conversion_time_and_space_conversion.csv", "a");
        if (fout == NULL)
            printf("Writing results fails.\n");
        fprintf(fout, "%s,%i,%i,%lld,%d,%f,%lf,%lf,%lf,%lf\n",
                filename, A->M, A->nnz, nums_Intermediate_product, C->nnz, compress_rate, time1, time2, csr_mem, space_compressed_mask);
        fclose(fout);
#endif
#if compute_step_time
        float Generates_auxiliary_mask_structure = time_tileOR + time_Form_tileCptr;
        fout = fopen("../../data/new_compressed_step_runtime.csv", "a");
        if (fout == NULL)
            printf("Writing results fails.\n");
        fprintf(fout, "%s,%i,%i,%lld,%d,%f,%lf,%lf,%lf,%lf\n",
                filename, A->M, A->nnz, nums_Intermediate_product, C->nnz, compress_rate, Generates_auxiliary_mask_structure, time_malloc, time_get_Ccol, time_form_cval);
        fclose(fout); // time2 is ms
#endif
    }
}
