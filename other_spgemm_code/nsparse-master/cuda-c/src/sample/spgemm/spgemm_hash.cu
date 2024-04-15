#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <algorithm>

#include <cuda.h>
#include <helper_cuda.h>

#include <nsparse.h>


void spgemm_csr(sfCSR *a, sfCSR *b, sfCSR *c)
{

    int i;
  
    long long int flop_count;
    cudaEvent_t event[2];
    float  flops;
  
    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
  
    /* Memcpy A and B from Host to Device */
    csr_memcpy(a);
    csr_memcpy(b);
  
    /* Count flop of SpGEMM computation */
    get_spgemm_flop(a, b, a->M, &flop_count);
    double symbolic_time = 0;
    double numeric_time = 0;
    /* Execution of SpGEMM on Device */
    // ave_msec = 0;
    for (i = 0; i < SPGEMM_TRI_NUM; i++) {
        if (i > 0) {
            release_csr(*c);
        }
        // cudaEventRecord(event[0], 0);
        spgemm_kernel_hash(a, b, c,symbolic_time,numeric_time);
        // cudaEventRecord(event[1], 0);
        // cudaThreadSynchronize();
        // cudaEventElapsedTime(&msec, event[0], event[1]);

        // if (i > 0) {
        //     ave_msec += msec;
        // }
    }
    // ave_msec /= SPGEMM_TRI_NUM - 1;
    symbolic_time /= SPGEMM_TRI_NUM;
    numeric_time /= SPGEMM_TRI_NUM;
    double total_time = symbolic_time + numeric_time;
    printf("symbolic_time is %lf, numeric_time is %lf, total_time is %lf\n",symbolic_time,numeric_time,total_time);

    // flops = (float)(flop_count) / 1000 / 1000 / ave_msec;
    flops = (float)(flop_count) / 1000 / 1000 / total_time;
    float numeric_flops = (float)(flop_count) / 1000 / 1000 / numeric_time;
    printf(" numeric_flops is %lf\n",numeric_flops);
    printf("SpGEMM using CSR format (Hash-based): %s, %f[GFLOPS], %f[ms]\n", a->matrix_name, flops, total_time);

    csr_memcpyDtH(c);
    release_csr(*c);
    
    /* Check answer */
#ifdef sfDEBUG
    sfCSR 
    
    
    ;
    spgemm_cu_csr(a, b, &ans);

    printf("(nnz of A): %d =>\n(Num of intermediate products): %ld =>\n(nnz of C): %d\n", a->nnz, flop_count / 2, c->nnz);
    check_spgemm_answer(*c, ans);

    release_cpu_csr(ans);
#endif
  
    //write result

    // char filename[100];
    // char* lastSlash = strrchr(a->matrix_name, '/');
    // char* lastDot = strrchr(a->matrix_name, '.');

    // if (lastSlash != NULL && lastDot != NULL && lastDot > lastSlash) {
    //     size_t length = lastDot - (lastSlash + 1);
    //     strncpy(filename, lastSlash + 1, length);
    //     filename[length] = '\0';
    // } else {
    //     strcpy(filename, a->matrix_name);
    // }

    char filename[100];
    char* lastSlash = strrchr(a->matrix_name, '/');
    char* lastDot = strrchr(a->matrix_name, '.');

    if (lastSlash != NULL && lastDot != NULL && lastDot > lastSlash) {
        size_t length = lastDot - (lastSlash + 1);
        strncpy(filename, lastSlash + 1, length);
        filename[length] = '\0';
    } else {
        strcpy(filename, a->matrix_name);
    }
    // printf("i am here\n");
    double compress_rate = flop_count / 2 / c->nnz;
    // FILE *fout = fopen("/home/wm/open_source_code/spGEMM/nsparse-master/cuda-c/data/Nsparse_new_repres_result.csv", "a");
    FILE *fout = fopen("/home/wm/open_source_code/spGEMM/nsparse-master/cuda-c/data/Nsparse_Newcompress_result.csv", "a");
    if (fout == NULL)
        printf("Writing results fails.\n");
    fprintf(fout, "%s,%.0lfK,%.1lfM,%.1lfM,%.1lfM,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
             filename, round((double)a->M/1000.0),round((double)a->nnz/1000000.0*10.0)/10.0,round((double)flop_count/1000000.0*10.0)/10.0, round((double)c->nnz/1000000.0*10.0)/10.0, compress_rate,total_time,flops,
                symbolic_time, numeric_time, numeric_flops);
            //  A->matrix_name, (double)A->M/1000.0,(double)A->nnz/1000000.0, (double)nums_Intermediate_product/1000000.0, (double)C->nnz/1000000.0, compress_rate, time2*1000, gflops,numerical_gflops);
    fclose(fout);

    release_csr(*a);
    release_csr(*b);
    for (i = 0; i < 2; i++) {
        cudaEventDestroy(event[i]);
    }

}

/* Main Function */
int main(int argc, char **argv)
{
    sfCSR mat_a, mat_b, mat_c;
  
    /* Set CSR reading from MM file */
    init_csr_matrix_from_file(&mat_a, argv[1]);
    init_csr_matrix_from_file(&mat_b, argv[1]);
  
    spgemm_csr(&mat_a, &mat_b, &mat_c);

    release_cpu_csr(mat_a);
    release_cpu_csr(mat_b);
    release_cpu_csr(mat_c);
    
    return 0;
}
