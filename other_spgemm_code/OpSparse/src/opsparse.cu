
#include "kernel_wrapper.cuh"
#include <fstream>
#include <cuda_profiler_api.h>
#include <cub/cub.cuh>
#include "cusparse_spgemm.h"
#include "Timings.h"
__global__ void check_array_in_device_int(int length,int* d_array){
        if(length==0){
            printf("the length is 0\n");
        }
        for(int i=0;i<length;i++){
                    printf("d_array[%d] is %d\n",i,d_array[i]);
                    // printf("d_array[%d] is %llu\n",i,d_array[i]);
        }
}

void opsparse(const CSR& A, const CSR& B, CSR& C, Meta& meta, Timings& timing){
    
    double t0, t1;
    t1 = t0 = fast_clock_time();
    C.M = A.M;
    C.N = B.N;
    C.nnz = 0;
    h_setup(A, B, C, meta, timing);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.setup = fast_clock_time() - t0;

    // symbolic binning
    t0 = fast_clock_time();
    h_symbolic_binning(C, meta);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.symbolic_binning = fast_clock_time() - t0;

    // symbolic phase
    t0 = fast_clock_time();
    h_symbolic(A, B, C, meta);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.symbolic = fast_clock_time() - t0;


    // numeric binning
    t0 = fast_clock_time();
    h_numeric_binning(C, meta);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.numeric_binning = fast_clock_time() - t0;

    // malloc C
    t0 = fast_clock_time();
    C.nnz = *meta.total_nnz;
    // printf("C.nnz is %d\n",C.nnz);
    CHECK_ERROR(cudaMalloc(&C.d_val, C.nnz * sizeof(mdouble)));
    CHECK_ERROR(cudaMalloc(&C.d_col, C.nnz * sizeof(mint)));
    timing.allocate = fast_clock_time() - t0;
    // printf("the  C.nnz is %d\n",C.nnz);

    // prefix sum and malloc
    t0 = fast_clock_time();
    cudaError_t err = cub::DeviceScan::ExclusiveSum(meta.d_cub_storage, meta.cub_storage_size, C.d_rpt, C.d_rpt, C.M + 1);
    // if (err != cudaSuccess) {
    //  // 函数执行失败，进行错误处理
    //     printf("ExclusiveSum is failed\n");
    // } else {
    //         // 函数执行成功
    //         printf("ExclusiveSum is success\n");
    // }
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.prefix = fast_clock_time() - t0;
    
    // numeric   
    t0 = fast_clock_time();
    h_numeric_full_occu(A, B, C, meta);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.numeric= fast_clock_time() - t0;

    // cleanup
    t0 = fast_clock_time();
    meta.release();
    timing.cleanup = fast_clock_time() - t0;
    timing.total = fast_clock_time() - t1;
}

int main(int argc, char **argv)
{
    std::string mat1, mat2;
    // mat1 = "can_24";
    // mat2 = "can_24";
    if(argc == 2){
        mat1 = argv[1];
        mat2 = argv[1];
    }
    if(argc >= 3){
        mat1 = argv[1];
        mat2 = argv[2];
    }
    std::string mat1_file;
    mat1_file = mat1;
    // if(mat1.find("ER") != std::string::npos){
    //     mat1_file = "../matrix/ER/" + mat1 +".mtx";
    // }
    // else if(mat1.find("G500") != std::string::npos){
    //     mat1_file = "../matrix/G500/" + mat1 +".mtx";
    // }
    // else{
    //     mat1_file = "/home/wm/total_matrix/" + mat1 + ".mtx";
    // }
    std::string mat2_file;
    mat2_file = mat2;
    // if(mat2.find("ER") != std::string::npos){
    //     mat2_file = "../matrix/ER/" + mat2 +".mtx";
    // }
    // else if(mat2.find("G500") != std::string::npos){
    //     mat2_file = "../matrix/G500/" + mat2 +".mtx";
    // }
    // else{
    //     mat2_file = "/home/wm/Matrix_set/" +  mat2 +".mtx";
    // }
	
    CSR A, B;
    A.construct(mat1_file);
    if(mat1 == mat2){
        B = A;
    }
    else{
        B.construct(mat2_file);
        if(A.N == B.M){
            // do nothing
        }
        else if(A.N < B.M){
            CSR tmp(B, A.N, B.N, 0, 0);
            B = tmp;
        }
        else{
            CSR tmp(A, A.M, B.M, 0, 0);
            A = tmp;
        }
    }

    A.H2D();
    B.H2D();

    printf("the A.M is %d,A.N is %d\n", A.M, A.N);
    long total_flop = compute_flop(A, B);
    CSR C;
    cudaruntime_warmup();
    Meta meta;
#if compute_share
    {
        Timings timing;
        opsparse(A, B, C, meta, timing);
        C.release();
    }
#endif

#if compute_total_time
    mint iter = 10;
    Timings timing, bench_timing;
    for(mint i = 0; i < iter; i++){
        opsparse(A, B, C, meta, timing);
        bench_timing += timing;
        if(i < iter - 1){
            C.release();
        }
    }
    bench_timing /= iter;

    printf("%s ",mat1.c_str());
    bench_timing.print(total_flop * 2);

    char filename[100];
    char* lastSlash = const_cast<char*>(strrchr(mat1.c_str(), '/'));
    char* lastDot = const_cast<char*>(strrchr(mat1.c_str(), '.'));

    if (lastSlash != NULL && lastDot != NULL && lastDot > lastSlash) {
        size_t length = lastDot - (lastSlash + 1);
        strncpy(filename, lastSlash + 1, length);
        filename[length] = '\0';
    } else {
        strcpy(filename, mat1.c_str());
    }
    
    double symbolic_time = bench_timing.total*1000 - (bench_timing.numeric + bench_timing.cleanup)*1000;
    double numeric_time = (bench_timing.numeric + bench_timing.cleanup)*1000;
    printf("the symbolic_time is %lf,numeric_time is %lf\n",symbolic_time,numeric_time);
    

    // FILE *fout_mem = fopen("./data/OpSparse_compress_total_time.csv", "a");
    // FILE *fout_mem = fopen("./data/OpSparse_new_repres_total_time.csv", "a");
    FILE *fout_mem = fopen("./data/OpSparse_result3090ti.csv", "a");
    double sum_total = bench_timing.setup + bench_timing.symbolic_binning + bench_timing.symbolic + bench_timing.numeric_binning
        + bench_timing.reduce + bench_timing.prefix + bench_timing.allocate + bench_timing.numeric + bench_timing.cleanup;
    double total_flop_G = (double)(total_flop * 2);
        total_flop_G /=1000000000;
        printf("the total_flop * 2 is %ld\n",total_flop * 2);
        total_flop_G /= bench_timing.total;
    double numeric_Gflops = (double)(total_flop * 2);
            numeric_Gflops /=1000000000;
            numeric_Gflops /=(bench_timing.numeric + bench_timing.cleanup);
    printf("the numeric_Gflops is %lf\n",numeric_Gflops);
    printf("total_flop_G is %lf,bench_timing.total is %lf,total_flop * 2 is %ld\n",total_flop_G,bench_timing.total*1000,total_flop * 2);
   
    if (fout_mem == NULL)
        printf("Writing results fails.\n");
    fprintf(fout_mem, "%s,%i,%ld,%lf,%lf,%lf,%lf,%lf\n",
             filename, A.M,
    total_flop * 2,
    bench_timing.total*1000,
    total_flop_G,
    symbolic_time,
    numeric_time,
    numeric_Gflops);
    fclose(fout_mem);
#endif
    // compare result

    C.D2H();
    // CSR C_ref;
    // cusparse_spgemm(&A, &B, &C_ref);
    // C_ref.D2H();
    // if(C == C_ref){
    //    printf("pass\n");
    // }
    // else{
    //    printf("error\n");
    // }
    
    A.release();
    C.release();
    B.release();

    C.release();
    return 0;
}


