#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <fstream>
#include <cub/cub.cuh>
#define DateTypeStoreMask unsigned long long
#define DateTypeStoreCompressMask unsigned long long
#define value_t double
#define index_t int
#define Sy_blockdim 128
#define loop_times_OR 3
#define num_of_SM 84
#define blocks_div_up_SMs 2
#define WSIZE 32
#define WSIZE_FOR_small_num 4
#define Wsize_for_compute_tile_nums 32
#define Nnz_per_partion 4096
#define NnzB_per_partion_for_initial_mask 1024
#define Numtile_per_partion_for_initial_compress_mask 4096
#define NnzA_per_partion_for_form_cptr 1024
#define PWARP_FOR_TileOR 8
#define PWARP_ROWS__FOR_TileOR 128
#define TileOR_PWARP_BLOCK_SIZE (PWARP_FOR_TileOR * PWARP_ROWS__FOR_TileOR)
#define PWARP_TSIZE 32
#define PWARP_FOR_CtilePtr 4
#define PWARP_ROWS_FOR_CtilePtr 256
#define Block_size_FOR_CtilePtr (PWARP_ROWS_FOR_CtilePtr * PWARP_FOR_CtilePtr)
#define PWARP_TSIZE_FOR_CtilePtr 32
#define div_up(a, b) ((a + b - 1) / b)
#define THRESH_SCALE 0.8
#define SYMBOLIC_SCALE_LARGE 1
#define NUMERIC_SCALE_LARGE 2
#define NUM_BIN 8
#define HASH_SCALE 107
#define HASH_SINGLE
#define threshold_value_selected 4096

#define INDEX_TYPE index_t
#define VALUE_TYPE value_t

#define TIMING 1

#define BLOCKS_PER_SM 4
#define MAX_STATIC_SHARED 49152

#define NUM_BIN_FOR_Ccol 14
#define Cnnz_ctile_rate_Threshold 0
#define Critical_bin_id 10
#define compute_share 0
#define compute_total_time 1
#define compute_conversion_time_and_space 1
#define compute_step_time 1
#define checek_kernel 0
#define output_bin 0
#define checek_release 0
#define SPGEMM_TRI_NUM 1
#define compute_variance 0
#define CHECK_RESULT 0
#define test_NHC 1
#define AAT 0
#define UNINT32_MAX 4294967295
typedef struct
{
    index_t *rowPtr;
    index_t *col;
    value_t *val;
    index_t *d_ptr;
    index_t *d_col;
    value_t *d_val;
    index_t *d_tem_ptr;
    int M;
    int N;
    index_t nnz;
    int nnz_max;
    int isSymmetric;
    char *matrix_name;
    index_t *d_tile_ptr;
    index_t *d_tile_col;
    index_t *d_tem_tile_ptr;
    index_t *tile_ptr;
    DateTypeStoreCompressMask *d_mask_num;
} NHC_CSR;

typedef struct
{
    DateTypeStoreMask *mask_matrix;
    DateTypeStoreMask *after_or_mask_matrix;
    int shift_of_pitch;
    int pitch;
    int *d_nums_over_limit_hashtable;
    int nums_over_limit_hashtable;
    int *d_array_store_Crow_id_over_limit;
    int num_unit;
    int *d_Brow_id;
    int size_arr[5] = {12288, 24576, 49152, 98304, 196608};
} NHC_mask_matrix;

typedef struct
{
    int *bin;
    int *d_bin;
    int max_rows_partion;
    int num_of_partion;
    int *d_max_nnz_in_onerow;
} NHC_bin;

typedef struct
{
    int num_of_partion;
    int *bin_size;
    int *bin_offset;
    int *max_row_nnz;
    int *total_nnz;
    int *bins;
    int *d_bins;
    int *d_bin_size;
    int *d_bin_offset;
    int *d_total_nnz;
    int *d_max_row_nnz;
    cudaStream_t *streams;
    int *d_global_mem_pool;
    size_t global_mem_pool_size;
    bool global_mem_pool_malloced;
    void *d_temp_storage;
    size_t temp_storage_bytes = 0;
} compressed_bin;

typedef struct
{
    int *d_tile_ptr;
    int *d_tile_col;
    int *d_tem_tile_ptr;
    int *tile_ptr;
    DateTypeStoreCompressMask *d_mask_num;
} NHC_tile;
