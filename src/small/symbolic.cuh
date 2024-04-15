#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <type_traits>
using namespace cub;
using namespace std;
__global__ void check_array_in_device_double(int length, double *d_array)
{
    if (length == 0)
    {
        printf("the length is 0\n");
    }
    for (int i = 0; i < length; i++)
    {
        printf("d_array[%d] is %lf\n", i, d_array[i]);
    }
}
__global__ void check_array_in_device_int(int length, int *d_array)
{
    if (length == 0)
    {
        printf("the length is 0\n");
    }
    for (int i = 0; i < length; i++)
    {
        printf("d_array[%d] is %d\n", i, d_array[i]);
    }
}
__global__ void check_array_in_device_unit32(int length, unsigned int *d_array)
{
    if (length == 0)
    {
        printf("the length is 0\n");
    }
    for (int i = 0; i < length; i++)
    {
        printf("d_array[%d] is %u\n", i, d_array[i]);
    }
}
__global__ void check_array_in_device_unit64(int length, unsigned long long *d_array)
{
    if (length == 0)
    {
        printf("the length is 0\n");
    }
    for (int i = 0; i < length; i++)
    {
        printf("d_array[%d] is %llu\n", i, d_array[i]);
    }
}
int get_interval(int num_unit, int *size_arr)
{
    int i = 0;
    while (size_arr[i] < num_unit)
    {
        i++;
    }
    return i;
}
// partion for form cptr
int partion_A(int *a_ptr, int *bin, int nums_row)
{
    int num_partition = 0;
    int Upper_limit = NnzB_per_partion_for_initial_mask;
    int i = 0;
    while (i <= nums_row)
    {
        if (a_ptr[i] > Upper_limit)
        {
            bin[num_partition] = i - 1;
            Upper_limit = a_ptr[i - 1] + NnzA_per_partion_for_form_cptr;
            num_partition++;
        }
        i++;
    }
    bin[num_partition] = nums_row;
    return num_partition + 1;
}
void matrixA_partion_for_Cptr(int *a_ptr, NHC_bin *c_bin, int nums_row)
{
    c_bin->num_of_partion = partion_A(a_ptr, c_bin->bin + 1, nums_row);
    cudaMalloc((void **)&c_bin->d_bin, (c_bin->num_of_partion + 1) * sizeof(int)); //
    cudaMemcpy(c_bin->d_bin, c_bin->bin, (c_bin->num_of_partion + 1) * sizeof(int), cudaMemcpyHostToDevice);
}

template <int size_hash>
__global__ void kernel_Form_Cptr_with_partion(int *d_bin, index_t *a_dptr, index_t *a_dcol, index_t *d_tem_ptr, int pitch, int num_unit, DateTypeStoreMask *mask_matrix, DateTypeStoreMask *after_or_mask_matrix, int *d_nums_over_limit_hashtable)
{
    __shared__ DateTypeStoreMask target[size_hash];
    __shared__ DateTypeStoreMask buffer[size_hash];
    __shared__ int nnz_of_oneCrow[1];
    int index, i;
    int start_row = d_bin[blockIdx.x];
    int end_row = d_bin[blockIdx.x + 1];
    int start_nnz;
    int acol, num_of_one;
    DateTypeStoreMask *row_point;
    for (index = start_row; index < end_row; index++)
    {
        for (i = threadIdx.x; i < num_unit; i += blockDim.x)
        {
            target[i] = 0;
        }
        __syncthreads();
        for (start_nnz = a_dptr[index]; start_nnz < a_dptr[index + 1]; start_nnz++)
        {
            acol = a_dcol[start_nnz];
            row_point = (DateTypeStoreMask *)((char *)mask_matrix + (acol * pitch));
            for (i = threadIdx.x; i < num_unit; i += blockDim.x)
            {
                buffer[i] = row_point[i];
                target[i] |= buffer[i];
            }
        }
        if (threadIdx.x == 0)
        {
            nnz_of_oneCrow[0] = 0;
        }
        __syncthreads();
        num_of_one = 0;
        for (i = threadIdx.x; i < num_unit; i += blockDim.x)
        {
            num_of_one += __popcll(target[i]);
        }
        if (num_of_one)
        {
            atomicAdd(nnz_of_oneCrow, num_of_one);
        }
        __syncthreads();
        row_point = (DateTypeStoreMask *)((char *)after_or_mask_matrix + (index * pitch));
        for (i = threadIdx.x; i < num_unit; i += blockDim.x)
        {
            row_point[i] = target[i];
        }
        if (threadIdx.x == 0)
        {
            if (nnz_of_oneCrow[0] > (size_hash << 1))
            {
                atomicAdd(d_nums_over_limit_hashtable, 1);
            }
            d_tem_ptr[index] = nnz_of_oneCrow[0];
        }
        __syncthreads();
    }
}

void Form_Cptr_with_partion(NHC_CSR *A, NHC_mask_matrix *mask_matrixB, NHC_CSR *C, NHC_bin *c_bin)
{
    size_t pitch;
    cudaMallocPitch((void **)&mask_matrixB->after_or_mask_matrix, &pitch, mask_matrixB->num_unit * sizeof(DateTypeStoreMask), A->M);
    cudaMalloc((void **)&mask_matrixB->d_nums_over_limit_hashtable, sizeof(int));
    cudaMemset(mask_matrixB->d_nums_over_limit_hashtable, 0, sizeof(int));
    mask_matrixB->nums_over_limit_hashtable = 1;
    cudaMalloc((void **)&C->d_ptr, (A->M + 1) * sizeof(index_t));
    C->rowPtr = new index_t[A->M + 1]();
    int size_arr[5] = {191, 383, 767, 1535, 3071};
    int interval_num = get_interval(mask_matrixB->num_unit, size_arr);
    int GS, BS;
    switch (interval_num)
    {
    case 0:
        BS = 64;
        GS = (ceil(2048 * num_of_SM / BS)) * blocks_div_up_SMs;
        GS = min(GS, A->M);
        kernel_Form_Cptr_with_partion<191><<<GS, BS>>>(c_bin->d_bin, A->d_ptr, A->d_col, C->d_ptr, mask_matrixB->pitch, mask_matrixB->num_unit, mask_matrixB->mask_matrix, mask_matrixB->after_or_mask_matrix, mask_matrixB->d_nums_over_limit_hashtable);
        break;
    case 1:
        BS = 128;
        GS = (ceil(2048 * num_of_SM / BS)) * blocks_div_up_SMs;
        GS = min(GS, A->M);
        kernel_Form_Cptr_with_partion<383><<<GS, BS>>>(c_bin->d_bin, A->d_ptr, A->d_col, C->d_ptr, mask_matrixB->pitch, mask_matrixB->num_unit, mask_matrixB->mask_matrix, mask_matrixB->after_or_mask_matrix, mask_matrixB->d_nums_over_limit_hashtable);
        break;
    case 2:
        BS = 256;
        GS = (ceil(2048 * num_of_SM / BS)) * blocks_div_up_SMs;
        GS = min(GS, A->M);
        kernel_Form_Cptr_with_partion<767><<<GS, BS>>>(c_bin->d_bin, A->d_ptr, A->d_col, C->d_ptr, mask_matrixB->pitch, mask_matrixB->num_unit, mask_matrixB->mask_matrix, mask_matrixB->after_or_mask_matrix, mask_matrixB->d_nums_over_limit_hashtable);
        break;
    case 3:
        BS = 512;
        GS = (ceil(2048 * num_of_SM / BS)) * blocks_div_up_SMs;
        GS = min(GS, A->M);
        kernel_Form_Cptr_with_partion<1535><<<GS, BS>>>(c_bin->d_bin, A->d_ptr, A->d_col, C->d_ptr, mask_matrixB->pitch, mask_matrixB->num_unit, mask_matrixB->mask_matrix, mask_matrixB->after_or_mask_matrix, mask_matrixB->d_nums_over_limit_hashtable);
        break;
    case 4:
        BS = 1024;
        GS = (ceil(2048 * num_of_SM / BS)) * blocks_div_up_SMs;
        GS = min(GS, A->M);
        kernel_Form_Cptr_with_partion<3071><<<GS, BS>>>(c_bin->d_bin, A->d_ptr, A->d_col, C->d_ptr, mask_matrixB->pitch, mask_matrixB->num_unit, mask_matrixB->mask_matrix, mask_matrixB->after_or_mask_matrix, mask_matrixB->d_nums_over_limit_hashtable);
        break;
    }
    cudaFree(c_bin->d_bin);
    // Presum use cube
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cudaDeviceSynchronize();
    cudaFree(mask_matrixB->mask_matrix);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, C->d_ptr, C->d_ptr, A->M + 1, 0);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, C->d_ptr, C->d_ptr, A->M + 1, 0);
    cudaFree(d_temp_storage);
    cudaDeviceSynchronize();
    cudaMemcpy(C->rowPtr, C->d_ptr, (A->M + 1) * sizeof(index_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&mask_matrixB->nums_over_limit_hashtable, mask_matrixB->d_nums_over_limit_hashtable, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemset(mask_matrixB->d_nums_over_limit_hashtable, 0, sizeof(int));
    C->nnz = C->rowPtr[A->M];
}

template <int size_hash>
__global__ void kernel_Form_Cptr(int *d_max_nnz_in_onerow, int A_M, index_t *a_dptr, index_t *a_dcol, index_t *d_tem_ptr, int pitch, int num_unit, DateTypeStoreMask *mask_matrix, DateTypeStoreMask *after_or_mask_matrix)
{
    int row_id;
    __shared__ DateTypeStoreMask target[size_hash];
    __shared__ DateTypeStoreMask buffer[size_hash];
    __shared__ int nnz_of_oneCrow[1];
    for (row_id = blockIdx.x; row_id < A_M; row_id += gridDim.x)
    {
        if (threadIdx.x == 0)
        {
            nnz_of_oneCrow[0] = 0;
        }
        int start_nnz = a_dptr[row_id];
        int end_Annz = a_dptr[row_id + 1];
        int index;
        if (end_Annz == start_nnz)
        {
            if (threadIdx.x == 0)
            {
                d_tem_ptr[row_id] = 0;
            }
        }
        else
        {
            int acol = a_dcol[start_nnz];
            DateTypeStoreMask *row_point = (DateTypeStoreMask *)((char *)mask_matrix + (acol * pitch));
            for (index = threadIdx.x; index < num_unit; index += blockDim.x)
            {
                target[index] = row_point[index];
            }
            __syncthreads();
            for (start_nnz++; start_nnz < end_Annz; start_nnz++)
            {
                acol = a_dcol[start_nnz];
                row_point = (DateTypeStoreMask *)((char *)mask_matrix + (acol * pitch));
                for (index = threadIdx.x; index < num_unit; index += blockDim.x)
                {
                    buffer[index] = row_point[index];
                    target[index] |= buffer[index];
                }
            }
            __syncthreads();
            int num_of_one = 0;
            for (index = threadIdx.x; index < num_unit; index += blockDim.x)
            {
                num_of_one += __popcll(target[index]);
            }
            if (num_of_one)
            {
                atomicAdd(nnz_of_oneCrow, num_of_one);
            }
            __syncthreads();
            DateTypeStoreMask *new_row_point = (DateTypeStoreMask *)((char *)after_or_mask_matrix + (row_id * pitch));
            for (index = threadIdx.x; index < num_unit; index += blockDim.x)
            {
                new_row_point[index] = target[index];
            }
            if (threadIdx.x == 0)
            {
                if (nnz_of_oneCrow[0] > (size_hash << 1))
                {
                    atomicMax(d_max_nnz_in_onerow, nnz_of_oneCrow[0]);
                }
                d_tem_ptr[row_id] = nnz_of_oneCrow[0];
            }
        }
        __syncthreads();
    }
}

void Form_Cptr(NHC_CSR *A, NHC_mask_matrix *mask_matrixB, NHC_CSR *C, int *d_max_nnz_in_onerow)
{
    size_t pitch;
    cudaMallocPitch((void **)&mask_matrixB->after_or_mask_matrix, &pitch, mask_matrixB->num_unit * sizeof(DateTypeStoreMask), A->M);
    cudaMalloc((void **)&C->d_ptr, (A->M + 1) * sizeof(index_t));
    int size_arr[5] = {191, 383, 767, 1535, 3071};
    int interval_num = get_interval(mask_matrixB->num_unit, size_arr);
    int GS, BS;
    switch (interval_num)
    {
    case 0:
        BS = 64;
        GS = (ceil(2048 * num_of_SM / BS)) * blocks_div_up_SMs;
        GS = min(GS, A->M);
        kernel_Form_Cptr<191><<<GS, BS>>>(d_max_nnz_in_onerow, A->M, A->d_ptr, A->d_col, C->d_ptr, mask_matrixB->pitch, mask_matrixB->num_unit, mask_matrixB->mask_matrix, mask_matrixB->after_or_mask_matrix);
        break;
    case 1:
        BS = 128;
        GS = (ceil(2048 * num_of_SM / BS)) * blocks_div_up_SMs;
        GS = min(GS, A->M);
        kernel_Form_Cptr<383><<<GS, BS>>>(d_max_nnz_in_onerow, A->M, A->d_ptr, A->d_col, C->d_ptr, mask_matrixB->pitch, mask_matrixB->num_unit, mask_matrixB->mask_matrix, mask_matrixB->after_or_mask_matrix);
        break;
    case 2:
        BS = 256;
        GS = (ceil(2048 * num_of_SM / BS)) * blocks_div_up_SMs;
        GS = min(GS, A->M);
        kernel_Form_Cptr<767><<<GS, BS>>>(d_max_nnz_in_onerow, A->M, A->d_ptr, A->d_col, C->d_ptr, mask_matrixB->pitch, mask_matrixB->num_unit, mask_matrixB->mask_matrix, mask_matrixB->after_or_mask_matrix);
        break;
    case 3:
        BS = 512;
        GS = (ceil(2048 * num_of_SM / BS)) * blocks_div_up_SMs;
        GS = min(GS, A->M);
        kernel_Form_Cptr<1535><<<GS, BS>>>(d_max_nnz_in_onerow, A->M, A->d_ptr, A->d_col, C->d_ptr, mask_matrixB->pitch, mask_matrixB->num_unit, mask_matrixB->mask_matrix, mask_matrixB->after_or_mask_matrix);
        break;
    case 4:
        BS = 1024;
        GS = (ceil(2048 * num_of_SM / BS)) * blocks_div_up_SMs;
        GS = min(GS, A->M);
        kernel_Form_Cptr<3071><<<GS, BS>>>(d_max_nnz_in_onerow, A->M, A->d_ptr, A->d_col, C->d_ptr, mask_matrixB->pitch, mask_matrixB->num_unit, mask_matrixB->mask_matrix, mask_matrixB->after_or_mask_matrix);
        break;
    }
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cudaDeviceSynchronize();
    cudaFree(mask_matrixB->mask_matrix);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, C->d_ptr, C->d_ptr, A->M + 1, 0);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, C->d_ptr, C->d_ptr, A->M + 1, 0);
    cudaFree(d_temp_storage);
}

template <int size_hash>
__global__ void kernel_Form_Ccol(int A_M, index_t *a_dptr, index_t *a_dcol, index_t *c_dptr, index_t *c_dcol, int pitch, int num_unit, DateTypeStoreMask *after_or_mask_matrix)
{
    int row_id;
    __shared__ DateTypeStoreMask target[size_hash];
    __shared__ int buffer[size_hash << 1];
    __shared__ int shared_offset[1];
    for (row_id = blockIdx.x; row_id < A_M; row_id += gridDim.x)
    {
        int start_nnz = c_dptr[row_id];
        int end_nnz = c_dptr[row_id + 1];
        int Cnnz_this_row = end_nnz - start_nnz;
        int index;
        if (threadIdx.x == 0)
        {
            shared_offset[0] = 0;
        }
        if (Cnnz_this_row)
        {
            DateTypeStoreMask *row_point = (DateTypeStoreMask *)((char *)after_or_mask_matrix + (row_id * pitch));
            for (index = threadIdx.x; index < num_unit; index += blockDim.x)
            {
                target[index] = row_point[index];
            }
            __syncthreads();
            int loc_offset;
            for (index = threadIdx.x; index < num_unit; index += blockDim.x)
            {
                if (target[index])
                {
                    DateTypeStoreMask temp = target[index];
                    while (temp)
                    {
                        loc_offset = atomicAdd(shared_offset, 1);
                        buffer[loc_offset] = __ffsll(temp) - 1 + (index << 6);
                        temp &= (temp - 1);
                    }
                }
            }
            __syncthreads();
            int count, ntarget;
            for (int j = threadIdx.x; j < Cnnz_this_row; j += blockDim.x)
            {
                ntarget = buffer[j];
                count = 0;
                for (int k = 0; k < Cnnz_this_row; k++)
                {
                    count += (unsigned int)(buffer[k] - ntarget) >> 31;
                }
                c_dcol[start_nnz + count] = buffer[j];
            }
        }
    }
}

template <int threads_onerow>
__global__ void kernel_Form_Ccol_with_partition(
    int pitch, int num_unit,
    const int *__restrict__ d_crpt,
    DateTypeStoreMask *after_or_mask_matrix,
    const int *__restrict__ d_bins,
    int *__restrict__ d_ccol)
{
    int loc_par_id = blockIdx.x;
    int off_row_id = d_bins[loc_par_id];
    int end_row_id = d_bins[loc_par_id + 1];

    int tid = threadIdx.x & (threads_onerow - 1);
    int wrapid = threadIdx.x / threads_onerow; // threads_onerow is 32
    int nums_wrap = 16;

    extern __shared__ int shared_mem[]; // total size is up to bin
    int *shared_col = shared_mem;
    int *col_table;
    int *shared_offset = shared_mem + 4096; // shared_offset size is rows_oneblock
    int j, k;
    for (j = threadIdx.x; j < 4096; j += blockDim.x)
    {
        shared_col[j] = -1;
    }
    for (j = threadIdx.x; j < 2048; j += blockDim.x)
    {
        shared_offset[j] = 0;
    }
    __syncthreads();

    DateTypeStoreMask target;
    int loc_offset;
    for (int row_id = wrapid + off_row_id; row_id < end_row_id; row_id += nums_wrap)
    {
        DateTypeStoreMask *row_point = (DateTypeStoreMask *)((char *)after_or_mask_matrix + (row_id * pitch));
        col_table = shared_col + d_crpt[row_id] - d_crpt[off_row_id];
        for (int index = tid; index < num_unit; index += threads_onerow)
        {
            target = row_point[index];
            if (target)
            {
                DateTypeStoreMask temp = target;
                while (temp)
                {
                    loc_offset = atomicAdd(shared_offset + row_id - off_row_id, 1);
                    col_table[loc_offset] = __ffsll(temp) - 1 + (index << 6);
                    temp &= (temp - 1);
                }
            }
        }
    }

    __syncthreads();
    int count, ntarget;
    for (int row_id = wrapid + off_row_id; row_id < end_row_id; row_id += nums_wrap)
    {
        int start_nnz_this_row = d_crpt[row_id];
        int Cnnz_this_row = d_crpt[row_id + 1] - start_nnz_this_row;
        col_table = shared_col + d_crpt[row_id] - d_crpt[off_row_id];
        for (int j = tid; j < Cnnz_this_row; j += threads_onerow)
        {
            ntarget = col_table[j];
            count = 0;
            for (k = 0; k < Cnnz_this_row; k++)
            {
                count += (unsigned int)(col_table[k] - ntarget) >> 31;
            }
            d_ccol[start_nnz_this_row + count] = col_table[j];
        }
    }
}

void Form_Ccol(NHC_CSR *C, NHC_mask_matrix *mask_matrixB, NHC_CSR *A, NHC_bin *c_bin)
{
    int max_nnz_in_onerow;
    cudaMemcpy(&max_nnz_in_onerow, c_bin->d_max_nnz_in_onerow, sizeof(int), cudaMemcpyDeviceToHost);
    int selected_num = max(mask_matrixB->num_unit, div_up(max_nnz_in_onerow, 2));
    int size_arr[5] = {191, 383, 767, 1535, 3071};
    int interval_num = get_interval(selected_num, size_arr);
    int GS, BS;
    switch (interval_num)
    {
    case 0:
        BS = 64;
        GS = ceil(2048 * num_of_SM / BS) * blocks_div_up_SMs;
        GS = min(GS, A->M);
        kernel_Form_Ccol<191><<<GS, BS>>>(A->M, A->d_ptr, A->d_col, C->d_ptr, C->d_col, mask_matrixB->pitch, mask_matrixB->num_unit, mask_matrixB->after_or_mask_matrix);
        break;
    case 1:
        BS = 128;
        GS = ceil(2048 * num_of_SM / BS) * blocks_div_up_SMs;
        GS = min(GS, A->M);
        kernel_Form_Ccol<383><<<GS, BS>>>(A->M, A->d_ptr, A->d_col, C->d_ptr, C->d_col, mask_matrixB->pitch, mask_matrixB->num_unit, mask_matrixB->after_or_mask_matrix);
        break;
    case 2:
        BS = 256;
        GS = ceil(2048 * num_of_SM / BS) * blocks_div_up_SMs;
        GS = min(GS, A->M);
        kernel_Form_Ccol<767><<<GS, BS>>>(A->M, A->d_ptr, A->d_col, C->d_ptr, C->d_col, mask_matrixB->pitch, mask_matrixB->num_unit, mask_matrixB->after_or_mask_matrix);
        break;
    case 3:
        BS = 512;
        GS = ceil(2048 * num_of_SM / BS) * blocks_div_up_SMs;
        GS = min(GS, A->M);
        kernel_Form_Ccol<1535><<<GS, BS>>>(A->M, A->d_ptr, A->d_col, C->d_ptr, C->d_col, mask_matrixB->pitch, mask_matrixB->num_unit, mask_matrixB->after_or_mask_matrix);
        break;
    case 4:
        BS = 1024;
        GS = ceil(2048 * num_of_SM / BS) * blocks_div_up_SMs;
        GS = min(GS, A->M);
        kernel_Form_Ccol<3071><<<GS, BS>>>(A->M, A->d_ptr, A->d_col, C->d_ptr, C->d_col, mask_matrixB->pitch, mask_matrixB->num_unit, mask_matrixB->after_or_mask_matrix);
        break;
    }
}
