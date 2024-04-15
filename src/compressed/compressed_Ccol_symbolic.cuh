template <int threads_onerow,
          int rows_oneblock,
          int nnz_onerow>
__global__ void k_formccol_shared_for_many_rows(
    const int *__restrict__ d_crpt,
    const int *__restrict__ d_ctilerpt, const int *__restrict__ d_ctilecol,
    const DateTypeStoreCompressMask *__restrict__ d_cmask,
    const int *__restrict__ d_bins,
    int bin_size,
    int *__restrict__ d_ccol)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x & (threads_onerow - 1);
    int rid = i / threads_onerow;
    int block_rid = rid & (rows_oneblock - 1); // warp id in one block

    extern __shared__ int shared_mem[]; // total size is up to bin
    int *shared_col = shared_mem;
    int *shared_offset = shared_mem + nnz_onerow * rows_oneblock; // shared_offset size is rows_oneblock
    int j, k;
    for (j = threadIdx.x; j < nnz_onerow * rows_oneblock; j += blockDim.x)
    {
        shared_col[j] = -1;
    }
    if (threadIdx.x < rows_oneblock)
    {
        shared_offset[threadIdx.x] = 0;
    }
    if (rid >= bin_size)
    {
        return;
    }
    __syncthreads();
    int *col_table = shared_col + block_rid * nnz_onerow; // every row's col Starting address
    rid = d_bins[rid];
    int c_tilecol, loc_offset;
    DateTypeStoreCompressMask cmask;
    for (j = d_ctilerpt[rid] + tid; j < d_ctilerpt[rid + 1]; j += threads_onerow)
    { // pwarp per row, thread per a item, thread per b row
        c_tilecol = d_ctilecol[j];
        cmask = d_cmask[j];
        while (cmask)
        {
            loc_offset = atomicAdd(shared_offset + block_rid, 1);
            col_table[loc_offset] = __ffsll(cmask) - 1 + (c_tilecol << 6);
            cmask &= (cmask - 1);
        }
    }
    __syncthreads();
    int count, ntarget;
    int start_nnz_this_row = d_crpt[rid];
    int Cnnz_this_row = d_crpt[rid + 1] - start_nnz_this_row;
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
// also get the c total nnz
template <int threads_onerow,
          int rows_oneblock,
          int nnz_onerow>
__global__ void k_formccol_global_for_many_rows(
    const int *__restrict__ d_crpt,
    const int *__restrict__ d_ctilerpt, const int *__restrict__ d_ctilecol,
    const DateTypeStoreCompressMask *__restrict__ d_cmask,
    const int *__restrict__ d_bins,
    int bin_size,
    int *__restrict__ d_ccol)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x & (threads_onerow - 1);
    int rid = i / threads_onerow;
    int block_rid = rid & (rows_oneblock - 1); // warp id in one block
    __shared__ int shared_offset[1];
    int j, k;
    if (threadIdx.x < rows_oneblock)
    {
        shared_offset[threadIdx.x] = 0;
    }
    if (rid >= bin_size)
    {
        return;
    }
    __syncthreads();
    rid = d_bins[rid];
    int c_tilecol, loc_offset;
    DateTypeStoreCompressMask cmask;
    for (j = d_ctilerpt[rid] + tid; j < d_ctilerpt[rid + 1]; j += threads_onerow)
    { // pwarp per row, thread per a item, thread per b row
        c_tilecol = d_ctilecol[j];
        cmask = d_cmask[j];
        while (cmask)
        {
            loc_offset = atomicAdd(shared_offset + block_rid, 1);
            d_ccol[loc_offset] = __ffsll(cmask) - 1 + (c_tilecol << 6);
            cmask &= (cmask - 1);
        }
    }
    // __syncthreads();
    // int count, ntarget;
    // int start_nnz_this_row=d_crpt[rid];
    // int Cnnz_this_row = d_crpt[rid+1] - start_nnz_this_row;
    // for (int j = tid; j < Cnnz_this_row; j += threads_onerow) {
    //     ntarget = col_table[j];
    //     count = 0;
    //     for (k = 0; k < Cnnz_this_row; k++) {
    //         count += (unsigned int)(col_table[k] - ntarget) >> 31;
    //     }
    //     d_ccol[start_nnz_this_row + count] = col_table[j];
    // }
}

void h_form_ccol(compressed_bin *compressed_bin, NHC_CSR *c)
{
    int gs, bs = 1024;
    if (compressed_bin->bin_size[12])
    {
        gs = compressed_bin->bin_size[12];
        cudaFuncSetAttribute(k_formccol_shared_for_many_rows<1024, 1, 24576>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        k_formccol_shared_for_many_rows<1024, 1, 24576><<<gs, bs, 98304, compressed_bin->streams[12]>>>(c->d_ptr, c->d_tile_ptr, c->d_tile_col, c->d_mask_num,
                                                                                                        compressed_bin->d_bins + compressed_bin->bin_offset[12], compressed_bin->bin_size[12], c->d_col);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX k_formccol_shared_for_many_rows bin_size[12] is failed\n");
            }
            else
            {
                printf("/////////// k_formccol_shared_for_many_rows bin_size[12] is cudaSuccess\n");
            }
        }
#endif
    }
    if (compressed_bin->bin_size[11])
    {
        gs = compressed_bin->bin_size[11];
        k_formccol_shared_for_many_rows<1024, 1, 12288><<<gs, bs, 49152, compressed_bin->streams[11]>>>(c->d_ptr, c->d_tile_ptr, c->d_tile_col, c->d_mask_num,
                                                                                                        compressed_bin->d_bins + compressed_bin->bin_offset[11], compressed_bin->bin_size[11], c->d_col);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX k_formccol_shared_for_many_rows bin_size[11] is failed\n");
            }
            else
            {
                printf("/////////// k_formccol_shared_for_many_rows bin_size[11] is cudaSuccess\n");
            }
        }
#endif
    }
    if (compressed_bin->bin_size[10])
    {
        gs = compressed_bin->bin_size[10];
        k_formccol_shared_for_many_rows<1024, 1, 8192><<<gs, bs, 49152, compressed_bin->streams[10]>>>(c->d_ptr, c->d_tile_ptr, c->d_tile_col, c->d_mask_num,
                                                                                                       compressed_bin->d_bins + compressed_bin->bin_offset[10], compressed_bin->bin_size[10], c->d_col);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX k_formccol_shared_for_many_rows bin_size[10] is failed\n");
            }
            else
            {
                printf("/////////// k_formccol_shared_for_many_rows bin_size[10] is cudaSuccess\n");
            }
        }
#endif
    }

    if (compressed_bin->bin_size[9])
    {
        gs = (compressed_bin->bin_size[9] + 1) >> 1;
        k_formccol_shared_for_many_rows<512, 2, 4096><<<gs, bs, 49152, compressed_bin->streams[9]>>>(c->d_ptr, c->d_tile_ptr, c->d_tile_col, c->d_mask_num,
                                                                                                     compressed_bin->d_bins + compressed_bin->bin_offset[9], compressed_bin->bin_size[9], c->d_col);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX k_formccol_shared_for_many_rows bin_size[9] is failed\n");
            }
            else
            {
                printf("/////////// k_formccol_shared_for_many_rows bin_size[9] is cudaSuccess\n");
            }
        }
#endif
    }
    if (compressed_bin->bin_size[8])
    {
        gs = (compressed_bin->bin_size[8] + 3) >> 2;
        k_formccol_shared_for_many_rows<256, 4, 2048><<<gs, bs, 49152, compressed_bin->streams[8]>>>(c->d_ptr, c->d_tile_ptr, c->d_tile_col, c->d_mask_num,
                                                                                                     compressed_bin->d_bins + compressed_bin->bin_offset[8], compressed_bin->bin_size[8], c->d_col);

#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX k_formccol_shared_for_many_rows bin_size[8] is failed\n");
            }
            else
            {
                printf("/////////// k_formccol_shared_for_many_rows bin_size[8] is cudaSuccess\n");
            }
        }
#endif
    }
    if (compressed_bin->bin_size[7])
    {
        gs = (compressed_bin->bin_size[7] + 7) >> 3;
        k_formccol_shared_for_many_rows<128, 8, 1024><<<gs, bs, 49152, compressed_bin->streams[7]>>>(c->d_ptr, c->d_tile_ptr, c->d_tile_col, c->d_mask_num,
                                                                                                     compressed_bin->d_bins + compressed_bin->bin_offset[7], compressed_bin->bin_size[7], c->d_col);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX k_formccol_shared_for_many_rows bin_size[7] is failed\n");
            }
            else
            {
                printf("/////////// k_formccol_shared_for_many_rows bin_size[7] is cudaSuccess\n");
            }
        }
#endif
    }
    if (compressed_bin->bin_size[6])
    {
        gs = (compressed_bin->bin_size[6] + 15) >> 4;
        k_formccol_shared_for_many_rows<64, 16, 512><<<gs, bs, 49152, compressed_bin->streams[6]>>>(c->d_ptr, c->d_tile_ptr, c->d_tile_col, c->d_mask_num,
                                                                                                    compressed_bin->d_bins + compressed_bin->bin_offset[6], compressed_bin->bin_size[6], c->d_col);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX k_formccol_shared_for_many_rows bin_size[6] is failed\n");
            }
            else
            {
                printf("/////////// k_formccol_shared_for_many_rows bin_size[6] is cudaSuccess\n");
            }
        }
#endif
    }
    if (compressed_bin->bin_size[5])
    {
        gs = (compressed_bin->bin_size[5] + 31) >> 5;
        k_formccol_shared_for_many_rows<32, 32, 256><<<gs, bs, 49152, compressed_bin->streams[5]>>>(c->d_ptr, c->d_tile_ptr, c->d_tile_col, c->d_mask_num,
                                                                                                    compressed_bin->d_bins + compressed_bin->bin_offset[5], compressed_bin->bin_size[5], c->d_col);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX k_formccol_shared_for_many_rows bin_size[5] is failed\n");
            }
            else
            {
                printf("/////////// k_formccol_shared_for_many_rows bin_size[5] is cudaSuccess\n");
            }
        }
#endif
    }
    if (compressed_bin->bin_size[4])
    {
        gs = (compressed_bin->bin_size[4] + 63) >> 6;
        k_formccol_shared_for_many_rows<16, 64, 128><<<gs, bs, 49152, compressed_bin->streams[4]>>>(c->d_ptr, c->d_tile_ptr, c->d_tile_col, c->d_mask_num,
                                                                                                    compressed_bin->d_bins + compressed_bin->bin_offset[4], compressed_bin->bin_size[4], c->d_col);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX k_formccol_shared_for_many_rows bin_size[4] is failed\n");
            }
            else
            {
                printf("/////////// k_formccol_shared_for_many_rows bin_size[4] is cudaSuccess\n");
            }
        }
#endif
    }
    if (compressed_bin->bin_size[3])
    {
        gs = (compressed_bin->bin_size[3] + 127) >> 7;
        k_formccol_shared_for_many_rows<8, 128, 64><<<gs, bs, 49152, compressed_bin->streams[3]>>>(c->d_ptr, c->d_tile_ptr, c->d_tile_col, c->d_mask_num,
                                                                                                   compressed_bin->d_bins + compressed_bin->bin_offset[3], compressed_bin->bin_size[3], c->d_col);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX k_formccol_shared_for_many_rows bin_size[3] is failed\n");
            }
            else
            {
                printf("/////////// k_formccol_shared_for_many_rows bin_size[3] is cudaSuccess\n");
            }
        }
#endif
    }
    if (compressed_bin->bin_size[2])
    {
        gs = (compressed_bin->bin_size[2] + 255) >> 8;
        k_formccol_shared_for_many_rows<4, 256, 32><<<gs, bs, 49152, compressed_bin->streams[2]>>>(c->d_ptr, c->d_tile_ptr, c->d_tile_col, c->d_mask_num,
                                                                                                   compressed_bin->d_bins + compressed_bin->bin_offset[2], compressed_bin->bin_size[2], c->d_col);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX k_formccol_shared_for_many_rows bin_size[2] is failed\n");
            }
            else
            {
                printf("/////////// k_formccol_shared_for_many_rows bin_size[2] is cudaSuccess\n");
            }
        }
#endif
    }
    if (compressed_bin->bin_size[1])
    {
        gs = (compressed_bin->bin_size[1] + 511) >> 9;
        k_formccol_shared_for_many_rows<2, 512, 16><<<gs, bs, 49152, compressed_bin->streams[1]>>>(c->d_ptr, c->d_tile_ptr, c->d_tile_col, c->d_mask_num,
                                                                                                   compressed_bin->d_bins + compressed_bin->bin_offset[1], compressed_bin->bin_size[1], c->d_col);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX k_formccol_shared_for_many_rows bin_size[1] is failed\n");
            }
            else
            {
                printf("/////////// k_formccol_shared_for_many_rows bin_size[1] is cudaSuccess\n");
            }
        }
#endif
    }
    if (compressed_bin->bin_size[0])
    {
        gs = (compressed_bin->bin_size[0] + 1023) >> 10;
        k_formccol_shared_for_many_rows<1, 1024, 8><<<gs, bs, 49152, compressed_bin->streams[0]>>>(c->d_ptr, c->d_tile_ptr, c->d_tile_col, c->d_mask_num,
                                                                                                   compressed_bin->d_bins, compressed_bin->bin_size[0], c->d_col);
    }
#if checek_kernel
    {
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("XXXXXXXXXXXXX k_formccol_shared_for_many_rows bin_size[0] is failed\n");
        }
        else
        {
            printf("/////////// k_formccol_shared_for_many_rows bin_size[0] is cudaSuccess\n");
        }
    }
#endif
}
void h_get_Ccol_from_Ctile(compressed_bin *compressed_bin, NHC_CSR *C)
{
    h_form_ccol(compressed_bin, C);
}
