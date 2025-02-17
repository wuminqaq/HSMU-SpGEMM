template <int threads_oneAnnz,
          int threads_onerow,
          int rows_oneblock,
          int nnz_onerow>
__global__ void k_formcval_shared_for_many_rows(
    const index_t *__restrict__ d_arpt, const index_t *__restrict__ d_acol,
    const value_t *__restrict__ d_aval,
    const index_t *__restrict__ d_brpt, const index_t *__restrict__ d_bcol,
    const value_t *__restrict__ d_bval,
    int *d_bins, int bin_size,
    index_t *d_crpt, const index_t *d_ccol, value_t *d_cval)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x & (threads_onerow - 1);
    int rid = i / threads_onerow;
    int block_rid = rid & (rows_oneblock - 1); // warp id in one block,warpid

    extern __shared__ int shared_mem[]; // total size is 4096
    int tsize = 4096;
    int *shared_col = shared_mem;
    value_t *shared_cnum = (value_t *)(shared_mem + tsize);
    int j, k;
    for (j = threadIdx.x; j < tsize; j += blockDim.x)
    {
        shared_col[j] = -1;
        shared_cnum[j] = 0;
    }
    if (rid >= bin_size)
    {
        return;
    }
    __syncthreads();
    int *col_table = shared_col + block_rid * nnz_onerow; // every row's col Starting address
    value_t *num_table = shared_cnum + block_rid * nnz_onerow;
    rid = d_bins[rid];
    int c_start_nnz = d_crpt[rid];
    int Cnnz_this_row = d_crpt[rid + 1] - c_start_nnz;
    int a_start_nnz = d_arpt[rid];
    int a_end_nnz = d_arpt[rid + 1];
    for (j = tid; j < Cnnz_this_row; j += threads_onerow)
    {
        col_table[j] = d_ccol[c_start_nnz + j];
    }
    __syncthreads();
    int inner_wid = tid / threads_oneAnnz;
    int inner_nw = threads_onerow / threads_oneAnnz;
    int inner_tid = tid & (threads_oneAnnz - 1);
    index_t acol, bcol, hash;
    value_t aval, bval;

    for (j = a_start_nnz + inner_wid; j < a_end_nnz; j += inner_nw)
    {
        aval = d_aval[j];
        acol = d_acol[j];
        hash = 0;
        for (k = d_brpt[acol] + inner_tid; k < d_brpt[acol + 1]; k += threads_oneAnnz)
        {
            bcol = d_bcol[k];
            bval = d_bval[k];
            hash = Binary_search_for_hash_loction(col_table, hash, Cnnz_this_row - 1, bcol);
            atomicAdd(num_table + hash, aval * bval);
        }
    }
    __syncthreads();
    for (j = tid; j < Cnnz_this_row; j += threads_onerow)
    {
        d_cval[c_start_nnz + j] = num_table[j];
    }
}


__global__ void k_formcval_shared_for_one_large_row(
    const index_t *__restrict__ d_arpt, const index_t *__restrict__ d_acol,
    const value_t *__restrict__ d_aval,
    const index_t *__restrict__ d_brpt, const index_t *__restrict__ d_bcol,
    const value_t *__restrict__ d_bval,
    int *d_bins,
    index_t *d_crpt, const index_t *d_ccol, value_t *d_cval)
{
    int tid = threadIdx.x;
    int block_rid = blockIdx.x; // warp id in one block,warpid
    int threads_oneAnnz = 32;
    extern __shared__ int shared_mem[]; // total size is 4096
    int tsize = 8192;
    int *shared_col = shared_mem;
    value_t *shared_cnum = (value_t *)(shared_mem + tsize);
    int j, k;
    for (j = threadIdx.x; j < tsize; j += blockDim.x)
    {
        shared_col[j] = -1;
        shared_cnum[j] = 0;
    }
    __syncthreads();
    int rid = d_bins[block_rid];
    int c_start_nnz = d_crpt[rid];
    int Cnnz_this_row = d_crpt[rid + 1] - c_start_nnz;
    int a_start_nnz = d_arpt[rid];
    int a_end_nnz = d_arpt[rid + 1];
    for (j = tid; j < Cnnz_this_row; j += blockDim.x)
    {
        shared_col[j] = d_ccol[c_start_nnz + j];
    }
    __syncthreads();
    int inner_wid = tid / threads_oneAnnz;
    int inner_nw = blockDim.x / threads_oneAnnz;
    int inner_tid = tid & (threads_oneAnnz - 1);
    index_t acol, bcol, hash;
    value_t aval, bval;

    for (j = a_start_nnz + inner_wid; j < a_end_nnz; j += inner_nw)
    {
        aval = d_aval[j];
        acol = d_acol[j];
        hash = 0;
        for (k = d_brpt[acol] + inner_tid; k < d_brpt[acol + 1]; k += threads_oneAnnz)
        {
            bcol = d_bcol[k];
            bval = d_bval[k];
            hash = Binary_search_for_hash_loction(shared_col, hash, Cnnz_this_row - 1, bcol);
            atomicAdd(shared_cnum + hash, aval * bval);
        }
    }
    __syncthreads();
    for (j = tid; j < Cnnz_this_row; j += blockDim.x)
    {
        d_cval[c_start_nnz + j] = shared_cnum[j];
    }
}

template <int threads_oneAnnz,
          int threads_onerow,
          int nnz_onerow>
__global__ void k_formcval_only_col_shared_for_one_row(
    const index_t *__restrict__ d_arpt, const index_t *__restrict__ d_acol,
    const value_t *__restrict__ d_aval,
    const index_t *__restrict__ d_brpt, const index_t *__restrict__ d_bcol,
    const value_t *__restrict__ d_bval,
    int *d_bins,
    index_t *d_crpt, const index_t *d_ccol, value_t *d_cval)
{
    int tid = threadIdx.x;
    int rid = blockIdx.x;

    extern __shared__ int shared_mem[];
    int tsize = nnz_onerow;
    int *shared_col = shared_mem;
    int j, k;
    for (j = threadIdx.x; j < tsize; j += blockDim.x)
    {
        shared_col[j] = -1;
    }
    __syncthreads();
    rid = d_bins[rid];
    int c_start_nnz = d_crpt[rid];
    int Cnnz_this_row = d_crpt[rid + 1] - c_start_nnz;
    value_t *num_table = d_cval + c_start_nnz;
    int a_start_nnz = d_arpt[rid];
    int a_end_nnz = d_arpt[rid + 1];
    for (j = tid; j < Cnnz_this_row; j += threads_onerow)
    {
        shared_col[j] = d_ccol[c_start_nnz + j];
    }
    __syncthreads();
    int inner_wid = tid / threads_oneAnnz;
    int inner_nw = threads_onerow / threads_oneAnnz;
    int inner_tid = tid & (threads_oneAnnz - 1);
    index_t acol, bcol, hash;
    value_t aval, bval;
    for (j = a_start_nnz + inner_wid; j < a_end_nnz; j += inner_nw)
    {
        aval = d_aval[j];
        acol = d_acol[j];
        hash = 0;
        for (k = d_brpt[acol] + inner_tid; k < d_brpt[acol + 1]; k += threads_oneAnnz)
        {
            bcol = d_bcol[k];
            bval = d_bval[k];
            hash = Binary_search_for_hash_loction(shared_col, hash, Cnnz_this_row - 1, bcol);
            atomicAdd(num_table + hash, aval * bval);
        }
    }
}


__global__ void compute_num_nnz(int *dptr, int *d_bin, int bin_size)
{
    long long tmp_nnz = 0;
    for (int i = 0; i < bin_size; i++)
    {
        int rowid = d_bin[i];
        tmp_nnz += (dptr[rowid + 1] - dptr[rowid]);
    }
    printf("the tmp_nnz is %lld----------------\n", tmp_nnz);
}

template <class GlobalMapRowOffsets>
void h_form_cval_with_dense_accumulator(compressed_bin *compressed_bin, NHC_CSR *A, NHC_CSR *B, NHC_CSR *C,
                                        GlobalMapRowOffsets *__restrict__ maps, INDEX_TYPE mapCount)
{
    double time0, time1;
    int gs, bs = 1024;
    unsigned long long practical_need = 0;
    unsigned long long true_need = 0;
    if (compressed_bin->bin_size[8])
    {
        gs = (compressed_bin->bin_size[8] + 1) >> 1;
        k_formcval_shared_for_many_rows<32, 512, 2, 2048><<<gs, bs, 49152, compressed_bin->streams[8]>>>(A->d_ptr, A->d_col, A->d_val, B->d_ptr, B->d_col, B->d_val,
                                                                                                         compressed_bin->d_bins + compressed_bin->bin_offset[8], compressed_bin->bin_size[8],
                                                                                                         C->d_ptr, C->d_col, C->d_val);
#if compute_share
        practical_need += (unsigned long long)gs * 49152;
        // compute_num_nnz<<<1,1>>>(C->d_ptr,compressed_bin->d_bins+compressed_bin->bin_offset[8],compressed_bin->bin_size[8]);
        printf("the gs*49152 is %d\n", gs * 49152);
#endif
#if checek_kernel
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("compressed_bin->bin_size[8] is failed\n");
        }
        else
        {
            printf("compressed_bin->bin_size[8] is cudaSuccess\n");
        }
#endif
    }
    if (compressed_bin->bin_size[9])
    {
        gs = compressed_bin->bin_size[9];
        k_formcval_shared_for_many_rows<32, 1024, 1, 4096><<<gs, bs, 49152, compressed_bin->streams[9]>>>(A->d_ptr, A->d_col, A->d_val, B->d_ptr, B->d_col, B->d_val,
                                                                                                          compressed_bin->d_bins + compressed_bin->bin_offset[9], compressed_bin->bin_size[9],
                                                                                                          C->d_ptr, C->d_col, C->d_val);
#if compute_share
        practical_need += (unsigned long long)gs * 49152;
        // compute_num_nnz<<<1,1>>>(C->d_ptr,compressed_bin->d_bins+compressed_bin->bin_offset[8],compressed_bin->bin_size[8]);
        printf("the gs*49152 is %d\n", gs * 49152);
#endif
#if checek_kernel
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("compressed_bin->bin_size[9] is failed\n");
        }
        else
        {
            printf("compressed_bin->bin_size[9] is cudaSuccess\n");
        }
#endif
    }
    if (compressed_bin->bin_size[7])
    {
        gs = (compressed_bin->bin_size[7] + 3) >> 2;
        k_formcval_shared_for_many_rows<32, 256, 4, 1024><<<gs, bs, 49152, compressed_bin->streams[7]>>>(A->d_ptr, A->d_col, A->d_val, B->d_ptr, B->d_col, B->d_val,
                                                                                                         compressed_bin->d_bins + compressed_bin->bin_offset[7], compressed_bin->bin_size[7],
                                                                                                         C->d_ptr, C->d_col, C->d_val);
#if compute_share
        practical_need += (unsigned long long)gs * 49152;
        // compute_num_nnz<<<1,1>>>(C->d_ptr,compressed_bin->d_bins+compressed_bin->bin_offset[7],compressed_bin->bin_size[7]);
        printf("the gs*49152 is %d\n", gs * 49152);
#endif
#if checek_kernel
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("compressed_bin->bin_size[7] is failed\n");
        }
        else
        {
            printf("compressed_bin->bin_size[7] is cudaSuccess\n");
        }
#endif
    }
    uint32_t *d_combined_pointers;
    index_t *rowOffsetMapIndices = nullptr;
    index_t *d_longestRowALength = nullptr;
    if (compressed_bin->bin_size[NUM_BIN_FOR_Ccol - 1])
    {
        uint32_t *d_rowOperations = nullptr;
        uint32_t *d_rowMaxOperations = nullptr;
        uint32_t *d_maxElementsPerRow = nullptr;
        uint32_t *d_sumProducts = nullptr;
        uint32_t *d_rowColMinMax = nullptr;
        uint32_t *d_maxComputationsPerRow = nullptr;

        uint32_t sumProducts = 0;
        uint32_t maxComputationsPerRow = 0;
        size_t cubTempBytesScan = 0;
        size_t cubTmpBytesReduce = 0;
        size_t cubTmpBytesActual = 0;

        void *cubTmp = nullptr;

        {
            cub::DeviceScan::ExclusiveSum(cubTmp, cubTempBytesScan, C->rowPtr, C->rowPtr, C->M + 1);
            cub::DeviceReduce::Sum(cubTmp, cubTmpBytesReduce, C->rowPtr, C->rowPtr, C->M);
            cubTmpBytesReduce = std::max(cubTempBytesScan, cubTmpBytesReduce); // have changed
        }
        size_t d_combined_pointers_size = sizeof(int) * (3 + 3 * A->M);
        CHECK_CUDA(cudaMalloc((void **)&d_combined_pointers, d_combined_pointers_size));
        CHECK_CUDA(cudaMemsetAsync(d_combined_pointers, 0, d_combined_pointers_size));
        d_maxElementsPerRow = d_combined_pointers; //
        /* keep this order */
        d_sumProducts = (uint32_t *)&d_maxElementsPerRow[1];
        d_maxComputationsPerRow = (uint32_t *)&d_sumProducts[1];
        d_rowOperations = &d_maxComputationsPerRow[1];
        d_rowMaxOperations = &d_rowOperations[A->M];
        d_rowColMinMax = (uint32_t *)&d_rowMaxOperations[A->M];
        {
            const int threadsPerBlock = 128;
            int cudaCores = 10752;
            // limit to threadsPerBlock rows!
            // -> and always try to stay slightly below the threads per block size, because if you are slightly above, it is way more expensive than being far below
            int rowsPerBlock = std::min(threadsPerBlock, std::max(1, (threadsPerBlock - 8) / std::max(1, int(A->nnz / A->M))));
            rowsPerBlock = std::max(1, std::min(rowsPerBlock, (A->M) / (4 * cudaCores / threadsPerBlock)));
            readOperations<threadsPerBlock><<<div_up((A->M), rowsPerBlock), threadsPerBlock>>>(
                A->d_ptr, A->d_col, A->M, A->nnz, B->d_ptr, B->d_col, B->M, B->nnz, d_rowOperations, rowsPerBlock, d_maxComputationsPerRow, d_rowColMinMax, d_rowMaxOperations, d_sumProducts);
            cudaDeviceSynchronize();

            uint32_t tmpArr[2];
            CHECK_CUDA(cudaMemcpy(&tmpArr, d_sumProducts, sizeof(int) * 2, cudaMemcpyDeviceToHost));
            sumProducts = tmpArr[0];           // sumProducts is the number of Intermediate product
            maxComputationsPerRow = tmpArr[1]; // maxComputationsPerRow represents the largest intermediate row corresponding to A
            sumProducts = max(sumProducts, 1);
        }
        {
            size_t globalMapMaxSize = sizeof(GlobalMapRowOffsets);
            mapCount = min(compressed_bin->bin_size[NUM_BIN_FOR_Ccol - 1], num_of_SM * BLOCKS_PER_SM);
            const int maxRowsPerBlock = 32;
            index_t rowOffsetMapElementsPer = 0;
            cudaMalloc(&d_longestRowALength, sizeof(index_t));
            cudaMemset(d_longestRowALength, 0, sizeof(index_t));
            const uint32_t _threads = 256;
            const uint32_t rowsPerThread = 2;
            const uint32_t blocks = div_up(index_t(A->M), _threads * rowsPerThread);
            getLongestRowA<index_t, _threads, rowsPerThread><<<blocks, _threads>>>(A->d_ptr, d_longestRowALength, A->M);
            cudaMemcpy(&rowOffsetMapElementsPer, d_longestRowALength, sizeof(index_t), cudaMemcpyDeviceToHost);

            const int sharedBytesPerWarpNumeric = MAX_STATIC_SHARED / 32 - 24; // 24 byte is the maximum static shared memory per block
            const int entriesPerWarpNumeric = sharedBytesPerWarpNumeric / (sizeof(index_t) + sizeof(value_t));
            int elementsPerMap = rowOffsetMapElementsPer * 5 / 4;
            if (elementsPerMap * 2 * sizeof(index_t) > 32 * entriesPerWarpNumeric * (sizeof(index_t) + sizeof(value_t)))
            {
                cudaMalloc((void **)&maps, globalMapMaxSize * mapCount); // Each dense row requires a global table
                cudaMalloc((void **)&rowOffsetMapIndices, sizeof(index_t) * mapCount * (rowOffsetMapElementsPer + maxRowsPerBlock + 1));
                initializeGlobalMapsNoVal<GlobalMapRowOffsets, index_t><<<mapCount, 1024, 0, compressed_bin->streams[0]>>>((GlobalMapRowOffsets *)maps, mapCount, rowOffsetMapIndices, rowOffsetMapElementsPer, maxRowsPerBlock);
            }
        }
        const int warpsNumeric = 32;
        const int staticSharedMemPerBlockNumeric = 24;
        const int MAX_DYNAMIC_SHARED = 98304;
        const int dynamicSharedBytesPerWarpNumeric = MAX_DYNAMIC_SHARED / warpsNumeric - staticSharedMemPerBlockNumeric; // 24 byte is the maximum static shared memory per block
        const int dynamicSharedBytesPerBlockNumeric = dynamicSharedBytesPerWarpNumeric * warpsNumeric;                   // true expression
        // choose 97536 or 98304
        cudaFuncSetAttribute(denseSpGEMMNumeric<GlobalMapRowOffsets, dynamicSharedBytesPerBlockNumeric, true, 1024>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, dynamicSharedBytesPerBlockNumeric);

        cudaDeviceSynchronize();
        gs = compressed_bin->bin_size[NUM_BIN_FOR_Ccol - 1];
        denseSpGEMMNumeric<GlobalMapRowOffsets, dynamicSharedBytesPerBlockNumeric, true, 1024><<<gs, bs, dynamicSharedBytesPerBlockNumeric, compressed_bin->streams[NUM_BIN_FOR_Ccol - 1]>>>(
            B->nnz, B->M, B->N, A->d_ptr, B->d_ptr, A->d_col, B->d_col, A->d_val, B->d_val, maps, mapCount, C->d_col, C->d_val, C->d_ptr,
            d_rowOperations, d_rowColMinMax, d_rowMaxOperations,
            compressed_bin->d_bins + compressed_bin->bin_offset[NUM_BIN_FOR_Ccol - 1]);
#if checek_kernel
        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                printf("XXXXXXXXXXXXX denseSpGEMMNumeric bin_size[NUM_BIN_FOR_Ccol -1] is failed\n");
            }
            else
            {
                printf("/////////// denseSpGEMMNumeric bin_size[NUM_BIN_FOR_Ccol -1] is cudaSuccess\n");
            }
        }
#endif
    }

    if (compressed_bin->bin_size[10])
    {
        gs = compressed_bin->bin_size[10];
        cudaFuncSetAttribute(k_formcval_shared_for_one_large_row, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        k_formcval_shared_for_one_large_row<<<gs, bs, 98304, compressed_bin->streams[10]>>>(A->d_ptr, A->d_col, A->d_val, B->d_ptr, B->d_col, B->d_val,
                                                                                            compressed_bin->d_bins + compressed_bin->bin_offset[10],
                                                                                            C->d_ptr, C->d_col, C->d_val);
#if compute_share
        practical_need += (unsigned long long)gs * 49152;
        // compute_num_nnz<<<1,1>>>(C->d_ptr,compressed_bin->d_bins+compressed_bin->bin_offset[8],compressed_bin->bin_size[8]);
        printf("the gs*49152 is %d\n", gs * 49152);
#endif
#if checek_kernel
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("compressed_bin->bin_size[10] is failed\n");
        }
        else
        {
            printf("compressed_bin->bin_size[10] is cudaSuccess\n");
        }
#endif
    }
    if (compressed_bin->bin_size[11])
    {
        gs = compressed_bin->bin_size[11];
        k_formcval_only_col_shared_for_one_row<32, 1024, 12288><<<gs, bs, 49152, compressed_bin->streams[11]>>>(A->d_ptr, A->d_col, A->d_val, B->d_ptr, B->d_col, B->d_val,
                                                                                                                compressed_bin->d_bins + compressed_bin->bin_offset[11],
                                                                                                                C->d_ptr, C->d_col, C->d_val);
#if compute_share
        practical_need += (unsigned long long)gs * 49152;
        // compute_num_nnz<<<1,1>>>(C->d_ptr,compressed_bin->d_bins+compressed_bin->bin_offset[8],compressed_bin->bin_size[8]);
        printf("the gs*49152 is %d\n", gs * 49152);
#endif
#if checek_kernel
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("compressed_bin->bin_size[11] is failed\n");
        }
        else
        {
            printf("compressed_bin->bin_size[11] is cudaSuccess\n");
        }
#endif
    }
    if (compressed_bin->bin_size[12])
    {
        gs = compressed_bin->bin_size[12];
        cudaFuncSetAttribute(k_formcval_only_col_shared_for_one_row<32, 1024, 24576>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        k_formcval_only_col_shared_for_one_row<32, 1024, 24576><<<gs, bs, 98304, compressed_bin->streams[12]>>>(A->d_ptr, A->d_col, A->d_val, B->d_ptr, B->d_col, B->d_val,
                                                                                                                compressed_bin->d_bins + compressed_bin->bin_offset[12],
                                                                                                                C->d_ptr, C->d_col, C->d_val);
#if compute_share
        practical_need += (unsigned long long)gs * 49152;
        // compute_num_nnz<<<1,1>>>(C->d_ptr,compressed_bin->d_bins+compressed_bin->bin_offset[8],compressed_bin->bin_size[8]);
        printf("the gs*49152 is %d\n", gs * 49152);
#endif
#if checek_kernel
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("compressed_bin->bin_size[12] is failed\n");
        }
        else
        {
            printf("compressed_bin->bin_size[12] is cudaSuccess\n");
        }
#endif
    }
    if (compressed_bin->bin_size[6])
    {
        gs = (compressed_bin->bin_size[6] + 7) >> 3;
        k_formcval_shared_for_many_rows<32, 128, 8, 512><<<gs, bs, 49152, compressed_bin->streams[6]>>>(A->d_ptr, A->d_col, A->d_val, B->d_ptr, B->d_col, B->d_val,
                                                                                                        compressed_bin->d_bins + compressed_bin->bin_offset[6], compressed_bin->bin_size[6],
                                                                                                        C->d_ptr, C->d_col, C->d_val);
#if compute_share
        practical_need += (unsigned long long)gs * 49152;
        printf("the gs*49152 is %d\n", gs * 49152);
#endif
    }
    if (compressed_bin->bin_size[5])
    {
        gs = (compressed_bin->bin_size[5] + 15) >> 4;
        k_formcval_shared_for_many_rows<32, 64, 16, 256><<<gs, bs, 49152, compressed_bin->streams[5]>>>(A->d_ptr, A->d_col, A->d_val, B->d_ptr, B->d_col, B->d_val,
                                                                                                        compressed_bin->d_bins + compressed_bin->bin_offset[5], compressed_bin->bin_size[5],
                                                                                                        C->d_ptr, C->d_col, C->d_val);
#if compute_share
        practical_need += (unsigned long long)gs * 49152;
        printf("the gs*49152 is %d\n", gs * 49152);
#endif
    }
    if (compressed_bin->bin_size[4])
    {
        gs = (compressed_bin->bin_size[4] + 31) >> 5;
        k_formcval_shared_for_many_rows<16, 32, 32, 128><<<gs, bs, 49152, compressed_bin->streams[4]>>>(A->d_ptr, A->d_col, A->d_val, B->d_ptr, B->d_col, B->d_val,
                                                                                                        compressed_bin->d_bins + compressed_bin->bin_offset[4], compressed_bin->bin_size[4],
                                                                                                        C->d_ptr, C->d_col, C->d_val);
#if compute_share
        practical_need += (unsigned long long)gs * 49152;
        // compute_num_nnz<<<1,1>>>(C->d_ptr,compressed_bin->d_bins+compressed_bin->bin_offset[4],compressed_bin->bin_size[4]);
        printf("the gs is %d, the gs*49152 is %ld\n", gs, gs * 49152);
#endif
    }
    if (compressed_bin->bin_size[3])
    {
        gs = (compressed_bin->bin_size[3] + 63) >> 6;
        k_formcval_shared_for_many_rows<8, 16, 64, 64><<<gs, bs, 49152, compressed_bin->streams[3]>>>(A->d_ptr, A->d_col, A->d_val, B->d_ptr, B->d_col, B->d_val,
                                                                                                      compressed_bin->d_bins + compressed_bin->bin_offset[3], compressed_bin->bin_size[3],
                                                                                                      C->d_ptr, C->d_col, C->d_val);
#if compute_share
        practical_need += (unsigned long long)gs * 49152;
        // compute_num_nnz<<<1,1>>>(C->d_ptr,compressed_bin->d_bins+compressed_bin->bin_offset[3],compressed_bin->bin_size[3]);
        printf("the gs*49152 is %d\n", gs * 49152);
#endif
    }
    if (compressed_bin->bin_size[2])
    {
        gs = (compressed_bin->bin_size[2] + 127) >> 7;
        k_formcval_shared_for_many_rows<4, 8, 128, 32><<<gs, bs, 49152, compressed_bin->streams[2]>>>(A->d_ptr, A->d_col, A->d_val, B->d_ptr, B->d_col, B->d_val,
                                                                                                      compressed_bin->d_bins + compressed_bin->bin_offset[2], compressed_bin->bin_size[2],
                                                                                                      C->d_ptr, C->d_col, C->d_val);
#if compute_share
        practical_need += (unsigned long long)gs * 49152;
        // compute_num_nnz<<<1,1>>>(C->d_ptr,compressed_bin->d_bins+compressed_bin->bin_offset[2],compressed_bin->bin_size[2]);
        printf("the gs*49152 is %d\n", gs * 49152);
#endif
    }
    if (compressed_bin->bin_size[1])
    {
        gs = (compressed_bin->bin_size[1] + 255) >> 8;
        k_formcval_shared_for_many_rows<2, 4, 256, 16><<<gs, bs, 49152, compressed_bin->streams[1]>>>(A->d_ptr, A->d_col, A->d_val, B->d_ptr, B->d_col, B->d_val,
                                                                                                      compressed_bin->d_bins + compressed_bin->bin_offset[1], compressed_bin->bin_size[1],
                                                                                                      C->d_ptr, C->d_col, C->d_val);
#if compute_share
        practical_need += (unsigned long long)gs * 49152;
        // compute_num_nnz<<<1,1>>>(C->d_ptr,compressed_bin->d_bins+compressed_bin->bin_offset[1],compressed_bin->bin_size[1]);
        printf("the gs*49152 is %d\n", gs * 49152);
#endif
    }
    if (compressed_bin->bin_size[0])
    {
        gs = (compressed_bin->bin_size[0] + 511) >> 9;
        k_formcval_shared_for_many_rows<1, 2, 512, 8><<<gs, bs, 49152, compressed_bin->streams[0]>>>(A->d_ptr, A->d_col, A->d_val, B->d_ptr, B->d_col, B->d_val,
                                                                                                     compressed_bin->d_bins, compressed_bin->bin_size[0],
                                                                                                     C->d_ptr, C->d_col, C->d_val);
#if compute_share
        practical_need += (unsigned long long)gs * 49152;
        // compute_num_nnz<<<1,1>>>(C->d_ptr,compressed_bin->d_bins,compressed_bin->bin_size[0]);
        printf("the gs*49152 is %d\n", gs * 49152);
#endif
    }

    if (compressed_bin->bin_size[NUM_BIN_FOR_Ccol - 1])
    {
        cudaFree(d_combined_pointers);
        cudaFree(d_longestRowALength);
        if (maps != nullptr)
        {
            cudaFree(maps);
        }
        if (rowOffsetMapIndices != nullptr)
        {
            cudaFree(rowOffsetMapIndices);
        }

#if checek_kernel
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf(" cudaFree(d_combined_pointers) is failed\n");
        }
        else
        {
            printf(" cudaFree(d_combined_pointers) is cudaSuccess\n");
        }
#endif
    }
    // compute use of shared memory
#if compute_share
    true_need = (unsigned long long)(C->nnz) * 12;
    printf(" the practical_need is % lld\n", practical_need);
    printf(" the true_need is % lld\n", true_need);
    double use_of_shared = (double)((double)true_need / (double)practical_need);
    printf(" the use_of_shared is % lf\n", use_of_shared);
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

    FILE *fout_mem = fopen("~/NHC_SPGEMM/data/NHC_new_repres_use_of_share.csv", "a");
    if (fout_mem == NULL)
        printf("Writing results fails.\n");
    fprintf(fout_mem, "%s,%i,%f\n",
            filename, A->M, use_of_shared);
    fclose(fout_mem);
#endif
}
