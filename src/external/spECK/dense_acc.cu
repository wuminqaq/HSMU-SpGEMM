#include "commom.cu"
template <uint32_t MAX_ELEMENTS_BLOCK, bool SUPPORT_GLOBAL, uint32_t THREADS, uint32_t SHIFT, bool useRowOffsets>
__device__ __forceinline__ void iterateMatrixDenseNumeric(
	uint32_t startId, uint32_t lastIdExcl, uint32_t elementsPerBlock, INDEX_TYPE *rowCursor,
	INDEX_TYPE rowsB, INDEX_TYPE nnzB, const INDEX_TYPE startRow,
	uint32_t resultStartId, const INDEX_TYPE *__restrict__ colIdsA, const INDEX_TYPE *__restrict__ colIdsB,
	const INDEX_TYPE *__restrict__ rowOffsetsB,
	const VALUE_TYPE *__restrict__ valuesA, const VALUE_TYPE *__restrict__ valuesB,
	INDEX_TYPE *__restrict__ colIdsC, VALUE_TYPE *__restrict__ valuesC, const uint32_t minCol, const uint32_t maxCol,
	INDEX_TYPE *prefix, INDEX_TYPE *prefix2,
	INDEX_TYPE *currentRowMinOffset, VALUE_TYPE *values,
	void *temp_storage)
{
	typedef cub::BlockScan<INDEX_TYPE, THREADS> BlockScan;
	uint32_t colOffset = minCol;
	uint32_t accumulatedPrefix = 0;
	while (colOffset < maxCol)
	{
		for (int i = threadIdx.x; i < elementsPerBlock; i += THREADS)
		{
			values[i] = 0.0f;
		}
		for (int i = threadIdx.x; i < elementsPerBlock / 32 + 1; i += THREADS)
		{
			prefix[i] = 0;
		}
		__syncthreads();
		if (useRowOffsets)
		{
			for (INDEX_TYPE _idA = startId; _idA < lastIdExcl; _idA += THREADS >> SHIFT)
			{
				const uint32_t idA = _idA + (threadIdx.x >> SHIFT);
				const bool valid = idA < lastIdExcl;
				const uint32_t colA = valid ? colIdsA[idA] : 0;
				const uint32_t rowStart = valid ? rowOffsetsB[colA] : 0; //
				const uint32_t startIdB = valid ? rowStart + (useRowOffsets ? rowCursor[idA - startId] : 0) : INT_MAX;
				const uint32_t lastIdBExcl = valid ? (min(startIdB + elementsPerBlock, colA + 1 < rowsB ? rowOffsetsB[colA + 1] : nnzB)) : 0;
				const VALUE_TYPE valueA = valid ? valuesA[idA] : 0;
				if (threadIdx.x % (1 << SHIFT) == 0 && valid)
				{
					currentRowMinOffset[threadIdx.x >> SHIFT] = lastIdBExcl - rowStart;
				}
				__syncthreads();
				for (uint32_t idB = startIdB + (threadIdx.x % (1 << SHIFT)); idB < lastIdBExcl; idB += (1 << SHIFT))
				{
					uint32_t colB = colIdsB[idB] - colOffset;
					if (colB < elementsPerBlock)
					{
						atomicAdd(&values[colB], valueA * valuesB[idB]);
						atomicOr(&prefix[colB / 32], 1 << (colB % 32));
					}
					else
					{
						atomicMin(&currentRowMinOffset[threadIdx.x >> SHIFT], idB - rowStart);
						break;
					}
				}
				__syncthreads();
				if (threadIdx.x % (1 << SHIFT) == 0 && valid)
				{
					rowCursor[idA - startId] = currentRowMinOffset[threadIdx.x >> SHIFT];
				}
			}
		}
		else
		{
			for (INDEX_TYPE idA = startId + (threadIdx.x >> SHIFT); idA < lastIdExcl; idA += THREADS >> SHIFT)
			{
				const INDEX_TYPE colA = colIdsA[idA];
				const INDEX_TYPE rowStart = rowOffsetsB[colA];
				const INDEX_TYPE startIdB = rowStart;
				const INDEX_TYPE lastIdBExcl = (min(startIdB + elementsPerBlock, colA + 1 < rowsB ? rowOffsetsB[colA + 1] : nnzB));
				const VALUE_TYPE valueA = valuesA[idA];
				for (INDEX_TYPE idB = startIdB + (threadIdx.x % (1 << SHIFT)); idB < lastIdBExcl; idB += 1 << SHIFT)
				{
					auto colB = colIdsB[idB] - colOffset;
					if (colB < elementsPerBlock)
					{
						atomicAdd(&values[colB], valueA * valuesB[idB]);
						atomicOr(&prefix[colB / 32], 1 << (colB % 32));
					}
				}
			}
		}
		__syncthreads();
		const uint32_t localElements = (MAX_ELEMENTS_BLOCK + THREADS - 1) / THREADS; // ceil -> (a + b - 1) / b;
		INDEX_TYPE thread_data[localElements];
		for (int i = 0; i < localElements; i++)
		{
			uint32_t id = threadIdx.x * localElements + i;
			thread_data[i] = id < elementsPerBlock / 32 + 1 ? __popc(prefix[id]) : 0U;
		}
		BlockScan(*((typename BlockScan::TempStorage *)temp_storage)).InclusiveSum(thread_data, thread_data);
		for (int i = 0; i < localElements; i++)
		{
			uint32_t id = threadIdx.x * localElements + i;
			if (id < elementsPerBlock / 32 + 1)
				prefix2[id] = thread_data[i];
		}
		__syncthreads();
		for (int i = threadIdx.x; i < elementsPerBlock; i += THREADS)
		{
			INDEX_TYPE warpPrefix = i / 32 == 0 ? 0 : prefix2[i / 32 - 1];
			bool isNonZero = prefix[i / 32] & (1 << (i % 32));
			warpPrefix += __popc(prefix[i / 32] & ((1 << (i % 32)) - 1));
			if (isNonZero)
			{
				uint32_t id = warpPrefix + accumulatedPrefix + resultStartId;
				colIdsC[id] = (INDEX_TYPE)(colOffset + i);
				valuesC[id] = values[i];
			}
		}
		colOffset += elementsPerBlock;
		if (colOffset >= maxCol)
			return;

		accumulatedPrefix += prefix2[elementsPerBlock / 32];
		__syncthreads();
	}
}

template <class GlobalRowOffsetsMap, uint32_t SHARED_MEM_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
__device__ __forceinline__ void denseSpGEMMNumericImplementation(
	const INDEX_TYPE nnzB, const INDEX_TYPE rowsB,
	const INDEX_TYPE *__restrict__ rowOffsetsB, const INDEX_TYPE *__restrict__ colIdsA, const INDEX_TYPE *__restrict__ colIdsB,
	const VALUE_TYPE *__restrict__ valuesA, const VALUE_TYPE *__restrict__ valuesB,
	GlobalRowOffsetsMap *__restrict__ maps, INDEX_TYPE mapCount,
	INDEX_TYPE *__restrict__ colIdsC, VALUE_TYPE *__restrict__ valuesC,
	const uint32_t *__restrict__ rowOperations,
	uint32_t *rowMaxOperations, uint32_t startRow,
	uint32_t resultStartId, const uint32_t minCol, const uint32_t maxCol, const uint32_t startId, const uint32_t lastIdExcl)
{
	extern __shared__ int dynamicShared[];
	typedef cub::BlockScan<INDEX_TYPE, THREADS> BlockScan;

	struct SMEM
	{
		// this map contains the current cursor in all rows of B which are referenced by this block. only used if we need multiple iterations
		GlobalRowOffsetsMap *globalOffsetsMap;
		INDEX_TYPE *rowOffsets;
	};
	__shared__ SMEM sMem;
	VALUE_TYPE *values;
	INDEX_TYPE *prefix;
	INDEX_TYPE *prefix2;
	INDEX_TYPE *currentRowMinOffset;
	INDEX_TYPE *rowOffsets;
	typename BlockScan::TempStorage *temp_storage;

	// 32/64 bit for float/double + 1 prefix + 1 prefix2
	const uint32_t bitsPerElement = sizeof(VALUE_TYPE) * 8 + 2;

	// 32 indexType for currentRowMinOffset
	const INDEX_TYPE freeBytesBlock = (SHARED_MEM_SIZE - sizeof(BlockScan::TempStorage) - 32 * sizeof(INDEX_TYPE)) - 64; // just leave some space free
	const INDEX_TYPE maxElementsBlock = freeBytesBlock * 8 / bitsPerElement;
	INDEX_TYPE elementsPerBlock = maxElementsBlock;
	bool useGlobalOffsetsMap = (maxCol - minCol) < elementsPerBlock ? false : (lastIdExcl - startId) * sizeof(INDEX_TYPE) > freeBytesBlock * 3 / 4;
	bool useRowOffsets = true;
	if (useGlobalOffsetsMap)
	{
		return;
		if (threadIdx.x == 0)
		{
			sMem.globalOffsetsMap = reserveMap<GlobalRowOffsetsMap>(maps, mapCount);
			sMem.rowOffsets = sMem.globalOffsetsMap->ids;
		}
		__syncthreads();
		temp_storage = (typename BlockScan::TempStorage *)(void *)dynamicShared;
		values = (VALUE_TYPE *)&((char *)dynamicShared)[sizeof(BlockScan::TempStorage)];
		prefix = (INDEX_TYPE *)&values[elementsPerBlock];
		prefix2 = (INDEX_TYPE *)&prefix[elementsPerBlock / 32 + 1];
		currentRowMinOffset = (INDEX_TYPE *)&prefix2[elementsPerBlock / 32 + 1];
		__syncthreads();
		__threadfence();
		rowOffsets = sMem.rowOffsets;
		for (int i = threadIdx.x; i < sMem.globalOffsetsMap->getSize(); ++i)
			rowOffsets[i] = 0;
	}
	else
	{
		uint32_t offsetElements = (maxCol - minCol) < elementsPerBlock ? 0 : lastIdExcl - startId + 1;
		useRowOffsets = offsetElements > 0;
		elementsPerBlock -= offsetElements * sizeof(INDEX_TYPE) * 8 / bitsPerElement;
		if (threadIdx.x == 0)
		{
			sMem.globalOffsetsMap = nullptr;
		}

		temp_storage = (typename BlockScan::TempStorage *)(void *)dynamicShared;
		values = (value_t *)&((char *)dynamicShared)[sizeof(BlockScan::TempStorage)];
		prefix = (INDEX_TYPE *)&values[elementsPerBlock];
		prefix2 = (INDEX_TYPE *)&prefix[elementsPerBlock / 32 + 1];
		rowOffsets = (INDEX_TYPE *)&prefix2[elementsPerBlock / 32 + 1];
		currentRowMinOffset = (INDEX_TYPE *)&rowOffsets[offsetElements];
		for (int i = threadIdx.x; i < offsetElements; i += THREADS)
		{
			rowOffsets[i] = 0;
		}
		__syncthreads();
	}
	uint32_t shift = getThreadShiftNew(rowOperations[startRow], rowMaxOperations[startRow], 5U, 31U - __clz(THREADS), lastIdExcl - startId);
#define iterate(SHIFT, useRowOffsets) iterateMatrixDenseNumeric<maxElementsBlock, SUPPORT_GLOBAL, THREADS, SHIFT, useRowOffsets>(startId, lastIdExcl, elementsPerBlock, rowOffsets, \
																																 rowsB, nnzB, startRow,                             \
																																 resultStartId, colIdsA, colIdsB,                   \
																																 rowOffsetsB,                                       \
																																 valuesA, valuesB,                                  \
																																 colIdsC, valuesC, minCol, maxCol,                  \
																																 prefix, prefix2, currentRowMinOffset, values, temp_storage)
	switch (shift)
	{
	case 10:
		useRowOffsets ? iterate(10, true) : iterate(10, false);
		break;
	case 9:
		useRowOffsets ? iterate(9, true) : iterate(9, false);
		break;
	case 8:
		useRowOffsets ? iterate(8, true) : iterate(8, false);
		break;
	case 7:
		useRowOffsets ? iterate(7, true) : iterate(7, false);
		break;
	case 6:
		useRowOffsets ? iterate(6, true) : iterate(6, false);
		break;
	default:
		useRowOffsets ? iterate(5, true) : iterate(5, false);
		break;
	}

	if (sMem.globalOffsetsMap != nullptr)
	{
		for (int i = threadIdx.x; i < 2; ++i)
		{
			sMem.globalOffsetsMap->ids[i] = sMem.globalOffsetsMap->UNUSED();
		}

		if (threadIdx.x == 0)
		{
			freeMap(sMem.globalOffsetsMap);
		}
	}
}

template <class GlobalOffsetMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
__global__ void denseSpGEMMNumeric(
	const INDEX_TYPE nnzB, const INDEX_TYPE rowsB, const INDEX_TYPE colsB,
	const INDEX_TYPE *__restrict__ rowOffsetsA, const INDEX_TYPE *__restrict__ rowOffsetsB, const INDEX_TYPE *__restrict__ colIdsA, const INDEX_TYPE *__restrict__ colIdsB,
	const VALUE_TYPE *__restrict__ valuesA, const VALUE_TYPE *__restrict__ valuesB,
	GlobalOffsetMap *__restrict__ maps, INDEX_TYPE mapCount, INDEX_TYPE *__restrict__ colIdsC,
	VALUE_TYPE *__restrict__ valuesC, const INDEX_TYPE *__restrict__ rowOffsetsC,
	const uint32_t *__restrict__ rowOperations,
	const uint32_t *__restrict__ rowColMinMax,
	uint32_t *rowMaxOperations, const INDEX_TYPE *__restrict__ d_bin)
{
	INDEX_TYPE startRow = d_bin[blockIdx.x];
	const uint32_t startId = rowOffsetsA[startRow];
	const uint32_t lastIdExcl = rowOffsetsA[startRow + 1];
	const uint32_t resultStartId = rowOffsetsC[startRow];
	const uint32_t minCol = rowColMinMax != nullptr ? rowColMinMaxtoMinCol(rowColMinMax[startRow]) : 0;
	const uint32_t rowColSize = rowColMinMax != nullptr ? rowColMinMaxtoRowLength(rowColMinMax[startRow]) : INT_MAX;
	const uint32_t colRange = min(colsB - minCol, rowColSize);
	denseSpGEMMNumericImplementation<GlobalOffsetMap, SHARED_HASH_SIZE, SUPPORT_GLOBAL, THREADS>(
		nnzB, rowsB, rowOffsetsB, colIdsA, colIdsB, valuesA, valuesB,
		maps, mapCount, colIdsC, valuesC, rowOperations,
		rowMaxOperations, startRow, resultStartId, minCol, minCol + colRange, startId, lastIdExcl);
}
