#include "struct.h"
#define BLOCK_SIZE 512U
#define CONFLICT_FREE_OFFSET(n) (0)
__device__ __host__ __forceinline__ uint32_t toBlockRange(uint32_t startRow, uint32_t numRows)
{
	return (startRow << 5) + (numRows - 1);
}

__device__ __forceinline__ uint32_t rowColMinMaxtoMinCol(uint32_t rowColMinMax)
{
	return rowColMinMax & ((1 << 27) - 1);
}
__device__ __forceinline__ uint32_t rowColMinMaxtoRowLength(uint32_t rowColMinMax)
{
	return 1 << (rowColMinMax >> 27);
}

__device__ __forceinline__ uint32_t toRowColMinMax(uint32_t minCol, uint32_t maxCol)
{
	uint32_t width = 32U - __clz(maxCol - minCol);
	return minCol + (width << 27);
}

__device__ __forceinline__ int getThreadShiftNew(uint32_t sumOps, INDEX_TYPE maxOpsPerCol, int minShift, int maxShift, INDEX_TYPE cols) // cols为A中非零元的数量
{
	const INDEX_TYPE maxThreads = 1 << maxShift;

	INDEX_TYPE opsPerNnz = max(1, (sumOps - maxOpsPerCol) / max(1, cols - 1));

	if (opsPerNnz > 64)
		minShift = max(5, minShift);

	INDEX_TYPE shift = max(minShift, min(maxShift, 31 - __clz(opsPerNnz)));
	if ((1 << shift) * 3 < opsPerNnz * 2 && shift < maxShift)
		++shift;

	INDEX_TYPE colIters = div_up(cols, maxThreads / (1U << shift));
	INDEX_TYPE maxIters = div_up(maxOpsPerCol, (1U << shift));

	if (maxIters > colIters * 2)
		shift += min(maxShift - shift, max(1, 31 - __clz(maxIters / colIters / 2)));

	colIters = div_up(cols, maxThreads / (1U << shift));
	maxIters = div_up(maxOpsPerCol, (1U << shift));

	if ((1 << shift) * 2 > opsPerNnz && colIters > maxIters * 2)
		shift -= min(shift / 2, max(1, 31 - __clz(colIters / maxIters)));

	shift = max(minShift, shift);

	INDEX_TYPE concurrentOps = cols << shift;

	if (concurrentOps < maxThreads)
		shift += 31 - __clz(maxThreads / concurrentOps);

	shift = max(minShift, min(maxShift, shift));
	return shift;
}

template <class HashMap>
__device__ __forceinline__ HashMap *reserveMap(HashMap *maps, int count)
{
	int index = blockIdx.x % count;

	while (true)
	{
		if (atomicCAS(&maps[index].reserved, 0, 1) == 0)
		{
			return &maps[index];
		}
		index = (index + 1) % count;
	}
}

template <class HashMap>
__device__ __forceinline__ void freeMap(HashMap *map)
{
	if (map == nullptr)
		return;
	map->reserved = 0;
	map = nullptr;
}

template <typename T, class CombineOp, class InputIterator, class OutputIterator, int KERNEL_COUNT>
__global__ void prescanArrayKernelNew(InputIterator in, OutputIterator out, int numElements,
									  CombineOp combine, int actualKernelCount, int *offsetCounters)
{

	__shared__ T temp[BLOCK_SIZE * 2 + BLOCK_SIZE / 8];
	__shared__ char prefix[BLOCK_SIZE * 2];
	int tid = threadIdx.x;
	int start = (BLOCK_SIZE * 2) * blockIdx.x;
	int aj, bj;
	aj = tid;
	bj = tid + BLOCK_SIZE;
	int bankOffsetA = CONFLICT_FREE_OFFSET(aj);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bj);

	for (int i = threadIdx.x; i < BLOCK_SIZE * 2; i += BLOCK_SIZE)
	{
		prefix[i] = 0;
	}

	if (numElements > start + aj)
		temp[aj + bankOffsetA] = in[start + aj];
	else
		temp[aj + bankOffsetA] = T();
	if (numElements > start + bj)
		temp[bj + bankOffsetB] = in[start + bj];
	else
		temp[bj + bankOffsetB] = T();

	int offset = 1;

#pragma unroll
	for (int d = BLOCK_SIZE; d > 0; d >>= 1)
	{
		__syncthreads();
		if (tid < d)
		{
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			auto tempA = temp[ai];
			auto tempB = combine(tempA, temp[bi]);

			if (tempB.first != tempA.first && tempA.first < numElements)
			{
				prefix[ai] = tempA.kernelScale + 1;
			}
			else
			{
				temp[bi] = tempB;
			}
		}
		offset *= 2;
	}
	__syncthreads();

	int lastId = BLOCK_SIZE * 2 - 1;
	if (threadIdx.x == BLOCK_SIZE - 1 && temp[lastId].first < numElements)
	{
		prefix[lastId] = temp[lastId].kernelScale + 1;
	}

	typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
	__shared__ typename BlockScan::TempStorage tempStorage;
	__shared__ int blockOffset;

	int threadIn[2];
	int threadOut[2];

	for (int kernelScale = 0; kernelScale < actualKernelCount; kernelScale++)
	{
		int kernelIndex = actualKernelCount - 1 - kernelScale;

		for (int i = 0; i < 2; i++)
		{
			if (prefix[tid * 2 + i] == kernelScale + 1)
				threadIn[i] = 1;
			else
				threadIn[i] = 0;
		}
		BlockScan(tempStorage).ExclusiveSum(threadIn, threadOut);

		if (threadIdx.x == BLOCK_SIZE - 1)
			blockOffset = atomicAdd(offsetCounters + (KERNEL_COUNT - 1 - kernelScale), threadOut[1] + threadIn[1]) + numElements * kernelIndex;
		__syncthreads();

		for (int i = 0; i < 2; i++)
		{
			if (threadIn[i] != 0)
			{
				int index = blockOffset;
				index += threadOut[i];
				auto tmpElement = temp[tid * 2 + i];

				out[index] = toBlockRange(tmpElement.first, tmpElement.numRows);
			}
		}
	}
}

template <typename T, int KERNEL_COUNT, class InputIterator, class OutputIterator, class CombineOp>
void prescanArrayOrdered(InputIterator &inputIterator, OutputIterator &outputIterator,
						 CombineOp &combine, int numElements, int actualKernelCount, int *offsetCounters)
{
	dim3 dim_block, dim_grid;
	dim_block.x = BLOCK_SIZE;
	dim_block.y = dim_block.z = 1;

	dim_grid.x = ceil((float)(numElements / (float)(dim_block.x * 2)));
	dim_grid.y = dim_grid.z = 1;

	prescanArrayKernelNew<T, CombineOp, InputIterator, OutputIterator, KERNEL_COUNT><<<dim_grid, dim_block>>>(
		inputIterator, outputIterator, numElements, combine, actualKernelCount, offsetCounters);
}

template <int THREADS>
__global__ void readOperations(
	int *d_aptr, int *d_acol, int nums_Arow, int nnz_A, int *d_bptr, int *d_bcol, int nums_Brow, int nnz_B,
	uint32_t *out, int rowsPerBlock,
	uint32_t *maxComputationsPerRow, uint32_t *rowColMinMax, uint32_t *rowOperationsMax, uint32_t *sumProducts)
{
	INDEX_TYPE startRow = blockIdx.x * rowsPerBlock;
	INDEX_TYPE lastRowExcl = min(INDEX_TYPE((blockIdx.x + 1) * rowsPerBlock), INDEX_TYPE(nums_Arow));
	bool checkCols = rowColMinMax != nullptr;
	bool checkRowOpsMax = rowOperationsMax != nullptr;

	if (startRow >= nums_Arow)
		return;

	__shared__ INDEX_TYPE rowOpsCounter[THREADS];
	__shared__ INDEX_TYPE rowOffsets[THREADS];
	__shared__ INDEX_TYPE rowMaxOps[THREADS];
	__shared__ INDEX_TYPE rowMinCols[THREADS];
	__shared__ INDEX_TYPE rowMaxCols[THREADS];
	__shared__ INDEX_TYPE blockProducts;
	__shared__ INDEX_TYPE blockMaxOps;

	rowOpsCounter[threadIdx.x] = 0U;
	rowMaxOps[threadIdx.x] = 0U;
	rowMinCols[threadIdx.x] = INT_MAX;
	rowMaxCols[threadIdx.x] = 0U;
	rowOffsets[threadIdx.x] = (startRow + threadIdx.x <= lastRowExcl) ? d_aptr[startRow + threadIdx.x] : nnz_A;
	if (threadIdx.x == 0)
	{
		blockProducts = 0;
		blockMaxOps = 0;
	}

	__syncthreads();

	uint32_t startId = rowOffsets[0];
	uint32_t lastIdExcl = lastRowExcl < nums_Arow ? rowOffsets[rowsPerBlock] : nnz_A;

	uint32_t currentRow = INT_MAX;
	uint32_t currentRowOps = 0;
	uint32_t currentMin = INT_MAX;
	uint32_t currentMax = 0;
	uint32_t currentRowMaxOps = 0;
	for (uint32_t id = threadIdx.x + startId; id < lastIdExcl; id += blockDim.x)
	{
		INDEX_TYPE rowA = 0;
		for (; rowA < rowsPerBlock; ++rowA)
		{
			if (rowOffsets[rowA] <= id && (rowA + startRow + 1 < nums_Arow ? rowOffsets[rowA + 1] : nnz_A) > id)
				break;
		}
		if (currentRow != rowA)
		{
			if (currentRow != INT_MAX)
			{
				if (checkCols)
				{
					atomicMin(&rowMinCols[currentRow], currentMin);
					atomicMax(&rowMaxCols[currentRow], currentMax);
				}
				if (checkRowOpsMax)
					atomicMax(&rowMaxOps[currentRow], currentRowMaxOps);
				atomicAdd(&rowOpsCounter[currentRow], currentRowOps);
			}
			currentMin = INT_MAX;
			currentMax = 0;
			currentRowMaxOps = 0;
			currentRow = rowA;
			currentRowOps = 0;
		}

		INDEX_TYPE rowB = d_acol[id];
		INDEX_TYPE startIdB = d_bptr[rowB];
		INDEX_TYPE lastIdBExcl = rowB + 1 <= nums_Brow ? d_bptr[rowB + 1] : nnz_B;
		INDEX_TYPE operations = lastIdBExcl - startIdB;

		if (checkCols && startIdB < lastIdBExcl)
		{
			currentMin = min(currentMin, d_bcol[startIdB]);
			if (lastIdBExcl > 0)
				currentMax = max(currentMax, d_bcol[lastIdBExcl - 1]);
		}
		currentRowOps += operations;
		if (checkRowOpsMax)
			currentRowMaxOps = max(currentRowMaxOps, operations);
	}

	if (currentRow != INT_MAX)
	{
		if (checkCols)
		{
			atomicMin(&rowMinCols[currentRow], currentMin);
			atomicMax(&rowMaxCols[currentRow], currentMax);
		}
		if (checkRowOpsMax)
			atomicMax(&rowMaxOps[currentRow], currentRowMaxOps);
		atomicAdd(&rowOpsCounter[currentRow], currentRowOps);
	}
	__syncthreads();
	if (rowsPerBlock > 1)
	{
		INDEX_TYPE rowProducts = rowOpsCounter[threadIdx.x];
		for (int i = 16; i > 0; i /= 2)
			rowProducts += __shfl_down_sync(0xFFFFFFFF, rowProducts, i);

		if (threadIdx.x % 32 == 0 && rowProducts > 0)
			atomicAdd(&blockProducts, rowProducts);

		INDEX_TYPE maxRowLength = rowOpsCounter[threadIdx.x];
		for (int i = 16; i > 0; i /= 2)
			maxRowLength = max(maxRowLength, __shfl_down_sync(0xFFFFFFFF, maxRowLength, i));

		if (threadIdx.x % 32 == 0 && maxRowLength > 0)
			atomicMax(&blockMaxOps, maxRowLength);

		__syncthreads();
	}

	if (threadIdx.x < rowsPerBlock && (threadIdx.x + startRow) < nums_Arow)
	{
		out[startRow + threadIdx.x] = rowOpsCounter[threadIdx.x];
		if (checkCols)
			rowColMinMax[startRow + threadIdx.x] = toRowColMinMax(rowMinCols[threadIdx.x], rowMaxCols[threadIdx.x]);
		if (checkRowOpsMax)
			rowOperationsMax[startRow + threadIdx.x] = rowMaxOps[threadIdx.x];
	}
	if (threadIdx.x == blockDim.x - 1)
	{
		if (rowsPerBlock == 1)
		{
			atomicMax(maxComputationsPerRow, rowOpsCounter[0]);
			atomicAdd(sumProducts, rowOpsCounter[0]);
		}
		else
		{
			atomicMax(maxComputationsPerRow, blockMaxOps);
			atomicAdd(sumProducts, blockProducts);
		}
	}
}
template <typename INDEX_type, uint32_t THREADS, uint32_t rowsPerThreads>
__global__ void getLongestRowA(const INDEX_type *__restrict__ rowOffsets, INDEX_type *__restrict__ longestRow, const INDEX_type rows)
{
	typedef cub::BlockReduce<INDEX_type, THREADS> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	INDEX_type rowLength[rowsPerThreads];
	for (int i = 0; i < rowsPerThreads; ++i)
		rowLength[i] = 0;

	INDEX_type startRow = blockIdx.x * THREADS * rowsPerThreads + threadIdx.x * rowsPerThreads;
	INDEX_type lastRowExcl = min(rows, blockIdx.x * THREADS * rowsPerThreads + (threadIdx.x + 1) * rowsPerThreads);
	if (lastRowExcl > startRow)
	{
		INDEX_type prevOffset = rowOffsets[startRow];
		for (int i = 1; i <= lastRowExcl - startRow; ++i)
		{
			INDEX_type currentRowOffset = rowOffsets[i + startRow];
			rowLength[i - 1] = currentRowOffset - prevOffset;
			prevOffset = currentRowOffset;
		}
	}

	INDEX_type longestRowBlock = BlockReduce(temp_storage).Reduce(rowLength, cub::Max());

	if (threadIdx.x == 0)
		atomicMax(longestRow, longestRowBlock);
}

template <typename HashMap, typename INDEX_type>
__global__ void initializeGlobalMapsNoVal(HashMap *maps, int count, INDEX_type *ids, INDEX_type elementsPerMap, int maxRowsPerBlock)
{
	if (threadIdx.x == 0)
	{
		maps[blockIdx.x].ids = ids + ((elementsPerMap + maxRowsPerBlock + 1) * blockIdx.x);
		maps[blockIdx.x].occupancyPerRow = &maps[blockIdx.x].ids[elementsPerMap];
		maps[blockIdx.x].occupancy = &maps[blockIdx.x].occupancyPerRow[maxRowsPerBlock];
		maps[blockIdx.x].capacity = elementsPerMap;
		maps[blockIdx.x].reserved = 0;
	}
	__syncthreads();
	maps[blockIdx.x].init(threadIdx.x == 0);
}
