#include "cuda_runtime.h"
#include "Executor.h"
#include "Multiply.h"
#include "DataLoader.h"
#include <iomanip>
#include "Config.h"
#include "Compare.h"
#include <cuSparseMultiply.h>
#include "Timings.h"
#include "spECKConfig.h"
#include "common.h"
#include "cuda_common.h"
#define compute_total_time 0
template <typename DataType>
long compt_flop(const CSR<DataType> &A, const CSR<DataType> &B){
	int M = A.rows;
	long total_flop = 0;
	for(int i = 0; i < M; i++){
	    for(int j = A.row_offsets[i]; j < A.row_offsets[i+1]; j++){
	    	total_flop += B.row_offsets[A.col_ids[j]+1] - B.row_offsets[A.col_ids[j]];
	    }
	}
	return total_flop;
}


template <typename ValueType>
int Executor<ValueType>::run()
{
	iterationsWarmup = Config::getInt(Config::IterationsWarmUp, 1);
	iterationsExecution = Config::getInt(Config::IterationsExecution, 10);
	//iterationsWarmup = 1;
	//iterationsExecution = 1;
	DataLoader<ValueType> data(runConfig.filePath, runConfig.filePath2);
	//std::cout << runConfig.filePath << std::endl;
	auto& matrices = data.matrices;
	//std::cout << "Matrix: " << matrices.cpuA.rows << "x" << matrices.cpuA.cols << ": " << matrices.cpuA.nnz << " nonzeros\n";

	long total_flops = compt_flop(matrices.cpuA, matrices.cpuB);

	dCSR<ValueType> dCsrHiRes, dCsrReference;
	Timings timings, warmupTimings, benchTimings;
	//bool measureAll = Config::getBool(Config::TrackIndividualTimes, false);
	bool measureAll = false;
	bool measureCompleteTimes = Config::getBool(Config::TrackCompleteTimes, true);
	auto config = spECK::spECKConfig::initialize(0);

	//bool compareData = false;
	bool compareData = true;

	if(Config::getBool(Config::CompareResult))
	{
		unsigned cuSubdiv_nnz = 0;
		cuSPARSE::CuSparseTest<ValueType> cusparse;
		cusparse.Multiply(matrices.gpuA, matrices.gpuB, dCsrReference, cuSubdiv_nnz);

		if(!compareData)
		{
			cudaFree(dCsrReference.data);
			dCsrReference.data = nullptr;
		}
	}

	// Warmup iterations for multiplication
	for (int i = 0; i < iterationsWarmup; ++i)
	{
		timings = Timings();
		timings.measureAll = measureAll;
		timings.measureCompleteTime = measureCompleteTimes;
		spECK::MultiplyspECK<ValueType, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>(matrices.gpuA, matrices.gpuB, dCsrHiRes, config, timings);
		warmupTimings += timings;

		if (dCsrHiRes.data != nullptr && dCsrHiRes.col_ids != nullptr && Config::getBool(Config::CompareResult))
		{
			printf("compare data \n");
			//if (!spECK::Compare(dCsrReference, dCsrHiRes, false))
			if (!spECK::Compare(dCsrReference, dCsrHiRes, compareData))
				printf("Error: Matrix incorrect\n");
		}
		dCsrHiRes.reset();
	}

	// Multiplication
	for (int i = 0; i < iterationsExecution; ++i)
	{
		timings = Timings();
		timings.measureAll = measureAll;
		timings.measureCompleteTime = measureCompleteTimes;
		spECK::MultiplyspECK<ValueType, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>
		(matrices.gpuA, matrices.gpuB, dCsrHiRes, config, timings);
		benchTimings += timings;

//		if (dCsrHiRes.data != nullptr && dCsrHiRes.col_ids != nullptr && Config::getBool(Config::CompareResult))
//		{
//			if (!spECK::Compare(dCsrReference, dCsrHiRes, compareData))
//				printf("Error: Matrix incorrect\n");
//		}
		dCsrHiRes.reset();
	}
	
	benchTimings /= iterationsExecution;
	benchTimings.reg_print(total_flops * 2);

	

	return 0;
}

template <typename ValueType>
int Executor<ValueType>::run_detail()
{
	// iterationsWarmup = Config::getInt(Config::IterationsWarmUp, 1);
	// iterationsExecution = Config::getInt(Config::IterationsExecution, 10);
	iterationsWarmup = 1;
	iterationsExecution = 1;
	DataLoader<ValueType> data(runConfig.filePath, runConfig.filePath2);
	// std::cout << runConfig.filePath << std::endl;//runConfig.filePath这个就是矩阵的名称
	auto& matrices = data.matrices;
	// std::cout << "Matrix: " << matrices.cpuA.rows << "x" << matrices.cpuA.cols << ": " << matrices.cpuA.nnz << " nonzeros\n";

	long total_flops = compt_flop(matrices.cpuA, matrices.cpuB);

	dCSR<ValueType> dCsrHiRes, dCsrReference;
	Timings timings, warmupTimings, benchTimings;
	bool measureAll = true;
	bool measureCompleteTimes = Config::getBool(Config::TrackCompleteTimes, true);
	auto config = spECK::spECKConfig::initialize(0);

	//bool compareData = false;
	bool compareData = true;

	if(Config::getBool(Config::CompareResult))
	{
		unsigned cuSubdiv_nnz = 0;
		cuSPARSE::CuSparseTest<ValueType> cusparse;
		cusparse.Multiply(matrices.gpuA, matrices.gpuB, dCsrReference, cuSubdiv_nnz);

		if(!compareData)
		{
			cudaFree(dCsrReference.data);
			dCsrReference.data = nullptr;
		}
	}

	// Warmup iterations for multiplication
	for (int i = 0; i < iterationsWarmup; ++i)
	{
		timings = Timings();
		timings.measureAll = measureAll;
		timings.measureCompleteTime = measureCompleteTimes;
		spECK::MultiplyspECK<ValueType, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>(matrices.gpuA, matrices.gpuB, dCsrHiRes, config, timings);
		warmupTimings += timings;

		if (dCsrHiRes.data != nullptr && dCsrHiRes.col_ids != nullptr && Config::getBool(Config::CompareResult))
		{
			printf("compare data \n");
			//if (!spECK::Compare(dCsrReference, dCsrHiRes, false))
			if (!spECK::Compare(dCsrReference, dCsrHiRes, compareData))
				printf("Error: Matrix incorrect\n");
		}
		dCsrHiRes.reset();
	}

	// Multiplication
	for (int i = 0; i < iterationsExecution; ++i)
	{
		timings = Timings();
		timings.measureAll = measureAll;
		timings.measureCompleteTime = measureCompleteTimes;
		spECK::MultiplyspECK<ValueType, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>
		(matrices.gpuA, matrices.gpuB, dCsrHiRes, config, timings);
		benchTimings += timings;

//		if (dCsrHiRes.data != nullptr && dCsrHiRes.col_ids != nullptr && Config::getBool(Config::CompareResult))
//		{
//			if (!spECK::Compare(dCsrReference, dCsrHiRes, compareData))
//				printf("Error: Matrix incorrect\n");
//		}
		dCsrHiRes.reset();
	}
	
	benchTimings /= iterationsExecution;
	benchTimings.print(total_flops * 2);
	string matrix_name = runConfig.filePath;
	char cstr_matrix_name[100];
	strcpy(cstr_matrix_name, matrix_name.c_str());
	printf("the cstr_matrix_name is %s",cstr_matrix_name);

	char filename[100];
    char* lastSlash = strrchr(cstr_matrix_name, '/');
    char* lastDot = strrchr(cstr_matrix_name, '.');

    if (lastSlash != NULL && lastDot != NULL && lastDot > lastSlash) {
        size_t length = lastDot - (lastSlash + 1);
        strncpy(filename, lastSlash + 1, length);
        filename[length] = '\0';
    } else {
        strcpy(filename, runConfig.filePath.c_str());
    }

#if compute_total_time
	//write
	// printf("%s \n",runConfig.filePath.c_str());
	// printf("total time is  %lf\n",benchTimings.total);
    // FILE *fout_mem = fopen("/home/wm/open_source_code/OpSparse-main/spECK/data/spECK_high_flops.csv", "a");
    // FILE *fout_mem = fopen("/home/wm/open_source_code/OpSparse-main/spECK/data/spECK_small_result.csv", "a");
    // FILE *fout_mem = fopen("/home/wm/open_source_code/OpSparse-main/spECK/data/spECK_common_result.csv", "a");

    // FILE *fout_mem = fopen("/home/wm/open_source_code/spGEMM/OpSparse-main/spECK/data/spECK_new_repres_total_time.csv", "a");
    FILE *fout_mem = fopen("/home/wm/open_source_code/spGEMM/OpSparse-main/spECK/data/spECK_compress_total_time.csv", "a");
	 float total_flop_d = float(total_flops * 2)/1000000;
	float total_flop_G = total_flop_d/benchTimings.total;
	float Gflops_of_numeric = total_flop_d/(benchTimings.numeric+benchTimings.cleanup);
	float symbolic_time = benchTimings.total - benchTimings.numeric - benchTimings.cleanup;
	float numeric_time = benchTimings.numeric + benchTimings.cleanup;
	printf("the Gflops_of_numeric is %f\n",Gflops_of_numeric);
    if (fout_mem == NULL)
        printf("Writing results fails.\n");
    fprintf(fout_mem, "%s,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",
            //  runConfig.filePath.c_str(),
             filename,
	benchTimings.setup,
    benchTimings.symbolic_binning,
    benchTimings.symbolic,
    benchTimings.numeric_binning,
    benchTimings.allocate,
    benchTimings.prefix,
    benchTimings.numeric,
    benchTimings.cleanup,
    benchTimings.total,
    total_flop_G,
	symbolic_time,
	numeric_time,
	Gflops_of_numeric);
#endif
	return 0;
}

template class Executor<double>;
