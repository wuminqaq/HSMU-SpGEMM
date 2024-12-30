# HSMU-SpGEMM
A High Shared Memory Utilization Algorithm for Parallel Sparse General Matrix-Matrix Multiplication on GPUs
## Introduction
Sparse general matrix-matrix multiplication (SpGEMM) is fundamental to numerous scientific applications. Traditional hash-based approaches fail to strike a trade-off between reducing hash collisions and efficiently utilizing fast shared memory. This significantly hinders the performance improvement of SpGEMM on GPUs. To tackle this issue, this paper proposes a parallel algorithm with high shared memory utilization, called HSMU-SpGEMM, for SpGEMM on modern GPUs. Our approach introduces a novel accumulator design to remedy the underuse of shared memory of traditional hash-based accumulators. Moreover, we devise distinct symbolic stages for the proposed accumulator.
Note that we named our method 'NHC' earlier. All 'NHC' in the code refers to HSMU.

## Artifact Setup (incl. Inputs)
Hardware: To better reproduce experiment results, we sug- gest an NVIDIA GPU with compute capability 8.6. To store input dataset, ensure that there is at least 200GB of free disk space.
Software: The artifact has been tested on Ubuntu 22.04.4, GCC 9.5.0, CUDA 11.4; For script execution, the following python libraries are required: Pandas,Matplotlib,Numpy.
Datasets / Inputs: HSMU-SpGEMM currently supports input matrix files with the .mtx extension. All of the matrices we tested can be downloaded from the SuiteSparse Matrix Collection (https://sparse.tamu.edu/). The 18 representative matrix sets tested in the paper are placed
in the folder <HSMU_dir>/evaluation/18matries. The 338 matrices tested in the paper can be downloaded by executing the command: $python3 <HSMU_dir>/evaluation/338MatrixSet/matrix download.py (requires 9 hours or more).
Deployment:
Compile with the commands $cd evaluation/script followed by $make to generate an executable file named test.
Then, run HSMU-SpGEMM with the following command: $ ./test <path/to/matrix set>

## Artifact Execution
In the directory <HSMU_dir>/evaluation/script, there are scripts to reproduce the paper data.
Run the command:
$bash simple run test.sh
This script uses the matrix ”cant” as a test case to prove that the artifact is available and correct.
Run the command:
$bash test threshold matrix.sh
This script executes the experiment corresponding to Figure 6 and generates Figure 6. (180 minutes or more)
Run the command:
$bash test four extremely large matrices.sh
This script performs the experiment corresponding to Figure 7 and generates the data corresponding to HSMU-SpGEMM. 
Run the command:
$bash test338matrices.sh
This script tests the 338 matrices tested in the paper and generates the data of HSMU-SpGEMM in Tables 3, 4, and Figure 9. (20 minutes or more)
Run the command:
$bash test18matrices.sh
This script tests the experiment corresponding to Figure 10 and obtains the corresponding results of HSMU-SpGEMM. (2 minutes)
Run the command:
$bash test peak memory.sh
This script tests the peak memory of 18 representative matrices of HSMU-SpGEMM, corresponding to Figure 11. Run the command:
$bash testAATmatrices.sh
This script reproduces the data for HSMU-SpGEMM in Figure 12.
Run the command:
$bash additional evaluation.sh
This script tests the time proportion of each stage of HSMU-SpGEMM, as well as the time and space overhead of the new format, and finally generates Figures 13, 14, and 15.

## Artifact Analysis (incl. Outputs)
For each matrix tested, the output information is as follows: lines 1-3 provide basic information about the tested matrix, including its name, number of rows, number of columns, and the number of non-zero elements.
line 4 outputs the loading time of the matrix file (in seconds). line 5 outputs the format conversion time (in milliseconds) (corresponds to Figure 15).
line 6 outputs the space overhead of the new format (corresponds to Figure 14).
line 7-11 present the time taken by each stage of HSMU- SpGEMM (in milliseconds) (corresponds to Figure 13).
line 12-14 provide information about matrix C, including the number of intermediate products, the number of non-zero elements in C, and the compression ratio.
line 15-16 output the running time and GFLOPS.