#!/bin/bash
Critical_bin_id_list=(5 6 7 8 9 11 12 13 10)
Cnnz_ctile_rate_Threshold_list=(0)

if [ $? -eq 0 ];then
    filenames=$(cat matrix338_list.txt)
    Source="test_threshold_matrix.txt"
    Source1="test_threshold_python_result.txt"
    echo "" > $Source
    echo "" > $Source1
    for Critical_bin_id in "${Critical_bin_id_list[@]}"; do
        sed -i "s/#define Critical_bin_id .*/#define Critical_bin_id ${Critical_bin_id}/" NHC_spgemm.h
        for Cnnz_ctile_rate_Threshold in "${Cnnz_ctile_rate_Threshold_list[@]}"; do
            echo "Critical_bin_id is $Critical_bin_id, Cnnz_ctile_rate_Threshold is $Cnnz_ctile_rate_Threshold ----------------------" >> merged_file.csv
            echo "" > ./data/NHC_4080S_result.csv
            echo "Critical_bin_id is $Critical_bin_id, Cnnz_ctile_rate_Threshold is $Cnnz_ctile_rate_Threshold ----------------------" >> $Source
            echo “Critical_bin_id is $Critical_bin_id, "Cnnz_ctile_rate_Threshold is $Cnnz_ctile_rate_Threshold-----------------------" >> $Source1
            sed -i "s/^#define Cnnz_ctile_rate_Threshold [0-9]*/#define Cnnz_ctile_rate_Threshold ${Cnnz_ctile_rate_Threshold}/" NHC_spgemm.h
            nvcc  -arch=compute_86 -code=sm_86 -O3 -Xcompiler -lrt -lcusparse -I /usr/local/cuda-11.4/include/cub test.cu -o test
            for filename in $filenames; do
                ./test ../338Matrixset/$filename.mtx >> $Source
            done
            python3 handle_338_matrix.py >> $Source1
        done
    done
    python3 draw_for_one_threshold.py
else
    echo "compile failed"
fi