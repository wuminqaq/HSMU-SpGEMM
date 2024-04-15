
Matrixset1=$(cat /home/wm/name_lists_of_matrix/compressed_matrix.txt)
# Matrixset1=$(cat /home/wm/name_lists_of_matrix/new_repres_matrix.txt)
# nvcc  -arch=compute_80 -code=sm_80 -O3 -Xcompiler -lrt -lcusparse -I /usr/local/cuda-11.4/include/cub test.cu -o test

if [ $? -eq 0 ]; then
       # echo > /home/wm/open_source_code/nsparse-master/Nsparse_Original_result.txt
       echo > /home/wm/open_source_code/spGEMM/nsparse-master/Nsparse_shared_Original_result.txt
       while read -r i; do
              i="${i}.mtx"
              echo "******************** $i is being tested with NHC_spgemm ***********************************"
              Path="/home/wm/total_matrix/$i"
              # Source="/home/wm/open_source_code/nsparse-master/Nsparse_Original_result.txt"
              Source="/home/wm/open_source_code/spGEMM/nsparse-master/Nsparse_shared_Original_result.txt"
              ./bin/spgemm_hash_d "$Path" >> "$Source"
              echo "**************************** NHC_spgemm test $i is done *******************************"
       done <<< "$Matrixset1"
       # python3 NHC_find.py
else
   echo "the compile is failed!" 
fi