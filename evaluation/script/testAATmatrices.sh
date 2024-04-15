#!/bin/bash
#Ensure that the NHC_spgemm.h parameters in the file are set as follows: "#define AAT 1"
filenames=$(cat AATmatrix_list.txt)
make
for filename in $filenames; do
    ./test ../18representMatrixSet/$filename.mtx
done

