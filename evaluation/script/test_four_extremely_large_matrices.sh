#!/bin/bash
filenames=$(cat four_extremely_large_matrices.txt)
make
for filename in $filenames; do
    ./test ../338Matrixset/$filename.mtx
done
