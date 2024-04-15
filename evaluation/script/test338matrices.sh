#!/bin/bash
filenames=$(cat matrix338_list.txt)
make
for filename in $filenames; do
    ./test ../338Matrixset/$filename.mtx
done
python3 handle_338_matrix.py
