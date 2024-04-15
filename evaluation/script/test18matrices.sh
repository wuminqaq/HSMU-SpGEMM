#!/bin/bash

filenames=$(cat represent_matrix18_list.txt)
make
for filename in $filenames; do
    ./test ../18representMatrixSet/$filename.mtx
done

