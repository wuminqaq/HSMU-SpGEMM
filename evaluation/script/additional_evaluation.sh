#!/bin/bash
filenames=$(cat matrix338_list.txt)
make
for filename in $filenames; do
    ./test ../338Matrixset/$filename.mtx
done
python3 draw_time_and_space_cost.py
python3 draw_stacked_bar_chart_338result.py