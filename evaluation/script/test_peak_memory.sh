#!/bin/bash

output_file="gpu_memory_usage.txt"

record_gpu_memory_usage() {
    while true; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1}' >> "$output_file"
        sleep 0.0000001
    done
}

record_gpu_memory_usage &
background_pid=$!
echo "Background loop PID: $background_pid"

filenames=$(cat represent_matrix18_list.txt)
for filename in $filenames; do
    echo "$filename result is below:" >> "$output_file"
    ./test ../18representMatrixSet/$filename.mtx
done

kill $background_pid 2>/dev/null

python3 extract_max_memory.py