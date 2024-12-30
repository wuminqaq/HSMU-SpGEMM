# nvcc -arch=compute_86 -code=sm_86 -O3 -Xcompiler -lrt  -lrt -lcudart -lcusparse -I /usr/local/cuda-11.4/include/cub ../src/test.cu -o test
make
if [ $? -eq 0 ]; then
    ./test ../18representMatrixSet/cant.mtx
else
   echo "the compile is failed!" 
fi

#gcc is 9.5,cuda is 11.4 