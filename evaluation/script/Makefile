CXX := nvcc
#Replace cuda path with actual cuda path
CXXFLAGS := -arch=compute_86 -code=sm_86 -O3 -Xcompiler -lrt -lrt -lcudart -lcusparse -I/usr/local/cuda-11.4/include/cub

#Replace with the actual file path
SRC := ../../src/test.cu
TARGET := test

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@
clean:
	rm -f $(TARGET)
