CXX = g++
NVCC = nvcc

#CUDAFLAGS = $(GENCODE)  -g -lineinfo 
CUDAFLAGS =  $(GENCODE)  -O3

#CUDAFLAGS = $(GENCODE)  -g -G

#gcc7 cuda11.4 is success!
GENCODE = -arch=compute_86 -code=sm_86 
CUDAFLAGS += -Xcompiler -fopenmp 

# for Device Code
CUDA_PATH = /usr/local/cuda
LDFLAGS += -lcusparse $(CUDAFLAGS)
INCLUDE += -I${CUDA_PATH}/include
INCLUDE += -I${CUDA_PATH}/samples/common/inc
INCLUDE += -I./inc

BIN = ./bin
SRC = ./src
OBJ = ./obj
INC = ./inc

OBJ_LIB = $(OBJ)/Meta.o $(OBJ)/CSR.o $(OBJ)/Timings.o

COMMON_DEP = $(INC)/cuda_common.h $(INC)/define.h 

$(OBJ)/%.o : $(SRC)/%.cu $(INC)/%.h $(COMMON_DEP)
	mkdir -p $(dir $@)
	@echo $^
	$(NVCC) -c $(CUDAFLAGS) $(INCLUDE) -o $@ $<  && echo "Compiled $< successfully!" || echo "Failed to compile $<!"

$(OBJ)/%.o : $(SRC)/%.cu $(COMMON_DEP) 
	mkdir -p $(dir $@)
	@echo $^
	$(NVCC) -c $(CUDAFLAGS) $(INCLUDE) -o $@ $< -lcusparse  && echo "Compiled $< successfully!" || echo "Failed to compile $<!"

opsparse : $(OBJ_LIB) $(OBJ)/opsparse.o
	$(NVCC)  -o $@ $^ $(LDFLAGS) $(INCLUDE) && echo "Compiled $< successfully!" || echo "Failed to compile $<!" -g

reg_opsparse : $(OBJ_LIB) $(OBJ)/reg_opsparse.o
	$(NVCC)  -o $@ $^ $(LDFLAGS) $(INCLUDE)

reg_cusparse : $(OBJ_LIB) $(OBJ)/reg_cusparse.o
	$(NVCC)  -o $@ $^ $(LDFLAGS) $(INCLUDE)

clean :
	rm -rf $(BIN)
	rm -rf $(OBJ)

#编译运行成功:gcc7.5, cuda11.4
#kernel_wrapper.cuh里控制#define compute_share 0  #define compute_total_time 1