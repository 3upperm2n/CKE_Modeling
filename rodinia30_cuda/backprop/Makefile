CUDA_DIR = /usr/local/cuda-6.5
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")

# C compiler
CC = gcc
CC_FLAGS = -g  -O2

# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include

NVCC_FLAGS += -O2

ifeq ($(OS_SIZE), 64)
  CUDA_LIB_DIR = -L$(CUDA_DIR)/lib64
else
  CUDA_LIB_DIR = -L$(CUDA_DIR)/lib
endif



backprop: backprop.o facetrain.o imagenet.o backprop_cuda.o 
	$(CC) $(CC_FLAGS) backprop.o facetrain.o imagenet.o backprop_cuda.o -o backprop $(CUDA_LIB_DIR) -lcuda -lcudart -lm

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -c

backprop_cuda.o: backprop_cuda.cu backprop.h
	$(NVCC) $(NVCC_FLAGS) -c backprop_cuda.cu

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c

clean:
	rm -f *.o *~ backprop backprop_cuda.linkinfo
