# Check CUDA Version
CUDA_DIR = /usr/local/cuda-8.0

# 32-bit or 64-bit  
OS_TYPE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")

# Compilers
GCC ?= g++
NVC = $(CUDA_DIR)/bin/nvcc  -ccbin $(GCC)

NVCC_FLAGS   :=	
NVCC_FLAGS   += -O2 
NVCC_FLAGS   += -m${OS_TYPE}
NVCC_FLAGS   += #--default-stream per-thread

# Directories for Header Files 
NV_INC = -I$(CUDA_DIR)/include/ -I$(CUDA_DIR)/samples/common/inc

# Directories for Libraries
ifeq ($(OS_TYPE), 64)
  NV_LIB = -L$(CUDA_DIR)/lib64
else
  NV_LIB = -L$(CUDA_DIR)/lib
endif

LIB_LINKER = -lcudart

SMS ?= 30 35 50 52

ifeq ($(GENCODE_FLAGS),)
  $(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif

#-----------------------------------------------------------------------------#
# Build Targets
#-----------------------------------------------------------------------------#
all: build 

build: main_sep main_seq  main no_d2h h2d 3h2d s1

main_sep: main_sep.cu
	$(NVC) $(NV_INC) $(NV_LIB) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $^ $(LIB_LINKER)

main_seq: main_seq.cu
	$(NVC) $(NV_INC) $(NV_LIB) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $^ $(LIB_LINKER)

main: main.cu
	$(NVC) $(NV_INC) $(NV_LIB) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $^ $(LIB_LINKER)

no_d2h: no_d2h.cu
	$(NVC) $(NV_INC) $(NV_LIB) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $^ $(LIB_LINKER)

h2d: h2d.cu
	$(NVC) $(NV_INC) $(NV_LIB) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $^ $(LIB_LINKER)

3h2d: 3h2d.cu
	$(NVC) $(NV_INC) $(NV_LIB) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $^ $(LIB_LINKER)

s1: s1.cu
	$(NVC) $(NV_INC) $(NV_LIB) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $^ $(LIB_LINKER)

.PHONY: clean
clean:
	rm -rf main_seq main_sep main no_d2h h2d 3h2d s1
