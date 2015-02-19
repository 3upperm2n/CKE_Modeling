# CMakeLists.txt for CUDA
cmake_minimum_required( VERSION 2.8.0 )
find_package(CUDA QUIET REQUIRED)

# nvcc options
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O2 
-gencode arch=compute_20,code=sm_20 
-gencode arch=compute_30,code=sm_30 
-gencode arch=compute_35,code=sm_35 
-gencode arch=compute_37,code=sm_37 
-gencode arch=compute_50,code=sm_50 
-gencode arch=compute_52,code=sm_52 
)

# indicate include dir
include_directories(
/usr/local/cuda/include
/usr/local/cuda/samples/common/inc
) 

# indicate library dir 
link_directories(
/usr/local/cuda/lib64		
)


# Indicate target & source files to compile
cuda_add_executable(
matrixMul
matrixMul.cu
)

cuda_add_executable(
matrixMul_low
matrixMul_low.cu
)
	 
cuda_add_executable(
matrixMul_cke
matrixMul_cke.cu
)

# Indicate target & libraries to link
target_link_libraries(
matrixMul	
-lcudart
) 

target_link_libraries(
matrixMul_low	
-lcudart
) 

target_link_libraries(
matrixMul_cke	
-lcudart
) 
