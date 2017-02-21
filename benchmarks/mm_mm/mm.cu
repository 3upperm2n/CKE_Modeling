// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <helper_cuda.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

// -device=0   -wA=320 -hA=320 -wB=320 -hB=320


/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

//	int gid_x = threadIdx.x + blockIdx.x * blockDim.x;                                              
//	int gid_y = threadIdx.y + blockIdx.y * blockDim.y;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {


        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
//	if(gid_x == 0 && gid_y ==0)                                                                     
//		printf("C = %f\n",Csub);  
}

void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
void matrixMultiply(int argc, char **argv, int block_size, dim3 &dimsA, dim3 &dimsB)
{
	//-------------------//
	// cuda streams
	//-------------------//
	int nstreams = 2;                                                           
	cudaStream_t *streams = (cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
	for(int i = 0; i < nstreams; i++)                                           
		checkCudaErrors(cudaStreamCreate(&(streams[i])));

	//-------------//
	// A
	//-------------//
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;

	float *h_A1 = NULL;
	cudaMallocHost((void **) &h_A1, mem_size_A);
	float *h_A2 = NULL;
	cudaMallocHost((void **) &h_A2, mem_size_A);

	//-------------//
	// B
	//-------------//
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;

	float *h_B1 = NULL;
	cudaMallocHost((void **) &h_B1, mem_size_B);
	float *h_B2 = NULL;
	cudaMallocHost((void **) &h_B2, mem_size_B);

    // Initialize host memory
    constantInit(h_A1, size_A, 1.0f);
    constantInit(h_A2, size_A, 1.0f);

    constantInit(h_B1, size_B, 0.01f);
    constantInit(h_B2, size_B, 0.01f);

	//-------------//
	// C
	//-------------//
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);

	float *h_C1 = NULL;
	cudaMallocHost((void **) &h_C1, mem_size_C);
	float *h_C2 = NULL;
	cudaMallocHost((void **) &h_C2, mem_size_C);

	//----------------------------------------//
	// gpu
	//----------------------------------------//
    float *d_A1 = NULL;
	float *d_B1 = NULL;
	float *d_C1 = NULL;
    cudaMalloc((void **) &d_A1, mem_size_A);
    cudaMalloc((void **) &d_B1, mem_size_B);
    cudaMalloc((void **) &d_C1, mem_size_C);


    float *d_A2 = NULL;
	float *d_B2 = NULL;
	float *d_C2 = NULL;
    cudaMalloc((void **) &d_A2, mem_size_A);
    cudaMalloc((void **) &d_B2, mem_size_B);
    cudaMalloc((void **) &d_C2, mem_size_C);

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

	//----------------------------------------------
    // copy host memory to device
	//----------------------------------------------
    cudaMemcpyAsync(d_A1, h_A1, mem_size_A, cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(d_B1, h_B1, mem_size_B, cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(d_A2, h_A2, mem_size_A, cudaMemcpyHostToDevice, streams[1]);
    cudaMemcpyAsync(d_B2, h_B2, mem_size_B, cudaMemcpyHostToDevice, streams[1]);


	//----------------------------------------------
    //  kernel
	//----------------------------------------------
	matrixMulCUDA<32><<< grid, threads, 0, streams[0]>>>(d_C1, d_A1, d_B1, dimsA.x, dimsB.x);
	matrixMulCUDA<32><<< grid, threads, 0, streams[1]>>>(d_C2, d_A2, d_B2, dimsA.x, dimsB.x);

	//----------------------------------------------
    // device to host
	//----------------------------------------------
	cudaMemcpyAsync(h_C1, d_C1, mem_size_C, cudaMemcpyDeviceToHost, streams[0]);
	cudaMemcpyAsync(h_C2, d_C2, mem_size_C, cudaMemcpyDeviceToHost, streams[1]);


    cudaDeviceSynchronize();

	for (int i = 0; i < nstreams; i++) {                                     
		checkCudaErrors(cudaStreamDestroy(streams[i]));                         
	}

    // Clean up memory
    cudaFreeHost(h_A1);
    cudaFreeHost(h_B1);
    cudaFreeHost(h_C1);
    cudaFreeHost(h_A2);
    cudaFreeHost(h_B2);
    cudaFreeHost(h_C2);

    cudaFree(d_A1);
    cudaFree(d_B1);
    cudaFree(d_C1);
    cudaFree(d_A2);
    cudaFree(d_B2);
    cudaFree(d_C2);


    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
}


/**
 * Program main
 */
int main(int argc, char **argv)
{
	//   -device=0   -wA=320 -hA=320 -wB=320 -hB=320

    //printf("[Matrix Multiply Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "?"))
    {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices must be equal.\n");

        exit(EXIT_SUCCESS);
    }

    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    int devID = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        cudaSetDevice(devID);
    }

    cudaError_t error;

    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
    }
    error = cudaGetDeviceProperties(&deviceProp, devID);


    // Use a larger block size for Fermi and above
    int block_size = (deviceProp.major < 2) ? 16 : 32;

    dim3 dimsA(5*2*block_size, 5*2*block_size, 1);
    dim3 dimsB(5*4*block_size, 5*2*block_size, 1);

    // width of Matrix A
    if (checkCmdLineFlag(argc, (const char **)argv, "wA"))
    {
        dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
    }

    // height of Matrix A
    if (checkCmdLineFlag(argc, (const char **)argv, "hA"))
    {
        dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
    }

    // width of Matrix B
    if (checkCmdLineFlag(argc, (const char **)argv, "wB"))
    {
        dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
    }

    // height of Matrix B
    if (checkCmdLineFlag(argc, (const char **)argv, "hB"))
    {
        dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
    }


    if (dimsA.x != dimsB.y)
    {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
               dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    //printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

    matrixMultiply(argc, argv, block_size, dimsA, dimsB);
}
