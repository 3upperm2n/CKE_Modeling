#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define BLKSIZE 32
#define DEBUG 0

__global__ void matrixMulCUDA_cke(float *C, float *A, float *B, int wA, int wB, size_t offsetA, size_t offsetC)
{
    int bx = blockIdx.x; // B col
    int by = blockIdx.y; // A row

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = wA * BLKSIZE * by;
    int aEnd   = aBegin + wA - 1;
    int aStep  = BLKSIZE;

    int bBegin = BLKSIZE * bx;
    int bStep  = BLKSIZE * wB;

    float Csub = 0;

    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        __shared__ float As[BLKSIZE][BLKSIZE];
        __shared__ float Bs[BLKSIZE][BLKSIZE];

        As[ty][tx] = A[a + wA * ty + tx + offsetA];
        Bs[ty][tx] = B[b + wB * ty + tx];

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLKSIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    int c = wB * BLKSIZE * by + BLKSIZE * bx;
    C[c + wB * ty + tx + offsetC] = Csub;
}

void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

int matrixMultiply(int argc, char **argv, dim3 &dimsA, dim3 &dimsB)
{
	//-------------------------------------------------------------------------------------------//
	// Allocate host memory
	//-------------------------------------------------------------------------------------------//
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);

    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // Initialize host memory
    const float valB = 0.01f;
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, valB);

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C = (float *) malloc(mem_size_C);

	printf("A = %d x %d\n", dimsA.x, dimsA.y);
	printf("B = %d x %d\n", dimsB.x, dimsB.y);
	printf("C = %d x %d\n", dimsC.x, dimsC.y);

	// TODO 
	// Initialize cuda streams: 1st stream copies data, the others parallelly compute
	int nstreams = 3;    
	cudaStream_t *streams = (cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
	for(int i = 0; i < nstreams; i++)
		checkCudaErrors(cudaStreamCreate(&(streams[i])));

	// Pre-compute the workloads for each stream
	//size_t mem_size_A_per_stream = mem_size_A / (nstreams-1);
	size_t offsetA = size_A / (nstreams -1);
	size_t offsetC = dimsC.x * dimsC.y / (nstreams - 1);
	printf("offsetA = %ld\toffsetC= %ld\n", offsetA, offsetC);

	//-------------------------------------------------------------------------------------------//
	// Allocate device memory
	//-------------------------------------------------------------------------------------------//
    float *d_A, *d_B, *d_C;

    checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));


    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

    // Setup execution configuration 
    dim3 threads(BLKSIZE, BLKSIZE);
	int dimsA_per_stream = dimsA.y / (nstreams - 1);
    dim3 grid(dimsB.x / threads.x, dimsA_per_stream / threads.y);

	//size_t sm_size = threads.x * threads.y * sizeof(float) * 2;
	size_t sm_size = 0;
	printf("launch grid = %d x %d\n", grid.x, grid.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

	for(int i=1; i<nstreams; i++)
	{	
		size_t startpos = offsetA * (i-1);
		size_t outpos = offsetC * (i-1);
		matrixMulCUDA_cke <<< grid, threads, sm_size, streams[i] >>> (d_C, 
				                                                      d_B, 
																	  d_A, 
																	  dimsA.x, 
																	  dimsB.x, 
																	  startpos, 
																	  outpos);
	}

#if DEBUG
    checkCudaErrors(cudaDeviceSynchronize());
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
#endif

    printf("done\n");

#if !DEBUG

    cudaDeviceSynchronize();

    // cuda event for timing 
    cudaEvent_t start;
    checkCudaErrors(cudaEventCreate(&start));

    cudaEvent_t stop;
    checkCudaErrors(cudaEventCreate(&stop));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    cudaError_t error;

    // Execute the kernel
    int nIter = 300;

    for (int j = 0; j < nIter; j++)
    {
		//TODO
		for(int i=1; i<nstreams; i++)
		{	
			size_t startpos = offsetA * (i-1);
			size_t outpos = offsetC * (i-1);
			matrixMulCUDA_cke <<< grid, threads, sm_size, streams[i] >>> (d_C, 
					                                                      d_B, 
					                                                      d_A, 
					                                                      dimsA.x,
					                                                      dimsB.x, 
				                                                      	  startpos, 
					                                                      outpos);
		}
    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);


    // Copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

#endif


    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6 ; // machine zero

    for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++)
    {
        double abs_err = fabs(h_C[i] - (dimsA.x * valB));
        double dot_length = dimsA.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err/abs_val/dot_length ;

        if (rel_err > eps)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA.x*valB, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

	for(int i = 0; i < nstreams; i++)
		cudaStreamDestroy(streams[i]);
	free(streams);

#if !DEBUG
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
#endif

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");


    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    if (correct)
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    }
}


/**
 * Program main
 */
int main(int argc, char **argv)
{
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

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

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    // Use a larger block size for Fermi and above
    //int BLKSIZE = (deviceProp.major < 2) ? 16 : 32;

    dim3 dimsA(5*2*BLKSIZE, 5*2*BLKSIZE, 1);
    dim3 dimsB(5*4*BLKSIZE, 5*2*BLKSIZE, 1);

	//printf("dimsA.x = %d\t dimsA.y = %d\n", dimsA.x, dimsA.y);
	//printf("dimsB.x = %d\t dimsB.y = %d\n", dimsB.x, dimsB.y);

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

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

    int matrix_result = matrixMultiply(argc, argv, dimsA, dimsB);

    exit(matrix_result);
}
