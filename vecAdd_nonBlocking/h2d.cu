#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

const int N = 1 << 20;

#define FLTSIZE sizeof(float)

inline int BLK(int data, int blocksize)
{
	return (data + blocksize - 1) / blocksize;
}

__global__ void kernel_vectorAdd (const float* __restrict__ a_d, 
		const float* __restrict__ b_d,
		const int N,
		const int offset,
		float *c_d)
{
	int tid = threadIdx.x + __mul24(blockIdx.x, blockDim.x);

	if(tid < N) {
		c_d[tid + offset] = a_d[tid + offset] + b_d[tid + offset];	
	}
}

int main( int argc, char **argv)
{
	int devid = 0 ;

	int num_streams = 8;

	if(argc >= 2)
		num_streams = atoi(argv[1]);

	if(argc >= 3)
		devid = atoi(argv[2]);

	cudaSetDevice(devid);

	printf("\nrunning %d cuda streams on device %d\n", num_streams, devid);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, devid);
	printf("Device Number: %d\n", devid);
	printf("  Device name: %s\n", prop.name);
	printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
	printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
	printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
	printf("  Concurrent copy and execution: %s\n",  (prop.deviceOverlap ? "Yes" : "No"));
	printf("  Concurrent kernels: %d\n",  (prop.concurrentKernels));
	printf("  Concurrent copy and kernel execution: %s with %d copy engine(s)\n", 
			(prop.deviceOverlap ? "Yes" : "No"), prop.asyncEngineCount);


	// allocate streams
    cudaStream_t *streams = (cudaStream_t *) malloc(num_streams * sizeof(cudaStream_t));

	// init
    for (int i = 0; i < num_streams; i++) {
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }

	//------------------------------------------------------------------------//
	// allocate data on the host
	//------------------------------------------------------------------------//
	size_t databytes = N  * FLTSIZE; 

	//float *a_h = (float*) malloc ( N * num_streams * FLTSIZE);
	//float *b_h = (float*) malloc ( N * num_streams * FLTSIZE);
	//float *c_h = (float*) malloc ( N * num_streams * FLTSIZE);

	float *a_h = NULL;
    checkCudaErrors(cudaMallocHost((void **)&a_h, N * num_streams * FLTSIZE));

	float *b_h = NULL;
    checkCudaErrors(cudaMallocHost((void **)&b_h, N * num_streams * FLTSIZE));

	float *c_h = NULL;
    checkCudaErrors(cudaMallocHost((void **)&c_h, N * num_streams * FLTSIZE));

	for(int i=0; i< N * num_streams; i++) {
		a_h[i] = 1.1f;	
		b_h[i] = 2.2f;	
	}

	//------------------------------------------------------------------------//
	// allocate data on the device 
	//------------------------------------------------------------------------//
	float *a_d;
	float *b_d;
	float *c_d;
	cudaMalloc((void**)&a_d, N * num_streams * FLTSIZE);
	cudaMalloc((void**)&b_d, N * num_streams * FLTSIZE);
	cudaMalloc((void**)&c_d, N * num_streams * FLTSIZE);

	// kernel configuration
	dim3 threads = dim3(256, 1, 1);
	dim3 blocks  = dim3(BLK(N, threads.x), 1, 1);

	// create cuda event handles
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	cudaEventRecord(start,0);

	// copy data to deivce
	for (int i = 0; i < num_streams; i++) {
		int offset = i * N;
		cudaMemcpyAsync(&a_d[offset], &a_h[offset],  databytes, cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(&b_d[offset], &b_h[offset],  databytes, cudaMemcpyHostToDevice, streams[i]);
	}

	/*
	// launch one worker kernel per stream
	for (int i = 0; i < num_streams; i++) {
		int offset = i * N;
		kernel_vectorAdd <<< blocks, threads, 0, streams[i] >>> (a_d, 
				                                                 b_d, 
																 N, 
																 offset,
																 c_d);
	}

	// copy data back to host
	for (int i = 0; i < num_streams; i++) {
		int offset = i * N;
		cudaMemcpyAsync(&c_h[offset], &c_d[offset],  databytes, cudaMemcpyDeviceToHost, streams[i]);
	}
	*/

	// required for async copy
	//cudaDeviceSynchronize();

	//cudaEventSynchronize(stop);
    cudaEventRecord(stop, 0);

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter=0;
    while (cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter++;
    }


	
	float gpuTime_ms= 0;
	cudaEventElapsedTime(&gpuTime_ms, start, stop);

	printf("runtime (ms) : %f\n", gpuTime_ms);

	// check data
	bool success = 1;
	for(int i=0; i< N * num_streams; i++) {
		if (abs(c_h[i] - 3.3f) > 1e-6) {
			fprintf(stderr, "%d : %f  (error)!\n", i, c_h[i]);
			success = 0;
			break;
		}
	}

	if(success) {
		printf("\nSuccess! Exit.\n");	
	}

	//------------------------------------------------------------------------//
	// free 
	//------------------------------------------------------------------------//
    for (int i = 0; i < num_streams; i++) {
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

	//free(a_h);
	//free(b_h);
	//free(c_h);
	cudaFreeHost(a_h);
	cudaFreeHost(b_h);
	cudaFreeHost(c_h);

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);

	cudaDeviceReset();

	return 0;
}
