#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>


__global__ void globalmemory(float *a, float *b, int num_iterations)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i=0; i<num_iterations; i++)
    {
        a[gid] += 3.f;
    }

	b[gid] = a[gid];
}


int main(int argc, char **argv)
{
    int cuda_device = 0;
    int nstreams = atoi(argv[1]);
    int n = 128 * 1024 * 1024; 
    size_t nbytes = n * sizeof(float);
    dim3 threads, blocks;       

    checkCudaErrors(cudaSetDevice(cuda_device));

	// input
    float *h_a = NULL;                
	h_a = (float *) malloc(nbytes);
	//checkCudaErrors(cudaMallocHost((void **)&h_a, nbytes));

	// output
    float *h_b = NULL;                
	h_b = (float *) malloc(nbytes);
	//checkCudaErrors(cudaMallocHost((void **)&h_b, nbytes));

	for(int i = 0; i < n; i++)
	{
		h_a[i] = 1.f;
	}

	// device memory
    float *d_a = 0, *d_b = 0;             
    checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
    checkCudaErrors(cudaMalloc((void **)&d_b, nbytes));

    cudaStream_t *streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));

    for (int i = 0; i < nstreams; i++)
    {
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }


    threads = dim3(512,1);
    blocks  = dim3(n/(nstreams*threads.x),1);

	// asynchronously transfer 
	for (int i = 0; i < nstreams; i++)
	{
		checkCudaErrors(cudaMemcpyAsync(d_a + i * n / nstreams, h_a + i * n / nstreams, nbytes / nstreams, cudaMemcpyHostToDevice, streams[i]));
	}

	//
	for (int i = 0; i < nstreams; i++)
	{
		globalmemory<<<blocks, threads, 0, streams[i]>>>(d_a + i *n / nstreams, d_b, 100);
	}

	for (int i = 0; i < nstreams; i++)
	{
		checkCudaErrors(cudaMemcpyAsync(h_b + i * n / nstreams, d_b + i * n / nstreams, nbytes / nstreams, cudaMemcpyDeviceToHost, streams[i]));
	}


    // release resources
    for (int i = 0; i < nstreams; i++)
    {
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }

    // Free cudaMallocHost or Generic Host allocated memory (from CUDA 4.0)
	//cudaFreeHost(h_a);
	//cudaFreeHost(h_b);
	free(h_a);
	free(h_b);

    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));

    checkCudaErrors(cudaDeviceReset());

    return EXIT_SUCCESS;
}
