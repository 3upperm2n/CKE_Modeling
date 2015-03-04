// Molecular Scattering Simulation 
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <helper_cuda.h>
#include <math_constants.h>

#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // min_element
#include <iterator> // distance, iterator

#define FOUR_PI (4*CUDART_PI_F)
#define INV_PI  (1/CUDART_PI_F)
#define TK 1 // time kernel
#define DB 0 // debug 

using namespace std;


__device__ __constant__ float d_atomC[9]={2.31000,  1.02000,  1.58860,  0.865000, 20.8439,  10.2075, 0.568700, 51.6512,  0.2156};
__device__ __constant__ float d_atomH[9]={0.493002, 0.322912, 0.140191, 0.040810, 10.5109,  26.1257, 3.14236,  57.7997,  0.003038};
__device__ __constant__ float d_atomO[9]={3.04850,  2.28680,  1.54630,  0.867000, 13.2771,  5.70110, 0.323900, 32.9089,  0.2508};
__device__ __constant__ float d_atomN[9]={12.2126,  3.13220,  2.01250,  1.16630,  0.005700, 9.89330, 28.9975,  0.582600, -11.52};


texture<float4, 1, cudaReadModeElementType> crdTex;
texture<int,    1, cudaReadModeElementType> wokTex;

texture<float,  1, cudaReadModeElementType> dqTex;
texture<float4, 1, cudaReadModeElementType> dfactorTex;

// input atom quantity
__device__ __constant__ char	d_atomtype[60000]; // 60KB 


__device__ float3 operator-(const float3 &a, const float3 &b)
{
	return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ float4 operator-(const float4 &a, const float4 &b)
{
	return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w - b.w);
}

class PREPARE 
{
	public:
		void readpdb(const char *file); // read pdb file
		void readcor(const char *file);       // read cor file
		void cor2pdb(const char *file);       // transform cor file to pdb file
		void read(const char *file);

		vector<char>  atom_type;
		//vector<float3> crd;
		vector<float4> crd;
};


void PREPARE::read(const char *file)
{
	string str = file;	
	unsigned found = str.find_last_of(".");
	//cout << "file type: " << str.substr(found+1) << '\n';
	string filetype = str.substr(found+1);
	if (!filetype.compare("pdb"))
		readpdb(file);

	//if (!filetype.compare("cor"))
	//readcor(file);
}


void PREPARE::readpdb(const char *file)
{
	char line[1000];
	char p1[10];
	char p2[10];
	char p3[10];// type
	char p4[10];
	char p5[10];//x
	char p6[10];//y
	char p7[10];//z
	char p8[10];
	char p9[10];
	char p10[2];//type as well

	FILE *fp = fopen(file,"r");
	if(fp == NULL)
		perror("Error opening file!!!\n\n");

	while (fgets(line,1000,fp)!=NULL)
	{
		sscanf(line, "%s %s %s %s %s %s %s %s %s %s", p1, p2, p3, p4, p5,
				p6, p7, p8, p9, p10);
		atom_type.push_back(p3[0]); // type
		//crd.push_back(make_float3(atof(p5), atof(p6), atof(p7)));
		crd.push_back(make_float4(atof(p5), atof(p6), atof(p7), 0.f));
	}

	fclose(fp);
}



__global__ void calc_qr(float *d_q, float *d_R, int N, float inv_lamda, float inv_distance)
{
	size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < N)
	{
		float tmp, tmp_q, tmp_r;

		tmp = gid * 0.0732f;	
		//CUPRINTF("%d\t: %f\n", gid, tmp);

		tmp_r = inv_lamda * sin(0.5 * atan(tmp * inv_distance));
		tmp_q = FOUR_PI * tmp_r;	
		tmp_r += tmp_r;

		d_q[gid] = tmp_q;	
		d_R[gid] = tmp_r;	
		//CUPRINTF("%d\t: %f\n", gid, d_q[gid]);
	}
}


__global__ void calc_FactorTable(float *d_q, float4 *d_factor, int N)
{
	size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < N)
	{
		float tmp;
		float fc, fh, fo, fn;
		tmp = d_q[gid] * 0.25 * INV_PI;
		tmp = powf(tmp,2.0);

		// loop unrolling
		fc = d_atomC[0] * expf(-d_atomC[4] * tmp) +
			d_atomC[1] * expf(-d_atomC[5] * tmp) +
			d_atomC[2] * expf(-d_atomC[6] * tmp) +
			d_atomC[3] * expf(-d_atomC[7] * tmp) +
			d_atomC[8];

		fh = d_atomH[0] * expf(-d_atomH[4] * tmp) +
			d_atomH[1] * expf(-d_atomH[5] * tmp) +
			d_atomH[2] * expf(-d_atomH[6] * tmp) +
			d_atomH[3] * expf(-d_atomH[7] * tmp) +
			d_atomH[8];

		fo = d_atomO[0] * expf(-d_atomO[4] * tmp) +
			d_atomO[1] * expf(-d_atomO[5] * tmp) +
			d_atomO[2] * expf(-d_atomO[6] * tmp) +
			d_atomO[3] * expf(-d_atomO[7] * tmp) +
			d_atomO[8];

		fn = d_atomN[0] * expf(-d_atomN[4] * tmp) +
			d_atomN[1] * expf(-d_atomN[5] * tmp) +
			d_atomN[2] * expf(-d_atomN[6] * tmp) +
			d_atomN[3] * expf(-d_atomN[7] * tmp) +
			d_atomN[8];

		d_factor[gid] = make_float4(fc, fh, fo, fn);
		//CUPRINTF("%f \t %f \t %f \t %f\n", d_factor[gid].x, d_factor[gid].y, d_factor[gid].z, d_factor[gid].w);
	}
}

__global__ void calc_diffraction(const int workoffset,
		                         const int worklen,
		                         const int lastpos,
		                         const int N,
		                         float *d_Iq,
		                         float *d_Iqz,
		                         int streamID)
{
	size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

	if(gid < N)
	{
		float iq, iqz;
		iq = iqz = 0;

		float4 factor = tex1Dfetch(dfactorTex, gid);

		float q = tex1Dfetch(dqTex, gid);

		float fj, fk;

		for (int kk=0; kk<worklen; kk++) 
		{
			int startpos = lastpos - tex1Dfetch(wokTex, workoffset + kk); // adjust the index

			char t1 = d_atomtype[startpos]; // const 

			// read d_factor 1 time 
			if (t1 == 'C')
			{
				fj = factor.x;
			}
			else if (t1 == 'H')
			{
				fj = factor.y;
			}
			else if (t1 == 'O')
			{
				fj = factor.z;
			}
			else
			{
				fj = factor.w;
			}

			float4 crd_ref = tex1Dfetch(crdTex, startpos);

			for(int i = startpos + 1; i <= lastpos; ++i) // atoms to compare with the base atom
			{
				// read d_atomtype i times 
				char t2 = d_atomtype[i];

				// read d_factor i times
				if (t2 == 'C'){
					//fk = factor[tid].x;
					fk = factor.x;
				}
				else if (t2 == 'H'){
					//fk = factor[tid].y;
					fk = factor.y;
				}
				else if (t2 == 'O'){
					//fk = factor[tid].z;
					fk = factor.z;
				}else{
					//fk = factor[tid].w;
					fk = factor.w;
				}

				//float fj_fk = fj * fk;

				float4 cur_crd = tex1Dfetch(crdTex, i);

				float4 distance =  crd_ref - cur_crd;


				iq  += fj * fk * j0(q * sqrt(distance.x * distance.x + distance.y * distance.y));

				// Iq_z=Iq_z+fj.*fk.*exp(1i*rz.*q);
				// For complex Z=X+i*Y, exp(Z) = exp(X)*(COS(Y)+i*SIN(Y)) 
				// here, only calculate the real part
				iqz += fj * fk * cos(abs(distance.z) * q);
			} // end of loop

		}
		d_Iq[gid  + N * streamID] = iq;
		d_Iqz[gid + N * streamID] = iqz;
	}
}







int main(int argc, char*argv[])
{
	// read input
	char *inputfile;
	if (argc == 3)
	{
		inputfile = argv[1];	
	}
	else
	{
		cout << "Please specify file name!\nUsage: ./fiber inputfile\n" << endl;
		exit (EXIT_FAILURE);
	}

	PREPARE prepare;
	prepare.read(inputfile);

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    std::cout << "max texture1d linear: " << deviceProp.maxTexture1DLinear << std::endl;


	int nstreams = atoi(argv[2]);
	cudaStream_t *streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
	for (int i = 0 ; i < nstreams ; i++){
		checkCudaErrors(cudaStreamCreate(&(streams[i])));
	}


	int len_crd = prepare.crd.size();
	cout << "atom volume = " << len_crd << endl;

	cudaMemcpyToSymbol(d_atomtype, &prepare.atom_type[0], sizeof(char) * len_crd, 0, cudaMemcpyHostToDevice);
	
	// copy data to device
	float4 *d_crd;
	cudaMalloc((void**)&d_crd, sizeof(float4) * len_crd);

	cudaMemcpy(d_crd, &prepare.crd[0], sizeof(float4) * len_crd, cudaMemcpyHostToDevice);

	// bind texture
	cudaChannelFormatDesc float4Desc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaBindTexture(NULL, crdTex, d_crd, float4Desc));

	//-------------------------------------------------------------------------------------------//
	// calculate q and R 
	//-------------------------------------------------------------------------------------------//
#if TK
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsedTime;
#endif

	float lamda = 1.033f;
	float dist = 300.0f;
	int N = 2000;

	size_t bytes_n = sizeof(float) * N;

#if DB
	float *q,*R;
	q   = (float*)malloc(bytes_n);
	R   = (float*)malloc(bytes_n);
#endif

	// configure dimensions of the kernel 
	int block_size = 256;
	int grid_size  = (N+ block_size - 1)/block_size;

	// device arrays
	float *d_q, *d_R;
	cudaMalloc((void**)&d_q , bytes_n);
	cudaMalloc((void**)&d_R , bytes_n);

#if TK
	cudaEventRecord(start, 0);
#endif

	// launch kernel
	calc_qr <<< grid_size, block_size >>> (d_q, d_R, N, 1/lamda, 1/dist);

#if TK
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("calc_qr = %f ms\n", elapsedTime);
#endif

#if DB	
	cudaMemcpy(q , d_q , bytes_n, cudaMemcpyDeviceToHost);
	cudaMemcpy(R , d_R , bytes_n, cudaMemcpyDeviceToHost);
	printf("q=\n");
	for(int i=0; i<10; ++i)
	{
		printf("%6.5f ", q[i]);
	}
	printf("\n");

	printf("R=\n");
	for(int i=0; i<10; ++i)
	{
		printf("%f ", R[i]);
	}
	printf("\n");
#endif

	//-------------------------------------------------------------------------------------------//
	// calculate atom factors
	//-------------------------------------------------------------------------------------------//
	float4 *d_factor;
	cudaMalloc((void**)&d_factor, sizeof(float4) * N);

#if TK
	cudaEventRecord(start, 0);
#endif

	int blk_factor = 256;
	calc_FactorTable <<< (N + blk_factor - 1)/blk_factor, blk_factor >>> (d_q, d_factor, N);

#if TK
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("calc_FactorTable = %f ms\n", elapsedTime);
#endif

#if DB
	float4 *factor;
	factor = (float4 *)malloc(sizeof(float4) *  N);
	cudaMemcpy(factor, d_factor , sizeof(float4) * N, cudaMemcpyDeviceToHost);
	cout << "\tC\tH\tO\tN" << endl;
	for(int i=0; i < N; i++){
		cout << i << "\t" << factor[i].x << "\t" << factor[i].y << "\t" << factor[i].z << "\t" << factor[i].w << endl;
	}

#endif





	//-------------------------------------------------------------------------------------------//
	// calculate iq and iqz
	//-------------------------------------------------------------------------------------------//

	// host mem for output
	float *h_Iq  = NULL;
	float *h_Iqz = NULL;
	checkCudaErrors(cudaMallocHost((void**)&h_Iq,  sizeof(float) * N * nstreams));
	checkCudaErrors(cudaMallocHost((void**)&h_Iqz, sizeof(float) * N * nstreams));

	// device mem for output
	float *d_Iq;
	float *d_Iqz;
	cudaMalloc((void**)&d_Iq,  sizeof(float) * N * nstreams);
	cudaMalloc((void**)&d_Iqz, sizeof(float) * N * nstreams);


	int atomNum = prepare.atom_type.size();
	int lastpos = atomNum - 1;

	int blk_diffraction = 256;
	int grd_diffraction = (N + blk_diffraction - 1) / blk_diffraction; // grid size 8
	size_t sm_size = 0;


	int step = (atomNum - 1) / nstreams;
	vector<int> beginpos;
	vector<int> endpos;
	for(int i=0; i<nstreams; i++){
		beginpos.push_back(i * step);

		if(i == (nstreams-1)){
			endpos.push_back(atomNum-2);
		}else{
			endpos.push_back((i + 1) * step - 1);
		}
	}


	//----------------------------------------//
	// K partitioning : greedy
	//----------------------------------------//
	int colNum= atomNum -1;
	vector<int> workloads_per_col;
    for(int i=1; i<=colNum; i++){
        workloads_per_col.push_back(i);
    }

	int K = nstreams;
    vector<vector <int> > matrix;
    for(int i=0; i<K; i++){
    	vector<int> row;
    	matrix.push_back(row);
    }

    // initialize weight
    vector<int> weight(K,0);

    for(int i=0; i<colNum; i++)
	{
    	// find the min element in the weight
    	int min_pos = distance(weight.begin(), min_element(weight.begin(),weight.end()));

    	// assign the value from input array to matrix
    	int value = workloads_per_col.back();
    	workloads_per_col.pop_back();

    	// add the weight
    	weight[min_pos] += value;
    	matrix[min_pos].push_back(value);
    }

	// matrix contains the workloads(start position) for each stream

	// send these workloads to each cuda stream
	int *h_works = (int*) malloc(sizeof(int) * colNum);

	// device mem
	int *d_works;
	checkCudaErrors(cudaMalloc((void**)&d_works, sizeof(int) * colNum));

	vector<int> work_offset(K,0); // where the works start in the h_works array
	for(int i=0; i<K; i++){
		work_offset[i] = matrix[i].size();
	}

	// shift right 1 position
	for(int i=K-2; i>=0; i--)
		work_offset[i+1] = work_offset[i];
	work_offset[0] = 0;

	// histogram on offset postion
	for(int i=1; i<K; i++) {
		work_offset[i] += work_offset[i-1];	
	}

	// copy data from matrix to h_works
    for(int i=0; i<K; i++){
		for(int j=0; j<matrix[i].size(); j++){
			h_works[work_offset[i] + j] = matrix[i][j];	
		}
    }


	// bind texture to d_works
	cudaMemcpy(d_works, h_works, sizeof(int) * colNum, cudaMemcpyHostToDevice);

	// bind texture
	cudaChannelFormatDesc intDesc = cudaCreateChannelDesc<int>();
	checkCudaErrors(cudaBindTexture(NULL, wokTex, d_works, intDesc));

	cudaChannelFormatDesc floatDesc = cudaCreateChannelDesc<float>();
	checkCudaErrors(cudaBindTexture(NULL, dqTex, d_q, floatDesc));

	checkCudaErrors(cudaBindTexture(NULL, dfactorTex, d_factor, float4Desc));

#if TK
	cudaEventRecord(start, 0);
#endif


	for(int i=0; i<nstreams; i++)
	{
		calc_diffraction <<< grd_diffraction, blk_diffraction, sm_size, streams[i] >>> (work_offset[i], 
                                                                                        matrix[i].size(), 
                                                                                        lastpos, 
                                                                                        N, 
                                                                                        d_Iq, 
                                                                                        d_Iqz, 
                                                                                        i);
	}


#if TK
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("calc_diffraction = %f ms\n", elapsedTime);
#endif



	for(int i=0; i<nstreams; i++)
	{
		checkCudaErrors(cudaMemcpyAsync(&h_Iq[i * N], 
                                        &d_Iq[i * N],  
                                        sizeof(float) * N, 
                                        cudaMemcpyDeviceToHost, 
                                        streams[i]));

		checkCudaErrors(cudaMemcpyAsync(&h_Iqz[i * N],  
                                        &d_Iqz[i * N],  
                                        sizeof(float) * N, 
                                        cudaMemcpyDeviceToHost, 
                                        streams[i]));
	}

	cudaDeviceSynchronize();

	for(int i=0; i < N; i++){
		for(int s=1; s<nstreams; s++){
			h_Iq[i]  += h_Iq[i + s * N];	
			h_Iqz[i] += h_Iqz[i + s * N];	
		}
	}

	/* release resources */
	cudaUnbindTexture(crdTex);	
	cudaUnbindTexture(wokTex);	
	cudaUnbindTexture(dqTex);
	cudaUnbindTexture(dfactorTex);

	for (int i = 0 ; i < nstreams ; i++){
		cudaStreamDestroy(streams[i]);
	}
	free(streams);

	cudaFree(d_q);
	cudaFree(d_R);
	cudaFree(d_factor);
	cudaFree(d_Iq);
	cudaFree(d_Iqz);
	cudaFree(d_crd);
	cudaFree(d_works);

	cudaFreeHost(h_Iq);
	cudaFreeHost(h_Iqz);

	free(h_works);

#if DB
	free(q);
	free(R);
	free(factor);
#endif

	checkCudaErrors(cudaDeviceReset());

	exit (EXIT_SUCCESS);
}
