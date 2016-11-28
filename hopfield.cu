//nvcc -arch=sm_20 hopfield.cu -o hopfield
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/types.h>
#include <cuda_runtime_api.h>


#define sizeGrid 65535
#define sizeBlok 1024
#define sizeWarp 32


__global__ void training(int dimP, int nP, int *ps, float *ws)
{
	extern __shared__ float s[];
	int x;
	x = blockIdx.x*blockDim.x + threadIdx.x;
	for (int i = 0; i < nP; i++)	
		s[x] += (float)((2*ps[i*dimP+(x/dimP)]-1)*(2*ps[i*dimP+(x%dimP)]-1));
	s[((x/dimP)*dimP)+(x/dimP)] = 0.0f;
	ws[x] = s[x]/nP;
}


__global__ void hopActivation(int dimP, float *ws, int *pt, int *at)
{
        extern __shared__ float sdata [];
        int tid = blockDim.x*blockIdx.x+threadIdx.x;
	int wpS = sizeWarp;
	if (dimP < 32)
		wpS = dimP; 
        int wid = tid / wpS;
        int lane= tid % wpS;
        if (wid < wpS ){
                int start_neuron = (wid*dimP);
                int end_neuron = ((wid+1)*dimP);
                sdata[threadIdx.x]=0;
                for(int i=start_neuron+lane;i<end_neuron;i+=32)
                        sdata[threadIdx.x]+= ws[i] * (2*pt[i % dimP ] -1);
		__syncthreads();
                if (lane + 16 < dimP) sdata[threadIdx.x] += sdata[threadIdx.x+16]; __syncthreads();
		if (lane +  8 < dimP) sdata[threadIdx.x] += sdata[threadIdx.x+ 8]; __syncthreads();
                if (lane +  4 < dimP) sdata[threadIdx.x] += sdata[threadIdx.x+ 4]; __syncthreads();
                if (lane +  2 < dimP) sdata[threadIdx.x] += sdata[threadIdx.x+ 2]; __syncthreads();
                if (lane +  1 < dimP) sdata[threadIdx.x] += sdata[threadIdx.x+ 1];
                if (lane == 0)
	        	at[wid] = ((sdata[threadIdx.x] > 0) - (sdata[threadIdx.x] < 0)+1)/2;
        }
}


float * lState (int nPatterns, int dimPattern, int *patterns)
{
	int *ps;
	float *weights, *ws;
	int sizeP = dimPattern*sizeof(int);
	int sizeW = dimPattern*dimPattern*sizeof(float);

	if ((weights = (float*) malloc (sizeW)) == NULL ) return NULL;
	if ( cudaSuccess != cudaMalloc ( &ps, (sizeP*nPatterns))) return NULL;
	if ( cudaSuccess != cudaMalloc ( &ws, (sizeW))) return NULL;
	if ( cudaSuccess != cudaMemcpy (ps, patterns, sizeP*nPatterns, cudaMemcpyHostToDevice)) return NULL;

	dim3 GRID_DIM (1);
	dim3 BLOCK_DIM (dimPattern*dimPattern);
	training<<< GRID_DIM, BLOCK_DIM, dimPattern*dimPattern*sizeof(float) >>> (dimPattern, nPatterns, ps, ws);
  
	if (cudaSuccess != cudaMemcpy (weights, ws, sizeW, cudaMemcpyDeviceToHost)) return NULL;
   	return weights;
}


int * actFunc(int dP, int *pattern, float *weight)
{
	float *ws;
	int *pt, *activation, *at;
	if ( (activation = (int *) malloc (dP*sizeof(int))) == NULL) return NULL;
	if (cudaSuccess != cudaMalloc (&ws, dP*dP*sizeof(float))) return NULL;
	if (cudaSuccess != cudaMalloc (&pt, dP*sizeof(int))) return NULL;
	if (cudaSuccess != cudaMalloc (&at, dP*sizeof(int))) return NULL;
	if ( cudaSuccess != cudaMemcpy (ws, weight, dP*dP*sizeof(float), cudaMemcpyHostToDevice)) return NULL;
	if ( cudaSuccess != cudaMemcpy (pt, pattern, dP*sizeof(int), cudaMemcpyHostToDevice)) return NULL;

	dim3 GRID_DIM (1);
	dim3 BLOCK_DIM (dP*dP);
	hopActivation<<< GRID_DIM, BLOCK_DIM, dP*dP*sizeof(float) >>> (dP, ws, pt, at);

  	if (cudaSuccess != cudaMemcpy (activation, at, dP*sizeof(int), cudaMemcpyDeviceToHost)) return NULL;
	return activation;
}
