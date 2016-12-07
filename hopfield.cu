#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/types.h>
#include <cuda_runtime_api.h>

#define sizeGrid 65535
#define sizeBlock 1024
#define sizeWarp 32


__global__ void training(int dimP, int nP, int *ps, float *ws)
{
	float product;
	int x;
	x = blockIdx.x*blockDim.x + threadIdx.x;
	if (x < dimP*dimP){
		product = 0.0f;
		for (int i = 0; i < nP; i++)
			product += (float)((2*ps[i*dimP+(x/dimP)]-1)*(2*ps[i*dimP+(x%dimP)]-1));
		product = (!((((x/dimP)*dimP)+(x/dimP)) == x)) * product;
		ws[x] = product/nP;
	}
}


__global__ void hopActivation(int dimP, float *ws, int *pt, int *at)
{
        extern __shared__ float sdata [];
        int tid = blockDim.x*blockIdx.x+threadIdx.x; 
        int wid  = tid / sizeWarp;
        int lane = tid % sizeWarp;
        if (wid < dimP ){
                int start_neuron = (wid*dimP);
                int end_neuron = ((wid+1)*dimP);
                sdata[threadIdx.x]=0;
                for(int i=start_neuron+lane;i<end_neuron;i+=32)
                        sdata[threadIdx.x]+= ws[i] * (2*pt[i % dimP ] -1);
		__syncthreads();
                if (lane < 16) sdata[threadIdx.x] += sdata[threadIdx.x+16]; __syncthreads();
		if (lane <  8) sdata[threadIdx.x] += sdata[threadIdx.x+ 8]; __syncthreads();
                if (lane <  4) sdata[threadIdx.x] += sdata[threadIdx.x+ 4]; __syncthreads();
                if (lane <  2) sdata[threadIdx.x] += sdata[threadIdx.x+ 2]; __syncthreads();
                if (lane <  1) sdata[threadIdx.x] += sdata[threadIdx.x+ 1];
                if (lane == 0)
	        	at[wid] = ((sdata[threadIdx.x] > 0) - (sdata[threadIdx.x] < 0)+1)/2;
        }
}


float * lState (int nPatterns, int dimPattern, int *patterns)
{
	int *ps;
	float *weights, *ws;
	int sizeP = dimPattern*sizeof(int);
	int sizeW = dimPattern*dimPattern;

	if ((weights = (float*) malloc (sizeW*sizeof(float))) == NULL ) return NULL;
	if ( cudaSuccess != cudaMalloc ( &ps, (sizeP*nPatterns))) return NULL;
	if ( cudaSuccess != cudaMalloc ( &ws, (sizeW*sizeof(float)))) return NULL;
	if ( cudaSuccess != cudaMemcpy (ps, patterns, sizeP*nPatterns, cudaMemcpyHostToDevice)) return NULL;

	dim3 GRID_DIM ((sizeW+sizeBlock-1)/sizeBlock);
	dim3 BLOCK_DIM (sizeBlock);
	training<<< GRID_DIM, BLOCK_DIM, (sizeBlock)*sizeof(float) >>> (dimPattern, nPatterns, ps, ws);
  
	if (cudaSuccess != cudaMemcpy (weights, ws, sizeW*sizeof(float), cudaMemcpyDeviceToHost)) return NULL;
	cudaFree(ps);
	cudaFree(ws);
   	return weights;
}


int * actFunc(int dimP, int *pattern, float *weight)
{
	float *ws;
	int *pt, *activation, *at;
	if ( (activation = (int *) malloc (dimP*sizeof(int))) == NULL) return NULL;
	if (cudaSuccess != cudaMalloc (&ws, dimP*dimP*sizeof(float))) return NULL;
	if (cudaSuccess != cudaMalloc (&pt, dimP*sizeof(int))) return NULL;
	if (cudaSuccess != cudaMalloc (&at, dimP*sizeof(int))) return NULL;
	if ( cudaSuccess != cudaMemcpy (ws, weight, dimP*dimP*sizeof(float), cudaMemcpyHostToDevice)) return NULL;
	if ( cudaSuccess != cudaMemcpy (pt, pattern, dimP*sizeof(int), cudaMemcpyHostToDevice)) return NULL;

	dim3 GRID_DIM (((dimP*32)+sizeBlock-1)/sizeBlock);
	dim3 BLOCK_DIM (sizeBlock);
	hopActivation<<< GRID_DIM, BLOCK_DIM, sizeBlock*sizeof(float) >>> (dimP, ws, pt, at);

  	if (cudaSuccess != cudaMemcpy (activation, at, dimP*sizeof(int), cudaMemcpyDeviceToHost)) return NULL;
	cudaFree(ws);
	cudaFree(pt);
	cudaFree(at);
	return activation;
}
