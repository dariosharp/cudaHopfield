//nvcc -arch=sm_20 hopfield.cu -o hopfield
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/types.h>
#include <cuda_runtime_api.h>

#define sizeGrid 65535
#define sizeBlok 1024
#define sizeWarp 32


__global__ void training(int dimP, int nP, int *ps, float *ws){
	extern __shared__ float s[];
	int x;
	x = blockIdx.x*blockDim.x + threadIdx.x;
	for (int i = 0; i < nP; i++)	
		s[x] += (float)((2*ps[i*dimP+(x/dimP)]-1)*(2*ps[i*dimP+(x%dimP)]-1));
	s[((x/dimP)*dimP)+(x/dimP)] = 0.0f;
	//__syncthreads();
	ws[x] = s[x]/nP;
}


__global__ void hopActivation(int dimP, float *ws, int *pt, float *at){
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	float  product = 0;
	for (int i = 0; i < dimP; i++)
		product += ws[(x*dimP)+i] * pt[i];
	at[x] = product;
}


/*__device__ __forceinline__ double sigmoid (double a)
{
    return 1.0 / (1.0 + exp (-a));
}


__global__ void sigmoid_kernel (const double * __restrict__ src, 
                                double * __restrict__ dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        dst[i] = sigmoid (src[i]);
    }
}   
*/


float * lState (int nPatterns, int dimPattern, int *patterns){
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


float * actFunc(int dP, int *pattern, float *weight){
	float *ws, *at, *activation;
	int *pt;
	if ( (activation = (float *) malloc (dP*sizeof(float))) == NULL) return NULL;
	if (cudaSuccess != cudaMalloc (&ws, dP*dP*sizeof(float))) return NULL;
	if (cudaSuccess != cudaMalloc (&pt, dP*sizeof(int))) return NULL;
	if (cudaSuccess != cudaMalloc (&at, dP*sizeof(float))) return NULL;
	if ( cudaSuccess != cudaMemcpy (ws, weight, dP*dP*sizeof(float), cudaMemcpyHostToDevice)) return NULL;
	if ( cudaSuccess != cudaMemcpy (pt, pattern, dP*sizeof(int), cudaMemcpyHostToDevice)) return NULL;


	dim3 GRID_DIM (1);
	dim3 BLOCK_DIM (dP);
	
	hopActivation<<< GRID_DIM, BLOCK_DIM >>> (dP, ws, pt, at);
  	if (cudaSuccess != cudaMemcpy (activation, at, dP, cudaMemcpyDeviceToHost)) return NULL;
   	
	return activation;
	
}


int main(int argc, char *argv[]){
	int nPatterns, dimPattern;
	int * patterns;

	nPatterns = 3;
	dimPattern = 7;
	if ((patterns = (int*) malloc (dimPattern*nPatterns*sizeof(int))) == NULL ) return 1;

	for (int i = 0; i < nPatterns*dimPattern; i++) {
		patterns[i] = rand() % 2; 
	}
	for (int j = 0; j < nPatterns; j++){
		printf("[ ");
		for (int i = 0; i < dimPattern; i++) {
			printf("%d ", patterns[j*dimPattern + i]);
		}
		 printf("]\n");
	}

	float * weights = lState(nPatterns, dimPattern, patterns);
	if (weights == NULL){
		printf("Error on Learning\n");
		return 1;
	}

	printf("Weights:\n");
   	for(int i = 0; i < dimPattern; i++){
      		printf("[ ");
      		for (int j = 0; j < dimPattern; j++) {
         		printf("%.3f ", weights[i*dimPattern+j]);
      		}
      		printf("]\n"); 
   	}

	float * activation = actFunc(dimPattern, patterns, weights);
	if (activation == NULL){
		printf("Error on Activarion\n");
		return 1;
	}
	printf("activation [");
	for (int i = 0; i < dimPattern; i++)
		printf("%.3f ", activation[i]);
	printf("]\n");
}
