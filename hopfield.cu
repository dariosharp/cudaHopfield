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
		s[x] += (float)(ps[i*dimP+(x/dimP)]*ps[i*dimP+(x%dimP)]);
	s[((x/dimP)*dimP)+(x/dimP)] = 0.0f;
	__syncthreads();
	ws[x] = s[x]/nP;
}


int main(int argc, char *argv[]){
	int *patterns, *ps;
	float *ws, *weights;
	int nPatterns, dimPatterns;
	//printf("Insert number of patterns: ");
	//scanf("%d", &num);
	nPatterns = 2;
	dimPatterns = 4;

	int sizeP = dimPatterns*sizeof(int);
	int sizeW = dimPatterns*dimPatterns*sizeof(float);
	if ((patterns = (int*) malloc (sizeP*nPatterns)) == NULL ) return 1;
	if ((weights = (float*) malloc (sizeW)) == NULL ) return 1;
	cudaMalloc ( &ps, (sizeP*nPatterns));
	cudaMalloc ( &ws, (sizeW));
      
	for (int i = 0; i < nPatterns*dimPatterns; i++) {
		patterns[i] = rand() % 2 == 0 ? -1 : 1; 
	}
	cudaMemcpy (ps, patterns, sizeP*nPatterns, cudaMemcpyHostToDevice);

	for (int j = 0; j < nPatterns; j++){
		printf("[ ");
		for (int i = 0; i < dimPatterns; i++) {
			printf("%d ", patterns[j*dimPatterns + i]);
		}
		 printf("]\n");
	}
 
	dim3 GRID_DIM (1);//(int)((dimPatterns*dimPatterns)/sizeGrid)+1);
	dim3 BLOCK_DIM (dimPatterns*dimPatterns);
	training<<< GRID_DIM, BLOCK_DIM, dimPatterns*dimPatterns*sizeof(float) >>> (dimPatterns, nPatterns, ps, ws);
	cudaThreadSynchronize();
  
	cudaMemcpy (weights, ws, sizeW, cudaMemcpyDeviceToHost);	
   	printf("C:\n");
   	for(int i = 0; i < dimPatterns; i++){
      		printf("[ ");
      		for (int j = 0; j < dimPatterns; j++) {
         		printf("%.3f ", weights[i*dimPatterns+j]);
      		}
      		printf("]\n"); 
   	}	
}
