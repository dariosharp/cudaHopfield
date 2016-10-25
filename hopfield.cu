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
	int x;
	x = blockIdx.x*blockDim.x + threadIdx.x;
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
	cudaMalloc ( (void**) &ps, (sizeP*nPatterns));
	cudaMalloc ( (float**) &ws, (sizeW));
      
	for (int i = 0; i < nPatterns*dimPatterns; i++) {
		patterns[i] = rand() % 2; 
	}
	cudaMemcpy (ps, patterns, sizeP, cudaMemcpyHostToDevice);

	for (int j = 0; j < nPatterns; j++){
		printf("[ ");
		for (int i = 0; i < dimPatterns; i++) {
			printf("%d ", patterns[j*dimPatterns + i]);
		}
		 printf("]\n");
	}
 
	dim3 GRID_DIM ((int)((dimPatterns*dimPatterns)/sizeGrid)+1);
	dim3 BLOCK_DIM (dimPatterns*dimPatterns);
	training<<< GRID_DIM, BLOCK_DIM >>> (dimPatterns, nPatterns, ps, weights);
/*
//   cudaThreadSynchronize();
   
//   cudaMemcpy (sol, devSol, sizeW, cudaMemcpyDeviceToHost);	
   printf("C:\n");
   for(int i = 0; i < N; i++){
      printf("[ ");
      for (int j = 0; j < N; j++) {
         printf("%.3f ", sol[i*N+j]);
      }
      printf("]\n"); 
   }
*/ 
}
