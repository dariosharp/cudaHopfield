#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "hopfield.h"

void checkVal(float *matrix, int dimP, int nP, int nC)
{
	printf("In check val\n");
	/*
	int * epat = (int *)malloc (dimPattern*sizeof(int));
        epat[0] = 1;
        epat[1] = 0;
        epat[2] = 1;
        epat[3] = 0;
        epat[4] = 1;
        epat[5] = 1;
        epat[6] = 0;

        int * activation = actFunc(dimPattern, epat, weights);
        if (activation == NULL){
                printf("Error on Activarion\n");
                return 1;
        }
        printf("activation [");
        for (int i = 0; i < dimPattern; i++)
                printf("%i ", activation[i]);
        printf("]\n");
	*/
}


float * randomValue(int nPatterns, int dimPattern)
{
      	int * patterns;
	int i, j;
        if ((patterns = (int*) malloc (dimPattern*nPatterns*sizeof(int))) == NULL ) return NULL;
	srand(time(NULL));
        for (i = 0; i < nPatterns*dimPattern; i++) {
                patterns[i] = rand() % 2;
        }
        for (j = 0; j < nPatterns; j++){
                printf("[ ");
                for (i = 0; i < dimPattern; i++) {
                        printf("%d ", patterns[j*dimPattern + i]);
                }
                 printf("]\n");
        }

        float * weights = lState(nPatterns, dimPattern, patterns);
        if (weights == NULL){
                printf("Error on Learning\n");
                return NULL;
        }

        printf("Weights:\n");
        for(i = 0; i < dimPattern; i++){
                printf("[ ");
                for (j = 0; j < dimPattern; j++) {
                        printf("%.3f ", weights[i*dimPattern+j]);
                }
                printf("]\n");
        }
	return weights;
}


int main(int argc, char *argv[])
{
        int i, k;
        int dimP, nP, nC, type = 0;

        if (argc < 2){
		printf("	-pf [patterns file name] -rf [patterns recognize]					Get pattern from File\n");
		printf("	--inline -dimP [size one pattern] -nP [number of patterns] -nC [numbre of check]	Generate random Patterns\n");
		printf("	-h											Help\n");
		return 1;	
	}

	for (i = 1; i < argc; i++){
		if(strcmp(argv[i], "-pf")==0){
			printf("Pattens file name: %s\n", argv[++i]);		
		}
		if(strcmp(argv[i], "-rf")==0){
			printf("Pattenrs recognize file name: %s\n", argv[++i]);		
		}
		if(strcmp(argv[i], "--inline")==0){
			type += 0x10;
			for(k = 1; k <= 5; k++){
				if(strcmp(argv[i+k], "-dimP")==0){
					dimP = atoi(argv[i+(++k)]);
				}
				if(strcmp(argv[i+k], "-nP")==0){
					nP = atoi(argv[i+(++k)]);		
				}
				if(strcmp(argv[i+k], "-nC")==0){
					nC = atoi(argv[i+(++k)]);		
				}
			}
			i +=k; 
		}
	}

	if (type & 0x10){
		float * matrix = randomValue(nP, dimP);
		if(matrix == NULL)
			return 1;
		checkVal(matrix, dimP, nP, nC);
	}

	return 0;
}

