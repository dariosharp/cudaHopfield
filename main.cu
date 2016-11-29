#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include "hopfield.h"

int checkVal(float *weights, int dimPattern, int nPatterns)
{
	int i = 0;
	int * epat = (int *)malloc (dimPattern*sizeof(int));
        
	srand(time(NULL));
	printf("Input: [");
        for (i = 0; i < dimPattern; i++) {
                epat[i] = rand() % 2;
		printf(" %d", epat[i]);
        }
	printf(" ]\n");

	for (i = 0; i < 60; i++)
        	epat = actFunc(dimPattern, epat, weights);
        if (epat == NULL){
                printf("Error on Activarion\n");
                return 1;
        }

        printf("Activation: [");
        for (i = 0; i < dimPattern; i++)
                printf("%i ", epat[i]);
        printf("]\n");

	free(epat);
	return 0;
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

int parserFile(char * path, int patters)
{
	return 0;
	/*
	FILE* fp;
	char buffer[255];

	fp = fopen("path, "r");

	while(fgets(buffer, 255, (FILE*) fp)) {
    		
	}

	fclose(fp);
	return dimP;
	*/
	
}

int main(int argc, char *argv[])
{
        int i, k;
        int dimP, nP, type = 0;

        if (argc < 2){
		printf("	-pf [patterns file name] -rf [patterns recognize]		Get pattern from File\n");
		printf("	--inline -dimP [size one pattern] -nP [number of patterns]	Generate random Patterns\n");
		printf("	-h								Help\n");
		return 1;	
	}

	for (i = 1; i < argc; i++){
		if(strcmp(argv[i], "-pf")==0){
			type += 0x1;
			parserFile(argv[++i], 2);		
		}
		if(strcmp(argv[i], "-rf")==0){
			parserFile(argv[++i], 2);		
		}
		if(strcmp(argv[i], "--inline")==0){
			type += 0x10;
			for(k = 1; k <= 3; k++){
				if(strcmp(argv[i+k], "-dimP")==0){
					dimP = atoi(argv[i+(++k)]);
				}
				if(strcmp(argv[i+k], "-nP")==0){
					nP = atoi(argv[i+(++k)]);		
				}
			}
			i +=k; 
		}
	}

	if (type & 0x10){
		float * matrix = randomValue(nP, dimP);
		if(matrix == NULL)
			return 1;
		checkVal(matrix, dimP, nP);
		free(matrix);
	}

	return 0;
}

