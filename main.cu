#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <regex.h>

#include "hopfield.h"

#define CICLI 5
#define CHECKARG(arg, val) if(strcmp((arg), (val))==0)
#define PRINT_HELP	printf("\t--file        Get pattern from File\n");\
                	printf("\t--inline      Generate random Patterns\n");\
 			printf("\t-v            verbose mode\n");\
			printf("\t\t-pf [patterns file name]\n");\
                	printf("\t\t-rf [patterns recognize]\n");\
                	printf("\t\t-dimP [size of pattern]\n");\
                	printf("\t\t-nP [number of pattern]\n");\
                	printf("\t-v            verbose mode\n");


int checkVal(float *weights, int dimPattern, int nPatterns)
{
	int i = 0;
	int * epat = (int *)malloc (dimPattern*sizeof(int));
        
	srand(time(NULL));
	printf("Input:\t\t[");
        for (i = 0; i < dimPattern; i++) {
                epat[i] = rand() % 2;
		printf(" %d", epat[i]);
        }
	printf(" ]\n");

	for (i = 0; i < CICLI; i++)
        	epat = actFunc(dimPattern, epat, weights);
        if (epat == NULL) {
                printf("Error on Activarion\n");
                return 1;
        }

        printf("Activation:\t[ ");
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
	printf("Pattern Generated: \n");
        for (j = 0; j < nPatterns; j++) {
                printf("\t[ ");
                for (i = 0; i < dimPattern; i++) {
                        printf("%d ", patterns[j*dimPattern + i]);
                }
                 printf("]\n");
        }

        float * weights = lState(nPatterns, dimPattern, patterns);
        if (weights == NULL) {
                printf("Error on Learning\n");
                return NULL;
        }
	
	if (verbose_mode) {
        	printf("Weights:\n");
        	for (i = 0; i < dimPattern; i++) {
                	printf("[ ");
                	for (j = 0; j < dimPattern; j++) {
                        	printf("%.3f ", weights[i*dimPattern+j]);
                	}
               		printf("]\n");
        	}	
	}
	return weights;
}


int parserFile(char * path, int * pts)
{	
	FILE* fp;
	char buffer[255];
	const char *p;
 	regex_t re;
	regmatch_t match;

	fp = fopen(path, "r");
	if (fp == NULL) return 1;
   	if (regcomp(&re, "[0-1]", REG_EXTENDED) != 0) return 1;
 
	int i = 0;
	while (fgets(buffer, 255, (FILE*) fp)) {
		p = buffer;
     		while (regexec(&re, p, 1, &match, 0) == 0) {
        		pts[i] = p[match.rm_so] - '0';
        		p += match.rm_eo;
			i++;
    		}
	}

	fclose(fp);
    	regfree(&re);
	return 0;
}


int main(int argc, char *argv[])
{
        int i;
        int dimP = 0, nP = 0, type = 0;
	int *patterns, *recognize;
	char *pathPf = NULL, *pathRf = NULL;

        if (argc < 2) {
		PRINT_HELP;
		exit(1);	
	}
	
	for (i = 1; i < argc; i++) {
                CHECKARG(argv[i], "-v")
                        verbose_mode = 1;
		CHECKARG(argv[i], "--file")
			type = 0x1;
		CHECKARG(argv[i], "-pf")
			pathPf = argv[i+1];
		CHECKARG(argv[i], "-rf")
			pathRf = argv[i+1];
		CHECKARG(argv[i], "--inline")
			type = 0x10;
		CHECKARG(argv[i], "-dimP")
			dimP = atoi(argv[i+1]);
		CHECKARG(argv[i], "-nP")
			nP = atoi(argv[i+1]);		
	}

	if (type & 0x1) {		
		if (pathPf != NULL ||pathRf != NULL || dimP != 0 || nP != 0) {
			patterns = (int *) malloc(dimP*nP*sizeof(int));
			if (parserFile(pathPf, patterns)) {
				printf("Error in parse file\n");
				exit(1);
			}
			recognize = (int *) malloc(dimP*sizeof(int));
			if (parserFile(pathRf, recognize)) {
				printf("Error in parse file\n");
				exit(1);
			}				
		}
		else {
			PRINT_HELP;
			exit(1);
		}
	}

	if (type & 0x10) {
		float * matrix = randomValue(nP, dimP);
		if (matrix == NULL)
			return 1;
		checkVal(matrix, dimP, nP);
		free(matrix);
	}

	return 0;
}
