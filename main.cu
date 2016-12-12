#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <regex.h>

#include "hopfield.h"

#define CICLI 32
#define CHECKARG(arg, val) if(strcmp((arg), (val))==0)
#define PRINT_HELP	printf("\t--file        Get pattern from File\n");\
                	printf("\t--inline      Generate random Patterns\n");\
			printf("\t\t-pf [patterns file name]\n");\
                	printf("\t\t-rf [patterns recognize]\n");\
                	printf("\t\t-dimP [size of pattern]\n");\
                	printf("\t\t-nP [number of pattern]\n");\
 			printf("\t-v            verbose mode\n");
	

void print_weights(float *weights, int dimP)
{
	int i, j;
	printf("Weights:\n");
        for (i = 0; i < dimP; i++) {
        	printf("[ ");
               	for (j = 0; j < dimP; j++) {
               		printf("%.3f ", weights[i*dimP+j]);
                }
	printf("]\n");
        }
}


int checkVal(float *weights, int * epat, int dimPattern, int nPatterns)
{
	int i = 0;
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

	return 0;
}


float * randomValue(int nPatterns, int dimPattern)
{
      	int * patterns;
	int i, j;
        time_t t;
	if ((patterns = (int*) malloc (dimPattern*nPatterns*sizeof(int))) == NULL ) return NULL;

	srand((unsigned) time(&t));
        for (i = 0; i < nPatterns*dimPattern; i++) {
                patterns[i] = rand() % 2;
        }
	printf("Patterns Generated: \n");
        for (j = 0; j < nPatterns; j++) {
                printf("\t\t[ ");
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
        	print_weights(weights, dimPattern);
	}
	free(patterns);
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
			float * weights = lState(nP, dimP, patterns);
        		if (weights == NULL) {
                		printf("Error on Learning\n");
                		exit(1);
        		}
			if (verbose_mode) {
        			print_weights(weights, dimP);
			}

			recognize = (int *) malloc(dimP*sizeof(int));
			if (parserFile(pathRf, recognize)) {
				printf("Error in parse file\n");
				exit(1);
			}
			checkVal(weights, recognize, dimP, nP);
			free(weights);
			free(patterns);
			free(recognize);				
		}
		else {
			PRINT_HELP;
			exit(1);
		}
	}

	if (type & 0x10) {
		if (dimP != 0 || nP !=0){
			float * weights = randomValue(nP, dimP);
			if (weights == NULL){
				printf("Error on Learning");
			 	exit(1);
			}

	        	int * recognize = (int *)malloc (dimP*sizeof(int));        
        		srand((unsigned) *recognize);
       			printf("Input:\t\t[");
        		for (i = 0; i < dimP; i++) {
                		recognize[i] = rand() % 2;
               			printf(" %d", recognize[i]);
        		}
        		printf(" ]\n");
			checkVal(weights, recognize, dimP, nP);
			free(weights);
			free(recognize);
		}
		else {
			PRINT_HELP;
			exit(1);
		}
	}

	return 0;
}
