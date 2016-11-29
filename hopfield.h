#ifndef hopfiel_h
#define hopfiel_h

static int verbose_mode = 0;

void training(int dimP, int nP, int *ps, float *ws);
void hopActivation(int dimP, float *ws, int *pt, int *at);
float * lState (int nPatterns, int dimPattern, int *patterns);
int * actFunc(int dP, int *pattern, float *weight);


#endif
