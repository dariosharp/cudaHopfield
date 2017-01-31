# cudaHopfield

The aim of this Project is study how to implement a Neural Network algorithm in GPU architecture. The code is PoC about Hopfield Network developed in CUDA C referenced from https://dumas.ccsd.cnrs.fr/dumas-00636458/document.

## How to start
There is a makefile to compile the code:
```
$ cd cudaHopfield
$ make
```
Args setting keys are:
```
$ ./hopefield
	  --file        Get pattern from File
	  --inline      Generate random Patterns
		  -pf [patterns file name]
		  -rf [patterns recognize]
		  -dimP [size of pattern]
		  -nP [number of pattern]
	  -v            verbose mode
```

It's possible execute tests using files:
```
$ ./hopfield --file -pf pattern.txt -rf test.txt -dimP 1024 -nP 2
```
or using random values just for fan:
```
$ ./hopfield --inline -dimP 1024 -nP 2
```

####Hardware Project based
>Device: Tesla M2090
>
>GPU Clock rate: 1,30 Ghz
>
>Global Memory: 6 GByte
>
>Shared Memory per Block: 50 KBytes
>
>Warp size: 32
>
>Threads per Block: 1024
>
>Dimension Grid: 65535 x 65535 x 65535


