hopfield: main.cu hopfield.o
	nvcc -arch=sm_20 -o hopfield main.cu hopfield.o

hopfield.o: hopfield.cu hopfield.h
	nvcc -arch=sm_20 -dc hopfield.cu

clean:
	rm *.o
