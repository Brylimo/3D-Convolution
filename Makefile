
.PHONY: all clean

all : singlethread multithread gpu

singlethread: singlethread.c
	gcc -mavx2 -o singlethread singlethread.c

multithread: multithread.c Queue.c
	gcc -mavx2 -o multithread multithread.c Queue.c -pthread

gpu : ji_back.cu
	nvcc ji_back.cu -o gpu
clean :
	rm singlethread
	rm multithread
	rm gpu
