
.PHONY: all clean

all : singlethread multithread

singlethread: singlethread.c
	gcc -mavx2 -o singlethread singlethread.c

multithread: multithread.c Queue.c
	gcc -mavx2 -o multithread multithread.c Queue.c -pthread

clean :
	rm singlethread
	rm multithread
