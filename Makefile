
.PHONY: all clean

all : singlethread multithread

singlethread: singlethread.c
	gcc -mavx2 -o singlethread singlethread.c

multithread: multithread.c
	gcc -mavx2 -o multithread multithread.c

clean :
	rm singlethread
	rm multithread
