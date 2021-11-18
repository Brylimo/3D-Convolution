
.PHONY: all clean

all : singlethread

singlethread: singlethread.c
	gcc -mavx2 -o singlethread singlethread.c

clean :
	rm singlethread
