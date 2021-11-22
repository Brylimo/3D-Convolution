#ifndef Queue_h
#define Queue_h

#define TRUE 1
#define FALSE 0
#define BLOCK_LENGTH 18

typedef struct _block
{
        int start_col[BLOCK_LENGTH];
        int start_row[BLOCK_LENGTH];
        int start_dep[BLOCK_LENGTH];
        int num[BLOCK_LENGTH];
	int pluto;
} Block;

typedef Block Data;
typedef struct _node
{
	Data* data;
	struct _node* next;
}Node;
typedef struct _queue
{
	Node* front;
	Node* rear;
}Queue;

void queueInit(Queue* que);
int IsQueueEmpty(Queue* que);
int IsQueueFull(Queue* que);
void enqueue(Queue* que, Data* data);
Data* dequeue(Queue* que);
Data* queuePeek(Queue* que);

#endif /* Queue_h */
