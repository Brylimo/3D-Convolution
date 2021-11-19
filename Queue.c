#include <stdio.h>
#include <stdlib.h>
#include "Queue.h"

void queueInit(Queue* que)
{
	que->front = NULL;
	que->rear = NULL;
}

int IsQueueEmpty(Queue* que)
{
	if (que->front == NULL)
		return TRUE;
	else
		return FALSE;
}

void enqueue(Queue* que, Data* data)
{
	Node* newNode = (Node*)malloc(sizeof(Node));
	newNode->data = data;
	newNode->next = NULL;

	if (IsQueueEmpty(que))
	{
		que->front = newNode;
		que->rear = newNode;
	} else { 
		que->rear->next = newNode;
		que->rear = newNode;
	} 
}

Data* dequeue(Queue* que)
{
	if (IsQueueEmpty(que))
	{
		exit(-1);
	}

	Node* temp = que->front;
	Data* tempData = temp->data;
	que->front = que->front->next;

	free(temp);
	return tempData;
}

Data* queuePeek(Queue* que)
{
	if (IsQueueEmpty(que))
	{
		exit(-1);
	}
	return que->front->data;	
}
