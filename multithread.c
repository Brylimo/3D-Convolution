#include <stdio.h> 
#include <pthread.h>
#include <immintrin.h>
#include <stdint.h>
#include <x86intrin.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/time.h>
#include "Queue.h" 

#define CHECK(CALL)\
  if ((CALL) < 0) {perror (#CALL); exit (EXIT_FAILURE);}
#define ARRAY_LENGTH 8
#define ABS(X) ((X) < 0 ? -(X) : (X))
#define E 0.0001
#define BLOCK_SIZE 18

void* workerThread(void *arg);

typedef struct __container
{
	int zp, yp, xp; // real
	int z, y, x;    // with padding
	int padding;
	float *matrix;
} Container;

typedef struct __kernel
{
	int size;
	float *matrix;
} Kernel;

typedef struct __threadpool {
	int alive;
	int num_threads;
	pthread_t* pool;
} ThreadPool; 

Container in, out;
Kernel ker;
Queue buffer;
pthread_mutex_t lock;
pthread_mutex_t lock1;
pthread_mutex_t lock2;
pthread_mutex_t lock3;
pthread_mutex_t lock4;
pthread_cond_t signal;
float* ans;
int count = 0;
struct timeval start, stop;

int f_equal(float x, float y) // 부동소수점 근사값 계산 함수
{
	if (ABS(x - y) < E)
	{
		return 1;
	} else {
		return 0;
	}
}

void container_constructor(Container *container, int padd, int z, int y, int x)
{
	int size = (z+padd*2) * (y+padd*2) * (x+padd*2);
	
	container->padding = padd;
	container->z = z + padd*2;
	container->y = y + padd*2;
	container->x = x + padd*2;
	container->zp = z;
	container->yp = y;
	container->xp = x;
	container->matrix = (float*)malloc(sizeof(float)*size);
}

void container_destructor(Container *container)
{
	free(container->matrix);
}

void kernel_constructor(Kernel *kernel, int size)
{
	kernel->size = size;
	kernel->matrix = (float*)malloc(sizeof(float)*size*size*size);
}

void kernel_destructor(Kernel *kernel)
{
	free(kernel->matrix);
}

ThreadPool* thread_pool_constructor(int num_threads)
{
        ThreadPool* thread_pool = (ThreadPool *)malloc(sizeof(ThreadPool));
        thread_pool->num_threads = num_threads;
	thread_pool->alive = 0;
        thread_pool->pool = (pthread_t*)malloc(sizeof(pthread_t[num_threads]));

        for (int i = 0; i < num_threads; i++)
        {
		pthread_create(&thread_pool->pool[i], NULL, workerThread, (void*)thread_pool);
        }
        return thread_pool;
}

void thread_pool_destructor(ThreadPool * thread_pool)
{
	thread_pool->alive = 1;
	pthread_cond_signal(&signal);

        for (int i = 0; i < thread_pool->num_threads; i++)
        {
                int rc = pthread_join(thread_pool->pool[i], NULL);
                if (rc) {
                        printf("Error; return code from pthread_join() is %d\n", rc);
                        exit(-1);
                }
        }
        free(thread_pool->pool);
}

void multiThread()
{
	int total = in.xp * in.yp * in.zp, t, p, rc, start_row, start_col, start_dep;

	CHECK(gettimeofday(&start, NULL));
	for (t=0; t < total; t+=BLOCK_SIZE)
	{
		Block* block = (Block*)malloc(sizeof(Block));
		for (p = 0; p < BLOCK_SIZE; p++)
		{
			if (t+p < total) {
				block->start_row[p] = (t+p) % in.xp;
				block->start_col[p] = ((t+p) % (in.xp*in.yp)) / in.xp;
				block->start_dep[p] = (t+p) / (in.xp*in.yp);
				block->num[p] = t+p;
				block->pluto += 1;
			} else {
				break;
			}
		}

		pthread_mutex_lock(&lock3);
		if (IsQueueEmpty(&buffer))
		{
			enqueue(&buffer, block);
			pthread_cond_signal(&signal);
		} else {
			pthread_mutex_lock(&lock);
			enqueue(&buffer, block);
			pthread_mutex_unlock(&lock);
		}
		pthread_mutex_unlock(&lock3);
	}

	while (count != total) {};

	CHECK(gettimeofday(&stop, NULL));
}

void* workerThread(void *arg)
{
	ThreadPool* thread_pool = (ThreadPool*)arg;
	Block* block;
	
	while (!thread_pool->alive)
	{
		pthread_mutex_lock(&lock1);
		if (IsQueueEmpty(&buffer) && !thread_pool->alive) {
			pthread_cond_wait(&signal, &lock2);
			if (!thread_pool->alive)
				block = dequeue(&buffer);
		} else {
			pthread_mutex_lock(&lock);
			block = dequeue(&buffer);
			pthread_mutex_unlock(&lock);
		}
		pthread_mutex_unlock(&lock1);

		if (!thread_pool->alive) {
			int l, m, k, size, height=in.yp, width=in.xp, depth=in.zp;
			for (int z=0;z<block->pluto;z++) {
				float* vector_in = aligned_alloc(32, sizeof(float)*ARRAY_LENGTH);
				float* vector_ker = aligned_alloc(32, sizeof(float)*ARRAY_LENGTH);
				float* vsum = aligned_alloc(32, sizeof(float)*ARRAY_LENGTH);
				float zero = 0.000000, sum = 0.000000;
				__m256 result = _mm256_broadcast_ss(&zero);

				if (ker.size > ARRAY_LENGTH)	size = ARRAY_LENGTH;	
				else	size = ker.size;

				for(l=0;l<ker.size;l++)
				{
					for(m=0;m<ker.size;m++)
					{
						if (size == ARRAY_LENGTH) {
							int sub = ker.size;
							for(k=0;(sub=(sub-ARRAY_LENGTH))>0;k++)
							{
								memcpy(vector_in, &in.matrix[((block->start_dep[z]+l)*in.y*in.x)+((block->start_col[z]+m)*in.x)+block->start_row[z]+k*ARRAY_LENGTH], sizeof(float)*size);
								memcpy(vector_ker, &ker.matrix[l*ker.size*ker.size+m*ker.size+k*ARRAY_LENGTH], sizeof(float)*size);
								__m256 vin = _mm256_load_ps(vector_in);
								__m256 vker = _mm256_load_ps(vector_ker);	
								result = _mm256_add_ps(result, _mm256_mul_ps(vin, vker));
								if (sub < ARRAY_LENGTH) break;	
							}
							if (sub < ARRAY_LENGTH && sub > 0) {
								size = sub;
								memcpy(vector_in, &in.matrix[((block->start_dep[z]+l)*in.y*in.x)+((block->start_col[z]+m)*in.x)+block->start_row[z]], sizeof(float)*size);
								memcpy(vector_ker, &ker.matrix[l*ker.size*ker.size+m*ker.size], sizeof(float)*size);
								__m256 vin = _mm256_load_ps(vector_in);
								__m256 vker = _mm256_load_ps(vector_ker);	
								result = _mm256_add_ps(result, _mm256_mul_ps(vin, vker));
							}
						} else {
							memcpy(vector_in, &in.matrix[((block->start_dep[z]+l)*in.y*in.x)+((block->start_col[z]+m)*in.x)+block->start_row[z]], sizeof(float)*size);
							memcpy(vector_ker, &ker.matrix[l*ker.size*ker.size+m*ker.size], sizeof(float)*size);
							__m256 vin = _mm256_load_ps(vector_in);
							__m256 vker = _mm256_load_ps(vector_ker);	
							result = _mm256_add_ps(result, _mm256_mul_ps(vin, vker));
						}		
					}
				}

				_mm256_store_ps(vsum, result);
				for (m=0;m<ARRAY_LENGTH;m++)
				{	
					sum+=vsum[m];
				}
				ans[block->num[z]] = sum;
				pthread_mutex_lock(&lock4);
				count++;
				pthread_mutex_unlock(&lock4);
			}
		}
	}
	return NULL;
}

int main(int argc, char* argv[])
{
	FILE *fp, *fp2, *fp3;
	char input[255], kernel[255], output[255];
	int i, j, k, in_size[3], out_size[3], kernel_size, flag=0;
	ThreadPool* thread_pool;
	
	if (argc != 4) {
		printf("Usage: ./singlethread <input.txt> <kernel.txt> <output.txt>\n");
		exit(0);
	} else {
		char *filename1 = (char*)malloc(strlen(argv[1]));
		char *filename2 = (char*)malloc(strlen(argv[2]));
		char *filename3 = (char*)malloc(strlen(argv[3]));
		strcpy(filename1, argv[1]);
		strcpy(filename2, argv[2]);
		strcpy(filename3, argv[3]);

		if ((fp = fopen(filename1, "r")) == NULL) {
			printf("File open failed\n");
			exit(1);
		}
	
		if ((fp2 = fopen(filename2, "r")) == NULL) {
			printf("File open failed\n");
			exit(1);
		}

		if ((fp3 = fopen(filename3, "r")) == NULL) {
			printf("File open failed\n");
			exit(1);
		}

		pthread_mutex_init(&lock, NULL);
		pthread_mutex_init(&lock1, NULL);
		pthread_mutex_init(&lock2, NULL);
		pthread_mutex_init(&lock3, NULL);
		pthread_mutex_init(&lock4, NULL);
		pthread_cond_init(&signal, NULL);
		queueInit(&buffer);
		thread_pool = thread_pool_constructor(7);

		for (i = 0; i < 3; i++)
		{
			fscanf(fp, "%s", input);
			fscanf(fp3, "%s", output);
			in_size[i] = atoi(input);
			out_size[i] = atoi(output);
		}
		
		fscanf(fp2, "%s", kernel);
		kernel_size = atoi(kernel);

		container_constructor(&in, (kernel_size-1)/2, in_size[0], in_size[1], in_size[2]);
		container_constructor(&out, 0, out_size[0], out_size[1], out_size[2]);

		kernel_constructor(&ker, kernel_size);

		for(i=in.padding;i<in.z-in.padding;i++) //z
		{
			for(j=in.padding;j<in.y-in.padding;j++) //y
			{
				for(k=in.padding;k<in.x-in.padding;k++) //x
				{
					fscanf(fp, "%s", input);
					sscanf(input, "%f", &in.matrix[i*in.x*in.y+j*in.x+k]);
				}
			}
		}

		for(i=0; fscanf(fp2, "%s", kernel) != EOF; i++) 
		{
			sscanf(kernel, "%f", &ker.matrix[i]);
		}

		for(i=0; fscanf(fp3, "%s", output) != EOF; i++) 
		{
			sscanf(output, "%f", &out.matrix[i]);
		}

		ans = (float*)malloc(sizeof(float)*in.zp*in.yp*in.xp);
		multiThread();

		for (i=0; i < in.zp*in.yp*in.xp; i++)
		{
			if (!f_equal(out.matrix[i], ans[i]))
			{
				flag = 1;
				break;
			}
		}

		if (!flag) {
			printf("output.txt와 값이 동일합니다!\n");
		} else {
			printf("output.txt와 값이 다릅니다!!!\n");
		}

		fprintf(stderr, "Elapsed time with multi thread code = %f milliseconds\n",
				(stop.tv_sec-start.tv_sec)*1000 + (stop.tv_usec-start.tv_usec)*0.001);

		fclose(fp);
		fclose(fp2);
		fclose(fp3);
	}
	thread_pool_destructor(thread_pool);
	pthread_mutex_destroy(&lock);
	pthread_mutex_destroy(&lock1);
	pthread_mutex_destroy(&lock2);
	pthread_mutex_destroy(&lock3);
	pthread_mutex_destroy(&lock4);
	pthread_cond_destroy(&signal);
	container_destructor(&in);
	container_destructor(&out);
	kernel_destructor(&ker);
	return 0;  
}
