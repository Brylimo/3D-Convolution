#include <stdio.h>
#include <immintrin.h>
#include <stdint.h>
#include <x86intrin.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_LENGTH 8

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

float* conv(Container* in, Kernel* ker)
{
	float* ans = (float*)malloc(sizeof(float)*in->zp*in->yp*in->xp);
	float* vector_in = aligned_alloc(32, sizeof(float)*ARRAY_LENGTH);
	float* vector_ker = aligned_alloc(32, sizeof(float)*ARRAY_LENGTH);
	float* vsum = aligned_alloc(32, sizeof(float)*ARRAY_LENGTH);
	float zero = 0.0, sum =0.0;
	int i,j,k,l,m,a,size;
	int height = in->yp;
       	int width = in->xp;
        int depth = in->zp;
	__m256 result;

	if (ker->size > ARRAY_LENGTH)	size = ARRAY_LENGTH;	
	else	size = ker->size; 
	
	for (i=0;(ker->size + i) <= in->z;i++)
	{
		for (j=0;(ker->size + j) <= in->y;j++)
		{
			for (a=0;(ker->size + a) <= in->x;a++){
				result = _mm256_broadcast_ss(&zero);
				for(l=0;l<ker->size;l++) // 3D CONVOLUTION FOR ONE BLOCK
				{
					for(m=0;m<ker->size;m++)
					{
						if (size == ARRAY_LENGTH) {
							int portion = ker->size;
							int sub;
							for(k=0;(sub=portion-(ker->size))>0;k++)
							{
								if (sub < ARRAY_LENGTH) size = sub;
								memcpy(vector_in, &in->matrix[i*in->y*in->x+j*in->y+k*ARRAY_LENGTH], sizeof(float)*size);
								memcpy(vector_ker, &ker->matrix[l*ker->size*ker->size+m*ker->size+k*ARRAY_LENGTH], sizeof(float)*size);
								__m256 vin = _mm256_load_ps(vector_in);
								__m256 vker = _mm256_load_ps(vector_ker);	
								result = _mm256_add_ps(result, _mm256_mul_ps(vin, vker));	
							}
						} else {
							memcpy(vector_in, &in->matrix[((i+l)*in->y*in->x)+((j+m)*in->x)+a], sizeof(float)*size);
							memcpy(vector_ker, &ker->matrix[l*ker->size*ker->size+m*ker->size], sizeof(float)*size);
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
				ans[i*height*width+j*width+a] = sum;
				sum = 0.0;
			}
		}
	}
	return ans;
}

int main(int argc, char* argv[])
{
	FILE *fp, *fp2, *fp3;
	char input[255], kernel[255], output[255];
	Container in, out;
	Kernel ker;
	int i, j, k, in_size[3], out_size[3], kernel_size;
	uint64_t start, end;

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
					in.matrix[i*in.x*in.y+j*in.x+k] = atof(input);
				}
			}
		}

		for(i=0; fscanf(fp2, "%s", kernel) != EOF; i++) 
		{
			ker.matrix[i] = atof(kernel);
		}

		for(i=0; fscanf(fp3, "%s", output) != EOF; i++) 
		{
			out.matrix[i] = atof(output);
		}
		
		float* ans = conv(&in, &ker);
		for (i=0; i < in.zp*in.yp*in.xp; i++)
		{
			if (out.matrix[i] == ans[i])
			{
				printf("yeah\n");
			}
		}

		fclose(fp);
		fclose(fp2);
		fclose(fp3);
	}
	return 0;
}
