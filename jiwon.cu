#include <stdio.h>
#include<stdlib.h>

#define TILE_SIZE  6
#define BLOCK_SIZE 8
#define MASK_WIDTH 3

__constant__ float C_kernel[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void Convolution(float* N, float* P, int size){
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	int row_o = blockIdx.y * TILE_SIZE + ty;
	int col_o = blockIdx.x * TILE_SIZE + tx;
	int dep_o = blockIdx.z * TILE_SIZE + tz;

	int row_i = row_o -1;
	int col_i = col_o -1;
	int dep_i = dep_o -1;
	int Rows = size;
	int Columns = size;
	int Depth = size;

	float output =0.0f;
	__shared__ float Ns[TILE_SIZE + MASK_WIDTH-1][TILE_SIZE+ MASK_WIDTH -1][TILE_SIZE +MASK_WIDTH -1];

	if((row_i>=0)&&(row_i<Rows)&&(col_i>=0)&&(col_i<Columns)&&(dep_i>=0)&&(dep_i<Depth)){
		Ns[tz][ty][tx] = N[dep_i*Rows*Columns+row_i*Columns+col_i];//thisk;
	}
	else{
		Ns[tz][ty][tx] = 0.0f;
	}
	__syncthreads();

	if(ty<TILE_SIZE&&tx<TILE_SIZE&&tz<TILE_SIZE){
		for(int i=0;i<MASK_WIDTH;i++){
			for(int j=0;j<MASK_WIDTH;j++){
				for(int k=0;k<MASK_WIDTH;k++){
					output += C_kernel[i][j][k]*Ns[i+tz][j+ty][k+tx];
				}
			}
		}

		if(row_o<Rows&& col_o<Columns && dep_o<Depth)
			P[dep_o*Columns*Rows+row_o*Columns+col_o] = output;
	}
	
}

int main(int argc, char *argv[]){
	float* IN, *IN_c;
	float* kernel;
	float* OUT, *OUT_c;

	int size;

	printf("size : ");
	scanf("%d", &size);

	IN = (float*)malloc(sizeof(float)*size*size*size);
	kernel = (float*)malloc(sizeof(float)*27);
	OUT = (float*)malloc(sizeof(float)*size*size*size);

	for(int i=0;i<size*size*size;i++){
		IN[i]=1.0f;
	}
	for(int i=0;i<27;i++){
		kernel[i]=1.0f;
	}

	cudaMemcpyToSymbol(C_kernel, kernel, sizeof(float)*27);

	cudaMalloc((void**)&IN_c, sizeof(float)*size*size*size);
	cudaMalloc((void**)&OUT_c, sizeof(float)*size*size*size);

	cudaMemcpy(IN_c, IN, sizeof(float)*size*size*size, cudaMemcpyHostToDevice);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((size-1)/TILE_SIZE+1, (size-1)/TILE_SIZE+1, (size-1)/TILE_SIZE+1);

	Convolution<<<dimGrid,dimBlock>>>(IN_c,OUT_c, size);
	cudaMemcpy(OUT, OUT_c, sizeof(float)*size*size*size, cudaMemcpyDeviceToHost);
	free(IN);
	free(OUT);
	free(kernel);
	return 0;
}
	
