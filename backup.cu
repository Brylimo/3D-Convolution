#include <stdio.h>
#include<stdlib.h>

#define BLOCK_SIZE 8
#define MASK_WIDTH 3

__constant__ float C_kernel[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void Convolution(float* N, float* P, int x, int y, int z, int TILE_SIZE){
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	int row_o = blockIdx.y * TILE_SIZE + ty;
	int col_o = blockIdx.x * TILE_SIZE + tx;
	int dep_o = blockIdx.z * TILE_SIZE + tz;

	int row_i = row_o -(MASK_WIDTH-1)/2;
	int col_i = col_o -(MASK_WIDTH-1)/2;
	int dep_i = dep_o -(MASK_WIDTH-1)/2;
	int Columns = x;
	int Rows = y;
	int Depth = z;

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
	FILE *fp1, *fp2, *fp3;
	float* IN, *IN_c;
	float* kernel;
	float* OUT, *OUT_c;
	float* ANS;
	int input_x, input_y, input_z;
	int kernel_size;
	int output_x, output_y, output_z;

	if(argc !=4){
		printf("Usage : ./retry\n");
		exit(0);
	}
	
	char *filename1 = (char*)malloc(strlen(argv[1]));
	char *filename2 = (char*)malloc(strlen(argv[2]));
	char *filename3 = (char*)malloc(strlen(argv[3]));
	strcpy(filename1, argv[1]);
	strcpy(filename2, argv[2]);
	strcpy(filename3, argv[3]);
	printf("%s\n", filename1);
	if((fp1=fopen(filename1, "r"))==NULL){
		printf("file open failed\n");
		exit(0);
	}
	if((fp2 = fopen(filename2, "r")) ==NULL){
		printf("file open failed\n");
		exit(1);
	}
	if((fp3 = fopen(filename3, "r"))==NULL){
		printf("file open failed\n");
		exit(1);
	}
	
	for(int i=0;i<3;i++){
		if(i==0){
			fscanf(fp1, "%d", &input_z);
			fscanf(fp2, "%d", &kernel_size);
			fscanf(fp3, "%d", &output_z);
		}
		 if(i==1){
                        fscanf(fp1, "%d", &input_y);
                        fscanf(fp3, "%d", &output_y);
                }
		 if(i==2){
                        fscanf(fp1, "%d", &input_x);
                        fscanf(fp3, "%d", &output_x);
                }
	}
	int tile_size = BLOCK_SIZE-kernel_size+1;



	printf("%d %d %d\n", input_x, kernel_size, output_x);
	float temp;


	IN = (float*)malloc(sizeof(float)*input_x*input_y*input_z);
	kernel = (float*)malloc(sizeof(float)*kernel_size*kernel_size*kernel_size);
	OUT = (float*)malloc(sizeof(float)*output_x*output_y*output_z);
	ANS = (float*)malloc(sizeof(float)*output_x*output_y*output_z);
	for(int i=0;i<input_x*input_y*input_z;i++){
		fscanf(fp1, "%f", &temp);
		IN[i] = temp;
	}
	for(int i=0;i<kernel_size*kernel_size*kernel_size;i++){
		fscanf(fp2, "%f", &temp);
		kernel[i] = temp;
	}
	for(int i=0;i<output_x*output_y*output_z;i++){
		fscanf(fp3, "%f", &temp);
		ANS[i] = temp;
	}

	cudaMemcpyToSymbol(C_kernel, kernel, sizeof(float)*27);

	cudaMalloc((void**)&IN_c, sizeof(float)*input_x*input_y*input_z);
	cudaMalloc((void**)&OUT_c, sizeof(float)*output_x*output_y*output_z);

	cudaMemcpy(IN_c, IN, sizeof(float)*input_x*input_y*input_z, cudaMemcpyHostToDevice);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((input_x-1)/tile_size+1, (input_y-1)/tile_size+1, (input_z-1)/tile_size+1);

	Convolution<<<dimGrid,dimBlock>>>(IN_c,OUT_c, input_x, input_y, input_z, tile_size);
	cudaMemcpy(OUT, OUT_c, sizeof(float)*output_x*output_y*output_z, cudaMemcpyDeviceToHost);
	for(int i=0;i<output_x*output_y*output_z;i++){
		if(abs(OUT[i]-ANS[i])>=0.002f){
			printf("NOT EQUAL!\n");
			return 0;
		}
	}

	free(IN);
	free(OUT);
	free(kernel);
	printf("ok\n");
	return 0;
}
	
