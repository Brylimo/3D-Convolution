#include<stdio.h>
#include<stdlib.h>

int main(int argc, char* argv[]){
	FILE *fp1, *fp2, *fp3;
	float* IN1, IN2;
	float* kernel;
	float* OUT;
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
	 float temp;
	int input_z_1=input_z/2;//짝수 가정

        IN1 = (float*)malloc(sizeof(float)*input_x*input_y*input_z_1);
	IN2 = (float*)malloc(sizeof(float)*input_x*input_y*input_z_1);
        kernel = (float*)malloc(sizeof(float)*kernel_size*kernel_size*kernel_size);
        OUT = (float*)malloc(sizeof(float)*output_x*output_y*output_z);
        ANS = (float*)malloc(sizeof(float)*output_x*output_y*output_z);
        for(int i=0;i<input_x*input_y*input_z;i++){
                fscanf(fp1, "%f", &temp);
                IN1[i] = temp;
        }
	for(int i=0;i<input_x*input_y*input_z_1;i++){
		fscanf(fp1, "%f", &temp);
		IN2[i] = temp;
	}
        for(int i=0;i<kernel_size*kernel_size*kernel_size;i++){
                fscanf(fp2, "%f", &temp);
                kernel[i] = temp;
        }
        for(int i=0;i<output_x*output_y*output_z;i++){
                fscanf(fp3, "%f", &temp);
                ANS[i] = temp;
        }

	pthread_create();
	pthread_create();



}	



