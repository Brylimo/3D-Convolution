#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
	FILE *fp, *fp2, *fp3;
	char input[255], kernel[255], output[255];
	int i, j, k, in_size[3], out_size[3], kernel_size;

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

		fclose(fp);
		fclose(fp2);
		fclose(fp3);
	}
	return 0;
}
