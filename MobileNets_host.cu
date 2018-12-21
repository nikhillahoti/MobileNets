#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>

#include "MobileNets_kernel.cu"

#define INPUT_LAYER_SIZE 225 * 225 * 3
#define FIRST_LAYER_WEIGHT_SIZE 32 * 3 * 3 * 3
#define FIRST_LAYER_OUTPUT_SIZE 112 * 112 * 32

// Function declarations 
void NeuralNetwork();
double * read_Input_Weights(char * weightFileName);
void read_Input_File(char * inputFileName, double *Layer1_Neurons_CPU);

int main(){
    NeuralNetwork();    
}

void NeuralNetwork(){
    // Reading the input layer data 
    double * Layer1_Neurons_CPU = (double *) malloc(sizeof(double) * INPUT_LAYER_SIZE);
    // read_Input_File("data/Input_Layer_Data.txt", Layer1_Neurons_CPU);
    read_Input_File("data/fTemp.txt", Layer1_Neurons_CPU);

    // Reading the weights file
    double * Layer1_Weights_CPU = read_Input_Weights("data/First_Layer_Weights.txt");

    // Allocating memory for Output Matrix
    double * Layer2_Neurons_CPU = (double *) malloc(sizeof(double) * FIRST_LAYER_OUTPUT_SIZE);

    // Copy memory from Host to Kernel
    double *Layer1_Weights_GPU, *Layer1_Neurons_GPU, *Layer2_Neurons_GPU;

    cudaMalloc((void**) &Layer1_Neurons_GPU, sizeof(double) * INPUT_LAYER_SIZE); 
    cudaMalloc((void**) &Layer1_Weights_GPU, sizeof(double) * FIRST_LAYER_WEIGHT_SIZE);
    cudaMalloc((void**) &Layer2_Neurons_GPU, sizeof(double) * FIRST_LAYER_OUTPUT_SIZE);

    cudaMemcpy(Layer1_Neurons_GPU, Layer1_Neurons_CPU, sizeof(double) * INPUT_LAYER_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer1_Weights_GPU, Layer1_Weights_CPU, sizeof(double) * FIRST_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer2_Neurons_GPU, Layer2_Neurons_CPU, sizeof(double) * FIRST_LAYER_OUTPUT_SIZE, cudaMemcpyHostToDevice);

    // Kernel Launch 
    dim3 gridSizeA(32, 3, 3);
    dim3 blockSizeA(32,32);

    executeFirstLayer_partA<<< gridSizeA, blockSizeA>>>(Layer1_Neurons_GPU, Layer1_Weights_GPU, Layer2_Neurons_GPU);

    cudaDeviceSynchronize();

    dim3 gridSizeB(32, 7);
    dim3 blockSizeB(16, 16);

    executeFirstLayer_partB<<< gridSizeB, blockSizeB>>>(Layer1_Neurons_GPU, Layer1_Weights_GPU, Layer2_Neurons_GPU);

    cudaDeviceSynchronize();

    dim3 gridSizeC(32, 6);
    dim3 blockSizeC(16, 16);

    executeFirstLayer_partC<<< gridSizeC, blockSizeC>>>(Layer1_Neurons_GPU, Layer1_Weights_GPU, Layer2_Neurons_GPU);

    cudaDeviceSynchronize();


    // Get back the data from the kernel to the host
    cudaMemcpy(Layer2_Neurons_CPU, Layer2_Neurons_GPU, sizeof(double) * FIRST_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // Logic to save into the file to verify the results
    FILE * fOutput = fopen("data/output.txt", "w");
    int value = FIRST_LAYER_OUTPUT_SIZE;
    for(int i = 0 ; i < value ; i++){
        fprintf (fOutput, "%lf\n", Layer2_Neurons_CPU[i]);
    }
    fclose(fOutput);

    printf("\n\n Processing Done !!! ");

    // Free the memory at the end
    free(Layer1_Neurons_CPU);
    free(Layer1_Weights_CPU);
    free(Layer2_Neurons_CPU);

    cudaFree(Layer1_Neurons_GPU);
    cudaFree(Layer1_Weights_GPU);
    cudaFree(Layer2_Neurons_GPU);
}


double * read_Input_Weights(char * inputWeightFileName){
    
    // Allocate the memory 
    double * input_weight = (double *) malloc(sizeof(double) * FIRST_LAYER_WEIGHT_SIZE);

    FILE *fp = fopen(inputWeightFileName, "r");
    if (fp == NULL){
        printf("\n No input file present at the location \n");
        return NULL;
    }

    int counter = 0;
    ssize_t read;
    char * line = NULL;
    size_t len = 1000;

    while ((read = getline(&line, &len, fp)) != -1) 
        input_weight[counter++] = atof(line);
    fclose(fp);
    return input_weight;
}

void read_Input_File(char * inputFileName, double * Layer1_Neurons_CPU){
    FILE *fp = fopen(inputFileName, "r");

    int value = INPUT_LAYER_SIZE;
    printf("\n Value is %d", value);
    

    if (fp == NULL){
        printf("\n No input file present at the location \n");
        return;
    }

    int counter = 0;
    ssize_t read;
    char * line = NULL;
    size_t len = 1000;
    int index = 0;
    int lastRow = 0;

    while ((read = getline(&line, &len, fp)) != -1) {
        Layer1_Neurons_CPU[counter++] = atof(line);
        index++;
        // handle padding 
        if (index == 224){
            Layer1_Neurons_CPU[counter++] = 0;
            index = 0;
            lastRow++;
            if(lastRow == 224){
                lastRow = 0;
                int temp = 0;
                while (temp < 225) {
                    Layer1_Neurons_CPU[counter++] = 0;
                    temp++;
                }
            }
        }
    }

    printf("\n Total characters read ---> %d\n", counter);
    fclose(fp);
}

