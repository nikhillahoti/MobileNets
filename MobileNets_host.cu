#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>

#include "MobileNets_kernel.cu"

#define INPUT_LAYER_SIZE 225 * 225 * 3
#define FIRST_LAYER_WEIGHT_SIZE 32 * 3 * 3 * 3
#define FIRST_LAYER_OUTPUT_SIZE 114 * 114 * 32
#define FIRST_LAYER_CHANNELS 32

#define SECOND_LAYER_WEIGHT_SIZE 32 * 3 * 3
#define SECOND_LAYER_OUTPUT_SIZE 112 * 112 * 32
#define SECOND_LAYER_CHANNELS 32

#define THIRD_LAYER_WEIGHT_SIZE 64 * 32
#define THIRD_LAYER_OUTPUT_SIZE 113 * 113 * 64
#define THIRD_LAYER_CHANNELS 64

// Function declarations
void NeuralNetwork();
void read_File(const char * weightFileName, double *Layer1_Weights_CPU);
void read_Input_File(const char * inputFileName, double *Layer1_Neurons_CPU);

void Read_First_Layer_Data(double * Layer1_Neurons_CPU,
    double * Layer1_Weights_CPU,
    double * Layer1_Mean_CPU,
    double * Layer1_StanDev_CPU,
    double * Layer1_Gamma_CPU,
    double * Layer1_Beta_CPU
);

void Read_SecondLayer_Data(double *Layer1_Weights_CPU,
    double *Layer2_Mean_CPU,
    double *Layer2_StanDev_CPU,
    double *Layer2_Gamma_CPU,
    double *Layer2_Beta_CPU
);

void Read_ThirdLayer_Data(double *Layer3_Weights_CPU);

int main(){
    NeuralNetwork();
}

void NeuralNetwork(){
    // Reading the input layer data
    double * Layer1_Neurons_CPU = (double *) malloc(sizeof(double) * INPUT_LAYER_SIZE);
    double * Layer1_Weights_CPU = (double *) malloc(sizeof(double) * FIRST_LAYER_WEIGHT_SIZE);
    double * Layer1_Mean_CPU = (double *) malloc(sizeof(double) * FIRST_LAYER_CHANNELS);
    double * Layer1_StanDev_CPU = (double *) malloc(sizeof(double) * FIRST_LAYER_CHANNELS);
    double * Layer1_Gamma_CPU = (double *) malloc(sizeof(double) * FIRST_LAYER_CHANNELS);
    double * Layer1_Beta_CPU = (double *) malloc(sizeof(double) * FIRST_LAYER_CHANNELS);

    Read_First_Layer_Data(Layer1_Neurons_CPU,
                Layer1_Weights_CPU,
                Layer1_Mean_CPU,
                Layer1_StanDev_CPU,
                Layer1_Gamma_CPU,
                Layer1_Beta_CPU
    );

    // Allocating memory for Output Matrix
    double * Layer2_Neurons_CPU = (double *) malloc(sizeof(double) * FIRST_LAYER_OUTPUT_SIZE);

    // Copy memory from Host to Kernel
    double *Layer1_Weights_GPU,
           *Layer1_Neurons_GPU,
           *Layer2_Neurons_GPU,
           *Layer1_Mean_GPU,
           *Layer1_StanDev_GPU,
           *Layer1_Gamma_GPU,
           *Layer1_Beta_GPU;

    cudaMalloc((void**) &Layer1_Neurons_GPU, sizeof(double) * INPUT_LAYER_SIZE);
    cudaMalloc((void**) &Layer1_Weights_GPU, sizeof(double) * FIRST_LAYER_WEIGHT_SIZE);
    cudaMalloc((void**) &Layer2_Neurons_GPU, sizeof(double) * FIRST_LAYER_OUTPUT_SIZE);
    cudaMalloc((void**) &Layer1_Mean_GPU, sizeof(double) * FIRST_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer1_StanDev_GPU, sizeof(double) * FIRST_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer1_Gamma_GPU, sizeof(double) * FIRST_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer1_Beta_GPU, sizeof(double) * FIRST_LAYER_CHANNELS);

    cudaMemcpy(Layer1_Neurons_GPU, Layer1_Neurons_CPU, sizeof(double) * INPUT_LAYER_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer1_Weights_GPU, Layer1_Weights_CPU, sizeof(double) * FIRST_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer2_Neurons_GPU, Layer2_Neurons_CPU, sizeof(double) * FIRST_LAYER_OUTPUT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer1_Mean_GPU, Layer1_Mean_CPU, sizeof(double) * FIRST_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer1_StanDev_GPU, Layer1_StanDev_CPU, sizeof(double) * FIRST_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer1_Gamma_GPU, Layer1_Gamma_CPU, sizeof(double) * FIRST_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer1_Beta_GPU, Layer1_Beta_CPU, sizeof(double) * FIRST_LAYER_CHANNELS, cudaMemcpyHostToDevice);


    // Kernel Launch
    dim3 gridSizeA(32, 3, 3);
    dim3 blockSizeA(32,32);

    executeFirstLayer_partA<<< gridSizeA, blockSizeA>>>(Layer1_Neurons_GPU,
                        Layer1_Weights_GPU,
                        Layer2_Neurons_GPU,
                        Layer1_Mean_GPU,
                        Layer1_StanDev_GPU,
                        Layer1_Gamma_GPU,
                        Layer1_Beta_GPU
                    );

    dim3 gridSizeB(32, 7);
    dim3 blockSizeB(16, 16);

    executeFirstLayer_partB<<< gridSizeB, blockSizeB>>>(Layer1_Neurons_GPU,
                        Layer1_Weights_GPU,
                        Layer2_Neurons_GPU,
                        Layer1_Mean_GPU,
                        Layer1_StanDev_GPU,
                        Layer1_Gamma_GPU,
                        Layer1_Beta_GPU
                    );

    dim3 gridSizeC(32, 6);
    dim3 blockSizeC(16, 16);

    executeFirstLayer_partC<<< gridSizeC, blockSizeC>>>(Layer1_Neurons_GPU,
                        Layer1_Weights_GPU,
                        Layer2_Neurons_GPU,
                        Layer1_Mean_GPU,
                        Layer1_StanDev_GPU,
                        Layer1_Gamma_GPU,
                        Layer1_Beta_GPU
                    );

    
    double * Layer2_Weights_CPU = (double *) malloc(sizeof(double) * SECOND_LAYER_WEIGHT_SIZE);
    double * Layer3_Neurons_CPU = (double *) malloc(sizeof(double) * SECOND_LAYER_OUTPUT_SIZE);
    double * Layer2_Mean_CPU = (double *) malloc(sizeof(double) * SECOND_LAYER_CHANNELS);
    double * Layer2_StanDev_CPU = (double *) malloc(sizeof(double) * SECOND_LAYER_CHANNELS);
    double * Layer2_Gamma_CPU = (double *) malloc(sizeof(double) * SECOND_LAYER_CHANNELS);
    double * Layer2_Beta_CPU = (double *) malloc(sizeof(double) * SECOND_LAYER_CHANNELS);


    Read_SecondLayer_Data(Layer2_Weights_CPU,
                        Layer2_Mean_CPU,
                        Layer2_StanDev_CPU,
                        Layer2_Gamma_CPU,
                        Layer2_Beta_CPU
    );

    double *Layer3_Neurons_GPU,
           *Layer2_Weights_GPU,
           *Layer2_Mean_GPU,
           *Layer2_StanDev_GPU,
           *Layer2_Gamma_GPU,
           *Layer2_Beta_GPU;;

    cudaMalloc((void**) &Layer2_Weights_GPU, sizeof(double) * SECOND_LAYER_WEIGHT_SIZE);
    cudaMalloc((void**) &Layer3_Neurons_GPU, sizeof(double) * SECOND_LAYER_OUTPUT_SIZE);
    cudaMalloc((void**) &Layer2_Mean_GPU, sizeof(double) * SECOND_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer2_StanDev_GPU, sizeof(double) * SECOND_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer2_Gamma_GPU, sizeof(double) * SECOND_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer2_Beta_GPU, sizeof(double) * SECOND_LAYER_CHANNELS);

    cudaMemcpy(Layer2_Weights_GPU, Layer2_Weights_CPU, sizeof(double) * SECOND_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer2_Mean_GPU, Layer2_Mean_CPU, sizeof(double) * SECOND_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer2_StanDev_GPU, Layer2_StanDev_CPU, sizeof(double) * SECOND_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer2_Gamma_GPU, Layer2_Gamma_CPU, sizeof(double) * SECOND_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer2_Beta_GPU, Layer2_Beta_CPU, sizeof(double) * SECOND_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    executeSecondLayer_partA<<< gridSizeA, blockSizeA>>>(Layer2_Neurons_GPU,
                                            Layer2_Weights_GPU,
                                            Layer3_Neurons_GPU,
                                            Layer2_Mean_GPU,
                                            Layer2_StanDev_GPU,
                                            Layer2_Gamma_GPU,
                                            Layer2_Beta_GPU
                                        );

    executeSecondLayer_partB<<< gridSizeB, blockSizeB>>>(Layer2_Neurons_GPU,
                                            Layer2_Weights_GPU,
                                            Layer3_Neurons_GPU,
                                            Layer2_Mean_GPU,
                                            Layer2_StanDev_GPU,
                                            Layer2_Gamma_GPU,
                                            Layer2_Beta_GPU
    );

    executeSecondLayer_partC<<< gridSizeC, blockSizeC>>>(Layer2_Neurons_GPU,
                                            Layer2_Weights_GPU,
                                            Layer3_Neurons_GPU,
                                            Layer2_Mean_GPU,
                                            Layer2_StanDev_GPU,
                                            Layer2_Gamma_GPU,
                                            Layer2_Beta_GPU
    );

    /* ************************************************ THIRD LAYER ******************************************************** */
    double * Layer3_Weights_CPU = (double *) malloc(sizeof(double) * THIRD_LAYER_WEIGHT_SIZE);
    double * Layer4_Neurons_CPU = (double *) malloc(sizeof(double) * THIRD_LAYER_OUTPUT_SIZE);

    Read_ThirdLayer_Data(Layer3_Weights_CPU);

    double *Layer4_Neurons_GPU,
           *Layer3_Weights_GPU;

    cudaMalloc((void**) &Layer3_Weights_GPU, sizeof(double) * THIRD_LAYER_WEIGHT_SIZE);
    cudaMalloc((void**) &Layer4_Neurons_GPU, sizeof(double) * THIRD_LAYER_OUTPUT_SIZE);

    cudaMemcpy(Layer3_Weights_GPU, Layer3_Weights_CPU, sizeof(double) * THIRD_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    
    // Execution of the Third Layer
    dim3 gridSizeThirdLayerA(64, 3, 3);
    dim3 blockSizeThirdLayerA(32,32);

    executeThirdLayer_partA<<< gridSizeThirdLayerA, blockSizeThirdLayerA>>>(Layer3_Neurons_GPU,
                        Layer3_Weights_GPU,
                        Layer4_Neurons_GPU
                    );

    dim3 gridSizeThirdLayerB(64, 7);
    dim3 blockSizeThirdLayerB(16, 16);

    executeThirdLayer_partB<<< gridSizeThirdLayerB, blockSizeThirdLayerB>>>(Layer3_Neurons_GPU,
                        Layer3_Weights_GPU,
                        Layer4_Neurons_GPU
                    );

    dim3 gridSizeThirdLayerC(64, 6);
    dim3 blockSizeThirdLayerC(16, 16);

    executeThirdLayer_partC<<< gridSizeThirdLayerC, blockSizeThirdLayerC>>>(Layer3_Neurons_GPU,
                        Layer3_Weights_GPU,
                        Layer4_Neurons_GPU
                    );

    cudaDeviceSynchronize();

    // Get back the data from the kernel to the host.

    // Output of the Third Layer
    cudaMemcpy(Layer4_Neurons_CPU, Layer4_Neurons_GPU, sizeof(double) * THIRD_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // Logic to save into the file to verify the results
    FILE * fOutput = fopen("data/ThirdLayer/output.txt", "w");
    int value = THIRD_LAYER_OUTPUT_SIZE;
    for(int i = 0 ; i < value ; i++){
        fprintf (fOutput, "%0.6lf\n", Layer4_Neurons_CPU[i]);
    }
    fclose(fOutput);

    cudaMemcpy(Layer3_Neurons_CPU, Layer3_Neurons_GPU, sizeof(double) * SECOND_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // Logic to save into the file to verify the results
    fOutput = fopen("data/SecondLayer/output.txt", "w");
    value = SECOND_LAYER_OUTPUT_SIZE;
    for(int i = 0 ; i < value ; i++){
        fprintf (fOutput, "%0.6lf\n", Layer3_Neurons_CPU[i]);
    }
    fclose(fOutput);

    // Saving output of the first layer
    cudaMemcpy(Layer2_Neurons_CPU, Layer2_Neurons_GPU, sizeof(double) * FIRST_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // Logic to save into the file to verify the results
    fOutput = fopen("data/FirstLayer/output.txt", "w");
    value = FIRST_LAYER_OUTPUT_SIZE;
    for(int i = 0 ; i < value ; i++){
        fprintf (fOutput, "%0.6lf\n", Layer2_Neurons_CPU[i]);
    }
    fclose(fOutput);

    printf("\n\n Processing Done !!! \n\n");

    // Free the memory at the end

    // First Layer
    free(Layer1_Neurons_CPU);
    free(Layer1_Weights_CPU);
    free(Layer2_Neurons_CPU);
    free(Layer1_Mean_CPU);
    free(Layer1_StanDev_CPU);
    free(Layer1_Gamma_CPU);
    free(Layer1_Beta_CPU);

    // Second Layer
    free(Layer2_Weights_CPU);
    free(Layer3_Neurons_CPU);
    free(Layer2_Mean_CPU);
    free(Layer2_StanDev_CPU);
    free(Layer2_Gamma_CPU);
    free(Layer2_Beta_CPU);

    // Third Layer
    free(Layer3_Weights_CPU);
    free(Layer4_Neurons_CPU);

    cudaFree(Layer1_Neurons_GPU);
    cudaFree(Layer1_Weights_GPU);
    cudaFree(Layer2_Neurons_GPU);
    cudaFree(Layer1_Mean_GPU);
    cudaFree(Layer1_StanDev_GPU);
    cudaFree(Layer1_Gamma_GPU);
    cudaFree(Layer1_Beta_GPU);

    cudaFree(Layer2_Weights_GPU);
    cudaFree(Layer3_Neurons_GPU);
    cudaFree(Layer2_Mean_GPU);
    cudaFree(Layer2_StanDev_GPU);
    cudaFree(Layer2_Gamma_GPU);
    cudaFree(Layer2_Beta_GPU);

    cudaFree(Layer3_Weights_GPU);
    cudaFree(Layer4_Neurons_GPU);
}

void Read_First_Layer_Data(double * Layer1_Neurons_CPU,
    double * Layer1_Weights_CPU,
    double * Layer1_Mean_CPU,
    double * Layer1_StanDev_CPU,
    double * Layer1_Gamma_CPU,
    double * Layer1_Beta_CPU
){
    read_Input_File("data/FirstLayer/InputFiles/inputNorm.txt", Layer1_Neurons_CPU);
    read_File("data/FirstLayer/weightsNorm.txt", Layer1_Weights_CPU);
    read_File("data/FirstLayer/First_Layer_Mean.txt", Layer1_Mean_CPU);
    read_File("data/FirstLayer/First_Layer_StanDev.txt", Layer1_StanDev_CPU);
    read_File("data/FirstLayer/First_Layer_Gamma.txt", Layer1_Gamma_CPU);
    read_File("data/FirstLayer/First_Layer_Beta.txt", Layer1_Beta_CPU);
}

void Read_SecondLayer_Data(double *Layer2_Weights_CPU,
    double * Layer2_Mean_CPU,
    double * Layer2_StanDev_CPU,
    double * Layer2_Gamma_CPU,
    double * Layer2_Beta_CPU
){
    read_File("data/SecondLayer/weightsNorm.txt", Layer2_Weights_CPU);
    read_File("data/SecondLayer/Second_Layer_Mean.txt", Layer2_Mean_CPU);
    read_File("data/SecondLayer/Second_Layer_StanDev.txt", Layer2_StanDev_CPU);
    read_File("data/SecondLayer/Second_Layer_Gamma.txt", Layer2_Gamma_CPU);
    read_File("data/SecondLayer/Second_Layer_Beta.txt", Layer2_Beta_CPU);
}

void Read_ThirdLayer_Data(double *Layer3_Weights_CPU){
    read_File("data/ThirdLayer/weightsSet1.txt", Layer3_Weights_CPU);
}

void read_File(const char * input_FileName, double * input_values){

    FILE *fp = fopen(input_FileName, "r");
    if (fp == NULL){
        printf("\n No input file present at the location \n");
        return;
    }

    int counter = 0;
    ssize_t read;
    char * line = NULL;
    size_t len = 1000;

    while ((read = getline(&line, &len, fp)) != -1)
        input_values[counter++] = atof(line);
    fclose(fp);
}

void read_Input_File(const char * inputFileName, double * Layer1_Neurons_CPU){
    FILE *fp = fopen(inputFileName, "r");

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
