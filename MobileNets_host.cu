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

#define FOURTH_LAYER_WEIGHT_SIZE 3 * 3 * 64
#define FOURTH_LAYER_OUTPUT_SIZE 56 * 56 * 64
#define FOURTH_LAYER_CHANNELS 64

#define FIFTH_LAYER_WEIGHT_SIZE 64 * 128
#define FIFTH_LAYER_OUTPUT_SIZE 58 * 58 * 128
#define FIFTH_LAYER_CHANNELS 128

#define SIXTH_LAYER_WEIGHT_SIZE 3 * 3 * 128
#define SIXTH_LAYER_OUTPUT_SIZE 56 * 56 * 128
#define SIXTH_LAYER_CHANNELS 128

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

void Execute_First_Layer(double * Layer2_Neurons_GPU);

void Read_SecondLayer_Data(double *Layer1_Weights_CPU,
    double *Layer2_Mean_CPU,
    double *Layer2_StanDev_CPU,
    double *Layer2_Gamma_CPU,
    double *Layer2_Beta_CPU
);

void Execute_Second_Layer(
    double * Layer2_Neurons_GPU,
    double * Layer3_Neurons_GPU
);

void Read_ThirdLayer_Data(double *Layer3_Weights_CPU,
    double * Layer3_Mean_CPU,
    double * Layer3_StanDev_CPU,
    double * Layer3_Gamma_CPU,
    double * Layer3_Beta_CPU
);

void Execute_Third_Layer(
    double * Layer3_Neurons_GPU,
    double * Layer4_Neurons_GPU
);

void Read_FourthLayer_Data(double *Layer4_Weights_CPU,
    double * Layer4_Mean_CPU,
    double * Layer4_StanDev_CPU,
    double * Layer4_Gamma_CPU,
    double * Layer4_Beta_CPU
);

void Execute_Fourth_Layer(
    double * Layer4_Neurons_GPU,
    double * Layer5_Neurons_GPU
);

void Read_FifthLayer_Data(double *Layer5_Weights_CPU,
    double * Layer5_Mean_CPU,
    double * Layer5_StanDev_CPU,
    double * Layer5_Gamma_CPU,
    double * Layer5_Beta_CPU
);

void Execute_Fifth_Layer(
    double * Layer5_Neurons_GPU,
    double * Layer6_Neurons_GPU
);

void Read_SixthLayer_Data(double *Layer6_Weights_CPU,
    double * Layer6_Mean_CPU,
    double * Layer6_StanDev_CPU,
    double * Layer6_Gamma_CPU,
    double * Layer6_Beta_CPU
);

void Execute_Sixth_Layer(
    double * Layer6_Neurons_GPU,
    double * Layer7_Neurons_GPU
);

int main(){
    NeuralNetwork();
}

void NeuralNetwork(){
    FILE *fOutput;
    int value;

    /* ************************************************ FIRST LAYER ******************************************************** */
    double *Layer2_Neurons_GPU = NULL; 
    cudaMalloc((void**) &Layer2_Neurons_GPU, sizeof(double) * FIRST_LAYER_OUTPUT_SIZE);

    Execute_First_Layer(Layer2_Neurons_GPU);

    // Saving output of the first layer: Initially Not Saved
    bool SAVE_FIRST_LAYER_WEIGHTS = false;
    if(SAVE_FIRST_LAYER_WEIGHTS){
        
        double *Layer2_Neurons_CPU = (double *) malloc(sizeof(double) * FIRST_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer2_Neurons_CPU, Layer2_Neurons_GPU, sizeof(double) * FIRST_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/FirstLayer/output.txt", "w");
        value = FIRST_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer2_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer2_Neurons_CPU);
    }
    
    printf("\n Layer 1 Execution complete !!!");
    /* ************************************************ FIRST LAYER COMPLETE *********************************************** */

    /* ************************************************ SECOND LAYER ******************************************************** */
    double *Layer3_Neurons_GPU;
    cudaMalloc((void**) &Layer3_Neurons_GPU, sizeof(double) * SECOND_LAYER_OUTPUT_SIZE);

    Execute_Second_Layer(Layer2_Neurons_GPU, Layer3_Neurons_GPU);

    bool SAVE_SECOND_LAYER_WEIGHTS = false;
    if(SAVE_SECOND_LAYER_WEIGHTS){
        
        double * Layer3_Neurons_CPU = (double *) malloc(sizeof(double) * SECOND_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer3_Neurons_CPU, Layer3_Neurons_GPU, sizeof(double) * SECOND_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/SecondLayer/output.txt", "w");
        value = SECOND_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer3_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer3_Neurons_CPU);
    }

    printf("\n Layer 2 Execution complete !!!");
    /* ************************************************ SECOND LAYER COMPLETE *********************************************** */

    /* ************************************************ THIRD LAYER ******************************************************** */
    double *Layer4_Neurons_GPU;
    cudaMalloc((void**) &Layer4_Neurons_GPU, sizeof(double) * THIRD_LAYER_OUTPUT_SIZE);

    Execute_Third_Layer(Layer3_Neurons_GPU, Layer4_Neurons_GPU);

    bool SAVE_THIRD_LAYER_WEIGHTS = true;
    if(SAVE_THIRD_LAYER_WEIGHTS){
        double * Layer4_Neurons_CPU = (double *) malloc(sizeof(double) * THIRD_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer4_Neurons_CPU, Layer4_Neurons_GPU, sizeof(double) * THIRD_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/ThirdLayer/output.txt", "w");
        value = THIRD_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer4_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer4_Neurons_CPU);
    }

    printf("\n Layer 3 Execution complete !!!");
    /* ************************************************ THIRD LAYER COMPLETE *********************************************** */

    /* ************************************************ FOURTH LAYER ******************************************************** */
    double *Layer5_Neurons_GPU;
    cudaMalloc((void**) &Layer5_Neurons_GPU, sizeof(double) * FOURTH_LAYER_OUTPUT_SIZE);

    Execute_Fourth_Layer(Layer4_Neurons_GPU, Layer5_Neurons_GPU);

    bool SAVE_FOURTH_LAYER_WEIGHTS = true;
    if(SAVE_FOURTH_LAYER_WEIGHTS){
        double * Layer5_Neurons_CPU = (double *) malloc(sizeof(double) * FOURTH_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer5_Neurons_CPU, Layer5_Neurons_GPU, sizeof(double) * FOURTH_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/FourthLayer/output.txt", "w");
        value = FOURTH_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer5_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer5_Neurons_CPU);
    }

    printf("\n Layer 4 Execution complete !!!");
    /* ************************************************ FOURTH LAYER COMPLETE *********************************************** */

    /* ************************************************ FIFTH LAYER ******************************************************** */
    double *Layer6_Neurons_GPU;
    cudaMalloc((void**) &Layer6_Neurons_GPU, sizeof(double) * FIFTH_LAYER_OUTPUT_SIZE);

    Execute_Fifth_Layer(Layer5_Neurons_GPU, Layer6_Neurons_GPU);

    bool SAVE_FIFTH_LAYER_WEIGHTS = true;
    if(SAVE_FIFTH_LAYER_WEIGHTS){
        double * Layer6_Neurons_CPU = (double *) malloc(sizeof(double) * FIFTH_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer6_Neurons_CPU, Layer6_Neurons_GPU, sizeof(double) * FIFTH_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/FifthLayer/output.txt", "w");
        value = FIFTH_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer6_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer6_Neurons_CPU);
    }

    printf("\n Layer 5 Execution complete !!!");
    /* ************************************************ FIFTH LAYER COMPLETE *********************************************** */

    /* ************************************************ SIXTH LAYER ******************************************************** */
    double *Layer7_Neurons_GPU;
    cudaMalloc((void**) &Layer7_Neurons_GPU, sizeof(double) * SIXTH_LAYER_OUTPUT_SIZE);

    Execute_Sixth_Layer(Layer6_Neurons_GPU, Layer7_Neurons_GPU);

    bool SAVE_SIXTH_LAYER_WEIGHTS = true;
    if(SAVE_SIXTH_LAYER_WEIGHTS){
        double * Layer7_Neurons_CPU = (double *) malloc(sizeof(double) * SIXTH_LAYER_OUTPUT_SIZE);
        cudaMemcpy(Layer7_Neurons_CPU, Layer7_Neurons_GPU, sizeof(double) * SIXTH_LAYER_OUTPUT_SIZE, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        // Logic to save into the file to verify the results
        fOutput = fopen("data/SixthLayer/output.txt", "w");
        value = SIXTH_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer7_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer7_Neurons_CPU);
    }

    printf("\n Layer 6 Execution complete !!!");
    /* ************************************************ FIFTH LAYER COMPLETE *********************************************** */

    printf("\n\n Processing Done !!! \n\n");

    cudaFree(Layer2_Neurons_GPU);
    cudaFree(Layer3_Neurons_GPU);
    cudaFree(Layer4_Neurons_GPU);
}

void Execute_First_Layer(double *Layer2_Neurons_GPU)
{
    double *Layer1_Neurons_CPU = (double *) malloc(sizeof(double) * INPUT_LAYER_SIZE);
    double *Layer1_Weights_CPU = (double *) malloc(sizeof(double) * FIRST_LAYER_WEIGHT_SIZE);
    double *Layer1_Mean_CPU = (double *) malloc(sizeof(double) * FIRST_LAYER_CHANNELS);
    double *Layer1_StanDev_CPU = (double *) malloc(sizeof(double) * FIRST_LAYER_CHANNELS);
    double *Layer1_Gamma_CPU = (double *) malloc(sizeof(double) * FIRST_LAYER_CHANNELS);
    double *Layer1_Beta_CPU = (double *) malloc(sizeof(double) * FIRST_LAYER_CHANNELS);

    Read_First_Layer_Data(
        Layer1_Neurons_CPU,
        Layer1_Weights_CPU,        
        Layer1_Mean_CPU,
        Layer1_StanDev_CPU,
        Layer1_Gamma_CPU,
        Layer1_Beta_CPU
    );

    // Copy memory from Host to Kernel
    double *Layer1_Weights_GPU,
           *Layer1_Neurons_GPU,
           *Layer1_Mean_GPU,
           *Layer1_StanDev_GPU,
           *Layer1_Gamma_GPU,
           *Layer1_Beta_GPU;

    cudaMalloc((void**) &Layer1_Neurons_GPU, sizeof(double) * INPUT_LAYER_SIZE);
    cudaMalloc((void**) &Layer1_Weights_GPU, sizeof(double) * FIRST_LAYER_WEIGHT_SIZE);
    cudaMalloc((void**) &Layer1_Mean_GPU, sizeof(double) * FIRST_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer1_StanDev_GPU, sizeof(double) * FIRST_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer1_Gamma_GPU, sizeof(double) * FIRST_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer1_Beta_GPU, sizeof(double) * FIRST_LAYER_CHANNELS);

    cudaMemcpy(Layer1_Neurons_GPU, Layer1_Neurons_CPU, sizeof(double) * INPUT_LAYER_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer1_Weights_GPU, Layer1_Weights_CPU, sizeof(double) * FIRST_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer1_Mean_GPU, Layer1_Mean_CPU, sizeof(double) * FIRST_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer1_StanDev_GPU, Layer1_StanDev_CPU, sizeof(double) * FIRST_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer1_Gamma_GPU, Layer1_Gamma_CPU, sizeof(double) * FIRST_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer1_Beta_GPU, Layer1_Beta_CPU, sizeof(double) * FIRST_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer1_Neurons_CPU);
    free(Layer1_Weights_CPU);
    free(Layer1_Mean_CPU);
    free(Layer1_StanDev_CPU);
    free(Layer1_Gamma_CPU);
    free(Layer1_Beta_CPU);

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

    cudaDeviceSynchronize();

    // First Layer GPU Memory Free
    cudaFree(Layer1_Neurons_GPU);
    cudaFree(Layer1_Weights_GPU);
    cudaFree(Layer1_Mean_GPU);
    cudaFree(Layer1_StanDev_GPU);
    cudaFree(Layer1_Gamma_GPU);
    cudaFree(Layer1_Beta_GPU);
}

void Read_First_Layer_Data(
    double * Layer1_Neurons_CPU,
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

void Execute_Second_Layer(
    double * Layer2_Neurons_GPU,
    double * Layer3_Neurons_GPU
)
{
    double * Layer2_Weights_CPU = (double *) malloc(sizeof(double) * SECOND_LAYER_WEIGHT_SIZE);
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
    
    double *Layer2_Weights_GPU,
           *Layer2_Mean_GPU,
           *Layer2_StanDev_GPU,
           *Layer2_Gamma_GPU,
           *Layer2_Beta_GPU;;

    cudaMalloc((void**) &Layer2_Weights_GPU, sizeof(double) * SECOND_LAYER_WEIGHT_SIZE);
    cudaMalloc((void**) &Layer2_Mean_GPU, sizeof(double) * SECOND_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer2_StanDev_GPU, sizeof(double) * SECOND_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer2_Gamma_GPU, sizeof(double) * SECOND_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer2_Beta_GPU, sizeof(double) * SECOND_LAYER_CHANNELS);

    cudaMemcpy(Layer2_Weights_GPU, Layer2_Weights_CPU, sizeof(double) * SECOND_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer2_Mean_GPU, Layer2_Mean_CPU, sizeof(double) * SECOND_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer2_StanDev_GPU, Layer2_StanDev_CPU, sizeof(double) * SECOND_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer2_Gamma_GPU, Layer2_Gamma_CPU, sizeof(double) * SECOND_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer2_Beta_GPU, Layer2_Beta_CPU, sizeof(double) * SECOND_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer2_Weights_CPU);
    free(Layer2_Mean_CPU);
    free(Layer2_StanDev_CPU);
    free(Layer2_Gamma_CPU);
    free(Layer2_Beta_CPU);

    dim3 gridSizeA(32, 3, 3);
    dim3 blockSizeA(32,32);
    executeSecondLayer_partA<<< gridSizeA, blockSizeA>>>(Layer2_Neurons_GPU,
                                            Layer2_Weights_GPU,
                                            Layer3_Neurons_GPU,
                                            Layer2_Mean_GPU,
                                            Layer2_StanDev_GPU,
                                            Layer2_Gamma_GPU,
                                            Layer2_Beta_GPU
    );

    dim3 gridSizeB(32, 7);
    dim3 blockSizeB(16, 16);
    executeSecondLayer_partB<<< gridSizeB, blockSizeB>>>(Layer2_Neurons_GPU,
                                            Layer2_Weights_GPU,
                                            Layer3_Neurons_GPU,
                                            Layer2_Mean_GPU,
                                            Layer2_StanDev_GPU,
                                            Layer2_Gamma_GPU,
                                            Layer2_Beta_GPU
    );

    dim3 gridSizeC(32, 6);
    dim3 blockSizeC(16, 16);
    executeSecondLayer_partC<<< gridSizeC, blockSizeC>>>(Layer2_Neurons_GPU,
                                            Layer2_Weights_GPU,
                                            Layer3_Neurons_GPU,
                                            Layer2_Mean_GPU,
                                            Layer2_StanDev_GPU,
                                            Layer2_Gamma_GPU,
                                            Layer2_Beta_GPU
    );

    cudaFree(Layer2_Weights_GPU);    
    cudaFree(Layer2_Mean_GPU);
    cudaFree(Layer2_StanDev_GPU);
    cudaFree(Layer2_Gamma_GPU);
    cudaFree(Layer2_Beta_GPU);
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

void Execute_Third_Layer(
    double * Layer3_Neurons_GPU,
    double * Layer4_Neurons_GPU
){
    double * Layer3_Weights_CPU = (double *) malloc(sizeof(double) * THIRD_LAYER_WEIGHT_SIZE);
    double * Layer3_Mean_CPU = (double *) malloc(sizeof(double) * THIRD_LAYER_CHANNELS);
    double * Layer3_StanDev_CPU = (double *) malloc(sizeof(double) * THIRD_LAYER_CHANNELS);
    double * Layer3_Gamma_CPU = (double *) malloc(sizeof(double) * THIRD_LAYER_CHANNELS);
    double * Layer3_Beta_CPU = (double *) malloc(sizeof(double) * THIRD_LAYER_CHANNELS);

    Read_ThirdLayer_Data(Layer3_Weights_CPU,
                Layer3_Mean_CPU,
                Layer3_StanDev_CPU,
                Layer3_Gamma_CPU,
                Layer3_Beta_CPU
    );

    double *Layer3_Weights_GPU,
           *Layer3_Mean_GPU,
           *Layer3_StanDev_GPU,
           *Layer3_Gamma_GPU,
           *Layer3_Beta_GPU;

    cudaMalloc((void**) &Layer3_Weights_GPU, sizeof(double) * THIRD_LAYER_WEIGHT_SIZE);
    cudaMalloc((void**) &Layer3_Mean_GPU, sizeof(double) * THIRD_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer3_StanDev_GPU, sizeof(double) * THIRD_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer3_Gamma_GPU, sizeof(double) * THIRD_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer3_Beta_GPU, sizeof(double) * THIRD_LAYER_CHANNELS);

    cudaMemcpy(Layer3_Weights_GPU, Layer3_Weights_CPU, sizeof(double) * THIRD_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer3_Mean_GPU, Layer3_Mean_CPU, sizeof(double) * THIRD_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer3_StanDev_GPU, Layer3_StanDev_CPU, sizeof(double) * THIRD_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer3_Gamma_GPU, Layer3_Gamma_CPU, sizeof(double) * THIRD_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer3_Beta_GPU, Layer3_Beta_CPU, sizeof(double) * THIRD_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer3_Weights_CPU);
    free(Layer3_Mean_CPU);
    free(Layer3_StanDev_CPU);
    free(Layer3_Gamma_CPU);
    free(Layer3_Beta_CPU);
    
    // Execution of the Third Layer
    dim3 gridSizeThirdLayerA(64, 3, 3);
    dim3 blockSizeThirdLayerA(32,32);
    executeThirdLayer_partA<<< gridSizeThirdLayerA, blockSizeThirdLayerA>>>(Layer3_Neurons_GPU,
                        Layer3_Weights_GPU,
                        Layer4_Neurons_GPU,
                        Layer3_Mean_GPU,
                        Layer3_StanDev_GPU,
                        Layer3_Gamma_GPU,
                        Layer3_Beta_GPU
    );

    dim3 gridSizeThirdLayerB(64, 7);
    dim3 blockSizeThirdLayerB(16, 16);
    executeThirdLayer_partB<<< gridSizeThirdLayerB, blockSizeThirdLayerB>>>(Layer3_Neurons_GPU,
                        Layer3_Weights_GPU,
                        Layer4_Neurons_GPU,
                        Layer3_Mean_GPU,
                        Layer3_StanDev_GPU,
                        Layer3_Gamma_GPU,
                        Layer3_Beta_GPU
    );

    dim3 gridSizeThirdLayerC(64, 6);
    dim3 blockSizeThirdLayerC(16, 16);
    executeThirdLayer_partC<<< gridSizeThirdLayerC, blockSizeThirdLayerC>>>(Layer3_Neurons_GPU,
                        Layer3_Weights_GPU,
                        Layer4_Neurons_GPU,
                        Layer3_Mean_GPU,
                        Layer3_StanDev_GPU,
                        Layer3_Gamma_GPU,
                        Layer3_Beta_GPU
    );

    cudaDeviceSynchronize();

    cudaFree(Layer3_Weights_GPU);
    cudaFree(Layer3_Mean_GPU);
    cudaFree(Layer3_StanDev_GPU);
    cudaFree(Layer3_Gamma_GPU);
    cudaFree(Layer3_Beta_GPU);
}

void Read_ThirdLayer_Data(double *Layer3_Weights_CPU,
    double * Layer3_Mean_CPU,
    double * Layer3_StanDev_CPU,
    double * Layer3_Gamma_CPU,
    double * Layer3_Beta_CPU
){
    read_File("data/ThirdLayer/weightsNorm.txt", Layer3_Weights_CPU);
    read_File("data/ThirdLayer/Third_Layer_Mean.txt", Layer3_Mean_CPU);
    read_File("data/ThirdLayer/Third_Layer_StanDev.txt", Layer3_StanDev_CPU);
    read_File("data/ThirdLayer/Third_Layer_Gamma.txt", Layer3_Gamma_CPU);
    read_File("data/ThirdLayer/Third_Layer_Beta.txt", Layer3_Beta_CPU);
}

void Execute_Fourth_Layer(
    double * Layer4_Neurons_GPU,
    double * Layer5_Neurons_GPU
){  
    double * Layer4_Weights_CPU = (double *) malloc(sizeof(double) * FOURTH_LAYER_WEIGHT_SIZE);
    double * Layer4_Mean_CPU = (double *) malloc(sizeof(double) * FOURTH_LAYER_CHANNELS);
    double * Layer4_StanDev_CPU = (double *) malloc(sizeof(double) * FOURTH_LAYER_CHANNELS);
    double * Layer4_Gamma_CPU = (double *) malloc(sizeof(double) * FOURTH_LAYER_CHANNELS);
    double * Layer4_Beta_CPU = (double *) malloc(sizeof(double) * FOURTH_LAYER_CHANNELS);

    Read_FourthLayer_Data(Layer4_Weights_CPU,
                    Layer4_Mean_CPU,
                    Layer4_StanDev_CPU,
                    Layer4_Gamma_CPU,
                    Layer4_Beta_CPU
    );
    
    double *Layer4_Weights_GPU,
           *Layer4_Mean_GPU,
           *Layer4_StanDev_GPU,
           *Layer4_Gamma_GPU,
           *Layer4_Beta_GPU;

    cudaMalloc((void**) &Layer4_Weights_GPU, sizeof(double) * FOURTH_LAYER_WEIGHT_SIZE);
    cudaMalloc((void**) &Layer4_Mean_GPU, sizeof(double) * FOURTH_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer4_StanDev_GPU, sizeof(double) * FOURTH_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer4_Gamma_GPU, sizeof(double) * FOURTH_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer4_Beta_GPU, sizeof(double) * FOURTH_LAYER_CHANNELS);

    cudaMemcpy(Layer4_Weights_GPU, Layer4_Weights_CPU, sizeof(double) * FOURTH_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer4_Mean_GPU, Layer4_Mean_CPU, sizeof(double) * FOURTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer4_StanDev_GPU, Layer4_StanDev_CPU, sizeof(double) * FOURTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer4_Gamma_GPU, Layer4_Gamma_CPU, sizeof(double) * FOURTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer4_Beta_GPU, Layer4_Beta_CPU, sizeof(double) * FOURTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);

    free(Layer4_Weights_CPU);
    free(Layer4_Mean_CPU);
    free(Layer4_StanDev_CPU);
    free(Layer4_Gamma_CPU);
    free(Layer4_Beta_CPU);

    dim3 gridSizeFourthLayer(64);
    dim3 blockSizeFourthLayerA(32,32);
    executeFourthLayer_partA<<< gridSizeFourthLayer, blockSizeFourthLayerA>>>(Layer4_Neurons_GPU,
                        Layer4_Weights_GPU,
                        Layer5_Neurons_GPU,
                        Layer4_Mean_GPU,
                        Layer4_StanDev_GPU,
                        Layer4_Gamma_GPU,
                        Layer4_Beta_GPU
                    );

    dim3 blockSizeFourthLayerB(32, 24);
    executeFourthLayer_partB<<< gridSizeFourthLayer, blockSizeFourthLayerB>>>(Layer4_Neurons_GPU,
                        Layer4_Weights_GPU,
                        Layer5_Neurons_GPU,
                        Layer4_Mean_GPU,
                        Layer4_StanDev_GPU,
                        Layer4_Gamma_GPU,
                        Layer4_Beta_GPU
                    );

    
    dim3 blockSizeFourthLayerC(24, 32);
    executeFourthLayer_partC<<< gridSizeFourthLayer, blockSizeFourthLayerC>>>(Layer4_Neurons_GPU,
                        Layer4_Weights_GPU,
                        Layer5_Neurons_GPU,
                        Layer4_Mean_GPU,
                        Layer4_StanDev_GPU,
                        Layer4_Gamma_GPU,
                        Layer4_Beta_GPU
                    );

    
    dim3 blockSizeFourthLayerD(24, 24);
    executeFourthLayer_partD<<< gridSizeFourthLayer, blockSizeFourthLayerD>>>(Layer4_Neurons_GPU,
                        Layer4_Weights_GPU,
                        Layer5_Neurons_GPU,
                        Layer4_Mean_GPU,
                        Layer4_StanDev_GPU,
                        Layer4_Gamma_GPU,
                        Layer4_Beta_GPU
                    );

    cudaFree(Layer4_Weights_GPU);
    cudaFree(Layer4_Mean_GPU);
    cudaFree(Layer4_StanDev_GPU);
    cudaFree(Layer4_Gamma_GPU);
    cudaFree(Layer4_Beta_GPU);
}

void Read_FourthLayer_Data(double *Layer4_Weights_CPU,
    double * Layer4_Mean_CPU,
    double * Layer4_StanDev_CPU,
    double * Layer4_Gamma_CPU,
    double * Layer4_Beta_CPU
){
    read_File("data/FourthLayer/weightsNorm.txt", Layer4_Weights_CPU);
    read_File("data/FourthLayer/Fourth_Layer_Mean.txt", Layer4_Mean_CPU);
    read_File("data/FourthLayer/Fourth_Layer_StanDev.txt", Layer4_StanDev_CPU);
    read_File("data/FourthLayer/Fourth_Layer_Gamma.txt", Layer4_Gamma_CPU);
    read_File("data/FourthLayer/Fourth_Layer_Beta.txt", Layer4_Beta_CPU);
}

void Execute_Fifth_Layer(
    double * Layer5_Neurons_GPU,
    double * Layer6_Neurons_GPU
){  
    double * Layer5_Weights_CPU = (double *) malloc(sizeof(double) * FIFTH_LAYER_WEIGHT_SIZE);
    double * Layer5_Mean_CPU = (double *) malloc(sizeof(double) * FIFTH_LAYER_CHANNELS);
    double * Layer5_StanDev_CPU = (double *) malloc(sizeof(double) * FIFTH_LAYER_CHANNELS);
    double * Layer5_Gamma_CPU = (double *) malloc(sizeof(double) * FIFTH_LAYER_CHANNELS);
    double * Layer5_Beta_CPU = (double *) malloc(sizeof(double) * FIFTH_LAYER_CHANNELS);

    Read_FifthLayer_Data(Layer5_Weights_CPU,
                    Layer5_Mean_CPU,
                    Layer5_StanDev_CPU,
                    Layer5_Gamma_CPU,
                    Layer5_Beta_CPU
                );
    
    double *Layer5_Weights_GPU,
           *Layer5_Mean_GPU,
           *Layer5_StanDev_GPU,
           *Layer5_Gamma_GPU,
           *Layer5_Beta_GPU;

    cudaMalloc((void**) &Layer5_Weights_GPU, sizeof(double) * FIFTH_LAYER_WEIGHT_SIZE);
    cudaMalloc((void**) &Layer5_Mean_GPU, sizeof(double) * FIFTH_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer5_StanDev_GPU, sizeof(double) * FIFTH_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer5_Gamma_GPU, sizeof(double) * FIFTH_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer5_Beta_GPU, sizeof(double) * FIFTH_LAYER_CHANNELS);

    cudaMemcpy(Layer5_Weights_GPU, Layer5_Weights_CPU, sizeof(double) * FIFTH_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer5_Mean_GPU, Layer5_Mean_CPU, sizeof(double) * FIFTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer5_StanDev_GPU, Layer5_StanDev_CPU, sizeof(double) * FIFTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer5_Gamma_GPU, Layer5_Gamma_CPU, sizeof(double) * FIFTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer5_Beta_GPU, Layer5_Beta_CPU, sizeof(double) * FIFTH_LAYER_CHANNELS, cudaMemcpyHostToDevice); 

    free(Layer5_Weights_CPU);
    free(Layer5_Mean_CPU);
    free(Layer5_StanDev_CPU);
    free(Layer5_Gamma_CPU);
    free(Layer5_Beta_CPU);

    dim3 gridSizeFifthLayer(128);
    dim3 blockSizeFifthLayerA(32,32);
    executeFifthLayer_partA<<< gridSizeFifthLayer, blockSizeFifthLayerA>>>(Layer5_Neurons_GPU,
                        Layer5_Weights_GPU,
                        Layer6_Neurons_GPU,
                        Layer5_Mean_GPU,
                        Layer5_StanDev_GPU,
                        Layer5_Gamma_GPU,
                        Layer5_Beta_GPU
                    );
                    
    dim3 blockSizeFifthLayerB(32, 24);
    executeFifthLayer_partB<<< gridSizeFifthLayer, blockSizeFifthLayerB>>>(Layer5_Neurons_GPU,
                        Layer5_Weights_GPU,
                        Layer6_Neurons_GPU,
                        Layer5_Mean_GPU,
                        Layer5_StanDev_GPU,
                        Layer5_Gamma_GPU,
                        Layer5_Beta_GPU
                    );

    
    dim3 blockSizeFifthLayerC(24, 32);
    executeFifthLayer_partC<<< gridSizeFifthLayer, blockSizeFifthLayerC>>>(Layer5_Neurons_GPU,
                        Layer5_Weights_GPU,
                        Layer6_Neurons_GPU,
                        Layer5_Mean_GPU,
                        Layer5_StanDev_GPU,
                        Layer5_Gamma_GPU,
                        Layer5_Beta_GPU
                    );

    
    dim3 blockSizeFifthLayerD(24, 24);
    executeFifthLayer_partD<<< gridSizeFifthLayer, blockSizeFifthLayerD>>>(Layer5_Neurons_GPU,
                        Layer5_Weights_GPU,
                        Layer6_Neurons_GPU,
                        Layer5_Mean_GPU,
                        Layer5_StanDev_GPU,
                        Layer5_Gamma_GPU,
                        Layer5_Beta_GPU
                    );

    cudaFree(Layer5_Weights_GPU);
    cudaFree(Layer5_Mean_GPU);
    cudaFree(Layer5_StanDev_GPU);
    cudaFree(Layer5_Gamma_GPU);
    cudaFree(Layer5_Beta_GPU);
}

void Read_FifthLayer_Data(double *Layer5_Weights_CPU,
    double * Layer5_Mean_CPU,
    double * Layer5_StanDev_CPU,
    double * Layer5_Gamma_CPU,
    double * Layer5_Beta_CPU

){
    read_File("data/FifthLayer/weightsNorm.txt", Layer5_Weights_CPU);
    read_File("data/FifthLayer/Fifth_Layer_Mean.txt", Layer5_Mean_CPU);
    read_File("data/FifthLayer/Fifth_Layer_StanDev.txt", Layer5_StanDev_CPU);
    read_File("data/FifthLayer/Fifth_Layer_Gamma.txt", Layer5_Gamma_CPU);
    read_File("data/FifthLayer/Fifth_Layer_Beta.txt", Layer5_Beta_CPU);
}

void Execute_Sixth_Layer(
    double * Layer6_Neurons_GPU,
    double * Layer7_Neurons_GPU
){  
    double * Layer6_Weights_CPU = (double *) malloc(sizeof(double) * SIXTH_LAYER_WEIGHT_SIZE);
    double * Layer6_Mean_CPU = (double *) malloc(sizeof(double) * SIXTH_LAYER_CHANNELS);
    double * Layer6_StanDev_CPU = (double *) malloc(sizeof(double) * SIXTH_LAYER_CHANNELS);
    double * Layer6_Gamma_CPU = (double *) malloc(sizeof(double) * SIXTH_LAYER_CHANNELS);
    double * Layer6_Beta_CPU = (double *) malloc(sizeof(double) * SIXTH_LAYER_CHANNELS);

    Read_SixthLayer_Data(Layer6_Weights_CPU,
                    Layer6_Mean_CPU,
                    Layer6_StanDev_CPU,
                    Layer6_Gamma_CPU,
                    Layer6_Beta_CPU
                );
    
    double *Layer6_Weights_GPU,
           *Layer6_Mean_GPU,
           *Layer6_StanDev_GPU,
           *Layer6_Gamma_GPU,
           *Layer6_Beta_GPU;

    cudaMalloc((void**) &Layer6_Weights_GPU, sizeof(double) * SIXTH_LAYER_WEIGHT_SIZE);
    cudaMalloc((void**) &Layer6_Mean_GPU, sizeof(double) * SIXTH_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer6_StanDev_GPU, sizeof(double) * SIXTH_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer6_Gamma_GPU, sizeof(double) * SIXTH_LAYER_CHANNELS);
    cudaMalloc((void**) &Layer6_Beta_GPU, sizeof(double) * SIXTH_LAYER_CHANNELS);

    cudaMemcpy(Layer6_Weights_GPU, Layer6_Weights_CPU, sizeof(double) * SIXTH_LAYER_WEIGHT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer6_Mean_GPU, Layer6_Mean_CPU, sizeof(double) * SIXTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer6_StanDev_GPU, Layer6_StanDev_CPU, sizeof(double) * SIXTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer6_Gamma_GPU, Layer6_Gamma_CPU, sizeof(double) * SIXTH_LAYER_CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(Layer6_Beta_GPU, Layer6_Beta_CPU, sizeof(double) * SIXTH_LAYER_CHANNELS, cudaMemcpyHostToDevice); 

    free(Layer6_Weights_CPU);
    free(Layer6_Mean_CPU);
    free(Layer6_StanDev_CPU);
    free(Layer6_Gamma_CPU);
    free(Layer6_Beta_CPU);

    dim3 gridSizeSixthLayer(128);
    dim3 blockSizeSixthLayerA(32,32);
    executeSixthLayer_partA<<< gridSizeSixthLayer, blockSizeSixthLayerA>>>(Layer6_Neurons_GPU,
                        Layer6_Weights_GPU,
                        Layer7_Neurons_GPU,
                        Layer6_Mean_GPU,
                        Layer6_StanDev_GPU,
                        Layer6_Gamma_GPU,
                        Layer6_Beta_GPU
                    );
                    
    dim3 blockSizeSixthLayerB(32, 24);
    executeSixthLayer_partB<<< gridSizeSixthLayer, blockSizeSixthLayerB>>>(Layer6_Neurons_GPU,
                        Layer6_Weights_GPU,
                        Layer7_Neurons_GPU,
                        Layer6_Mean_GPU,
                        Layer6_StanDev_GPU,
                        Layer6_Gamma_GPU,
                        Layer6_Beta_GPU
                    );
    
    dim3 blockSizeSixthLayerC(24, 32);
    executeSixthLayer_partC<<< gridSizeSixthLayer, blockSizeSixthLayerC>>>(Layer6_Neurons_GPU,
                        Layer6_Weights_GPU,
                        Layer7_Neurons_GPU,
                        Layer6_Mean_GPU,
                        Layer6_StanDev_GPU,
                        Layer6_Gamma_GPU,
                        Layer6_Beta_GPU
                    );

    
    dim3 blockSizeSixthLayerD(24, 24);
    executeSixthLayer_partD<<< gridSizeSixthLayer, blockSizeSixthLayerD>>>(Layer6_Neurons_GPU,
                        Layer6_Weights_GPU,
                        Layer7_Neurons_GPU,
                        Layer6_Mean_GPU,
                        Layer6_StanDev_GPU,
                        Layer6_Gamma_GPU,
                        Layer6_Beta_GPU
                    );

    cudaFree(Layer6_Weights_GPU);
    cudaFree(Layer6_Mean_GPU);
    cudaFree(Layer6_StanDev_GPU);
    cudaFree(Layer6_Gamma_GPU);
    cudaFree(Layer6_Beta_GPU);
}

void Read_SixthLayer_Data(double *Layer6_Weights_CPU,
    double * Layer6_Mean_CPU,
    double * Layer6_StanDev_CPU,
    double * Layer6_Gamma_CPU,
    double * Layer6_Beta_CPU
){
    read_File("data/SixthLayer/weightsNorm.txt", Layer6_Weights_CPU);
    read_File("data/SixthLayer/Sixth_Layer_Mean.txt", Layer6_Mean_CPU);
    read_File("data/SixthLayer/Sixth_Layer_StanDev.txt", Layer6_StanDev_CPU);
    read_File("data/SixthLayer/Sixth_Layer_Gamma.txt", Layer6_Gamma_CPU);
    read_File("data/SixthLayer/Sixth_Layer_Beta.txt", Layer6_Beta_CPU);
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
    fclose(fp);
}