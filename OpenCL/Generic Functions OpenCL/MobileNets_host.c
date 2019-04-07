#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include <unistd.h>
#include <CL/opencl.h>

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

#define SEVENTH_LAYER_WEIGHT_SIZE 128 * 128
#define SEVENTH_LAYER_OUTPUT_SIZE 57 * 57 * 128
#define SEVENTH_LAYER_CHANNELS 128

#define EIGHTH_LAYER_WEIGHT_SIZE 3 * 3 * 128
#define EIGHTH_LAYER_OUTPUT_SIZE 28 * 28 * 128
#define EIGHTH_LAYER_CHANNELS 128

#define NINTH_LAYER_WEIGHT_SIZE  128 * 256
#define NINTH_LAYER_OUTPUT_SIZE 30 * 30 * 256
#define NINTH_LAYER_CHANNELS 256

#define TENTH_LAYER_WEIGHT_SIZE  9 * 256
#define TENTH_LAYER_OUTPUT_SIZE 28 * 28 * 256
#define TENTH_LAYER_CHANNELS 256

#define ELEVENTH_LAYER_WEIGHT_SIZE  256 * 256
#define ELEVENTH_LAYER_OUTPUT_SIZE 29 * 29 * 256
#define ELEVENTH_LAYER_CHANNELS 256

#define TWELFTH_LAYER_WEIGHT_SIZE  9 * 256
#define TWELFTH_LAYER_OUTPUT_SIZE 14 * 14 * 256
#define TWELFTH_LAYER_CHANNELS 256

#define THIRTEENTH_LAYER_WEIGHT_SIZE  512 * 256
#define THIRTEENTH_LAYER_OUTPUT_SIZE 16 * 16 * 512
#define THIRTEENTH_LAYER_CHANNELS 512

#define FOURTEENTH_LAYER_WEIGHT_SIZE  512 * 9
#define FOURTEENTH_LAYER_OUTPUT_SIZE 14 * 14 * 512
#define FOURTEENTH_LAYER_CHANNELS 512

#define FIFTEENTH_LAYER_WEIGHT_SIZE  512 * 512
#define FIFTEENTH_LAYER_OUTPUT_SIZE 16 * 16 * 512
#define FIFTEENTH_LAYER_CHANNELS 512

#define SIXTEENTH_LAYER_WEIGHT_SIZE  512 * 9
#define SIXTEENTH_LAYER_OUTPUT_SIZE 14 * 14 * 512
#define SIXTEENTH_LAYER_CHANNELS 512

#define SEVENTEENTH_LAYER_WEIGHT_SIZE  512 * 512
#define SEVENTEENTH_LAYER_OUTPUT_SIZE 16 * 16 * 512
#define SEVENTEENTH_LAYER_CHANNELS 512

#define EIGHTEENTH_LAYER_WEIGHT_SIZE  512 * 9
#define EIGHTEENTH_LAYER_OUTPUT_SIZE 14 * 14 * 512
#define EIGHTEENTH_LAYER_CHANNELS 512

#define NINETEENTH_LAYER_WEIGHT_SIZE  512 * 512
#define NINETEENTH_LAYER_OUTPUT_SIZE 16 * 16 * 512
#define NINETEENTH_LAYER_CHANNELS 512

#define TWENTY_LAYER_WEIGHT_SIZE  512 * 9
#define TWENTY_LAYER_OUTPUT_SIZE 14 * 14 * 512
#define TWENTY_LAYER_CHANNELS 512

#define TWENTYONE_LAYER_WEIGHT_SIZE  512 * 512
#define TWENTYONE_LAYER_OUTPUT_SIZE 16 * 16 * 512
#define TWENTYONE_LAYER_CHANNELS 512

#define TWENTYTWO_LAYER_WEIGHT_SIZE  512 * 9
#define TWENTYTWO_LAYER_OUTPUT_SIZE 14 * 14 * 512
#define TWENTYTWO_LAYER_CHANNELS 512

#define TWENTYTHREE_LAYER_WEIGHT_SIZE  512 * 512
#define TWENTYTHREE_LAYER_OUTPUT_SIZE 15 * 15 * 512
#define TWENTYTHREE_LAYER_CHANNELS 512

#define TWENTYFOUR_LAYER_WEIGHT_SIZE  9 * 512
#define TWENTYFOUR_LAYER_OUTPUT_SIZE 7 * 7 * 512
#define TWENTYFOUR_LAYER_CHANNELS 512

#define TWENTYFIVE_LAYER_WEIGHT_SIZE  1024 * 512
#define TWENTYFIVE_LAYER_OUTPUT_SIZE 9 * 9 * 1024
#define TWENTYFIVE_LAYER_CHANNELS 1024

#define TWENTYSIX_LAYER_WEIGHT_SIZE  1024 * 9
#define TWENTYSIX_LAYER_OUTPUT_SIZE 7 * 7 * 1024
#define TWENTYSIX_LAYER_CHANNELS 1024

#define TWENTYSEVEN_LAYER_WEIGHT_SIZE  1024 * 1024
#define TWENTYSEVEN_LAYER_OUTPUT_SIZE 7 * 7 * 1024
#define TWENTYSEVEN_LAYER_CHANNELS 1024

// Global Average Pooling Layer
#define TWENTYEIGHT_LAYER_OUTPUT_SIZE 1024

// Fully Connected Layer
#define TWENTYNINE_LAYER_OUTPUT_SIZE 1000
#define TWENTYNINE_LAYER_WEIGHT_SIZE 1024 * 1000

typedef enum
{
    false = ( 1 == 0 ),
    true = ( ! false )
} bool;
 
void NeuralNetwork();
void read_File(const char * weightFileName, double *Layer1_Weights_CPU);
void read_Input_File(const char * inputFileName, double *Layer1_Neurons_CPU);

char * read_Kernel_File();

void Read_First_Layer_Data(double * Layer1_Neurons_CPU,
    double * Layer1_Weights_CPU,
    double * Layer1_Mean_CPU,
    double * Layer1_StanDev_CPU,
    double * Layer1_Gamma_CPU,
    double * Layer1_Beta_CPU
);

void Execute_First_Layer(cl_mem Layer2_Neurons_GPU,
    cl_context context,
    cl_command_queue queue,
    cl_program program
);

void Read_SecondLayer_Data(double *Layer2_Weights_CPU,
    double *Layer2_Mean_CPU,
    double *Layer2_StanDev_CPU,
    double *Layer2_Gamma_CPU,
    double *Layer2_Beta_CPU
);

void Execute_Second_Layer(cl_mem Layer2_Neurons_GPU,
    cl_mem Layer3_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_ThirdLayer_Data(double *Layer3_Weights_CPU,
    double *Layer3_Mean_CPU,
    double *Layer3_StanDev_CPU,
    double *Layer3_Gamma_CPU,
    double *Layer3_Beta_CPU
);

void Execute_Third_Layer(cl_mem Layer3_Neurons_GPU,
    cl_mem Layer4_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_FourthLayer_Data(double *Layer4_Weights_CPU,
    double *Layer4_Mean_CPU,
    double *Layer4_StanDev_CPU,
    double *Layer4_Gamma_CPU,
    double *Layer4_Beta_CPU
);

void Execute_Fourth_Layer(cl_mem Layer4_Neurons_GPU,
    cl_mem Layer5_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_FifthLayer_Data(double *Layer5_Weights_CPU,
    double *Layer5_Mean_CPU,
    double *Layer5_StanDev_CPU,
    double *Layer5_Gamma_CPU,
    double *Layer5_Beta_CPU
);

void Execute_Fifth_Layer(cl_mem Layer5_Neurons_GPU,
    cl_mem Layer6_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_SixthLayer_Data(double *Layer6_Weights_CPU,
    double *Layer6_Mean_CPU,
    double *Layer6_StanDev_CPU,
    double *Layer6_Gamma_CPU,
    double *Layer6_Beta_CPU
);

void Execute_Sixth_Layer(cl_mem Layer6_Neurons_GPU,
    cl_mem Layer7_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_SeventhLayer_Data(double *Layer7_Weights_CPU,
    double *Layer7_Mean_CPU,
    double *Layer7_StanDev_CPU,
    double *Layer7_Gamma_CPU,
    double *Layer7_Beta_CPU
);

void Execute_Seventh_Layer(cl_mem Layer7_Neurons_GPU,
    cl_mem Layer8_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_EighthLayer_Data(double *Layer8_Weights_CPU,
    double *Layer8_Mean_CPU,
    double *Layer8_StanDev_CPU,
    double *Layer8_Gamma_CPU,
    double *Layer8_Beta_CPU
);

void Execute_Eighth_Layer(cl_mem Layer8_Neurons_GPU,
    cl_mem Layer9_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_NinthLayer_Data(double *Layer9_Weights_CPU,
    double *Layer9_Mean_CPU,
    double *Layer9_StanDev_CPU,
    double *Layer9_Gamma_CPU,
    double *Layer9_Beta_CPU
);

void Execute_Ninth_Layer(cl_mem Layer9_Neurons_GPU,
    cl_mem Layer10_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_TenthLayer_Data(double *Layer10_Weights_CPU,
    double *Layer10_Mean_CPU,
    double *Layer10_StanDev_CPU,
    double *Layer10_Gamma_CPU,
    double *Layer10_Beta_CPU
);

void Execute_Tenth_Layer(cl_mem Layer10_Neurons_GPU,
    cl_mem Layer11_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_EleventhLayer_Data(double *Layer11_Weights_CPU,
    double *Layer11_Mean_CPU,
    double *Layer11_StanDev_CPU,
    double *Layer11_Gamma_CPU,
    double *Layer11_Beta_CPU
);

void Execute_Eleventh_Layer(cl_mem Layer11_Neurons_GPU,
    cl_mem Layer12_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_TwelvethLayer_Data(double *Layer12_Weights_CPU,
    double *Layer12_Mean_CPU,
    double *Layer12_StanDev_CPU,
    double *Layer12_Gamma_CPU,
    double *Layer12_Beta_CPU
);

void Execute_Twelveth_Layer(cl_mem Layer12_Neurons_GPU,
    cl_mem Layer13_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_ThirteenthLayer_Data(double *Layer13_Weights_CPU,
    double *Layer13_Mean_CPU,
    double *Layer13_StanDev_CPU,
    double *Layer13_Gamma_CPU,
    double *Layer13_Beta_CPU
);

void Execute_Thirteenth_Layer(cl_mem Layer13_Neurons_GPU,
    cl_mem Layer14_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_FourteenthLayer_Data(double *Layer14_Weights_CPU,
    double *Layer14_Mean_CPU,
    double *Layer14_StanDev_CPU,
    double *Layer14_Gamma_CPU,
    double *Layer14_Beta_CPU
);

void Execute_Fourteenth_Layer(cl_mem Layer14_Neurons_GPU,
    cl_mem Layer15_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_FifteenthLayer_Data(double *Layer15_Weights_CPU,
    double *Layer15_Mean_CPU,
    double *Layer15_StanDev_CPU,
    double *Layer15_Gamma_CPU,
    double *Layer15_Beta_CPU
);

void Execute_Fifteenth_Layer(cl_mem Layer15_Neurons_GPU,
    cl_mem Layer16_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_SixteenthLayer_Data(double *Layer16_Weights_CPU,
    double *Layer16_Mean_CPU,
    double *Layer16_StanDev_CPU,
    double *Layer16_Gamma_CPU,
    double *Layer16_Beta_CPU
);

void Execute_Sixteenth_Layer(cl_mem Layer16_Neurons_GPU,
    cl_mem Layer17_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_SeventeenthLayer_Data(double *Layer17_Weights_CPU,
    double *Layer17_Mean_CPU,
    double *Layer17_StanDev_CPU,
    double *Layer17_Gamma_CPU,
    double *Layer17_Beta_CPU
);

void Execute_Seventeenth_Layer(cl_mem Layer17_Neurons_GPU,
    cl_mem Layer18_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_EighteenthLayer_Data(double *Layer18_Weights_CPU,
    double *Layer18_Mean_CPU,
    double *Layer18_StanDev_CPU,
    double *Layer18_Gamma_CPU,
    double *Layer18_Beta_CPU
);

void Execute_Eighteenth_Layer(cl_mem Layer18_Neurons_GPU,
    cl_mem Layer19_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_NineteenthLayer_Data(double *Layer19_Weights_CPU,
    double *Layer19_Mean_CPU,
    double *Layer19_StanDev_CPU,
    double *Layer19_Gamma_CPU,
    double *Layer19_Beta_CPU
);

void Execute_Nineteenth_Layer(cl_mem Layer19_Neurons_GPU,
    cl_mem Layer20_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_TwentyLayer_Data(double *Layer20_Weights_CPU,
    double *Layer20_Mean_CPU,
    double *Layer20_StanDev_CPU,
    double *Layer20_Gamma_CPU,
    double *Layer20_Beta_CPU
);

void Execute_Twenty_Layer(cl_mem Layer20_Neurons_GPU,
    cl_mem Layer21_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_TwentyOneLayer_Data(double *Layer20_Weights_CPU,
    double *Layer20_Mean_CPU,
    double *Layer20_StanDev_CPU,
    double *Layer20_Gamma_CPU,
    double *Layer20_Beta_CPU
);

void Execute_TwentyOne_Layer(cl_mem Layer20_Neurons_GPU,
    cl_mem Layer21_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_TwentyTwoLayer_Data(double *Layer22_Weights_CPU,
    double *Layer22_Mean_CPU,
    double *Layer22_StanDev_CPU,
    double *Layer22_Gamma_CPU,
    double *Layer22_Beta_CPU
);

void Execute_TwentyTwo_Layer(cl_mem Layer22_Neurons_GPU,
    cl_mem Layer23_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_TwentyThreeLayer_Data(double *Layer23_Weights_CPU,
    double *Layer23_Mean_CPU,
    double *Layer23_StanDev_CPU,
    double *Layer23_Gamma_CPU,
    double *Layer23_Beta_CPU
);

void Execute_TwentyThree_Layer(cl_mem Layer23_Neurons_GPU,
    cl_mem Layer24_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_TwentyFourLayer_Data(double *Layer24_Weights_CPU,
    double *Layer24_Mean_CPU,
    double *Layer24_StanDev_CPU,
    double *Layer24_Gamma_CPU,
    double *Layer24_Beta_CPU
);

void Execute_TwentyFour_Layer(cl_mem Layer24_Neurons_GPU,
    cl_mem Layer25_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_TwentyFiveLayer_Data(double *Layer25_Weights_CPU,
    double *Layer25_Mean_CPU,
    double *Layer25_StanDev_CPU,
    double *Layer25_Gamma_CPU,
    double *Layer25_Beta_CPU
);

void Execute_TwentyFive_Layer(cl_mem Layer25_Neurons_GPU,
    cl_mem Layer26_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_TwentySixLayer_Data(double *Layer26_Weights_CPU,
    double *Layer26_Mean_CPU,
    double *Layer26_StanDev_CPU,
    double *Layer26_Gamma_CPU,
    double *Layer26_Beta_CPU
);

void Execute_TwentySix_Layer(cl_mem Layer26_Neurons_GPU,
    cl_mem Layer27_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Read_TwentySevenLayer_Data(double *Layer27_Weights_CPU,
    double *Layer27_Mean_CPU,
    double *Layer27_StanDev_CPU,
    double *Layer27_Gamma_CPU,
    double *Layer27_Beta_CPU
);

void Execute_TwentySeven_Layer(cl_mem Layer27_Neurons_GPU,
    cl_mem Layer28_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

void Execute_TwentyEight_Layer(cl_mem Layer28_Neurons_GPU,
    cl_mem Layer29_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
);

int main( int argc, char* argv[] )
{
    NeuralNetwork();

    return 0;
}

void NeuralNetwork(){

    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel
 
    // Bind to platform
    cl_int err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    char* value;
    size_t valueSize;
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, valueSize, value, NULL);
    printf("\n Currently using device: %s\n", value);
    free(value);

    const char *kernelSource = read_Kernel_File();
 
    // Create a context 
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
 
    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
 
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                            (char **) &kernelSource, NULL, &err);
 
    // Build the program executable
    clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    FILE *fOutput;
    int value_size;
    /* ************************************************ FIRST LAYER ******************************************************** */
    cl_mem Layer2_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, FIRST_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_First_Layer(Layer2_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_FIRST_LAYER_WEIGHTS = false;
    if(SAVE_FIRST_LAYER_WEIGHTS){
        
        double *Layer2_Neurons_CPU = (double *) malloc(sizeof(double) * FIRST_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer2_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * FIRST_LAYER_OUTPUT_SIZE, Layer2_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/FirstLayer/output.txt", "w");
        value_size = FIRST_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer2_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer2_Neurons_CPU);
    }
    printf("\n Layer 1 Execution complete !!! \n");
    /* ************************************************ FIRST LAYER COMPLETE *********************************************** */

    /* ************************************************ SECOND LAYER START ******************************************************** */
    cl_mem Layer3_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SECOND_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_Second_Layer(Layer2_Neurons_GPU, Layer3_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_SECOND_LAYER_WEIGHTS = false;
    if(SAVE_SECOND_LAYER_WEIGHTS){
        
        double *Layer3_Neurons_CPU = (double *) malloc(sizeof(double) * SECOND_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer3_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * SECOND_LAYER_OUTPUT_SIZE, Layer3_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/SecondLayer/output.txt", "w");
        value_size = SECOND_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer3_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer3_Neurons_CPU);
    }
    clReleaseMemObject(Layer2_Neurons_GPU);
    printf("\n Layer 2 Execution complete !!! \n");
    /* ************************************************ SECOND LAYER COMPLETE *********************************************** */

    /* ************************************************ THIRD LAYER START ******************************************************** */
    cl_mem Layer4_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, THIRD_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_Third_Layer(Layer3_Neurons_GPU, Layer4_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_THIRD_LAYER_WEIGHTS = false;
    if(SAVE_THIRD_LAYER_WEIGHTS){
        
        double *Layer4_Neurons_CPU = (double *) malloc(sizeof(double) * THIRD_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer4_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * THIRD_LAYER_OUTPUT_SIZE, Layer4_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/ThirdLayer/output.txt", "w");
        value_size = THIRD_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer4_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer4_Neurons_CPU);
    }
    clReleaseMemObject(Layer3_Neurons_GPU);
    printf("\n Layer 3 Execution complete !!! \n");
    /* ************************************************ THIRD LAYER COMPLETE *********************************************** */

    /* ************************************************ FOURTH LAYER START ******************************************************** */
    cl_mem Layer5_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, FOURTH_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_Fourth_Layer(Layer4_Neurons_GPU, Layer5_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_FOURTH_LAYER_WEIGHTS = false;
    if(SAVE_FOURTH_LAYER_WEIGHTS){
        
        double *Layer5_Neurons_CPU = (double *) malloc(sizeof(double) * FOURTH_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer5_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * FOURTH_LAYER_OUTPUT_SIZE, Layer5_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/FourthLayer/output.txt", "w");
        value_size = FOURTH_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer5_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer5_Neurons_CPU);
    }
    clReleaseMemObject(Layer4_Neurons_GPU);
    printf("\n Layer 4 Execution complete !!! \n");
    /* ************************************************ FOURTH LAYER COMPLETE *********************************************** */

    /* ************************************************ FIFTH LAYER START ******************************************************** */
    cl_mem Layer6_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, FIFTH_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_Fifth_Layer(Layer5_Neurons_GPU, Layer6_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_FIFTH_LAYER_WEIGHTS = false;
    if(SAVE_FIFTH_LAYER_WEIGHTS){
        
        double *Layer6_Neurons_CPU = (double *) malloc(sizeof(double) * FIFTH_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer6_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * FIFTH_LAYER_OUTPUT_SIZE, Layer6_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/FifthLayer/output.txt", "w");
        value_size = FIFTH_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer6_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer6_Neurons_CPU);
    }
    clReleaseMemObject(Layer5_Neurons_GPU);
    printf("\n Layer 5 Execution complete !!! \n");
    /* ************************************************ FIFTH LAYER COMPLETE *********************************************** */
    
    /* ************************************************ SIXTH LAYER START ******************************************************** */
    cl_mem Layer7_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIXTH_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_Sixth_Layer(Layer6_Neurons_GPU, Layer7_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_SEVENTH_LAYER_WEIGHTS = false;
    if(SAVE_SEVENTH_LAYER_WEIGHTS){
        
        double *Layer7_Neurons_CPU = (double *) malloc(sizeof(double) * SIXTH_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer7_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * SIXTH_LAYER_OUTPUT_SIZE, Layer7_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/SixthLayer/output.txt", "w");
        value_size = SIXTH_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer7_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer7_Neurons_CPU);
    }
    clReleaseMemObject(Layer6_Neurons_GPU);
    printf("\n Layer 6 Execution complete !!! \n");
    /* ************************************************ SIXTH LAYER COMPLETE *********************************************** */

    /* ************************************************ SEVENTH LAYER START ******************************************************** */
    cl_mem Layer8_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SEVENTH_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_Seventh_Layer(Layer7_Neurons_GPU, Layer8_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_SIXTH_LAYER_WEIGHTS = false;
    if(SAVE_SIXTH_LAYER_WEIGHTS){
        
        double *Layer8_Neurons_CPU = (double *) malloc(sizeof(double) * SEVENTH_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer8_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * SEVENTH_LAYER_OUTPUT_SIZE, Layer8_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/SeventhLayer/output.txt", "w");
        value_size = SEVENTH_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer8_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer8_Neurons_CPU);
    }
    clReleaseMemObject(Layer7_Neurons_GPU);
    printf("\n Layer 7 Execution complete !!! \n");
    /* ************************************************ SEVENTH LAYER COMPLETE *********************************************** */

    /* ************************************************ EIGHTH LAYER START ******************************************************** */
    cl_mem Layer9_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SEVENTH_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_Eighth_Layer(Layer8_Neurons_GPU, Layer9_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_EIGHTH_LAYER_WEIGHTS = false;
    if(SAVE_EIGHTH_LAYER_WEIGHTS){
        
        double *Layer9_Neurons_CPU = (double *) malloc(sizeof(double) * EIGHTH_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer9_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * EIGHTH_LAYER_OUTPUT_SIZE, Layer9_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/EighthLayer/output.txt", "w");
        value_size = EIGHTH_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer9_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer9_Neurons_CPU);
    }
    clReleaseMemObject(Layer8_Neurons_GPU);
    printf("\n Layer 8 Execution complete !!! \n");
    /* ************************************************ EIGHTH LAYER COMPLETE *********************************************** */

    /* ************************************************ NINTH LAYER START ******************************************************** */
    cl_mem Layer10_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, NINTH_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_Ninth_Layer(Layer9_Neurons_GPU, Layer10_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_NINTH_LAYER_WEIGHTS = false;
    if(SAVE_NINTH_LAYER_WEIGHTS){
        
        double *Layer10_Neurons_CPU = (double *) malloc(sizeof(double) * NINTH_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer10_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * NINTH_LAYER_OUTPUT_SIZE, Layer10_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/NinthLayer/output.txt", "w");
        value_size = NINTH_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer10_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer10_Neurons_CPU);
    }
    clReleaseMemObject(Layer9_Neurons_GPU);
    printf("\n Layer 9 Execution complete !!! \n");
    /* ************************************************ NINTH LAYER COMPLETE *********************************************** */

    /* ************************************************ TENTH LAYER START ******************************************************** */
    cl_mem Layer11_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, TENTH_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_Tenth_Layer(Layer10_Neurons_GPU, Layer11_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_TENTH_LAYER_WEIGHTS = false;
    if(SAVE_TENTH_LAYER_WEIGHTS){
        
        double *Layer11_Neurons_CPU = (double *) malloc(sizeof(double) * TENTH_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer11_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * TENTH_LAYER_OUTPUT_SIZE, Layer11_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TenthLayer/output.txt", "w");
        value_size = TENTH_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer11_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer11_Neurons_CPU);
    }
    clReleaseMemObject(Layer10_Neurons_GPU);
    printf("\n Layer 10 Execution complete !!! \n");
    /* ************************************************ TENTH LAYER COMPLETE *********************************************** */

    /* ************************************************ ELEVENTH LAYER START ******************************************************** */
    cl_mem Layer12_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, ELEVENTH_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_Eleventh_Layer(Layer11_Neurons_GPU, Layer12_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_ELEVENTH_LAYER_WEIGHTS = false;
    if(SAVE_ELEVENTH_LAYER_WEIGHTS){
        
        double *Layer12_Neurons_CPU = (double *) malloc(sizeof(double) * ELEVENTH_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer12_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * ELEVENTH_LAYER_OUTPUT_SIZE, Layer12_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/EleventhLayer/output.txt", "w");
        value_size = ELEVENTH_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer12_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer12_Neurons_CPU);
    }
    clReleaseMemObject(Layer11_Neurons_GPU);
    printf("\n Layer 11 Execution complete !!! \n");
    /* ************************************************ TENTH LAYER COMPLETE *********************************************** */

    /* ************************************************ TWELFTH LAYER START ******************************************************** */
    cl_mem Layer13_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, TWELFTH_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_Twelveth_Layer(Layer12_Neurons_GPU, Layer13_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_TWELFTH_LAYER_WEIGHTS = false;
    if(SAVE_TWELFTH_LAYER_WEIGHTS){
        
        double *Layer13_Neurons_CPU = (double *) malloc(sizeof(double) * TWELFTH_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer13_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * TWELFTH_LAYER_OUTPUT_SIZE, Layer13_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwelvethLayer/output.txt", "w");
        value_size = TWELFTH_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer13_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer13_Neurons_CPU);
    }
    clReleaseMemObject(Layer12_Neurons_GPU);
    printf("\n Layer 12 Execution complete !!! \n");
    /* ************************************************ TWELFTH LAYER COMPLETE *********************************************** */

    /* ************************************************ THIRTEENTH LAYER START ******************************************************** */
    cl_mem Layer14_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, THIRTEENTH_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_Thirteenth_Layer(Layer13_Neurons_GPU, Layer14_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_THIRTEENTH_LAYER_WEIGHTS = false;
    if(SAVE_THIRTEENTH_LAYER_WEIGHTS){
        
        double *Layer14_Neurons_CPU = (double *) malloc(sizeof(double) * THIRTEENTH_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer14_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * THIRTEENTH_LAYER_OUTPUT_SIZE, Layer14_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/ThirteenthLayer/output.txt", "w");
        value_size = THIRTEENTH_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer14_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer14_Neurons_CPU);
    }
    clReleaseMemObject(Layer13_Neurons_GPU);
    printf("\n Layer 13 Execution complete !!! \n");
    /* ************************************************ THIRTEENTH LAYER COMPLETE *********************************************** */

    /* ************************************************ FOURTEENTH LAYER START ******************************************************** */
    cl_mem Layer15_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, FOURTEENTH_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_Fourteenth_Layer(Layer14_Neurons_GPU, Layer15_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_FOURTEENTH_LAYER_WEIGHTS = false;
    if(SAVE_FOURTEENTH_LAYER_WEIGHTS){
        
        double *Layer15_Neurons_CPU = (double *) malloc(sizeof(double) * FOURTEENTH_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer15_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * FOURTEENTH_LAYER_OUTPUT_SIZE, Layer15_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/FourteenthLayer/output.txt", "w");
        value_size = FOURTEENTH_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer15_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer15_Neurons_CPU);
    }
    clReleaseMemObject(Layer14_Neurons_GPU);
    printf("\n Layer 14 Execution complete !!! \n");
    /* ************************************************ FOURTEENTH LAYER COMPLETE *********************************************** */

    /* ************************************************ FIFTEENTH LAYER START ******************************************************** */
    cl_mem Layer16_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, FIFTEENTH_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_Fifteenth_Layer(Layer15_Neurons_GPU, Layer16_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_FIFTEENTH_LAYER_WEIGHTS = false;
    if(SAVE_FIFTEENTH_LAYER_WEIGHTS){
        
        double *Layer16_Neurons_CPU = (double *) malloc(sizeof(double) * FIFTEENTH_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer16_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * FIFTEENTH_LAYER_OUTPUT_SIZE, Layer16_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/FifteenthLayer/output.txt", "w");
        value_size = FIFTEENTH_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer16_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer16_Neurons_CPU);
    }
    clReleaseMemObject(Layer15_Neurons_GPU);
    printf("\n Layer 15 Execution complete !!! \n");
    /* ************************************************ FIFTEENTH LAYER COMPLETE *********************************************** */

    /* ************************************************ SIXTEENTH LAYER START ******************************************************** */
    cl_mem Layer17_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIXTEENTH_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_Sixteenth_Layer(Layer16_Neurons_GPU, Layer17_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_SIXTEENTH_LAYER_WEIGHTS = false;
    if(SAVE_SIXTEENTH_LAYER_WEIGHTS){
        
        double *Layer17_Neurons_CPU = (double *) malloc(sizeof(double) * SIXTEENTH_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer17_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * SIXTEENTH_LAYER_OUTPUT_SIZE, Layer17_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/SixteenthLayer/output.txt", "w");
        value_size = SIXTEENTH_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer17_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer17_Neurons_CPU);
    }
    clReleaseMemObject(Layer16_Neurons_GPU);
    printf("\n Layer 16 Execution complete !!! \n");
    /* ************************************************ SIXTEENTH LAYER COMPLETE *********************************************** */

    /* ************************************************ SEVENTEENTH LAYER START ******************************************************** */
    cl_mem Layer18_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SEVENTEENTH_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_Seventeenth_Layer(Layer17_Neurons_GPU, Layer18_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_SEVENTEENTH_LAYER_WEIGHTS = false;
    if(SAVE_SEVENTEENTH_LAYER_WEIGHTS){
        
        double *Layer18_Neurons_CPU = (double *) malloc(sizeof(double) * SEVENTEENTH_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer18_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * SEVENTEENTH_LAYER_OUTPUT_SIZE, Layer18_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/SeventeenthLayer/output.txt", "w");
        value_size = SEVENTEENTH_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer18_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer18_Neurons_CPU);
    }
    clReleaseMemObject(Layer17_Neurons_GPU);
    printf("\n Layer 17 Execution complete !!! \n");
    /* ************************************************ SEVENTEENTH LAYER COMPLETE *********************************************** */

    /* ************************************************ EIGHTEENTH LAYER START ******************************************************** */
    cl_mem Layer19_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, EIGHTEENTH_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_Eighteenth_Layer(Layer18_Neurons_GPU, Layer19_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_EIGHTEENTH_LAYER_WEIGHTS = false;
    if(SAVE_EIGHTEENTH_LAYER_WEIGHTS){
        
        double *Layer19_Neurons_CPU = (double *) malloc(sizeof(double) * EIGHTEENTH_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer19_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * EIGHTEENTH_LAYER_OUTPUT_SIZE, Layer19_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/EighteenthLayer/output.txt", "w");
        value_size = EIGHTEENTH_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer19_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer19_Neurons_CPU);
    }
    clReleaseMemObject(Layer18_Neurons_GPU);
    printf("\n Layer 18 Execution complete !!! \n");
    /* ************************************************ EIGHTEENTH LAYER COMPLETE *********************************************** */

    /* ************************************************ NINETEENTH LAYER START ******************************************************** */
    cl_mem Layer20_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, NINETEENTH_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_Nineteenth_Layer(Layer19_Neurons_GPU, Layer20_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_NINEEENTH_LAYER_WEIGHTS = false;
    if(SAVE_NINEEENTH_LAYER_WEIGHTS){
        
        double *Layer20_Neurons_CPU = (double *) malloc(sizeof(double) * NINETEENTH_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer20_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * NINETEENTH_LAYER_OUTPUT_SIZE, Layer20_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/NineteenthLayer/output.txt", "w");
        value_size = NINETEENTH_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer20_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer20_Neurons_CPU);
    }
    clReleaseMemObject(Layer19_Neurons_GPU);
    printf("\n Layer 19 Execution complete !!! \n");
    /* ************************************************ NINETEENTH LAYER COMPLETE *********************************************** */

    /* ************************************************ TWENTY LAYER START ******************************************************** */
    cl_mem Layer21_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, TWENTY_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_Twenty_Layer(Layer20_Neurons_GPU, Layer21_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_TWENTY_LAYER_WEIGHTS = false;
    if(SAVE_TWENTY_LAYER_WEIGHTS){
        
        double *Layer21_Neurons_CPU = (double *) malloc(sizeof(double) * TWENTY_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer21_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * TWENTY_LAYER_OUTPUT_SIZE, Layer21_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwentyLayer/output.txt", "w");
        value_size = TWENTY_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer21_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer21_Neurons_CPU);
    }
    clReleaseMemObject(Layer20_Neurons_GPU);
    printf("\n Layer 20 Execution complete !!! \n");
    /* ************************************************ TWENTY LAYER COMPLETE *********************************************** */

    /* ************************************************ TWENTY ONE LAYER START ******************************************************** */
    cl_mem Layer22_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, TWENTYONE_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_TwentyOne_Layer(Layer21_Neurons_GPU, Layer22_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_TWENTY_ONE_LAYER_WEIGHTS = false;
    if(SAVE_TWENTY_ONE_LAYER_WEIGHTS){
        
        double *Layer22_Neurons_CPU = (double *) malloc(sizeof(double) * TWENTYONE_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer22_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * TWENTYONE_LAYER_OUTPUT_SIZE, Layer22_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwentyOneLayer/output.txt", "w");
        value_size = TWENTYONE_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer22_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer22_Neurons_CPU);
    }
    clReleaseMemObject(Layer21_Neurons_GPU);
    printf("\n Layer 21 Execution complete !!! \n");
    /* ************************************************ TWENTY ONE LAYER COMPLETE *********************************************** */

    /* ************************************************ TWENTY TWO LAYER START ******************************************************** */
    cl_mem Layer23_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, TWENTYTWO_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_TwentyTwo_Layer(Layer22_Neurons_GPU, Layer23_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_TWENTY_TWO_LAYER_WEIGHTS = false;
    if(SAVE_TWENTY_TWO_LAYER_WEIGHTS){
        
        double *Layer23_Neurons_CPU = (double *) malloc(sizeof(double) * TWENTYTWO_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer23_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * TWENTYTWO_LAYER_OUTPUT_SIZE, Layer23_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwentyTwoLayer/output.txt", "w");
        value_size = TWENTYTWO_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer23_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer23_Neurons_CPU);
    }
    clReleaseMemObject(Layer22_Neurons_GPU);
    printf("\n Layer 22 Execution complete !!! \n");
    /* ************************************************ TWENTY TWO LAYER COMPLETE *********************************************** */

    /* ************************************************ TWENTY THREE LAYER START ******************************************************** */
    cl_mem Layer24_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, TWENTYTHREE_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_TwentyThree_Layer(Layer23_Neurons_GPU, Layer24_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_TWENTY_THREE_LAYER_WEIGHTS = false;
    if(SAVE_TWENTY_THREE_LAYER_WEIGHTS){
        
        double *Layer24_Neurons_CPU = (double *) malloc(sizeof(double) * TWENTYTHREE_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer24_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * TWENTYTHREE_LAYER_OUTPUT_SIZE, Layer24_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwentyThreeLayer/output.txt", "w");
        value_size = TWENTYTHREE_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer24_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer24_Neurons_CPU);
    }
    clReleaseMemObject(Layer23_Neurons_GPU);
    printf("\n Layer 23 Execution complete !!! \n");
    /* ************************************************ TWENTY THREE LAYER COMPLETE *********************************************** */

    /* ************************************************ TWENTY FOUR LAYER START ******************************************************** */
    cl_mem Layer25_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, TWENTYFOUR_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_TwentyFour_Layer(Layer24_Neurons_GPU, Layer25_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_TWENTY_FOUR_LAYER_WEIGHTS = false;
    if(SAVE_TWENTY_FOUR_LAYER_WEIGHTS){
        
        double *Layer25_Neurons_CPU = (double *) malloc(sizeof(double) * TWENTYFOUR_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer25_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * TWENTYFOUR_LAYER_OUTPUT_SIZE, Layer25_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwentyFourLayer/output.txt", "w");
        value_size = TWENTYFOUR_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer25_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer25_Neurons_CPU);
    }
    clReleaseMemObject(Layer24_Neurons_GPU);
    printf("\n Layer 24 Execution complete !!! \n");
    /* ************************************************ TWENTY FOUR LAYER COMPLETE *********************************************** */

    /* ************************************************ TWENTY FIVE LAYER START ******************************************************** */
    cl_mem Layer26_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, TWENTYFIVE_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_TwentyFive_Layer(Layer25_Neurons_GPU, Layer26_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_TWENTY_FIVE_LAYER_WEIGHTS = false;
    if(SAVE_TWENTY_FIVE_LAYER_WEIGHTS){
        
        double *Layer26_Neurons_CPU = (double *) malloc(sizeof(double) * TWENTYFIVE_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer26_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * TWENTYFIVE_LAYER_OUTPUT_SIZE, Layer26_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwentyFiveLayer/output.txt", "w");
        value_size = TWENTYFIVE_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer26_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer26_Neurons_CPU);
    }
    clReleaseMemObject(Layer25_Neurons_GPU);
    printf("\n Layer 25 Execution complete !!! \n");
    /* ************************************************ TWENTY FIVE LAYER COMPLETE *********************************************** */

    /* ************************************************ TWENTY SIX LAYER START ******************************************************** */
    cl_mem Layer27_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, TWENTYSIX_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_TwentySix_Layer(Layer26_Neurons_GPU, Layer27_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_TWENTY_SIX_LAYER_WEIGHTS = false;
    if(SAVE_TWENTY_SIX_LAYER_WEIGHTS){
        
        double *Layer27_Neurons_CPU = (double *) malloc(sizeof(double) * TWENTYSIX_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer27_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * TWENTYSIX_LAYER_OUTPUT_SIZE, Layer27_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwentySixLayer/output.txt", "w");
        value_size = TWENTYSIX_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer27_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer27_Neurons_CPU);
    }
    clReleaseMemObject(Layer26_Neurons_GPU);
    printf("\n Layer 26 Execution complete !!! \n");
    /* ************************************************ TWENTY SIX LAYER COMPLETE *********************************************** */

    /* ************************************************ TWENTY SEVEN LAYER START ******************************************************** */
    cl_mem Layer28_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, TWENTYSEVEN_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_TwentySeven_Layer(Layer27_Neurons_GPU, Layer28_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_TWENTY_SEVEN_LAYER_WEIGHTS = false;
    if(SAVE_TWENTY_SEVEN_LAYER_WEIGHTS){
        
        double *Layer28_Neurons_CPU = (double *) malloc(sizeof(double) * TWENTYSEVEN_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer28_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * TWENTYSEVEN_LAYER_OUTPUT_SIZE, Layer28_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwentySevenLayer/output.txt", "w");
        value_size = TWENTYSEVEN_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer28_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer28_Neurons_CPU);
    }
    clReleaseMemObject(Layer27_Neurons_GPU);
    printf("\n Layer 27 Execution complete !!! \n");
    /* ************************************************ TWENTY SEVEN LAYER COMPLETE *********************************************** */

    /* ************************************************ TWENTY EIGHT LAYER START ******************************************************** */
    cl_mem Layer29_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, TWENTYEIGHT_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_TwentyEight_Layer(Layer28_Neurons_GPU, Layer29_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_TWENTY_EIGHT_LAYER_WEIGHTS = false;
    if(SAVE_TWENTY_EIGHT_LAYER_WEIGHTS){
        
        double *Layer29_Neurons_CPU = (double *) malloc(sizeof(double) * TWENTYEIGHT_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer29_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * TWENTYEIGHT_LAYER_OUTPUT_SIZE, Layer29_Neurons_CPU, 0, NULL, NULL );

        clFinish(queue);

        // Logic to save into the file to verify the results
        fOutput = fopen("data/TwentyEightLayer/output.txt", "w");
        value_size = TWENTYEIGHT_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer29_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer29_Neurons_CPU);
    }
    clReleaseMemObject(Layer28_Neurons_GPU);
    printf("\n Layer 28 Execution complete !!! \n");
    /* ************************************************ TWENTY EIGHT LAYER COMPLETE *********************************************** */

    /* ************************************************ TWENTY NINE LAYER START ******************************************************** */
    cl_mem Layer30_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, TWENTYNINE_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_TwentyNine_Layer(Layer29_Neurons_GPU, Layer30_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_TWENTY_NINE_LAYER_WEIGHTS = true;
    if(SAVE_TWENTY_NINE_LAYER_WEIGHTS){
        double *Layer30_Neurons_CPU = (double *) malloc(sizeof(double) * TWENTYNINE_LAYER_OUTPUT_SIZE);
        clEnqueueReadBuffer(queue, Layer30_Neurons_GPU, CL_TRUE, 0,
                                sizeof(double) * TWENTYNINE_LAYER_OUTPUT_SIZE, Layer30_Neurons_CPU, 0, NULL, NULL );
        clFinish(queue);

        fOutput = fopen("data/TwentyNineLayer/output_w.txt", "w");
        value_size = TWENTYNINE_LAYER_OUTPUT_SIZE;
        for(int i = 0 ; i < value_size ; i++){
            fprintf (fOutput, "%0.6lf\n", Layer30_Neurons_CPU[i]);
        }
        fclose(fOutput);

        // Logic to save into the file to verify the results
        
        value_size = TWENTYNINE_LAYER_OUTPUT_SIZE;
        double sum = 0.0;
        for(int i = 0 ; i < value_size ; i++){
            sum += exp(Layer30_Neurons_CPU[i]);
        }

        for(int i = 0 ; i < value_size ; i++){
            Layer30_Neurons_CPU[i] = (exp(Layer30_Neurons_CPU[i]) * 1.0 / sum);
        }

        fOutput = fopen("data/TwentyNineLayer/outputf.txt", "w");
        for(int i = 0 ; i < value_size ; i++){
            fprintf(fOutput, "%0.6lf\n", Layer30_Neurons_CPU[i]);
        }
        fclose(fOutput);

        free(Layer30_Neurons_CPU);
    }
    clReleaseMemObject(Layer29_Neurons_GPU);
    clReleaseMemObject(Layer30_Neurons_GPU);
    printf("\n Layer 29 Execution complete !!! \n");
    /* ************************************************ TWENTY NINE LAYER COMPLETE *********************************************** */

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);    
}

void Execute_TwentyNine_Layer(cl_mem Layer29_Neurons_GPU,
    cl_mem Layer30_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer29_Weights_CPU = (double *) malloc(sizeof(double) * TWENTYNINE_LAYER_WEIGHT_SIZE);
    double *Layer29_Bias_CPU = (double *) malloc(sizeof(double) * TWENTYNINE_LAYER_OUTPUT_SIZE);

    Read_TwentyNineLayer_Data(
        Layer29_Weights_CPU,        
        Layer29_Bias_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer29_Weights_GPU,
           Layer29_Bias_GPU;

    cl_int err;
    
    Layer29_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYNINE_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer29_Bias_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYNINE_LAYER_OUTPUT_SIZE, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer29_Weights_GPU, CL_TRUE, 0, sizeof(double) * TWENTYNINE_LAYER_WEIGHT_SIZE, Layer29_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer29_Bias_GPU, CL_TRUE, 0, sizeof(double) * TWENTYNINE_LAYER_OUTPUT_SIZE, Layer29_Bias_CPU, 0, NULL, NULL);

    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeTwentyNineLayer_FullyConnected", NULL);
    
    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer29_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer30_Neurons_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer29_Weights_GPU);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &Layer29_Bias_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {1, 1000, 1};
    size_t localWorkSize_A[3] = {1, 1, 1};

    clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    free(Layer29_Weights_CPU);
    free(Layer29_Bias_CPU);

    clReleaseMemObject(Layer29_Weights_GPU);
    clReleaseMemObject(Layer29_Bias_GPU);
}

void Read_TwentyNineLayer_Data(double *Layer29_Weights_CPU,
    double * Layer29_Bias_CPU
){
    read_File("data/TwentyNineLayer/weightsNorm.txt", Layer29_Weights_CPU);
    read_File("data/TwentyNineLayer/biases.txt", Layer29_Bias_CPU);
}

void Execute_TwentyEight_Layer(cl_mem Layer28_Neurons_GPU,
    cl_mem Layer29_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeTwentyEightLayer_AvgPooling", NULL);
    
    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer28_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer29_Neurons_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {1, 1 * 32, 1 * 32};
    size_t localWorkSize_A[3] = {1, 32, 32};

    clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);
}

void Execute_TwentySeven_Layer(cl_mem Layer27_Neurons_GPU,
    cl_mem Layer28_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer27_Weights_CPU = (double *) malloc(sizeof(double) * TWENTYSEVEN_LAYER_WEIGHT_SIZE);
    double *Layer27_Mean_CPU = (double *) malloc(sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS);
    double *Layer27_StanDev_CPU = (double *) malloc(sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS);
    double *Layer27_Gamma_CPU = (double *) malloc(sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS);
    double *Layer27_Beta_CPU = (double *) malloc(sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS);

    Read_TwentySevenLayer_Data(
        Layer27_Weights_CPU,        
        Layer27_Mean_CPU,
        Layer27_StanDev_CPU,
        Layer27_Gamma_CPU,
        Layer27_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer27_Weights_GPU,
           Layer27_Mean_GPU,
           Layer27_StanDev_GPU,
           Layer27_Gamma_GPU,
           Layer27_Beta_GPU;

    cl_int err;
    
    Layer27_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYSEVEN_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer27_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS, NULL, NULL);
    Layer27_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS, NULL, NULL);
    Layer27_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS, NULL, NULL);
    Layer27_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer27_Weights_GPU, CL_TRUE, 0, sizeof(double) * TWENTYSEVEN_LAYER_WEIGHT_SIZE, Layer27_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer27_Mean_GPU, CL_TRUE, 0, sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS, Layer27_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer27_StanDev_GPU, CL_TRUE, 0, sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS, Layer27_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer27_Gamma_GPU, CL_TRUE, 0, sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS, Layer27_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer27_Beta_GPU, CL_TRUE, 0, sizeof(double) * TWENTYSEVEN_LAYER_CHANNELS, Layer27_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_PSC", &err);
    
    int inputWidth = 7, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 7, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 1024, channelSize = 1024, stride = 1, offset = 0;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer27_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer27_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer28_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer27_Mean_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer27_StanDev_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer27_Gamma_GPU);
    clSetKernelArg(kernel, 20, sizeof(cl_mem), (void *) &Layer27_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {1024, 1 * 7, 1 * 7};
    size_t localWorkSize_A[3] = {1, 7, 7};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    clReleaseMemObject(Layer27_Weights_GPU);
    clReleaseMemObject(Layer27_Mean_GPU);
    clReleaseMemObject(Layer27_StanDev_GPU);
    clReleaseMemObject(Layer27_Gamma_GPU);
    clReleaseMemObject(Layer27_Beta_GPU);

    free(Layer27_Weights_CPU);
    free(Layer27_Mean_CPU);
    free(Layer27_StanDev_CPU);
    free(Layer27_Gamma_CPU);
    free(Layer27_Beta_CPU);
}

void Read_TwentySevenLayer_Data(double *Layer27_Weights_CPU,
    double * Layer27_Mean_CPU,
    double * Layer27_StanDev_CPU,
    double * Layer27_Gamma_CPU,
    double * Layer27_Beta_CPU
){
    read_File("data/TwentySevenLayer/weightsNorm.txt", Layer27_Weights_CPU);
    read_File("data/TwentySevenLayer/TwentySeven_Layer_Mean.txt", Layer27_Mean_CPU);
    read_File("data/TwentySevenLayer/TwentySeven_Layer_StanDev.txt", Layer27_StanDev_CPU);
    read_File("data/TwentySevenLayer/TwentySeven_Layer_Gamma.txt", Layer27_Gamma_CPU);
    read_File("data/TwentySevenLayer/TwentySeven_Layer_Beta.txt", Layer27_Beta_CPU);
}

void Execute_TwentySix_Layer(cl_mem Layer26_Neurons_GPU,
    cl_mem Layer27_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer26_Weights_CPU = (double *) malloc(sizeof(double) * TWENTYSIX_LAYER_WEIGHT_SIZE);
    double *Layer26_Mean_CPU = (double *) malloc(sizeof(double) * TWENTYSIX_LAYER_CHANNELS);
    double *Layer26_StanDev_CPU = (double *) malloc(sizeof(double) * TWENTYSIX_LAYER_CHANNELS);
    double *Layer26_Gamma_CPU = (double *) malloc(sizeof(double) * TWENTYSIX_LAYER_CHANNELS);
    double *Layer26_Beta_CPU = (double *) malloc(sizeof(double) * TWENTYSIX_LAYER_CHANNELS);

    Read_TwentySixLayer_Data(
        Layer26_Weights_CPU,        
        Layer26_Mean_CPU,
        Layer26_StanDev_CPU,
        Layer26_Gamma_CPU,
        Layer26_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer26_Weights_GPU,
           Layer26_Mean_GPU,
           Layer26_StanDev_GPU,
           Layer26_Gamma_GPU,
           Layer26_Beta_GPU;

    cl_int err;
    
    Layer26_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYSIX_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer26_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYSIX_LAYER_CHANNELS, NULL, NULL);
    Layer26_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYSIX_LAYER_CHANNELS, NULL, NULL);
    Layer26_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYSIX_LAYER_CHANNELS, NULL, NULL);
    Layer26_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYSIX_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer26_Weights_GPU, CL_TRUE, 0, sizeof(double) * TWENTYSIX_LAYER_WEIGHT_SIZE, Layer26_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer26_Mean_GPU, CL_TRUE, 0, sizeof(double) * TWENTYSIX_LAYER_CHANNELS, Layer26_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer26_StanDev_GPU, CL_TRUE, 0, sizeof(double) * TWENTYSIX_LAYER_CHANNELS, Layer26_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer26_Gamma_GPU, CL_TRUE, 0, sizeof(double) * TWENTYSIX_LAYER_CHANNELS, Layer26_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer26_Beta_GPU, CL_TRUE, 0, sizeof(double) * TWENTYSIX_LAYER_CHANNELS, Layer26_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_DSC", &err);
    
    int inputWidth = 9, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 7, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 9, kernelSize = 3, stride = 1;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer26_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer26_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer27_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(cl_mem), (void *) &Layer26_Mean_GPU);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer26_StanDev_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer26_Gamma_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer26_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {1024, 1 * 7, 1 * 7};
    size_t localWorkSize_A[3] = {1, 7, 7};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    clReleaseMemObject(Layer26_Weights_GPU);
    clReleaseMemObject(Layer26_Mean_GPU);
    clReleaseMemObject(Layer26_StanDev_GPU);
    clReleaseMemObject(Layer26_Gamma_GPU);
    clReleaseMemObject(Layer26_Beta_GPU);

    free(Layer26_Weights_CPU);
    free(Layer26_Mean_CPU);
    free(Layer26_StanDev_CPU);
    free(Layer26_Gamma_CPU);
    free(Layer26_Beta_CPU);
}

void Read_TwentySixLayer_Data(double *Layer26_Weights_CPU,
    double * Layer26_Mean_CPU,
    double * Layer26_StanDev_CPU,
    double * Layer26_Gamma_CPU,
    double * Layer26_Beta_CPU
){
    read_File("data/TwentySixLayer/weightsNorm.txt", Layer26_Weights_CPU);
    read_File("data/TwentySixLayer/TwentySix_Layer_Mean.txt", Layer26_Mean_CPU);
    read_File("data/TwentySixLayer/TwentySix_Layer_StanDev.txt", Layer26_StanDev_CPU);
    read_File("data/TwentySixLayer/TwentySix_Layer_Gamma.txt", Layer26_Gamma_CPU);
    read_File("data/TwentySixLayer/TwentySix_Layer_Beta.txt", Layer26_Beta_CPU);
}

void Execute_TwentyFive_Layer(cl_mem Layer25_Neurons_GPU,
    cl_mem Layer26_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer25_Weights_CPU = (double *) malloc(sizeof(double) * TWENTYFIVE_LAYER_WEIGHT_SIZE);
    double *Layer25_Mean_CPU = (double *) malloc(sizeof(double) * TWENTYFIVE_LAYER_CHANNELS);
    double *Layer25_StanDev_CPU = (double *) malloc(sizeof(double) * TWENTYFIVE_LAYER_CHANNELS);
    double *Layer25_Gamma_CPU = (double *) malloc(sizeof(double) * TWENTYFIVE_LAYER_CHANNELS);
    double *Layer25_Beta_CPU = (double *) malloc(sizeof(double) * TWENTYFIVE_LAYER_CHANNELS);

    Read_TwentyFiveLayer_Data(
        Layer25_Weights_CPU,        
        Layer25_Mean_CPU,
        Layer25_StanDev_CPU,
        Layer25_Gamma_CPU,
        Layer25_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer25_Weights_GPU,
           Layer25_Mean_GPU,
           Layer25_StanDev_GPU,
           Layer25_Gamma_GPU,
           Layer25_Beta_GPU;

    cl_int err;
    
    Layer25_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYFIVE_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer25_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYFIVE_LAYER_CHANNELS, NULL, NULL);
    Layer25_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYFIVE_LAYER_CHANNELS, NULL, NULL);
    Layer25_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYFIVE_LAYER_CHANNELS, NULL, NULL);
    Layer25_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYFIVE_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer25_Weights_GPU, CL_TRUE, 0, sizeof(double) * TWENTYFIVE_LAYER_WEIGHT_SIZE, Layer25_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer25_Mean_GPU, CL_TRUE, 0, sizeof(double) * TWENTYFIVE_LAYER_CHANNELS, Layer25_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer25_StanDev_GPU, CL_TRUE, 0, sizeof(double) * TWENTYFIVE_LAYER_CHANNELS, Layer25_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer25_Gamma_GPU, CL_TRUE, 0, sizeof(double) * TWENTYFIVE_LAYER_CHANNELS, Layer25_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer25_Beta_GPU, CL_TRUE, 0, sizeof(double) * TWENTYFIVE_LAYER_CHANNELS, Layer25_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_PSC", &err);
    
    int inputWidth = 7, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 9, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 512, channelSize = 512, stride = 1, offset = 10;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer25_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer25_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer26_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer25_Mean_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer25_StanDev_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer25_Gamma_GPU);
    clSetKernelArg(kernel, 20, sizeof(cl_mem), (void *) &Layer25_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {1024, 1 * 7, 1 * 7};
    size_t localWorkSize_A[3] = {1, 7, 7};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    clReleaseMemObject(Layer25_Weights_GPU);
    clReleaseMemObject(Layer25_Mean_GPU);
    clReleaseMemObject(Layer25_StanDev_GPU);
    clReleaseMemObject(Layer25_Gamma_GPU);
    clReleaseMemObject(Layer25_Beta_GPU);

    free(Layer25_Weights_CPU);
    free(Layer25_Mean_CPU);
    free(Layer25_StanDev_CPU);
    free(Layer25_Gamma_CPU);
    free(Layer25_Beta_CPU);
}

void Read_TwentyFiveLayer_Data(double *Layer25_Weights_CPU,
    double * Layer25_Mean_CPU,
    double * Layer25_StanDev_CPU,
    double * Layer25_Gamma_CPU,
    double * Layer25_Beta_CPU
){
    read_File("data/TwentyFiveLayer/weightsNorm.txt", Layer25_Weights_CPU);
    read_File("data/TwentyFiveLayer/TwentyFive_Layer_Mean.txt", Layer25_Mean_CPU);
    read_File("data/TwentyFiveLayer/TwentyFive_Layer_StanDev.txt", Layer25_StanDev_CPU);
    read_File("data/TwentyFiveLayer/TwentyFive_Layer_Gamma.txt", Layer25_Gamma_CPU);
    read_File("data/TwentyFiveLayer/TwentyFive_Layer_Beta.txt", Layer25_Beta_CPU);
}

void Execute_TwentyFour_Layer(cl_mem Layer24_Neurons_GPU,
    cl_mem Layer25_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer24_Weights_CPU = (double *) malloc(sizeof(double) * TWENTYFOUR_LAYER_WEIGHT_SIZE);
    double *Layer24_Mean_CPU = (double *) malloc(sizeof(double) * TWENTYFOUR_LAYER_CHANNELS);
    double *Layer24_StanDev_CPU = (double *) malloc(sizeof(double) * TWENTYFOUR_LAYER_CHANNELS);
    double *Layer24_Gamma_CPU = (double *) malloc(sizeof(double) * TWENTYFOUR_LAYER_CHANNELS);
    double *Layer24_Beta_CPU = (double *) malloc(sizeof(double) * TWENTYFOUR_LAYER_CHANNELS);

    Read_TwentyFourLayer_Data(
        Layer24_Weights_CPU,        
        Layer24_Mean_CPU,
        Layer24_StanDev_CPU,
        Layer24_Gamma_CPU,
        Layer24_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer24_Weights_GPU,
           Layer24_Mean_GPU,
           Layer24_StanDev_GPU,
           Layer24_Gamma_GPU,
           Layer24_Beta_GPU;

    cl_int err;
    
    Layer24_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYFOUR_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer24_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYFOUR_LAYER_CHANNELS, NULL, NULL);
    Layer24_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYFOUR_LAYER_CHANNELS, NULL, NULL);
    Layer24_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYFOUR_LAYER_CHANNELS, NULL, NULL);
    Layer24_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYFOUR_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer24_Weights_GPU, CL_TRUE, 0, sizeof(double) * TWENTYFOUR_LAYER_WEIGHT_SIZE, Layer24_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer24_Mean_GPU, CL_TRUE, 0, sizeof(double) * TWENTYFOUR_LAYER_CHANNELS, Layer24_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer24_StanDev_GPU, CL_TRUE, 0, sizeof(double) * TWENTYFOUR_LAYER_CHANNELS, Layer24_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer24_Gamma_GPU, CL_TRUE, 0, sizeof(double) * TWENTYFOUR_LAYER_CHANNELS, Layer24_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer24_Beta_GPU, CL_TRUE, 0, sizeof(double) * TWENTYFOUR_LAYER_CHANNELS, Layer24_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_DSC", &err);
    
    int inputWidth = 15, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 7, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 9, kernelSize = 3, stride = 2;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer24_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer24_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer25_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(cl_mem), (void *) &Layer24_Mean_GPU);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer24_StanDev_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer24_Gamma_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer24_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {512, 1 * 7, 1 * 7};
    size_t localWorkSize_A[3] = {1, 7, 7};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    clReleaseMemObject(Layer24_Weights_GPU);
    clReleaseMemObject(Layer24_Mean_GPU);
    clReleaseMemObject(Layer24_StanDev_GPU);
    clReleaseMemObject(Layer24_Gamma_GPU);
    clReleaseMemObject(Layer24_Beta_GPU);

    free(Layer24_Weights_CPU);
    free(Layer24_Mean_CPU);
    free(Layer24_StanDev_CPU);
    free(Layer24_Gamma_CPU);
    free(Layer24_Beta_CPU);
}

void Read_TwentyFourLayer_Data(double *Layer24_Weights_CPU,
    double * Layer24_Mean_CPU,
    double * Layer24_StanDev_CPU,
    double * Layer24_Gamma_CPU,
    double * Layer24_Beta_CPU
){
    read_File("data/TwentyFourLayer/weightsNorm.txt", Layer24_Weights_CPU);
    read_File("data/TwentyFourLayer/TwentyFour_Layer_Mean.txt", Layer24_Mean_CPU);
    read_File("data/TwentyFourLayer/TwentyFour_Layer_StanDev.txt", Layer24_StanDev_CPU);
    read_File("data/TwentyFourLayer/TwentyFour_Layer_Gamma.txt", Layer24_Gamma_CPU);
    read_File("data/TwentyFourLayer/TwentyFour_Layer_Beta.txt", Layer24_Beta_CPU);
}

void Execute_TwentyThree_Layer(cl_mem Layer23_Neurons_GPU,
    cl_mem Layer24_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer23_Weights_CPU = (double *) malloc(sizeof(double) * TWENTYTHREE_LAYER_WEIGHT_SIZE);
    double *Layer23_Mean_CPU = (double *) malloc(sizeof(double) * TWENTYTHREE_LAYER_CHANNELS);
    double *Layer23_StanDev_CPU = (double *) malloc(sizeof(double) * TWENTYTHREE_LAYER_CHANNELS);
    double *Layer23_Gamma_CPU = (double *) malloc(sizeof(double) * TWENTYTHREE_LAYER_CHANNELS);
    double *Layer23_Beta_CPU = (double *) malloc(sizeof(double) * TWENTYTHREE_LAYER_CHANNELS);

    Read_TwentyThreeLayer_Data(
        Layer23_Weights_CPU,        
        Layer23_Mean_CPU,
        Layer23_StanDev_CPU,
        Layer23_Gamma_CPU,
        Layer23_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer23_Weights_GPU,
           Layer23_Mean_GPU,
           Layer23_StanDev_GPU,
           Layer23_Gamma_GPU,
           Layer23_Beta_GPU;

    cl_int err;
    
    Layer23_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYTHREE_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer23_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYTHREE_LAYER_CHANNELS, NULL, NULL);
    Layer23_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYTHREE_LAYER_CHANNELS, NULL, NULL);
    Layer23_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYTHREE_LAYER_CHANNELS, NULL, NULL);
    Layer23_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYTHREE_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer23_Weights_GPU, CL_TRUE, 0, sizeof(double) * TWENTYTHREE_LAYER_WEIGHT_SIZE, Layer23_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer23_Mean_GPU, CL_TRUE, 0, sizeof(double) * TWENTYTHREE_LAYER_CHANNELS, Layer23_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer23_StanDev_GPU, CL_TRUE, 0, sizeof(double) * TWENTYTHREE_LAYER_CHANNELS, Layer23_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer23_Gamma_GPU, CL_TRUE, 0, sizeof(double) * TWENTYTHREE_LAYER_CHANNELS, Layer23_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer23_Beta_GPU, CL_TRUE, 0, sizeof(double) * TWENTYTHREE_LAYER_CHANNELS, Layer23_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_PSC", &err);
    
    int inputWidth = 14, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 15, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 512, channelSize = 512, stride = 1, offset = 0;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer23_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer23_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer24_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer23_Mean_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer23_StanDev_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer23_Gamma_GPU);
    clSetKernelArg(kernel, 20, sizeof(cl_mem), (void *) &Layer23_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {512, 1 * 14, 1 * 14};
    size_t localWorkSize_A[3] = {1, 14, 14};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    clReleaseMemObject(Layer23_Weights_GPU);
    clReleaseMemObject(Layer23_Mean_GPU);
    clReleaseMemObject(Layer23_StanDev_GPU);
    clReleaseMemObject(Layer23_Gamma_GPU);
    clReleaseMemObject(Layer23_Beta_GPU);

    free(Layer23_Weights_CPU);
    free(Layer23_Mean_CPU);
    free(Layer23_StanDev_CPU);
    free(Layer23_Gamma_CPU);
    free(Layer23_Beta_CPU);
}

void Read_TwentyThreeLayer_Data(double *Layer23_Weights_CPU,
    double * Layer23_Mean_CPU,
    double * Layer23_StanDev_CPU,
    double * Layer23_Gamma_CPU,
    double * Layer23_Beta_CPU
){
    read_File("data/TwentyThreeLayer/weightsNorm.txt", Layer23_Weights_CPU);
    read_File("data/TwentyThreeLayer/TwentyThree_Layer_Mean.txt", Layer23_Mean_CPU);
    read_File("data/TwentyThreeLayer/TwentyThree_Layer_StanDev.txt", Layer23_StanDev_CPU);
    read_File("data/TwentyThreeLayer/TwentyThree_Layer_Gamma.txt", Layer23_Gamma_CPU);
    read_File("data/TwentyThreeLayer/TwentyThree_Layer_Beta.txt", Layer23_Beta_CPU);
}

void Execute_TwentyTwo_Layer(cl_mem Layer22_Neurons_GPU,
    cl_mem Layer23_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer22_Weights_CPU = (double *) malloc(sizeof(double) * TWENTYTWO_LAYER_WEIGHT_SIZE);
    double *Layer22_Mean_CPU = (double *) malloc(sizeof(double) * TWENTYTWO_LAYER_CHANNELS);
    double *Layer22_StanDev_CPU = (double *) malloc(sizeof(double) * TWENTYTWO_LAYER_CHANNELS);
    double *Layer22_Gamma_CPU = (double *) malloc(sizeof(double) * TWENTYTWO_LAYER_CHANNELS);
    double *Layer22_Beta_CPU = (double *) malloc(sizeof(double) * TWENTYTWO_LAYER_CHANNELS);

    Read_TwentyTwoLayer_Data(
        Layer22_Weights_CPU,        
        Layer22_Mean_CPU,
        Layer22_StanDev_CPU,
        Layer22_Gamma_CPU,
        Layer22_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer22_Weights_GPU,
           Layer22_Mean_GPU,
           Layer22_StanDev_GPU,
           Layer22_Gamma_GPU,
           Layer22_Beta_GPU;

    cl_int err;
    
    Layer22_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYTWO_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer22_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYTWO_LAYER_CHANNELS, NULL, NULL);
    Layer22_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYTWO_LAYER_CHANNELS, NULL, NULL);
    Layer22_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYTWO_LAYER_CHANNELS, NULL, NULL);
    Layer22_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTYTWO_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer22_Weights_GPU, CL_TRUE, 0, sizeof(double) * TWENTYTWO_LAYER_WEIGHT_SIZE, Layer22_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer22_Mean_GPU, CL_TRUE, 0, sizeof(double) * TWENTYTWO_LAYER_CHANNELS, Layer22_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer22_StanDev_GPU, CL_TRUE, 0, sizeof(double) * TWENTYTWO_LAYER_CHANNELS, Layer22_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer22_Gamma_GPU, CL_TRUE, 0, sizeof(double) * TWENTYTWO_LAYER_CHANNELS, Layer22_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer22_Beta_GPU, CL_TRUE, 0, sizeof(double) * TWENTYTWO_LAYER_CHANNELS, Layer22_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_DSC", &err);
    
    int inputWidth = 16, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 14, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 9, kernelSize = 3, stride = 1;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer22_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer22_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer23_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(cl_mem), (void *) &Layer22_Mean_GPU);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer22_StanDev_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer22_Gamma_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer22_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {512, 1 * 14, 1 * 14};
    size_t localWorkSize_A[3] = {1, 14, 14};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    clReleaseMemObject(Layer22_Weights_GPU);
    clReleaseMemObject(Layer22_Mean_GPU);
    clReleaseMemObject(Layer22_StanDev_GPU);
    clReleaseMemObject(Layer22_Gamma_GPU);
    clReleaseMemObject(Layer22_Beta_GPU);

    free(Layer22_Weights_CPU);
    free(Layer22_Mean_CPU);
    free(Layer22_StanDev_CPU);
    free(Layer22_Gamma_CPU);
    free(Layer22_Beta_CPU);
}

void Read_TwentyTwoLayer_Data(double *Layer22_Weights_CPU,
    double * Layer22_Mean_CPU,
    double * Layer22_StanDev_CPU,
    double * Layer22_Gamma_CPU,
    double * Layer22_Beta_CPU
){
    read_File("data/TwentyTwoLayer/weightsNorm.txt", Layer22_Weights_CPU);
    read_File("data/TwentyTwoLayer/TwentyTwo_Layer_Mean.txt", Layer22_Mean_CPU);
    read_File("data/TwentyTwoLayer/TwentyTwo_Layer_StanDev.txt", Layer22_StanDev_CPU);
    read_File("data/TwentyTwoLayer/TwentyTwo_Layer_Gamma.txt", Layer22_Gamma_CPU);
    read_File("data/TwentyTwoLayer/TwentyTwo_Layer_Beta.txt", Layer22_Beta_CPU);
}

void Execute_TwentyOne_Layer(cl_mem Layer21_Neurons_GPU,
    cl_mem Layer22_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer21_Weights_CPU = (double *) malloc(sizeof(double) * NINETEENTH_LAYER_WEIGHT_SIZE);
    double *Layer21_Mean_CPU = (double *) malloc(sizeof(double) * NINETEENTH_LAYER_CHANNELS);
    double *Layer21_StanDev_CPU = (double *) malloc(sizeof(double) * NINETEENTH_LAYER_CHANNELS);
    double *Layer21_Gamma_CPU = (double *) malloc(sizeof(double) * NINETEENTH_LAYER_CHANNELS);
    double *Layer21_Beta_CPU = (double *) malloc(sizeof(double) * NINETEENTH_LAYER_CHANNELS);

    Read_TwentyOneLayer_Data(
        Layer21_Weights_CPU,        
        Layer21_Mean_CPU,
        Layer21_StanDev_CPU,
        Layer21_Gamma_CPU,
        Layer21_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer21_Weights_GPU,
           Layer21_Mean_GPU,
           Layer21_StanDev_GPU,
           Layer21_Gamma_GPU,
           Layer21_Beta_GPU;

    cl_int err;
    
    Layer21_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * NINETEENTH_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer21_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * NINETEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer21_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * NINETEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer21_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * NINETEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer21_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * NINETEENTH_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer21_Weights_GPU, CL_TRUE, 0, sizeof(double) * NINETEENTH_LAYER_WEIGHT_SIZE, Layer21_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer21_Mean_GPU, CL_TRUE, 0, sizeof(double) * NINETEENTH_LAYER_CHANNELS, Layer21_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer21_StanDev_GPU, CL_TRUE, 0, sizeof(double) * NINETEENTH_LAYER_CHANNELS, Layer21_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer21_Gamma_GPU, CL_TRUE, 0, sizeof(double) * NINETEENTH_LAYER_CHANNELS, Layer21_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer21_Beta_GPU, CL_TRUE, 0, sizeof(double) * NINETEENTH_LAYER_CHANNELS, Layer21_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_PSC", &err);
    
    int inputWidth = 14, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 16, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 512, channelSize = 512, stride = 1, offset = 17;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer21_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer21_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer22_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer21_Mean_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer21_StanDev_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer21_Gamma_GPU);
    clSetKernelArg(kernel, 20, sizeof(cl_mem), (void *) &Layer21_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {512, 1 * 14, 1 * 14};
    size_t localWorkSize_A[3] = {1, 14, 14};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    clReleaseMemObject(Layer21_Weights_GPU);
    clReleaseMemObject(Layer21_Mean_GPU);
    clReleaseMemObject(Layer21_StanDev_GPU);
    clReleaseMemObject(Layer21_Gamma_GPU);
    clReleaseMemObject(Layer21_Beta_GPU);

    free(Layer21_Weights_CPU);
    free(Layer21_Mean_CPU);
    free(Layer21_StanDev_CPU);
    free(Layer21_Gamma_CPU);
    free(Layer21_Beta_CPU);
}

void Read_TwentyOneLayer_Data(double *Layer21_Weights_CPU,
    double * Layer21_Mean_CPU,
    double * Layer21_StanDev_CPU,
    double * Layer21_Gamma_CPU,
    double * Layer21_Beta_CPU
){
    read_File("data/TwentyOneLayer/weightsNorm.txt", Layer21_Weights_CPU);
    read_File("data/TwentyOneLayer/TwentyOne_Layer_Mean.txt", Layer21_Mean_CPU);
    read_File("data/TwentyOneLayer/TwentyOne_Layer_StanDev.txt", Layer21_StanDev_CPU);
    read_File("data/TwentyOneLayer/TwentyOne_Layer_Gamma.txt", Layer21_Gamma_CPU);
    read_File("data/TwentyOneLayer/TwentyOne_Layer_Beta.txt", Layer21_Beta_CPU);
}

void Execute_Twenty_Layer(cl_mem Layer20_Neurons_GPU,
    cl_mem Layer21_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer20_Weights_CPU = (double *) malloc(sizeof(double) * TWENTY_LAYER_WEIGHT_SIZE);
    double *Layer20_Mean_CPU = (double *) malloc(sizeof(double) * TWENTY_LAYER_CHANNELS);
    double *Layer20_StanDev_CPU = (double *) malloc(sizeof(double) * TWENTY_LAYER_CHANNELS);
    double *Layer20_Gamma_CPU = (double *) malloc(sizeof(double) * TWENTY_LAYER_CHANNELS);
    double *Layer20_Beta_CPU = (double *) malloc(sizeof(double) * TWENTY_LAYER_CHANNELS);

    Read_TwentyLayer_Data(
        Layer20_Weights_CPU,        
        Layer20_Mean_CPU,
        Layer20_StanDev_CPU,
        Layer20_Gamma_CPU,
        Layer20_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer20_Weights_GPU,
           Layer20_Mean_GPU,
           Layer20_StanDev_GPU,
           Layer20_Gamma_GPU,
           Layer20_Beta_GPU;

    cl_int err;
    
    Layer20_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTY_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer20_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTY_LAYER_CHANNELS, NULL, NULL);
    Layer20_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTY_LAYER_CHANNELS, NULL, NULL);
    Layer20_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTY_LAYER_CHANNELS, NULL, NULL);
    Layer20_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWENTY_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer20_Weights_GPU, CL_TRUE, 0, sizeof(double) * TWENTY_LAYER_WEIGHT_SIZE, Layer20_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer20_Mean_GPU, CL_TRUE, 0, sizeof(double) * TWENTY_LAYER_CHANNELS, Layer20_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer20_StanDev_GPU, CL_TRUE, 0, sizeof(double) * TWENTY_LAYER_CHANNELS, Layer20_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer20_Gamma_GPU, CL_TRUE, 0, sizeof(double) * TWENTY_LAYER_CHANNELS, Layer20_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer20_Beta_GPU, CL_TRUE, 0, sizeof(double) * TWENTY_LAYER_CHANNELS, Layer20_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_DSC", &err);
    
    int inputWidth = 16, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 14, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 9, kernelSize = 3, stride = 1;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer20_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer20_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer21_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(cl_mem), (void *) &Layer20_Mean_GPU);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer20_StanDev_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer20_Gamma_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer20_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {512, 1 * 14, 1 * 14};
    size_t localWorkSize_A[3] = {1, 14, 14};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    clReleaseMemObject(Layer20_Weights_GPU);
    clReleaseMemObject(Layer20_Mean_GPU);
    clReleaseMemObject(Layer20_StanDev_GPU);
    clReleaseMemObject(Layer20_Gamma_GPU);
    clReleaseMemObject(Layer20_Beta_GPU);

    free(Layer20_Weights_CPU);
    free(Layer20_Mean_CPU);
    free(Layer20_StanDev_CPU);
    free(Layer20_Gamma_CPU);
    free(Layer20_Beta_CPU);
}

void Read_TwentyLayer_Data(double *Layer20_Weights_CPU,
    double * Layer20_Mean_CPU,
    double * Layer20_StanDev_CPU,
    double * Layer20_Gamma_CPU,
    double * Layer20_Beta_CPU
){
    read_File("data/TwentyLayer/weightsNorm.txt", Layer20_Weights_CPU);
    read_File("data/TwentyLayer/Twenty_Layer_Mean.txt", Layer20_Mean_CPU);
    read_File("data/TwentyLayer/Twenty_Layer_StanDev.txt", Layer20_StanDev_CPU);
    read_File("data/TwentyLayer/Twenty_Layer_Gamma.txt", Layer20_Gamma_CPU);
    read_File("data/TwentyLayer/Twenty_Layer_Beta.txt", Layer20_Beta_CPU);
}

void Execute_Nineteenth_Layer(cl_mem Layer19_Neurons_GPU,
    cl_mem Layer20_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer19_Weights_CPU = (double *) malloc(sizeof(double) * NINETEENTH_LAYER_WEIGHT_SIZE);
    double *Layer19_Mean_CPU = (double *) malloc(sizeof(double) * NINETEENTH_LAYER_CHANNELS);
    double *Layer19_StanDev_CPU = (double *) malloc(sizeof(double) * NINETEENTH_LAYER_CHANNELS);
    double *Layer19_Gamma_CPU = (double *) malloc(sizeof(double) * NINETEENTH_LAYER_CHANNELS);
    double *Layer19_Beta_CPU = (double *) malloc(sizeof(double) * NINETEENTH_LAYER_CHANNELS);

    Read_NineteenthLayer_Data(
        Layer19_Weights_CPU,        
        Layer19_Mean_CPU,
        Layer19_StanDev_CPU,
        Layer19_Gamma_CPU,
        Layer19_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer19_Weights_GPU,
           Layer19_Mean_GPU,
           Layer19_StanDev_GPU,
           Layer19_Gamma_GPU,
           Layer19_Beta_GPU;

    cl_int err;
    
    Layer19_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * NINETEENTH_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer19_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * NINETEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer19_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * NINETEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer19_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * NINETEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer19_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * NINETEENTH_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer19_Weights_GPU, CL_TRUE, 0, sizeof(double) * NINETEENTH_LAYER_WEIGHT_SIZE, Layer19_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer19_Mean_GPU, CL_TRUE, 0, sizeof(double) * NINETEENTH_LAYER_CHANNELS, Layer19_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer19_StanDev_GPU, CL_TRUE, 0, sizeof(double) * NINETEENTH_LAYER_CHANNELS, Layer19_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer19_Gamma_GPU, CL_TRUE, 0, sizeof(double) * NINETEENTH_LAYER_CHANNELS, Layer19_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer19_Beta_GPU, CL_TRUE, 0, sizeof(double) * NINETEENTH_LAYER_CHANNELS, Layer19_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_PSC", &err);
    
    int inputWidth = 14, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 16, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 512, channelSize = 512, stride = 1, offset = 17;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer19_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer19_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer20_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer19_Mean_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer19_StanDev_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer19_Gamma_GPU);
    clSetKernelArg(kernel, 20, sizeof(cl_mem), (void *) &Layer19_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {512, 1 * 14, 1 * 14};
    size_t localWorkSize_A[3] = {1, 14, 14};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    clReleaseMemObject(Layer19_Weights_GPU);
    clReleaseMemObject(Layer19_Mean_GPU);
    clReleaseMemObject(Layer19_StanDev_GPU);
    clReleaseMemObject(Layer19_Gamma_GPU);
    clReleaseMemObject(Layer19_Beta_GPU);

    free(Layer19_Weights_CPU);
    free(Layer19_Mean_CPU);
    free(Layer19_StanDev_CPU);
    free(Layer19_Gamma_CPU);
    free(Layer19_Beta_CPU);
}

void Read_NineteenthLayer_Data(double *Layer19_Weights_CPU,
    double * Layer19_Mean_CPU,
    double * Layer19_StanDev_CPU,
    double * Layer19_Gamma_CPU,
    double * Layer19_Beta_CPU
){
    read_File("data/NineteenthLayer/weightsNorm.txt", Layer19_Weights_CPU);
    read_File("data/NineteenthLayer/Nineteenth_Layer_Mean.txt", Layer19_Mean_CPU);
    read_File("data/NineteenthLayer/Nineteenth_Layer_StanDev.txt", Layer19_StanDev_CPU);
    read_File("data/NineteenthLayer/Nineteenth_Layer_Gamma.txt", Layer19_Gamma_CPU);
    read_File("data/NineteenthLayer/Nineteenth_Layer_Beta.txt", Layer19_Beta_CPU);
}

void Execute_Eighteenth_Layer(cl_mem Layer18_Neurons_GPU,
    cl_mem Layer19_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer18_Weights_CPU = (double *) malloc(sizeof(double) * EIGHTEENTH_LAYER_WEIGHT_SIZE);
    double *Layer18_Mean_CPU = (double *) malloc(sizeof(double) * EIGHTEENTH_LAYER_CHANNELS);
    double *Layer18_StanDev_CPU = (double *) malloc(sizeof(double) * EIGHTEENTH_LAYER_CHANNELS);
    double *Layer18_Gamma_CPU = (double *) malloc(sizeof(double) * EIGHTEENTH_LAYER_CHANNELS);
    double *Layer18_Beta_CPU = (double *) malloc(sizeof(double) * EIGHTEENTH_LAYER_CHANNELS);

    Read_EighteenthLayer_Data(
        Layer18_Weights_CPU,        
        Layer18_Mean_CPU,
        Layer18_StanDev_CPU,
        Layer18_Gamma_CPU,
        Layer18_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer18_Weights_GPU,
           Layer18_Mean_GPU,
           Layer18_StanDev_GPU,
           Layer18_Gamma_GPU,
           Layer18_Beta_GPU;

    cl_int err;
    
    Layer18_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * EIGHTEENTH_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer18_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * EIGHTEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer18_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * EIGHTEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer18_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * EIGHTEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer18_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * EIGHTEENTH_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer18_Weights_GPU, CL_TRUE, 0, sizeof(double) * EIGHTEENTH_LAYER_WEIGHT_SIZE, Layer18_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer18_Mean_GPU, CL_TRUE, 0, sizeof(double) * EIGHTEENTH_LAYER_CHANNELS, Layer18_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer18_StanDev_GPU, CL_TRUE, 0, sizeof(double) * EIGHTEENTH_LAYER_CHANNELS, Layer18_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer18_Gamma_GPU, CL_TRUE, 0, sizeof(double) * EIGHTEENTH_LAYER_CHANNELS, Layer18_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer18_Beta_GPU, CL_TRUE, 0, sizeof(double) * EIGHTEENTH_LAYER_CHANNELS, Layer18_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_DSC", &err);
    
    int inputWidth = 16, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 14, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 9, kernelSize = 3, stride = 1;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer18_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer18_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer19_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(cl_mem), (void *) &Layer18_Mean_GPU);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer18_StanDev_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer18_Gamma_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer18_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {512, 1 * 14, 1 * 14};
    size_t localWorkSize_A[3] = {1, 14, 14};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    clReleaseMemObject(Layer18_Weights_GPU);
    clReleaseMemObject(Layer18_Mean_GPU);
    clReleaseMemObject(Layer18_StanDev_GPU);
    clReleaseMemObject(Layer18_Gamma_GPU);
    clReleaseMemObject(Layer18_Beta_GPU);

    free(Layer18_Weights_CPU);
    free(Layer18_Mean_CPU);
    free(Layer18_StanDev_CPU);
    free(Layer18_Gamma_CPU);
    free(Layer18_Beta_CPU);
}

void Read_EighteenthLayer_Data(double *Layer18_Weights_CPU,
    double * Layer18_Mean_CPU,
    double * Layer18_StanDev_CPU,
    double * Layer18_Gamma_CPU,
    double * Layer18_Beta_CPU
){
    read_File("data/EighteenthLayer/weightsNorm.txt", Layer18_Weights_CPU);
    read_File("data/EighteenthLayer/Eighteenth_Layer_Mean.txt", Layer18_Mean_CPU);
    read_File("data/EighteenthLayer/Eighteenth_Layer_StanDev.txt", Layer18_StanDev_CPU);
    read_File("data/EighteenthLayer/Eighteenth_Layer_Gamma.txt", Layer18_Gamma_CPU);
    read_File("data/EighteenthLayer/Eighteenth_Layer_Beta.txt", Layer18_Beta_CPU);
}

void Execute_Seventeenth_Layer(cl_mem Layer17_Neurons_GPU,
    cl_mem Layer18_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer17_Weights_CPU = (double *) malloc(sizeof(double) * SEVENTEENTH_LAYER_WEIGHT_SIZE);
    double *Layer17_Mean_CPU = (double *) malloc(sizeof(double) * SEVENTEENTH_LAYER_CHANNELS);
    double *Layer17_StanDev_CPU = (double *) malloc(sizeof(double) * SEVENTEENTH_LAYER_CHANNELS);
    double *Layer17_Gamma_CPU = (double *) malloc(sizeof(double) * SEVENTEENTH_LAYER_CHANNELS);
    double *Layer17_Beta_CPU = (double *) malloc(sizeof(double) * SEVENTEENTH_LAYER_CHANNELS);

    Read_SeventeenthLayer_Data(
        Layer17_Weights_CPU,        
        Layer17_Mean_CPU,
        Layer17_StanDev_CPU,
        Layer17_Gamma_CPU,
        Layer17_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer17_Weights_GPU,
           Layer17_Mean_GPU,
           Layer17_StanDev_GPU,
           Layer17_Gamma_GPU,
           Layer17_Beta_GPU;

    cl_int err;
    
    Layer17_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SEVENTEENTH_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer17_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SEVENTEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer17_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SEVENTEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer17_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SEVENTEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer17_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SEVENTEENTH_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer17_Weights_GPU, CL_TRUE, 0, sizeof(double) * SEVENTEENTH_LAYER_WEIGHT_SIZE, Layer17_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer17_Mean_GPU, CL_TRUE, 0, sizeof(double) * SEVENTEENTH_LAYER_CHANNELS, Layer17_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer17_StanDev_GPU, CL_TRUE, 0, sizeof(double) * SEVENTEENTH_LAYER_CHANNELS, Layer17_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer17_Gamma_GPU, CL_TRUE, 0, sizeof(double) * SEVENTEENTH_LAYER_CHANNELS, Layer17_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer17_Beta_GPU, CL_TRUE, 0, sizeof(double) * SEVENTEENTH_LAYER_CHANNELS, Layer17_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_PSC", &err);
    
    int inputWidth = 14, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 16, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 512, channelSize = 512, stride = 1, offset = 17;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer17_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer17_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer18_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer17_Mean_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer17_StanDev_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer17_Gamma_GPU);
    clSetKernelArg(kernel, 20, sizeof(cl_mem), (void *) &Layer17_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {512, 1 * 14, 1 * 14};
    size_t localWorkSize_A[3] = {1, 14, 14};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    clReleaseMemObject(Layer17_Weights_GPU);
    clReleaseMemObject(Layer17_Mean_GPU);
    clReleaseMemObject(Layer17_StanDev_GPU);
    clReleaseMemObject(Layer17_Gamma_GPU);
    clReleaseMemObject(Layer17_Beta_GPU);

    free(Layer17_Weights_CPU);
    free(Layer17_Mean_CPU);
    free(Layer17_StanDev_CPU);
    free(Layer17_Gamma_CPU);
    free(Layer17_Beta_CPU);
}

void Read_SeventeenthLayer_Data(double *Layer17_Weights_CPU,
    double * Layer17_Mean_CPU,
    double * Layer17_StanDev_CPU,
    double * Layer17_Gamma_CPU,
    double * Layer17_Beta_CPU
){
    read_File("data/SeventeenthLayer/weightsNorm.txt", Layer17_Weights_CPU);
    read_File("data/SeventeenthLayer/Seventeenth_Layer_Mean.txt", Layer17_Mean_CPU);
    read_File("data/SeventeenthLayer/Seventeenth_Layer_StanDev.txt", Layer17_StanDev_CPU);
    read_File("data/SeventeenthLayer/Seventeenth_Layer_Gamma.txt", Layer17_Gamma_CPU);
    read_File("data/SeventeenthLayer/Seventeenth_Layer_Beta.txt", Layer17_Beta_CPU);
}

void Execute_Sixteenth_Layer(cl_mem Layer16_Neurons_GPU,
    cl_mem Layer17_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer16_Weights_CPU = (double *) malloc(sizeof(double) * SIXTEENTH_LAYER_WEIGHT_SIZE);
    double *Layer16_Mean_CPU = (double *) malloc(sizeof(double) * SIXTEENTH_LAYER_CHANNELS);
    double *Layer16_StanDev_CPU = (double *) malloc(sizeof(double) * SIXTEENTH_LAYER_CHANNELS);
    double *Layer16_Gamma_CPU = (double *) malloc(sizeof(double) * SIXTEENTH_LAYER_CHANNELS);
    double *Layer16_Beta_CPU = (double *) malloc(sizeof(double) * SIXTEENTH_LAYER_CHANNELS);

    Read_SixteenthLayer_Data(
        Layer16_Weights_CPU,        
        Layer16_Mean_CPU,
        Layer16_StanDev_CPU,
        Layer16_Gamma_CPU,
        Layer16_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer16_Weights_GPU,
           Layer16_Mean_GPU,
           Layer16_StanDev_GPU,
           Layer16_Gamma_GPU,
           Layer16_Beta_GPU;

    cl_int err;
    
    Layer16_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SIXTEENTH_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer16_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SIXTEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer16_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SIXTEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer16_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SIXTEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer16_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SIXTEENTH_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer16_Weights_GPU, CL_TRUE, 0, sizeof(double) * SIXTEENTH_LAYER_WEIGHT_SIZE, Layer16_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer16_Mean_GPU, CL_TRUE, 0, sizeof(double) * SIXTEENTH_LAYER_CHANNELS, Layer16_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer16_StanDev_GPU, CL_TRUE, 0, sizeof(double) * SIXTEENTH_LAYER_CHANNELS, Layer16_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer16_Gamma_GPU, CL_TRUE, 0, sizeof(double) * SIXTEENTH_LAYER_CHANNELS, Layer16_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer16_Beta_GPU, CL_TRUE, 0, sizeof(double) * SIXTEENTH_LAYER_CHANNELS, Layer16_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_DSC", &err);
    
    int inputWidth = 16, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 14, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 9, kernelSize = 3, stride = 1;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer16_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer16_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer17_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(cl_mem), (void *) &Layer16_Mean_GPU);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer16_StanDev_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer16_Gamma_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer16_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {512, 1 * 14, 1 * 14};
    size_t localWorkSize_A[3] = {1, 14, 14};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    clReleaseMemObject(Layer16_Weights_GPU);
    clReleaseMemObject(Layer16_Mean_GPU);
    clReleaseMemObject(Layer16_StanDev_GPU);
    clReleaseMemObject(Layer16_Gamma_GPU);
    clReleaseMemObject(Layer16_Beta_GPU);

    free(Layer16_Weights_CPU);
    free(Layer16_Mean_CPU);
    free(Layer16_StanDev_CPU);
    free(Layer16_Gamma_CPU);
    free(Layer16_Beta_CPU);
}

void Read_SixteenthLayer_Data(double *Layer16_Weights_CPU,
    double * Layer16_Mean_CPU,
    double * Layer16_StanDev_CPU,
    double * Layer16_Gamma_CPU,
    double * Layer16_Beta_CPU
){
    read_File("data/SixteenthLayer/weightsNorm.txt", Layer16_Weights_CPU);
    read_File("data/SixteenthLayer/Sixteenth_Layer_Mean.txt", Layer16_Mean_CPU);
    read_File("data/SixteenthLayer/Sixteenth_Layer_StanDev.txt", Layer16_StanDev_CPU);
    read_File("data/SixteenthLayer/Sixteenth_Layer_Gamma.txt", Layer16_Gamma_CPU);
    read_File("data/SixteenthLayer/Sixteenth_Layer_Beta.txt", Layer16_Beta_CPU);
}

void Execute_Fifteenth_Layer(cl_mem Layer15_Neurons_GPU,
    cl_mem Layer16_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer15_Weights_CPU = (double *) malloc(sizeof(double) * FIFTEENTH_LAYER_WEIGHT_SIZE);
    double *Layer15_Mean_CPU = (double *) malloc(sizeof(double) * FIFTEENTH_LAYER_CHANNELS);
    double *Layer15_StanDev_CPU = (double *) malloc(sizeof(double) * FIFTEENTH_LAYER_CHANNELS);
    double *Layer15_Gamma_CPU = (double *) malloc(sizeof(double) * FIFTEENTH_LAYER_CHANNELS);
    double *Layer15_Beta_CPU = (double *) malloc(sizeof(double) * FIFTEENTH_LAYER_CHANNELS);

    Read_FifteenthLayer_Data(
        Layer15_Weights_CPU,        
        Layer15_Mean_CPU,
        Layer15_StanDev_CPU,
        Layer15_Gamma_CPU,
        Layer15_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer15_Weights_GPU,
           Layer15_Mean_GPU,
           Layer15_StanDev_GPU,
           Layer15_Gamma_GPU,
           Layer15_Beta_GPU;

    cl_int err;
    
    Layer15_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FIFTEENTH_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer15_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FIFTEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer15_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FIFTEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer15_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FIFTEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer15_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FIFTEENTH_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer15_Weights_GPU, CL_TRUE, 0, sizeof(double) * FIFTEENTH_LAYER_WEIGHT_SIZE, Layer15_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer15_Mean_GPU, CL_TRUE, 0, sizeof(double) * FIFTEENTH_LAYER_CHANNELS, Layer15_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer15_StanDev_GPU, CL_TRUE, 0, sizeof(double) * FIFTEENTH_LAYER_CHANNELS, Layer15_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer15_Gamma_GPU, CL_TRUE, 0, sizeof(double) * FIFTEENTH_LAYER_CHANNELS, Layer15_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer15_Beta_GPU, CL_TRUE, 0, sizeof(double) * FIFTEENTH_LAYER_CHANNELS, Layer15_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_PSC", &err);
    
    int inputWidth = 14, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 16, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 512, channelSize = 512, stride = 1, offset = 17;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer15_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer15_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer16_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer15_Mean_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer15_StanDev_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer15_Gamma_GPU);
    clSetKernelArg(kernel, 20, sizeof(cl_mem), (void *) &Layer15_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {512, 1 * 14, 1 * 14};
    size_t localWorkSize_A[3] = {1, 14, 14};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    clReleaseMemObject(Layer15_Weights_GPU);
    clReleaseMemObject(Layer15_Mean_GPU);
    clReleaseMemObject(Layer15_StanDev_GPU);
    clReleaseMemObject(Layer15_Gamma_GPU);
    clReleaseMemObject(Layer15_Beta_GPU);

    free(Layer15_Weights_CPU);
    free(Layer15_Mean_CPU);
    free(Layer15_StanDev_CPU);
    free(Layer15_Gamma_CPU);
    free(Layer15_Beta_CPU);
}

void Read_FifteenthLayer_Data(double *Layer15_Weights_CPU,
    double * Layer15_Mean_CPU,
    double * Layer15_StanDev_CPU,
    double * Layer15_Gamma_CPU,
    double * Layer15_Beta_CPU
){
    read_File("data/FifteenthLayer/weightsNorm.txt", Layer15_Weights_CPU);
    read_File("data/FifteenthLayer/Fifteenth_Layer_Mean.txt", Layer15_Mean_CPU);
    read_File("data/FifteenthLayer/Fifteenth_Layer_StanDev.txt", Layer15_StanDev_CPU);
    read_File("data/FifteenthLayer/Fifteenth_Layer_Gamma.txt", Layer15_Gamma_CPU);
    read_File("data/FifteenthLayer/Fifteenth_Layer_Beta.txt", Layer15_Beta_CPU);
}

void Execute_Fourteenth_Layer(cl_mem Layer14_Neurons_GPU,
    cl_mem Layer15_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer14_Weights_CPU = (double *) malloc(sizeof(double) * FOURTEENTH_LAYER_WEIGHT_SIZE);
    double *Layer14_Mean_CPU = (double *) malloc(sizeof(double) * FOURTEENTH_LAYER_CHANNELS);
    double *Layer14_StanDev_CPU = (double *) malloc(sizeof(double) * FOURTEENTH_LAYER_CHANNELS);
    double *Layer14_Gamma_CPU = (double *) malloc(sizeof(double) * FOURTEENTH_LAYER_CHANNELS);
    double *Layer14_Beta_CPU = (double *) malloc(sizeof(double) * FOURTEENTH_LAYER_CHANNELS);

    Read_FourteenthLayer_Data(
        Layer14_Weights_CPU,        
        Layer14_Mean_CPU,
        Layer14_StanDev_CPU,
        Layer14_Gamma_CPU,
        Layer14_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer14_Weights_GPU,
           Layer14_Mean_GPU,
           Layer14_StanDev_GPU,
           Layer14_Gamma_GPU,
           Layer14_Beta_GPU;

    cl_int err;
    
    Layer14_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FOURTEENTH_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer14_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FOURTEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer14_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FOURTEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer14_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FOURTEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer14_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FOURTEENTH_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer14_Weights_GPU, CL_TRUE, 0, sizeof(double) * FOURTEENTH_LAYER_WEIGHT_SIZE, Layer14_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer14_Mean_GPU, CL_TRUE, 0, sizeof(double) * FOURTEENTH_LAYER_CHANNELS, Layer14_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer14_StanDev_GPU, CL_TRUE, 0, sizeof(double) * FOURTEENTH_LAYER_CHANNELS, Layer14_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer14_Gamma_GPU, CL_TRUE, 0, sizeof(double) * FOURTEENTH_LAYER_CHANNELS, Layer14_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer14_Beta_GPU, CL_TRUE, 0, sizeof(double) * FOURTEENTH_LAYER_CHANNELS, Layer14_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_DSC", &err);
    
    int inputWidth = 16, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 14, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 9, kernelSize = 3, stride = 1;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer14_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer14_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer15_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(cl_mem), (void *) &Layer14_Mean_GPU);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer14_StanDev_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer14_Gamma_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer14_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {512, 1 * 14, 1 * 14};
    size_t localWorkSize_A[3] = {1, 14, 14};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    clReleaseMemObject(Layer14_Weights_GPU);
    clReleaseMemObject(Layer14_Mean_GPU);
    clReleaseMemObject(Layer14_StanDev_GPU);
    clReleaseMemObject(Layer14_Gamma_GPU);
    clReleaseMemObject(Layer14_Beta_GPU);

    free(Layer14_Weights_CPU);
    free(Layer14_Mean_CPU);
    free(Layer14_StanDev_CPU);
    free(Layer14_Gamma_CPU);
    free(Layer14_Beta_CPU);
}

void Read_FourteenthLayer_Data(double *Layer14_Weights_CPU,
    double * Layer14_Mean_CPU,
    double * Layer14_StanDev_CPU,
    double * Layer14_Gamma_CPU,
    double * Layer14_Beta_CPU
){
    read_File("data/FourteenthLayer/weightsNorm.txt", Layer14_Weights_CPU);
    read_File("data/FourteenthLayer/Fourteenth_Layer_Mean.txt", Layer14_Mean_CPU);
    read_File("data/FourteenthLayer/Fourteenth_Layer_StanDev.txt", Layer14_StanDev_CPU);
    read_File("data/FourteenthLayer/Fourteenth_Layer_Gamma.txt", Layer14_Gamma_CPU);
    read_File("data/FourteenthLayer/Fourteenth_Layer_Beta.txt", Layer14_Beta_CPU);
}

void Execute_Thirteenth_Layer(cl_mem Layer13_Neurons_GPU,
    cl_mem Layer14_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer13_Weights_CPU = (double *) malloc(sizeof(double) * THIRTEENTH_LAYER_WEIGHT_SIZE);
    double *Layer13_Mean_CPU = (double *) malloc(sizeof(double) * THIRTEENTH_LAYER_CHANNELS);
    double *Layer13_StanDev_CPU = (double *) malloc(sizeof(double) * THIRTEENTH_LAYER_CHANNELS);
    double *Layer13_Gamma_CPU = (double *) malloc(sizeof(double) * THIRTEENTH_LAYER_CHANNELS);
    double *Layer13_Beta_CPU = (double *) malloc(sizeof(double) * THIRTEENTH_LAYER_CHANNELS);

    Read_ThirteenthLayer_Data(
        Layer13_Weights_CPU,        
        Layer13_Mean_CPU,
        Layer13_StanDev_CPU,
        Layer13_Gamma_CPU,
        Layer13_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer13_Weights_GPU,
           Layer13_Mean_GPU,
           Layer13_StanDev_GPU,
           Layer13_Gamma_GPU,
           Layer13_Beta_GPU;

    cl_int err;
    
    Layer13_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * THIRTEENTH_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer13_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * THIRTEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer13_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * THIRTEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer13_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * THIRTEENTH_LAYER_CHANNELS, NULL, NULL);
    Layer13_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * THIRTEENTH_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer13_Weights_GPU, CL_TRUE, 0, sizeof(double) * THIRTEENTH_LAYER_WEIGHT_SIZE, Layer13_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer13_Mean_GPU, CL_TRUE, 0, sizeof(double) * THIRTEENTH_LAYER_CHANNELS, Layer13_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer13_StanDev_GPU, CL_TRUE, 0, sizeof(double) * THIRTEENTH_LAYER_CHANNELS, Layer13_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer13_Gamma_GPU, CL_TRUE, 0, sizeof(double) * THIRTEENTH_LAYER_CHANNELS, Layer13_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer13_Beta_GPU, CL_TRUE, 0, sizeof(double) * THIRTEENTH_LAYER_CHANNELS, Layer13_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_PSC", &err);
    
    int inputWidth = 14, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 16, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 256, channelSize = 256, stride = 1, offset = 17;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer13_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer13_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer14_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer13_Mean_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer13_StanDev_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer13_Gamma_GPU);
    clSetKernelArg(kernel, 20, sizeof(cl_mem), (void *) &Layer13_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {512, 1 * 14, 1 * 14};
    size_t localWorkSize_A[3] = {1, 14, 14};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    clReleaseMemObject(Layer13_Weights_GPU);
    clReleaseMemObject(Layer13_Mean_GPU);
    clReleaseMemObject(Layer13_StanDev_GPU);
    clReleaseMemObject(Layer13_Gamma_GPU);
    clReleaseMemObject(Layer13_Beta_GPU);

    free(Layer13_Weights_CPU);
    free(Layer13_Mean_CPU);
    free(Layer13_StanDev_CPU);
    free(Layer13_Gamma_CPU);
    free(Layer13_Beta_CPU);
}

void Read_ThirteenthLayer_Data(double *Layer13_Weights_CPU,
    double * Layer13_Mean_CPU,
    double * Layer13_StanDev_CPU,
    double * Layer13_Gamma_CPU,
    double * Layer13_Beta_CPU
){
    read_File("data/ThirteenthLayer/weightsNorm.txt", Layer13_Weights_CPU);
    read_File("data/ThirteenthLayer/Thirteenth_Layer_Mean.txt", Layer13_Mean_CPU);
    read_File("data/ThirteenthLayer/Thirteenth_Layer_StanDev.txt", Layer13_StanDev_CPU);
    read_File("data/ThirteenthLayer/Thirteenth_Layer_Gamma.txt", Layer13_Gamma_CPU);
    read_File("data/ThirteenthLayer/Thirteenth_Layer_Beta.txt", Layer13_Beta_CPU);
}

void Execute_Twelveth_Layer(cl_mem Layer12_Neurons_GPU,
    cl_mem Layer13_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer12_Weights_CPU = (double *) malloc(sizeof(double) * TWELFTH_LAYER_WEIGHT_SIZE);
    double *Layer12_Mean_CPU = (double *) malloc(sizeof(double) * TWELFTH_LAYER_CHANNELS);
    double *Layer12_StanDev_CPU = (double *) malloc(sizeof(double) * TWELFTH_LAYER_CHANNELS);
    double *Layer12_Gamma_CPU = (double *) malloc(sizeof(double) * TWELFTH_LAYER_CHANNELS);
    double *Layer12_Beta_CPU = (double *) malloc(sizeof(double) * TWELFTH_LAYER_CHANNELS);

    Read_TwelvethLayer_Data(
        Layer12_Weights_CPU,        
        Layer12_Mean_CPU,
        Layer12_StanDev_CPU,
        Layer12_Gamma_CPU,
        Layer12_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer12_Weights_GPU,
           Layer12_Mean_GPU,
           Layer12_StanDev_GPU,
           Layer12_Gamma_GPU,
           Layer12_Beta_GPU;

    cl_int err;
    
    Layer12_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWELFTH_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer12_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWELFTH_LAYER_CHANNELS, NULL, NULL);
    Layer12_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWELFTH_LAYER_CHANNELS, NULL, NULL);
    Layer12_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWELFTH_LAYER_CHANNELS, NULL, NULL);
    Layer12_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TWELFTH_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer12_Weights_GPU, CL_TRUE, 0, sizeof(double) * TWELFTH_LAYER_WEIGHT_SIZE, Layer12_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer12_Mean_GPU, CL_TRUE, 0, sizeof(double) * TWELFTH_LAYER_CHANNELS, Layer12_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer12_StanDev_GPU, CL_TRUE, 0, sizeof(double) * TWELFTH_LAYER_CHANNELS, Layer12_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer12_Gamma_GPU, CL_TRUE, 0, sizeof(double) * TWELFTH_LAYER_CHANNELS, Layer12_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer12_Beta_GPU, CL_TRUE, 0, sizeof(double) * TWELFTH_LAYER_CHANNELS, Layer12_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_DSC", &err);
    
    int inputWidth = 29, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 14, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 9, kernelSize = 3, stride = 2;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer12_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer12_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer13_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(cl_mem), (void *) &Layer12_Mean_GPU);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer12_StanDev_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer12_Gamma_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer12_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {256, 1 * 14, 1 * 14};
    size_t localWorkSize_A[3] = {1, 14, 14};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    clReleaseMemObject(Layer12_Weights_GPU);
    clReleaseMemObject(Layer12_Mean_GPU);
    clReleaseMemObject(Layer12_StanDev_GPU);
    clReleaseMemObject(Layer12_Gamma_GPU);
    clReleaseMemObject(Layer12_Beta_GPU);

    free(Layer12_Weights_CPU);
    free(Layer12_Mean_CPU);
    free(Layer12_StanDev_CPU);
    free(Layer12_Gamma_CPU);
    free(Layer12_Beta_CPU);
}

void Read_TwelvethLayer_Data(double *Layer12_Weights_CPU,
    double * Layer12_Mean_CPU,
    double * Layer12_StanDev_CPU,
    double * Layer12_Gamma_CPU,
    double * Layer12_Beta_CPU
){
    read_File("data/TwelvethLayer/weightsNorm.txt", Layer12_Weights_CPU);
    read_File("data/TwelvethLayer/Twelveth_Layer_Mean.txt", Layer12_Mean_CPU);
    read_File("data/TwelvethLayer/Twelveth_Layer_StanDev.txt", Layer12_StanDev_CPU);
    read_File("data/TwelvethLayer/Twelveth_Layer_Gamma.txt", Layer12_Gamma_CPU);
    read_File("data/TwelvethLayer/Twelveth_Layer_Beta.txt", Layer12_Beta_CPU);
}

void Execute_Eleventh_Layer(cl_mem Layer11_Neurons_GPU,
    cl_mem Layer12_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer11_Weights_CPU = (double *) malloc(sizeof(double) * ELEVENTH_LAYER_WEIGHT_SIZE);
    double *Layer11_Mean_CPU = (double *) malloc(sizeof(double) * ELEVENTH_LAYER_CHANNELS);
    double *Layer11_StanDev_CPU = (double *) malloc(sizeof(double) * ELEVENTH_LAYER_CHANNELS);
    double *Layer11_Gamma_CPU = (double *) malloc(sizeof(double) * ELEVENTH_LAYER_CHANNELS);
    double *Layer11_Beta_CPU = (double *) malloc(sizeof(double) * ELEVENTH_LAYER_CHANNELS);

    Read_EleventhLayer_Data(
        Layer11_Weights_CPU,        
        Layer11_Mean_CPU,
        Layer11_StanDev_CPU,
        Layer11_Gamma_CPU,
        Layer11_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer11_Weights_GPU,
           Layer11_Mean_GPU,
           Layer11_StanDev_GPU,
           Layer11_Gamma_GPU,
           Layer11_Beta_GPU;

    cl_int err;
    
    Layer11_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * ELEVENTH_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer11_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * ELEVENTH_LAYER_CHANNELS, NULL, NULL);
    Layer11_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * ELEVENTH_LAYER_CHANNELS, NULL, NULL);
    Layer11_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * ELEVENTH_LAYER_CHANNELS, NULL, NULL);
    Layer11_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * ELEVENTH_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer11_Weights_GPU, CL_TRUE, 0, sizeof(double) * ELEVENTH_LAYER_WEIGHT_SIZE, Layer11_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer11_Mean_GPU, CL_TRUE, 0, sizeof(double) * ELEVENTH_LAYER_CHANNELS, Layer11_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer11_StanDev_GPU, CL_TRUE, 0, sizeof(double) * ELEVENTH_LAYER_CHANNELS, Layer11_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer11_Gamma_GPU, CL_TRUE, 0, sizeof(double) * ELEVENTH_LAYER_CHANNELS, Layer11_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer11_Beta_GPU, CL_TRUE, 0, sizeof(double) * ELEVENTH_LAYER_CHANNELS, Layer11_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_PSC", &err);
    
    int inputWidth = 28, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 29, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 256, channelSize = 256, stride = 1, offset = 0;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer11_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer11_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer12_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer11_Mean_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer11_StanDev_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer11_Gamma_GPU);
    clSetKernelArg(kernel, 20, sizeof(cl_mem), (void *) &Layer11_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {256, 1 * 28, 1 * 28};
    size_t localWorkSize_A[3] = {1, 28, 28};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    clReleaseMemObject(Layer11_Weights_GPU);
    clReleaseMemObject(Layer11_Mean_GPU);
    clReleaseMemObject(Layer11_StanDev_GPU);
    clReleaseMemObject(Layer11_Gamma_GPU);
    clReleaseMemObject(Layer11_Beta_GPU);

    free(Layer11_Weights_CPU);
    free(Layer11_Mean_CPU);
    free(Layer11_StanDev_CPU);
    free(Layer11_Gamma_CPU);
    free(Layer11_Beta_CPU);
}

void Read_EleventhLayer_Data(double *Layer11_Weights_CPU,
    double * Layer11_Mean_CPU,
    double * Layer11_StanDev_CPU,
    double * Layer11_Gamma_CPU,
    double * Layer11_Beta_CPU
){
    read_File("data/EleventhLayer/weightsNorm.txt", Layer11_Weights_CPU);
    read_File("data/EleventhLayer/Eleventh_Layer_Mean.txt", Layer11_Mean_CPU);
    read_File("data/EleventhLayer/Eleventh_Layer_StanDev.txt", Layer11_StanDev_CPU);
    read_File("data/EleventhLayer/Eleventh_Layer_Gamma.txt", Layer11_Gamma_CPU);
    read_File("data/EleventhLayer/Eleventh_Layer_Beta.txt", Layer11_Beta_CPU);
}

void Execute_Tenth_Layer(cl_mem Layer10_Neurons_GPU,
    cl_mem Layer11_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer10_Weights_CPU = (double *) malloc(sizeof(double) * TENTH_LAYER_WEIGHT_SIZE);
    double *Layer10_Mean_CPU = (double *) malloc(sizeof(double) * TENTH_LAYER_CHANNELS);
    double *Layer10_StanDev_CPU = (double *) malloc(sizeof(double) * TENTH_LAYER_CHANNELS);
    double *Layer10_Gamma_CPU = (double *) malloc(sizeof(double) * TENTH_LAYER_CHANNELS);
    double *Layer10_Beta_CPU = (double *) malloc(sizeof(double) * TENTH_LAYER_CHANNELS);

    Read_TenthLayer_Data(
        Layer10_Weights_CPU,        
        Layer10_Mean_CPU,
        Layer10_StanDev_CPU,
        Layer10_Gamma_CPU,
        Layer10_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer10_Weights_GPU,
           Layer10_Mean_GPU,
           Layer10_StanDev_GPU,
           Layer10_Gamma_GPU,
           Layer10_Beta_GPU;

    cl_int err;
    
    Layer10_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TENTH_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer10_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TENTH_LAYER_CHANNELS, NULL, NULL);
    Layer10_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TENTH_LAYER_CHANNELS, NULL, NULL);
    Layer10_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TENTH_LAYER_CHANNELS, NULL, NULL);
    Layer10_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * TENTH_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer10_Weights_GPU, CL_TRUE, 0, sizeof(double) * TENTH_LAYER_WEIGHT_SIZE, Layer10_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer10_Mean_GPU, CL_TRUE, 0, sizeof(double) * TENTH_LAYER_CHANNELS, Layer10_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer10_StanDev_GPU, CL_TRUE, 0, sizeof(double) * TENTH_LAYER_CHANNELS, Layer10_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer10_Gamma_GPU, CL_TRUE, 0, sizeof(double) * TENTH_LAYER_CHANNELS, Layer10_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer10_Beta_GPU, CL_TRUE, 0, sizeof(double) * TENTH_LAYER_CHANNELS, Layer10_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_DSC", &err);
    
    int inputWidth = 30, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 28, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 9, kernelSize = 3, stride = 1;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer10_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer10_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer11_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(cl_mem), (void *) &Layer10_Mean_GPU);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer10_StanDev_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer10_Gamma_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer10_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {256, 1 * 28, 1 * 28};
    size_t localWorkSize_A[3] = {1, 28, 28};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    clReleaseMemObject(Layer10_Weights_GPU);
    clReleaseMemObject(Layer10_Mean_GPU);
    clReleaseMemObject(Layer10_StanDev_GPU);
    clReleaseMemObject(Layer10_Gamma_GPU);
    clReleaseMemObject(Layer10_Beta_GPU);

    free(Layer10_Weights_CPU);
    free(Layer10_Mean_CPU);
    free(Layer10_StanDev_CPU);
    free(Layer10_Gamma_CPU);
    free(Layer10_Beta_CPU);
}

void Read_TenthLayer_Data(double *Layer10_Weights_CPU,
    double * Layer10_Mean_CPU,
    double * Layer10_StanDev_CPU,
    double * Layer10_Gamma_CPU,
    double * Layer10_Beta_CPU
){
    read_File("data/TenthLayer/weightsNorm.txt", Layer10_Weights_CPU);
    read_File("data/TenthLayer/Tenth_Layer_Mean.txt", Layer10_Mean_CPU);
    read_File("data/TenthLayer/Tenth_Layer_StanDev.txt", Layer10_StanDev_CPU);
    read_File("data/TenthLayer/Tenth_Layer_Gamma.txt", Layer10_Gamma_CPU);
    read_File("data/TenthLayer/Tenth_Layer_Beta.txt", Layer10_Beta_CPU);
}

void Execute_Ninth_Layer(cl_mem Layer9_Neurons_GPU,
    cl_mem Layer10_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer9_Weights_CPU = (double *) malloc(sizeof(double) * NINTH_LAYER_WEIGHT_SIZE);
    double *Layer9_Mean_CPU = (double *) malloc(sizeof(double) * NINTH_LAYER_CHANNELS);
    double *Layer9_StanDev_CPU = (double *) malloc(sizeof(double) * NINTH_LAYER_CHANNELS);
    double *Layer9_Gamma_CPU = (double *) malloc(sizeof(double) * NINTH_LAYER_CHANNELS);
    double *Layer9_Beta_CPU = (double *) malloc(sizeof(double) * NINTH_LAYER_CHANNELS);

    Read_NinthLayer_Data(
        Layer9_Weights_CPU,        
        Layer9_Mean_CPU,
        Layer9_StanDev_CPU,
        Layer9_Gamma_CPU,
        Layer9_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer9_Weights_GPU,
           Layer9_Mean_GPU,
           Layer9_StanDev_GPU,
           Layer9_Gamma_GPU,
           Layer9_Beta_GPU;

    cl_int err;
    
    Layer9_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * NINTH_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer9_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * NINTH_LAYER_CHANNELS, NULL, NULL);
    Layer9_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * NINTH_LAYER_CHANNELS, NULL, NULL);
    Layer9_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * NINTH_LAYER_CHANNELS, NULL, NULL);
    Layer9_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * NINTH_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer9_Weights_GPU, CL_TRUE, 0, sizeof(double) * NINTH_LAYER_WEIGHT_SIZE, Layer9_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer9_Mean_GPU, CL_TRUE, 0, sizeof(double) * NINTH_LAYER_CHANNELS, Layer9_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer9_StanDev_GPU, CL_TRUE, 0, sizeof(double) * NINTH_LAYER_CHANNELS, Layer9_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer9_Gamma_GPU, CL_TRUE, 0, sizeof(double) * NINTH_LAYER_CHANNELS, Layer9_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer9_Beta_GPU, CL_TRUE, 0, sizeof(double) * NINTH_LAYER_CHANNELS, Layer9_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_PSC", &err);
    
    int inputWidth = 28, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 30, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 128, channelSize = 128, stride = 1, offset = 31;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer9_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer9_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer10_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer9_Mean_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer9_StanDev_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer9_Gamma_GPU);
    clSetKernelArg(kernel, 20, sizeof(cl_mem), (void *) &Layer9_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {256, 1 * 28, 1 * 28};
    size_t localWorkSize_A[3] = {1, 28, 28};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    clReleaseMemObject(Layer9_Weights_GPU);
    clReleaseMemObject(Layer9_Mean_GPU);
    clReleaseMemObject(Layer9_StanDev_GPU);
    clReleaseMemObject(Layer9_Gamma_GPU);
    clReleaseMemObject(Layer9_Beta_GPU);

    free(Layer9_Weights_CPU);
    free(Layer9_Mean_CPU);
    free(Layer9_StanDev_CPU);
    free(Layer9_Gamma_CPU);
    free(Layer9_Beta_CPU);
}

void Read_NinthLayer_Data(double *Layer9_Weights_CPU,
    double * Layer9_Mean_CPU,
    double * Layer9_StanDev_CPU,
    double * Layer9_Gamma_CPU,
    double * Layer9_Beta_CPU
){
    read_File("data/NinthLayer/weightsNorm.txt", Layer9_Weights_CPU);
    read_File("data/NinthLayer/Ninth_Layer_Mean.txt", Layer9_Mean_CPU);
    read_File("data/NinthLayer/Ninth_Layer_StanDev.txt", Layer9_StanDev_CPU);
    read_File("data/NinthLayer/Ninth_Layer_Gamma.txt", Layer9_Gamma_CPU);
    read_File("data/NinthLayer/Ninth_Layer_Beta.txt", Layer9_Beta_CPU);
}

void Execute_Eighth_Layer(cl_mem Layer8_Neurons_GPU,
    cl_mem Layer9_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer8_Weights_CPU = (double *) malloc(sizeof(double) * EIGHTH_LAYER_WEIGHT_SIZE);
    double *Layer8_Mean_CPU = (double *) malloc(sizeof(double) * EIGHTH_LAYER_CHANNELS);
    double *Layer8_StanDev_CPU = (double *) malloc(sizeof(double) * EIGHTH_LAYER_CHANNELS);
    double *Layer8_Gamma_CPU = (double *) malloc(sizeof(double) * EIGHTH_LAYER_CHANNELS);
    double *Layer8_Beta_CPU = (double *) malloc(sizeof(double) * EIGHTH_LAYER_CHANNELS);

    Read_EighthLayer_Data(
        Layer8_Weights_CPU,        
        Layer8_Mean_CPU,
        Layer8_StanDev_CPU,
        Layer8_Gamma_CPU,
        Layer8_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer8_Weights_GPU,
           Layer8_Mean_GPU,
           Layer8_StanDev_GPU,
           Layer8_Gamma_GPU,
           Layer8_Beta_GPU;

    cl_int err;
    
    Layer8_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * EIGHTH_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer8_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * EIGHTH_LAYER_CHANNELS, NULL, NULL);
    Layer8_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * EIGHTH_LAYER_CHANNELS, NULL, NULL);
    Layer8_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * EIGHTH_LAYER_CHANNELS, NULL, NULL);
    Layer8_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * EIGHTH_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer8_Weights_GPU, CL_TRUE, 0, sizeof(double) * EIGHTH_LAYER_WEIGHT_SIZE, Layer8_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer8_Mean_GPU, CL_TRUE, 0, sizeof(double) * EIGHTH_LAYER_CHANNELS, Layer8_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer8_StanDev_GPU, CL_TRUE, 0, sizeof(double) * EIGHTH_LAYER_CHANNELS, Layer8_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer8_Gamma_GPU, CL_TRUE, 0, sizeof(double) * EIGHTH_LAYER_CHANNELS, Layer8_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer8_Beta_GPU, CL_TRUE, 0, sizeof(double) * EIGHTH_LAYER_CHANNELS, Layer8_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_DSC", &err);
    
    int inputWidth = 57, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 28, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 9, kernelSize = 3, stride = 2;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer8_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer8_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer9_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(cl_mem), (void *) &Layer8_Mean_GPU);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer8_StanDev_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer8_Gamma_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer8_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {256, 1 * 28, 1 * 28};
    size_t localWorkSize_A[3] = {1, 28, 28};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    clReleaseMemObject(Layer8_Weights_GPU);
    clReleaseMemObject(Layer8_Mean_GPU);
    clReleaseMemObject(Layer8_StanDev_GPU);
    clReleaseMemObject(Layer8_Gamma_GPU);
    clReleaseMemObject(Layer8_Beta_GPU);

    free(Layer8_Weights_CPU);
    free(Layer8_Mean_CPU);
    free(Layer8_StanDev_CPU);
    free(Layer8_Gamma_CPU);
    free(Layer8_Beta_CPU);
}

void Read_EighthLayer_Data(double *Layer8_Weights_CPU,
    double * Layer8_Mean_CPU,
    double * Layer8_StanDev_CPU,
    double * Layer8_Gamma_CPU,
    double * Layer8_Beta_CPU
){
    read_File("data/EighthLayer/weightsNorm.txt", Layer8_Weights_CPU);
    read_File("data/EighthLayer/Eighth_Layer_Mean.txt", Layer8_Mean_CPU);
    read_File("data/EighthLayer/Eighth_Layer_StanDev.txt", Layer8_StanDev_CPU);
    read_File("data/EighthLayer/Eighth_Layer_Gamma.txt", Layer8_Gamma_CPU);
    read_File("data/EighthLayer/Eighth_Layer_Beta.txt", Layer8_Beta_CPU);
}

void Execute_Seventh_Layer(cl_mem Layer7_Neurons_GPU,
    cl_mem Layer8_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer7_Weights_CPU = (double *) malloc(sizeof(double) * SEVENTH_LAYER_WEIGHT_SIZE);
    double *Layer7_Mean_CPU = (double *) malloc(sizeof(double) * SEVENTH_LAYER_CHANNELS);
    double *Layer7_StanDev_CPU = (double *) malloc(sizeof(double) * SEVENTH_LAYER_CHANNELS);
    double *Layer7_Gamma_CPU = (double *) malloc(sizeof(double) * SEVENTH_LAYER_CHANNELS);
    double *Layer7_Beta_CPU = (double *) malloc(sizeof(double) * SEVENTH_LAYER_CHANNELS);

    Read_SeventhLayer_Data(
        Layer7_Weights_CPU,        
        Layer7_Mean_CPU,
        Layer7_StanDev_CPU,
        Layer7_Gamma_CPU,
        Layer7_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer7_Weights_GPU,
           Layer7_Mean_GPU,
           Layer7_StanDev_GPU,
           Layer7_Gamma_GPU,
           Layer7_Beta_GPU;

    cl_int err;
    
    Layer7_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SEVENTH_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer7_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SEVENTH_LAYER_CHANNELS, NULL, NULL);
    Layer7_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SEVENTH_LAYER_CHANNELS, NULL, NULL);
    Layer7_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SEVENTH_LAYER_CHANNELS, NULL, NULL);
    Layer7_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SEVENTH_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer7_Weights_GPU, CL_TRUE, 0, sizeof(double) * SEVENTH_LAYER_WEIGHT_SIZE, Layer7_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer7_Mean_GPU, CL_TRUE, 0, sizeof(double) * SEVENTH_LAYER_CHANNELS, Layer7_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer7_StanDev_GPU, CL_TRUE, 0, sizeof(double) * SEVENTH_LAYER_CHANNELS, Layer7_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer7_Gamma_GPU, CL_TRUE, 0, sizeof(double) * SEVENTH_LAYER_CHANNELS, Layer7_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer7_Beta_GPU, CL_TRUE, 0, sizeof(double) * SEVENTH_LAYER_CHANNELS, Layer7_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_PSC", &err);
    
    int inputWidth = 56, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 57, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 128, channelSize = 128, stride = 1, offset = 0;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer7_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer7_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer8_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer7_Mean_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer7_StanDev_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer7_Gamma_GPU);
    clSetKernelArg(kernel, 20, sizeof(cl_mem), (void *) &Layer7_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {128, 1 * 32, 1 * 32};
    size_t localWorkSize_A[3] = {1, 32, 32};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    cl_kernel kernelB = clCreateKernel(program, "executeGenericFunctions_PSC", &err);

    inputWidth = 56, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 32, blockOffset2 = 0;
    outputWidth = 57, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 32, blockOffset2Out = 0;

    // Setting arguments
    clSetKernelArg(kernelB, 0, sizeof(cl_mem), (void *) &Layer7_Neurons_GPU);
    clSetKernelArg(kernelB, 1, sizeof(cl_mem), (void *) &Layer7_Weights_GPU);
    clSetKernelArg(kernelB, 2, sizeof(cl_mem), (void *) &Layer8_Neurons_GPU);
    clSetKernelArg(kernelB, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernelB, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernelB, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernelB, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernelB, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernelB, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernelB, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernelB, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernelB, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernelB, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernelB, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernelB, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernelB, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernelB, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernelB, 17, sizeof(cl_mem), (void *) &Layer7_Mean_GPU);
    clSetKernelArg(kernelB, 18, sizeof(cl_mem), (void *) &Layer7_StanDev_GPU);
    clSetKernelArg(kernelB, 19, sizeof(cl_mem), (void *) &Layer7_Gamma_GPU);
    clSetKernelArg(kernelB, 20, sizeof(cl_mem), (void *) &Layer7_Beta_GPU);


    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_B[3] = {128, 1 * 32, 1 * 24};
    size_t localWorkSize_B[3] = {1, 32, 24};

    err = clEnqueueNDRangeKernel(queue, kernelB, 3, NULL, globalWorkSize_B, localWorkSize_B,
                                                              0, NULL, NULL);

    clFinish(queue);

    cl_kernel kernelC = clCreateKernel(program, "executeGenericFunctions_PSC", &err);

    inputWidth = 56, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 56 * 32, blockOffset2 = 0;
    outputWidth = 57, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 57 * 32, blockOffset2Out = 0;

    // Setting arguments
    clSetKernelArg(kernelC, 0, sizeof(cl_mem), (void *) &Layer7_Neurons_GPU);
    clSetKernelArg(kernelC, 1, sizeof(cl_mem), (void *) &Layer7_Weights_GPU);
    clSetKernelArg(kernelC, 2, sizeof(cl_mem), (void *) &Layer8_Neurons_GPU);
    clSetKernelArg(kernelC, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernelC, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernelC, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernelC, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernelC, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernelC, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernelC, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernelC, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernelC, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernelC, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernelC, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernelC, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernelC, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernelC, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernelC, 17, sizeof(cl_mem), (void *) &Layer7_Mean_GPU);
    clSetKernelArg(kernelC, 18, sizeof(cl_mem), (void *) &Layer7_StanDev_GPU);
    clSetKernelArg(kernelC, 19, sizeof(cl_mem), (void *) &Layer7_Gamma_GPU);
    clSetKernelArg(kernelC, 20, sizeof(cl_mem), (void *) &Layer7_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_C[3] = {128, 1 * 24, 1 * 32};
    size_t localWorkSize_C[3] = {1, 24, 32};

    err = clEnqueueNDRangeKernel(queue, kernelC, 3, NULL, globalWorkSize_C, localWorkSize_C,
                                                              0, NULL, NULL);
    
    cl_kernel kernelD = clCreateKernel(program, "executeGenericFunctions_PSC", &err);

    inputWidth = 56, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 56 * 32, blockOffset2 = 32;
    outputWidth = 57, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 57 * 32, blockOffset2Out = 32;

    // Setting arguments
    clSetKernelArg(kernelD, 0, sizeof(cl_mem), (void *) &Layer7_Neurons_GPU);
    clSetKernelArg(kernelD, 1, sizeof(cl_mem), (void *) &Layer7_Weights_GPU);
    clSetKernelArg(kernelD, 2, sizeof(cl_mem), (void *) &Layer8_Neurons_GPU);
    clSetKernelArg(kernelD, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernelD, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernelD, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernelD, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernelD, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernelD, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernelD, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernelD, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernelD, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernelD, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernelD, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernelD, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernelD, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernelD, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernelD, 17, sizeof(cl_mem), (void *) &Layer7_Mean_GPU);
    clSetKernelArg(kernelD, 18, sizeof(cl_mem), (void *) &Layer7_StanDev_GPU);
    clSetKernelArg(kernelD, 19, sizeof(cl_mem), (void *) &Layer7_Gamma_GPU);
    clSetKernelArg(kernelD, 20, sizeof(cl_mem), (void *) &Layer7_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_D[3] = {128, 1 * 24, 1 * 24};
    size_t localWorkSize_D[3] = {1, 24, 24};

    err = clEnqueueNDRangeKernel(queue, kernelD, 3, NULL, globalWorkSize_D, localWorkSize_D,
                                                              0, NULL, NULL);

    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    clReleaseMemObject(Layer7_Weights_GPU);
    clReleaseMemObject(Layer7_Mean_GPU);
    clReleaseMemObject(Layer7_StanDev_GPU);
    clReleaseMemObject(Layer7_Gamma_GPU);
    clReleaseMemObject(Layer7_Beta_GPU);

    free(Layer7_Weights_CPU);
    free(Layer7_Mean_CPU);
    free(Layer7_StanDev_CPU);
    free(Layer7_Gamma_CPU);
    free(Layer7_Beta_CPU);
}

void Read_SeventhLayer_Data(double *Layer7_Weights_CPU,
    double * Layer7_Mean_CPU,
    double * Layer7_StanDev_CPU,
    double * Layer7_Gamma_CPU,
    double * Layer7_Beta_CPU
){
    read_File("data/SeventhLayer/weightsNorm.txt", Layer7_Weights_CPU);
    read_File("data/SeventhLayer/Seventh_Layer_Mean.txt", Layer7_Mean_CPU);
    read_File("data/SeventhLayer/Seventh_Layer_StanDev.txt", Layer7_StanDev_CPU);
    read_File("data/SeventhLayer/Seventh_Layer_Gamma.txt", Layer7_Gamma_CPU);
    read_File("data/SeventhLayer/Seventh_Layer_Beta.txt", Layer7_Beta_CPU);
}

void Execute_Sixth_Layer(cl_mem Layer6_Neurons_GPU,
    cl_mem Layer7_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer6_Weights_CPU = (double *) malloc(sizeof(double) * SIXTH_LAYER_WEIGHT_SIZE);
    double *Layer6_Mean_CPU = (double *) malloc(sizeof(double) * SIXTH_LAYER_CHANNELS);
    double *Layer6_StanDev_CPU = (double *) malloc(sizeof(double) * SIXTH_LAYER_CHANNELS);
    double *Layer6_Gamma_CPU = (double *) malloc(sizeof(double) * SIXTH_LAYER_CHANNELS);
    double *Layer6_Beta_CPU = (double *) malloc(sizeof(double) * SIXTH_LAYER_CHANNELS);

    Read_SixthLayer_Data(
        Layer6_Weights_CPU,        
        Layer6_Mean_CPU,
        Layer6_StanDev_CPU,
        Layer6_Gamma_CPU,
        Layer6_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer6_Weights_GPU,
           Layer6_Mean_GPU,
           Layer6_StanDev_GPU,
           Layer6_Gamma_GPU,
           Layer6_Beta_GPU;

    cl_int err;
    
    Layer6_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SIXTH_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer6_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SIXTH_LAYER_CHANNELS, NULL, NULL);
    Layer6_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SIXTH_LAYER_CHANNELS, NULL, NULL);
    Layer6_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SIXTH_LAYER_CHANNELS, NULL, NULL);
    Layer6_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SIXTH_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer6_Weights_GPU, CL_TRUE, 0, sizeof(double) * SIXTH_LAYER_WEIGHT_SIZE, Layer6_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer6_Mean_GPU, CL_TRUE, 0, sizeof(double) * SIXTH_LAYER_CHANNELS, Layer6_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer6_StanDev_GPU, CL_TRUE, 0, sizeof(double) * SIXTH_LAYER_CHANNELS, Layer6_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer6_Gamma_GPU, CL_TRUE, 0, sizeof(double) * SIXTH_LAYER_CHANNELS, Layer6_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer6_Beta_GPU, CL_TRUE, 0, sizeof(double) * SIXTH_LAYER_CHANNELS, Layer6_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_DSC", &err);
    
    int inputWidth = 58, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 56, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 9, kernelSize = 3, stride = 1;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer6_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer6_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer7_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(cl_mem), (void *) &Layer6_Mean_GPU);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer6_StanDev_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer6_Gamma_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer6_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {128, 1 * 32, 1 * 32};
    size_t localWorkSize_A[3] = {1, 32, 32};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    cl_kernel kernelB = clCreateKernel(program, "executeGenericFunctions_DSC", &err);

    inputWidth = 58, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 32;
    outputWidth = 56, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 32;

    // Setting arguments
    clSetKernelArg(kernelB, 0, sizeof(cl_mem), (void *) &Layer6_Neurons_GPU);
    clSetKernelArg(kernelB, 1, sizeof(cl_mem), (void *) &Layer6_Weights_GPU);
    clSetKernelArg(kernelB, 2, sizeof(cl_mem), (void *) &Layer7_Neurons_GPU);
    clSetKernelArg(kernelB, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernelB, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernelB, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernelB, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernelB, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernelB, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernelB, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernelB, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernelB, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernelB, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernelB, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernelB, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernelB, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernelB, 16, sizeof(cl_mem), (void *) &Layer6_Mean_GPU);
    clSetKernelArg(kernelB, 17, sizeof(cl_mem), (void *) &Layer6_StanDev_GPU);
    clSetKernelArg(kernelB, 18, sizeof(cl_mem), (void *) &Layer6_Gamma_GPU);
    clSetKernelArg(kernelB, 19, sizeof(cl_mem), (void *) &Layer6_Beta_GPU);


    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_B[3] = {128, 1 * 32, 1 * 24};
    size_t localWorkSize_B[3] = {1, 32, 24};

    err = clEnqueueNDRangeKernel(queue, kernelB, 3, NULL, globalWorkSize_B, localWorkSize_B,
                                                              0, NULL, NULL);

    clFinish(queue);

    cl_kernel kernelC = clCreateKernel(program, "executeGenericFunctions_DSC", &err);

    inputWidth = 58, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 32 * 58, blockOffset2 = 0;
    outputWidth = 56, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 56 * 32, blockOffset2Out = 0;

    // Setting arguments
    clSetKernelArg(kernelC, 0, sizeof(cl_mem), (void *) &Layer6_Neurons_GPU);
    clSetKernelArg(kernelC, 1, sizeof(cl_mem), (void *) &Layer6_Weights_GPU);
    clSetKernelArg(kernelC, 2, sizeof(cl_mem), (void *) &Layer7_Neurons_GPU);
    clSetKernelArg(kernelC, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernelC, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernelC, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernelC, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernelC, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernelC, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernelC, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernelC, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernelC, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernelC, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernelC, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernelC, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernelC, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernelC, 16, sizeof(cl_mem), (void *) &Layer6_Mean_GPU);
    clSetKernelArg(kernelC, 17, sizeof(cl_mem), (void *) &Layer6_StanDev_GPU);
    clSetKernelArg(kernelC, 18, sizeof(cl_mem), (void *) &Layer6_Gamma_GPU);
    clSetKernelArg(kernelC, 19, sizeof(cl_mem), (void *) &Layer6_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_C[3] = {128, 1 * 24, 1 * 32};
    size_t localWorkSize_C[3] = {1, 24, 32};

    err = clEnqueueNDRangeKernel(queue, kernelC, 3, NULL, globalWorkSize_C, localWorkSize_C,
                                                              0, NULL, NULL);
    
    cl_kernel kernelD = clCreateKernel(program, "executeGenericFunctions_DSC", &err);

    inputWidth = 58, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 32 * 58, blockOffset2 = 32;
    outputWidth = 56, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 56 * 32, blockOffset2Out = 32;

    // Setting arguments
    clSetKernelArg(kernelD, 0, sizeof(cl_mem), (void *) &Layer6_Neurons_GPU);
    clSetKernelArg(kernelD, 1, sizeof(cl_mem), (void *) &Layer6_Weights_GPU);
    clSetKernelArg(kernelD, 2, sizeof(cl_mem), (void *) &Layer7_Neurons_GPU);
    clSetKernelArg(kernelD, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernelD, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernelD, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernelD, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernelD, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernelD, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernelD, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernelD, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernelD, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernelD, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernelD, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernelD, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernelD, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernelD, 16, sizeof(cl_mem), (void *) &Layer6_Mean_GPU);
    clSetKernelArg(kernelD, 17, sizeof(cl_mem), (void *) &Layer6_StanDev_GPU);
    clSetKernelArg(kernelD, 18, sizeof(cl_mem), (void *) &Layer6_Gamma_GPU);
    clSetKernelArg(kernelD, 19, sizeof(cl_mem), (void *) &Layer6_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_D[3] = {128, 1 * 24, 1 * 24};
    size_t localWorkSize_D[3] = {1, 24, 24};

    err = clEnqueueNDRangeKernel(queue, kernelD, 3, NULL, globalWorkSize_D, localWorkSize_D,
                                                              0, NULL, NULL);
    
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    clReleaseMemObject(Layer6_Weights_GPU);
    clReleaseMemObject(Layer6_Mean_GPU);
    clReleaseMemObject(Layer6_StanDev_GPU);
    clReleaseMemObject(Layer6_Gamma_GPU);
    clReleaseMemObject(Layer6_Beta_GPU);

    free(Layer6_Weights_CPU);
    free(Layer6_Mean_CPU);
    free(Layer6_StanDev_CPU);
    free(Layer6_Gamma_CPU);
    free(Layer6_Beta_CPU);
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

void Execute_Fifth_Layer(cl_mem Layer5_Neurons_GPU,
    cl_mem Layer6_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer5_Weights_CPU = (double *) malloc(sizeof(double) * FIFTH_LAYER_WEIGHT_SIZE);
    double *Layer5_Mean_CPU = (double *) malloc(sizeof(double) * FIFTH_LAYER_CHANNELS);
    double *Layer5_StanDev_CPU = (double *) malloc(sizeof(double) * FIFTH_LAYER_CHANNELS);
    double *Layer5_Gamma_CPU = (double *) malloc(sizeof(double) * FIFTH_LAYER_CHANNELS);
    double *Layer5_Beta_CPU = (double *) malloc(sizeof(double) * FIFTH_LAYER_CHANNELS);

    Read_FifthLayer_Data(
        Layer5_Weights_CPU,        
        Layer5_Mean_CPU,
        Layer5_StanDev_CPU,
        Layer5_Gamma_CPU,
        Layer5_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer5_Weights_GPU,
           Layer5_Mean_GPU,
           Layer5_StanDev_GPU,
           Layer5_Gamma_GPU,
           Layer5_Beta_GPU;

    cl_int err;
    
    Layer5_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FIFTH_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer5_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FIFTH_LAYER_CHANNELS, NULL, NULL);
    Layer5_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FIFTH_LAYER_CHANNELS, NULL, NULL);
    Layer5_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FIFTH_LAYER_CHANNELS, NULL, NULL);
    Layer5_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FIFTH_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer5_Weights_GPU, CL_TRUE, 0, sizeof(double) * FIFTH_LAYER_WEIGHT_SIZE, Layer5_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer5_Mean_GPU, CL_TRUE, 0, sizeof(double) * FIFTH_LAYER_CHANNELS, Layer5_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer5_StanDev_GPU, CL_TRUE, 0, sizeof(double) * FIFTH_LAYER_CHANNELS, Layer5_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer5_Gamma_GPU, CL_TRUE, 0, sizeof(double) * FIFTH_LAYER_CHANNELS, Layer5_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer5_Beta_GPU, CL_TRUE, 0, sizeof(double) * FIFTH_LAYER_CHANNELS, Layer5_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_PSC", &err);
    
    int inputWidth = 56, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 58, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 64, channelSize = 64, stride = 1, offset = 59;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer5_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer5_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer6_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer5_Mean_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer5_StanDev_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer5_Gamma_GPU);
    clSetKernelArg(kernel, 20, sizeof(cl_mem), (void *) &Layer5_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {128, 1 * 32, 1 * 32};
    size_t localWorkSize_A[3] = {1, 32, 32};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    cl_kernel kernelB = clCreateKernel(program, "executeGenericFunctions_PSC", &err);

    inputWidth = 56, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 32, blockOffset2 = 0;
    outputWidth = 58, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 32, blockOffset2Out = 0;

    // Setting arguments
    clSetKernelArg(kernelB, 0, sizeof(cl_mem), (void *) &Layer5_Neurons_GPU);
    clSetKernelArg(kernelB, 1, sizeof(cl_mem), (void *) &Layer5_Weights_GPU);
    clSetKernelArg(kernelB, 2, sizeof(cl_mem), (void *) &Layer6_Neurons_GPU);
    clSetKernelArg(kernelB, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernelB, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernelB, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernelB, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernelB, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernelB, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernelB, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernelB, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernelB, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernelB, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernelB, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernelB, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernelB, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernelB, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernelB, 17, sizeof(cl_mem), (void *) &Layer5_Mean_GPU);
    clSetKernelArg(kernelB, 18, sizeof(cl_mem), (void *) &Layer5_StanDev_GPU);
    clSetKernelArg(kernelB, 19, sizeof(cl_mem), (void *) &Layer5_Gamma_GPU);
    clSetKernelArg(kernelB, 20, sizeof(cl_mem), (void *) &Layer5_Beta_GPU);


    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_B[3] = {128, 1 * 32, 1 * 24};
    size_t localWorkSize_B[3] = {1, 32, 24};

    err = clEnqueueNDRangeKernel(queue, kernelB, 3, NULL, globalWorkSize_B, localWorkSize_B,
                                                              0, NULL, NULL);

    clFinish(queue);

    cl_kernel kernelC = clCreateKernel(program, "executeGenericFunctions_PSC", &err);

    inputWidth = 56, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 56 * 32, blockOffset2 = 0;
    outputWidth = 58, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 58 * 32, blockOffset2Out = 0;

    // Setting arguments
    clSetKernelArg(kernelC, 0, sizeof(cl_mem), (void *) &Layer5_Neurons_GPU);
    clSetKernelArg(kernelC, 1, sizeof(cl_mem), (void *) &Layer5_Weights_GPU);
    clSetKernelArg(kernelC, 2, sizeof(cl_mem), (void *) &Layer6_Neurons_GPU);
    clSetKernelArg(kernelC, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernelC, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernelC, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernelC, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernelC, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernelC, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernelC, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernelC, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernelC, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernelC, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernelC, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernelC, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernelC, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernelC, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernelC, 17, sizeof(cl_mem), (void *) &Layer5_Mean_GPU);
    clSetKernelArg(kernelC, 18, sizeof(cl_mem), (void *) &Layer5_StanDev_GPU);
    clSetKernelArg(kernelC, 19, sizeof(cl_mem), (void *) &Layer5_Gamma_GPU);
    clSetKernelArg(kernelC, 20, sizeof(cl_mem), (void *) &Layer5_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_C[3] = {128, 1 * 24, 1 * 32};
    size_t localWorkSize_C[3] = {1, 24, 32};

    err = clEnqueueNDRangeKernel(queue, kernelC, 3, NULL, globalWorkSize_C, localWorkSize_C,
                                                              0, NULL, NULL);
    
    cl_kernel kernelD = clCreateKernel(program, "executeGenericFunctions_PSC", &err);

    inputWidth = 56, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 56 * 32, blockOffset2 = 32;
    outputWidth = 58, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 58 * 32, blockOffset2Out = 32;

    // Setting arguments
    clSetKernelArg(kernelD, 0, sizeof(cl_mem), (void *) &Layer5_Neurons_GPU);
    clSetKernelArg(kernelD, 1, sizeof(cl_mem), (void *) &Layer5_Weights_GPU);
    clSetKernelArg(kernelD, 2, sizeof(cl_mem), (void *) &Layer6_Neurons_GPU);
    clSetKernelArg(kernelD, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernelD, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernelD, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernelD, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernelD, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernelD, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernelD, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernelD, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernelD, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernelD, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernelD, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernelD, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernelD, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernelD, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernelD, 17, sizeof(cl_mem), (void *) &Layer5_Mean_GPU);
    clSetKernelArg(kernelD, 18, sizeof(cl_mem), (void *) &Layer5_StanDev_GPU);
    clSetKernelArg(kernelD, 19, sizeof(cl_mem), (void *) &Layer5_Gamma_GPU);
    clSetKernelArg(kernelD, 20, sizeof(cl_mem), (void *) &Layer5_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_D[3] = {128, 1 * 24, 1 * 24};
    size_t localWorkSize_D[3] = {1, 24, 24};

    err = clEnqueueNDRangeKernel(queue, kernelD, 3, NULL, globalWorkSize_D, localWorkSize_D,
                                                              0, NULL, NULL);

    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    clReleaseMemObject(Layer5_Weights_GPU);
    clReleaseMemObject(Layer5_Mean_GPU);
    clReleaseMemObject(Layer5_StanDev_GPU);
    clReleaseMemObject(Layer5_Gamma_GPU);
    clReleaseMemObject(Layer5_Beta_GPU);

    free(Layer5_Weights_CPU);
    free(Layer5_Mean_CPU);
    free(Layer5_StanDev_CPU);
    free(Layer5_Gamma_CPU);
    free(Layer5_Beta_CPU);
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

void Execute_Fourth_Layer(cl_mem Layer4_Neurons_GPU,
    cl_mem Layer5_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer4_Weights_CPU = (double *) malloc(sizeof(double) * FOURTH_LAYER_WEIGHT_SIZE);
    double *Layer4_Mean_CPU = (double *) malloc(sizeof(double) * FOURTH_LAYER_CHANNELS);
    double *Layer4_StanDev_CPU = (double *) malloc(sizeof(double) * FOURTH_LAYER_CHANNELS);
    double *Layer4_Gamma_CPU = (double *) malloc(sizeof(double) * FOURTH_LAYER_CHANNELS);
    double *Layer4_Beta_CPU = (double *) malloc(sizeof(double) * FOURTH_LAYER_CHANNELS);

    Read_FourthLayer_Data(
        Layer4_Weights_CPU,        
        Layer4_Mean_CPU,
        Layer4_StanDev_CPU,
        Layer4_Gamma_CPU,
        Layer4_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer4_Weights_GPU,
           Layer4_Mean_GPU,
           Layer4_StanDev_GPU,
           Layer4_Gamma_GPU,
           Layer4_Beta_GPU;

    cl_int err;
    
    Layer4_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FOURTH_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer4_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FOURTH_LAYER_CHANNELS, NULL, NULL);
    Layer4_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FOURTH_LAYER_CHANNELS, NULL, NULL);
    Layer4_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FOURTH_LAYER_CHANNELS, NULL, NULL);
    Layer4_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FOURTH_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer4_Weights_GPU, CL_TRUE, 0, sizeof(double) * FOURTH_LAYER_WEIGHT_SIZE, Layer4_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer4_Mean_GPU, CL_TRUE, 0, sizeof(double) * FOURTH_LAYER_CHANNELS, Layer4_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer4_StanDev_GPU, CL_TRUE, 0, sizeof(double) * FOURTH_LAYER_CHANNELS, Layer4_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer4_Gamma_GPU, CL_TRUE, 0, sizeof(double) * FOURTH_LAYER_CHANNELS, Layer4_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer4_Beta_GPU, CL_TRUE, 0, sizeof(double) * FOURTH_LAYER_CHANNELS, Layer4_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_DSC", &err);
    
    int inputWidth = 113, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 56, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 9, kernelSize = 3, stride = 2;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer4_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer4_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer5_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(cl_mem), (void *) &Layer4_Mean_GPU);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer4_StanDev_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer4_Gamma_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer4_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {64, 1 * 32, 1 * 32};
    size_t localWorkSize_A[3] = {1, 32, 32};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    cl_kernel kernelB = clCreateKernel(program, "executeGenericFunctions_DSC", &err);

    inputWidth = 113, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 32, blockOffset2 = 0;
    outputWidth = 56, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 32, blockOffset2Out = 0;

    // Setting arguments
    clSetKernelArg(kernelB, 0, sizeof(cl_mem), (void *) &Layer4_Neurons_GPU);
    clSetKernelArg(kernelB, 1, sizeof(cl_mem), (void *) &Layer4_Weights_GPU);
    clSetKernelArg(kernelB, 2, sizeof(cl_mem), (void *) &Layer5_Neurons_GPU);
    clSetKernelArg(kernelB, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernelB, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernelB, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernelB, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernelB, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernelB, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernelB, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernelB, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernelB, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernelB, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernelB, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernelB, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernelB, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernelB, 16, sizeof(cl_mem), (void *) &Layer4_Mean_GPU);
    clSetKernelArg(kernelB, 17, sizeof(cl_mem), (void *) &Layer4_StanDev_GPU);
    clSetKernelArg(kernelB, 18, sizeof(cl_mem), (void *) &Layer4_Gamma_GPU);
    clSetKernelArg(kernelB, 19, sizeof(cl_mem), (void *) &Layer4_Beta_GPU);


    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_B[3] = {64, 1 * 32, 1 * 24};
    size_t localWorkSize_B[3] = {1, 32, 24};

    err = clEnqueueNDRangeKernel(queue, kernelB, 3, NULL, globalWorkSize_B, localWorkSize_B,
                                                              0, NULL, NULL);

    clFinish(queue);

    cl_kernel kernelC = clCreateKernel(program, "executeGenericFunctions_DSC", &err);

    inputWidth = 113, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 32 * 113, blockOffset2 = 0;
    outputWidth = 56, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 56 * 32, blockOffset2Out = 0;

    // Setting arguments
    clSetKernelArg(kernelC, 0, sizeof(cl_mem), (void *) &Layer4_Neurons_GPU);
    clSetKernelArg(kernelC, 1, sizeof(cl_mem), (void *) &Layer4_Weights_GPU);
    clSetKernelArg(kernelC, 2, sizeof(cl_mem), (void *) &Layer5_Neurons_GPU);
    clSetKernelArg(kernelC, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernelC, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernelC, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernelC, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernelC, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernelC, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernelC, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernelC, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernelC, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernelC, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernelC, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernelC, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernelC, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernelC, 16, sizeof(cl_mem), (void *) &Layer4_Mean_GPU);
    clSetKernelArg(kernelC, 17, sizeof(cl_mem), (void *) &Layer4_StanDev_GPU);
    clSetKernelArg(kernelC, 18, sizeof(cl_mem), (void *) &Layer4_Gamma_GPU);
    clSetKernelArg(kernelC, 19, sizeof(cl_mem), (void *) &Layer4_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_C[3] = {64, 1 * 24, 1 * 32};
    size_t localWorkSize_C[3] = {1, 24, 32};

    err = clEnqueueNDRangeKernel(queue, kernelC, 3, NULL, globalWorkSize_C, localWorkSize_C,
                                                              0, NULL, NULL);
    
    cl_kernel kernelD = clCreateKernel(program, "executeGenericFunctions_DSC", &err);

    inputWidth = 113, blockMultiplier1 = 0, blockMultiplier2 = 0, blockOffset1 = 32 * 113, blockOffset2 = 32;
    outputWidth = 56, blockMultiplier1Out = 0, blockMultiplier2Out = 0, blockOffset1Out = 56 * 32, blockOffset2Out = 32;

    // Setting arguments
    clSetKernelArg(kernelD, 0, sizeof(cl_mem), (void *) &Layer4_Neurons_GPU);
    clSetKernelArg(kernelD, 1, sizeof(cl_mem), (void *) &Layer4_Weights_GPU);
    clSetKernelArg(kernelD, 2, sizeof(cl_mem), (void *) &Layer5_Neurons_GPU);
    clSetKernelArg(kernelD, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernelD, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernelD, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernelD, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernelD, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernelD, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernelD, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernelD, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernelD, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernelD, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernelD, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernelD, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernelD, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernelD, 16, sizeof(cl_mem), (void *) &Layer4_Mean_GPU);
    clSetKernelArg(kernelD, 17, sizeof(cl_mem), (void *) &Layer4_StanDev_GPU);
    clSetKernelArg(kernelD, 18, sizeof(cl_mem), (void *) &Layer4_Gamma_GPU);
    clSetKernelArg(kernelD, 19, sizeof(cl_mem), (void *) &Layer4_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_D[3] = {64, 1 * 24, 1 * 24};
    size_t localWorkSize_D[3] = {1, 24, 24};

    err = clEnqueueNDRangeKernel(queue, kernelD, 3, NULL, globalWorkSize_D, localWorkSize_D,
                                                              0, NULL, NULL);
    
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    clReleaseMemObject(Layer4_Weights_GPU);
    clReleaseMemObject(Layer4_Mean_GPU);
    clReleaseMemObject(Layer4_StanDev_GPU);
    clReleaseMemObject(Layer4_Gamma_GPU);
    clReleaseMemObject(Layer4_Beta_GPU);

    free(Layer4_Weights_CPU);
    free(Layer4_Mean_CPU);
    free(Layer4_StanDev_CPU);
    free(Layer4_Gamma_CPU);
    free(Layer4_Beta_CPU);
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

void Execute_Third_Layer(cl_mem Layer3_Neurons_GPU,
    cl_mem Layer4_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer3_Weights_CPU = (double *) malloc(sizeof(double) * THIRD_LAYER_WEIGHT_SIZE);
    double *Layer3_Mean_CPU = (double *) malloc(sizeof(double) * THIRD_LAYER_CHANNELS);
    double *Layer3_StanDev_CPU = (double *) malloc(sizeof(double) * THIRD_LAYER_CHANNELS);
    double *Layer3_Gamma_CPU = (double *) malloc(sizeof(double) * THIRD_LAYER_CHANNELS);
    double *Layer3_Beta_CPU = (double *) malloc(sizeof(double) * THIRD_LAYER_CHANNELS);

    Read_ThirdLayer_Data(
        Layer3_Weights_CPU,        
        Layer3_Mean_CPU,
        Layer3_StanDev_CPU,
        Layer3_Gamma_CPU,
        Layer3_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer3_Weights_GPU,
           Layer3_Mean_GPU,
           Layer3_StanDev_GPU,
           Layer3_Gamma_GPU,
           Layer3_Beta_GPU;

    cl_int err;
    
    Layer3_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * THIRD_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer3_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * THIRD_LAYER_CHANNELS, NULL, NULL);
    Layer3_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * THIRD_LAYER_CHANNELS, NULL, NULL);
    Layer3_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * THIRD_LAYER_CHANNELS, NULL, NULL);
    Layer3_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * THIRD_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer3_Weights_GPU, CL_TRUE, 0, sizeof(double) * THIRD_LAYER_WEIGHT_SIZE, Layer3_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer3_Mean_GPU, CL_TRUE, 0, sizeof(double) * THIRD_LAYER_CHANNELS, Layer3_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer3_StanDev_GPU, CL_TRUE, 0, sizeof(double) * THIRD_LAYER_CHANNELS, Layer3_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer3_Gamma_GPU, CL_TRUE, 0, sizeof(double) * THIRD_LAYER_CHANNELS, Layer3_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer3_Beta_GPU, CL_TRUE, 0, sizeof(double) * THIRD_LAYER_CHANNELS, Layer3_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_PSC", &err);
    
    int inputWidth = 112, blockMultiplier1 = 32, blockMultiplier2 = 32, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 113, blockMultiplier1Out = 32, blockMultiplier2Out = 32, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 32, channelSize = 32, stride = 1, offset = 0;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer3_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer3_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer4_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer3_Mean_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer3_StanDev_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer3_Gamma_GPU);
    clSetKernelArg(kernel, 20, sizeof(cl_mem), (void *) &Layer3_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {64, 3 * 32, 3 * 32};
    size_t localWorkSize_A[3] = {1, 32, 32};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    cl_kernel kernelB = clCreateKernel(program, "executeGenericFunctions_PSC", &err);

    inputWidth = 112, blockMultiplier1 = 16, blockMultiplier2 = 0, blockOffset1 = 96, blockOffset2 = 0;
    outputWidth = 113, blockMultiplier1Out = 16, blockMultiplier2Out = 0, blockOffset1Out = 96, blockOffset2Out = 0;

    // Setting arguments
    clSetKernelArg(kernelB, 0, sizeof(cl_mem), (void *) &Layer3_Neurons_GPU);
    clSetKernelArg(kernelB, 1, sizeof(cl_mem), (void *) &Layer3_Weights_GPU);
    clSetKernelArg(kernelB, 2, sizeof(cl_mem), (void *) &Layer4_Neurons_GPU);
    clSetKernelArg(kernelB, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernelB, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernelB, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernelB, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernelB, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernelB, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernelB, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernelB, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernelB, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernelB, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernelB, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernelB, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernelB, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernelB, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernelB, 17, sizeof(cl_mem), (void *) &Layer3_Mean_GPU);
    clSetKernelArg(kernelB, 18, sizeof(cl_mem), (void *) &Layer3_StanDev_GPU);
    clSetKernelArg(kernelB, 19, sizeof(cl_mem), (void *) &Layer3_Gamma_GPU);
    clSetKernelArg(kernelB, 20, sizeof(cl_mem), (void *) &Layer3_Beta_GPU);


    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_B[3] = {64, 7 * 16, 1 * 16};
    size_t localWorkSize_B[3] = {1, 16, 16};

    err = clEnqueueNDRangeKernel(queue, kernelB, 3, NULL, globalWorkSize_B, localWorkSize_B,
                                                              0, NULL, NULL);

    clFinish(queue);

    cl_kernel kernelC = clCreateKernel(program, "executeGenericFunctions_PSC", &err);

    inputWidth = 112, blockMultiplier1 = 0, blockMultiplier2 = 16, blockOffset1 = 96 * 112, blockOffset2 = 0;
    outputWidth = 113, blockMultiplier1Out = 0, blockMultiplier2Out = 16, blockOffset1Out = 96 * 113, blockOffset2Out = 0;

    // Setting arguments
    clSetKernelArg(kernelC, 0, sizeof(cl_mem), (void *) &Layer3_Neurons_GPU);
    clSetKernelArg(kernelC, 1, sizeof(cl_mem), (void *) &Layer3_Weights_GPU);
    clSetKernelArg(kernelC, 2, sizeof(cl_mem), (void *) &Layer4_Neurons_GPU);
    clSetKernelArg(kernelC, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernelC, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernelC, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernelC, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernelC, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernelC, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernelC, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernelC, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernelC, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernelC, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernelC, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernelC, 14, sizeof(int), (void *) &channelSize);
    clSetKernelArg(kernelC, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernelC, 16, sizeof(int), (void *) &offset);
    clSetKernelArg(kernelC, 17, sizeof(cl_mem), (void *) &Layer3_Mean_GPU);
    clSetKernelArg(kernelC, 18, sizeof(cl_mem), (void *) &Layer3_StanDev_GPU);
    clSetKernelArg(kernelC, 19, sizeof(cl_mem), (void *) &Layer3_Gamma_GPU);
    clSetKernelArg(kernelC, 20, sizeof(cl_mem), (void *) &Layer3_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_C[3] = {64, 1 * 16, 6 * 16};
    size_t localWorkSize_C[3] = {1, 16, 16};

    err = clEnqueueNDRangeKernel(queue, kernelC, 3, NULL, globalWorkSize_C, localWorkSize_C,
                                                              0, NULL, NULL);
    

    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    clReleaseMemObject(Layer3_Weights_GPU);
    clReleaseMemObject(Layer3_Mean_GPU);
    clReleaseMemObject(Layer3_StanDev_GPU);
    clReleaseMemObject(Layer3_Gamma_GPU);
    clReleaseMemObject(Layer3_Beta_GPU);

    free(Layer3_Weights_CPU);
    free(Layer3_Mean_CPU);
    free(Layer3_StanDev_CPU);
    free(Layer3_Gamma_CPU);
    free(Layer3_Beta_CPU);
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

void Execute_First_Layer(cl_mem Layer2_Neurons_GPU, cl_context context, cl_command_queue queue, cl_program program)
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
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer1_Weights_GPU,
           Layer1_Neurons_GPU,
           Layer1_Mean_GPU,
           Layer1_StanDev_GPU,
           Layer1_Gamma_GPU,
           Layer1_Beta_GPU;

    cl_int err;
    
    Layer1_Neurons_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * INPUT_LAYER_SIZE, NULL, NULL);
    Layer1_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FIRST_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer1_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FIRST_LAYER_CHANNELS, NULL, NULL);
    Layer1_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FIRST_LAYER_CHANNELS, NULL, NULL);
    Layer1_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FIRST_LAYER_CHANNELS, NULL, NULL);
    Layer1_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * FIRST_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer1_Neurons_GPU, CL_TRUE, 0, sizeof(double) * INPUT_LAYER_SIZE, Layer1_Neurons_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer1_Weights_GPU, CL_TRUE, 0, sizeof(double) * FIRST_LAYER_WEIGHT_SIZE, Layer1_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer1_Mean_GPU, CL_TRUE, 0, sizeof(double) * FIRST_LAYER_CHANNELS, Layer1_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer1_StanDev_GPU, CL_TRUE, 0, sizeof(double) * FIRST_LAYER_CHANNELS, Layer1_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer1_Gamma_GPU, CL_TRUE, 0, sizeof(double) * FIRST_LAYER_CHANNELS, Layer1_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer1_Beta_GPU, CL_TRUE, 0, sizeof(double) * FIRST_LAYER_CHANNELS, Layer1_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel_partA = clCreateKernel(program, "executeFirstLayer_CONV3D_partA", &err);

    // Setting arguments
    clSetKernelArg(kernel_partA, 0, sizeof(cl_mem), (void *) &Layer1_Neurons_GPU);
    clSetKernelArg(kernel_partA, 1, sizeof(cl_mem), (void *) &Layer1_Weights_GPU);
    clSetKernelArg(kernel_partA, 2, sizeof(cl_mem), (void *) &Layer2_Neurons_GPU);
    clSetKernelArg(kernel_partA, 3, sizeof(cl_mem), (void *) &Layer1_Mean_GPU);
    clSetKernelArg(kernel_partA, 4, sizeof(cl_mem), (void *) &Layer1_StanDev_GPU);
    clSetKernelArg(kernel_partA, 5, sizeof(cl_mem), (void *) &Layer1_Gamma_GPU);
    clSetKernelArg(kernel_partA, 6, sizeof(cl_mem), (void *) &Layer1_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {32, 3 * 32, 3 * 32};
    size_t localWorkSize_A[3] = {1, 32, 32};

    err = clEnqueueNDRangeKernel(queue, kernel_partA, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    // Reading the associated kernel - Part B
    cl_kernel kernel_partB = clCreateKernel(program, "executeFirstLayer_CONV3D_partB", &err);

    // Setting arguments
    clSetKernelArg(kernel_partB, 0, sizeof(cl_mem), (void *) &Layer1_Neurons_GPU);
    clSetKernelArg(kernel_partB, 1, sizeof(cl_mem), (void *) &Layer1_Weights_GPU);
    clSetKernelArg(kernel_partB, 2, sizeof(cl_mem), (void *) &Layer2_Neurons_GPU);
    clSetKernelArg(kernel_partB, 3, sizeof(cl_mem), (void *) &Layer1_Mean_GPU);
    clSetKernelArg(kernel_partB, 4, sizeof(cl_mem), (void *) &Layer1_StanDev_GPU);
    clSetKernelArg(kernel_partB, 5, sizeof(cl_mem), (void *) &Layer1_Gamma_GPU);
    clSetKernelArg(kernel_partB, 6, sizeof(cl_mem), (void *) &Layer1_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_B[3] = {32, 7 * 16, 1 * 16};
    size_t localWorkSize_B[3] = {1, 16, 16};

    err = clEnqueueNDRangeKernel(queue, kernel_partB, 3, NULL, globalWorkSize_B, localWorkSize_B,
                                                              0, NULL, NULL);

    // Reading the associated kernel - Part C
    cl_kernel kernel_partC = clCreateKernel(program, "executeFirstLayer_CONV3D_partC", &err);

    // Setting arguments
    clSetKernelArg(kernel_partC, 0, sizeof(cl_mem), (void *) &Layer1_Neurons_GPU);
    clSetKernelArg(kernel_partC, 1, sizeof(cl_mem), (void *) &Layer1_Weights_GPU);
    clSetKernelArg(kernel_partC, 2, sizeof(cl_mem), (void *) &Layer2_Neurons_GPU);
    clSetKernelArg(kernel_partC, 3, sizeof(cl_mem), (void *) &Layer1_Mean_GPU);
    clSetKernelArg(kernel_partC, 4, sizeof(cl_mem), (void *) &Layer1_StanDev_GPU);
    clSetKernelArg(kernel_partC, 5, sizeof(cl_mem), (void *) &Layer1_Gamma_GPU);
    clSetKernelArg(kernel_partC, 6, sizeof(cl_mem), (void *) &Layer1_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_C[3] = {32, 6 * 16, 1 * 16};
    size_t localWorkSize_C[3] = {1, 16, 16};

    err = clEnqueueNDRangeKernel(queue, kernel_partC, 3, NULL, globalWorkSize_C, localWorkSize_C,
                                                              0, NULL, NULL);


    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    clReleaseMemObject(Layer1_Neurons_GPU);
    clReleaseMemObject(Layer1_Weights_GPU);
    clReleaseMemObject(Layer1_Mean_GPU);
    clReleaseMemObject(Layer1_StanDev_GPU);
    clReleaseMemObject(Layer1_Gamma_GPU);
    clReleaseMemObject(Layer1_Beta_GPU);

    free(Layer1_Neurons_CPU);
    free(Layer1_Weights_CPU);
    free(Layer1_Mean_CPU);
    free(Layer1_StanDev_CPU);
    free(Layer1_Gamma_CPU);
    free(Layer1_Beta_CPU);
}

void Read_First_Layer_Data(
    double * Layer1_Neurons_CPU,
    double * Layer1_Weights_CPU,
    double * Layer1_Mean_CPU,
    double * Layer1_StanDev_CPU,
    double * Layer1_Gamma_CPU,
    double * Layer1_Beta_CPU
){
    read_Input_File("data/FirstLayer/InputFiles/inputsNorm.txt", Layer1_Neurons_CPU);
    read_File("data/FirstLayer/weightsNorm.txt", Layer1_Weights_CPU);
    read_File("data/FirstLayer/First_Layer_Mean.txt", Layer1_Mean_CPU);
    read_File("data/FirstLayer/First_Layer_StanDev.txt", Layer1_StanDev_CPU);
    read_File("data/FirstLayer/First_Layer_Gamma.txt", Layer1_Gamma_CPU);
    read_File("data/FirstLayer/First_Layer_Beta.txt", Layer1_Beta_CPU);
}

void Execute_Second_Layer(cl_mem Layer2_Neurons_GPU,
    cl_mem Layer3_Neurons_GPU, 
    cl_context context, 
    cl_command_queue queue, 
    cl_program program
){
    double *Layer2_Weights_CPU = (double *) malloc(sizeof(double) * SECOND_LAYER_WEIGHT_SIZE);
    double *Layer2_Mean_CPU = (double *) malloc(sizeof(double) * SECOND_LAYER_CHANNELS);
    double *Layer2_StanDev_CPU = (double *) malloc(sizeof(double) * SECOND_LAYER_CHANNELS);
    double *Layer2_Gamma_CPU = (double *) malloc(sizeof(double) * SECOND_LAYER_CHANNELS);
    double *Layer2_Beta_CPU = (double *) malloc(sizeof(double) * SECOND_LAYER_CHANNELS);

    Read_SecondLayer_Data(
        Layer2_Weights_CPU,        
        Layer2_Mean_CPU,
        Layer2_StanDev_CPU,
        Layer2_Gamma_CPU,
        Layer2_Beta_CPU
    );
    
    // allocating memory for thje matrices on the GPU
    cl_mem Layer2_Weights_GPU,
           Layer2_Mean_GPU,
           Layer2_StanDev_GPU,
           Layer2_Gamma_GPU,
           Layer2_Beta_GPU;

    cl_int err;
    
    Layer2_Weights_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SECOND_LAYER_WEIGHT_SIZE, NULL, NULL);
    Layer2_Mean_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SECOND_LAYER_CHANNELS, NULL, NULL);
    Layer2_StanDev_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SECOND_LAYER_CHANNELS, NULL, NULL);
    Layer2_Gamma_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SECOND_LAYER_CHANNELS, NULL, NULL);
    Layer2_Beta_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * SECOND_LAYER_CHANNELS, NULL, NULL);

    clEnqueueWriteBuffer(queue, Layer2_Weights_GPU, CL_TRUE, 0, sizeof(double) * SECOND_LAYER_WEIGHT_SIZE, Layer2_Weights_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer2_Mean_GPU, CL_TRUE, 0, sizeof(double) * SECOND_LAYER_CHANNELS, Layer2_Mean_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer2_StanDev_GPU, CL_TRUE, 0, sizeof(double) * SECOND_LAYER_CHANNELS, Layer2_StanDev_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer2_Gamma_GPU, CL_TRUE, 0, sizeof(double) * SECOND_LAYER_CHANNELS, Layer2_Gamma_CPU, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, Layer2_Beta_GPU, CL_TRUE, 0, sizeof(double) * SECOND_LAYER_CHANNELS, Layer2_Beta_CPU, 0, NULL, NULL);
    
    // Reading the associated kernel
    cl_kernel kernel = clCreateKernel(program, "executeGenericFunctions_DSC", &err);
    
    int inputWidth = 114, blockMultiplier1 = 32, blockMultiplier2 = 32, blockOffset1 = 0, blockOffset2 = 0;
    int outputWidth = 112, blockMultiplier1Out = 32, blockMultiplier2Out = 32, blockOffset1Out = 0, blockOffset2Out = 0;
    int weight_size = 9, kernelSize = 3, stride = 1;

    // Setting arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &Layer2_Neurons_GPU);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &Layer2_Weights_GPU);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &Layer3_Neurons_GPU);
    clSetKernelArg(kernel, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernel, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernel, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernel, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernel, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernel, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernel, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernel, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernel, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernel, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernel, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernel, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernel, 16, sizeof(cl_mem), (void *) &Layer2_Mean_GPU);
    clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *) &Layer2_StanDev_GPU);
    clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *) &Layer2_Gamma_GPU);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *) &Layer2_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_A[3] = {32, 3 * 32, 3 * 32};
    size_t localWorkSize_A[3] = {1, 32, 32};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize_A, localWorkSize_A,
                                                              0, NULL, NULL);

    clFinish(queue);

    cl_kernel kernelB = clCreateKernel(program, "executeGenericFunctions_DSC", &err);

    inputWidth = 114, blockMultiplier1 = 16, blockMultiplier2 = 0, blockOffset1 = 0, blockOffset2 = 96;
    outputWidth = 112, blockMultiplier1Out = 16, blockMultiplier2Out = 0, blockOffset1Out = 0, blockOffset2Out = 96;

    // Setting arguments
    clSetKernelArg(kernelB, 0, sizeof(cl_mem), (void *) &Layer2_Neurons_GPU);
    clSetKernelArg(kernelB, 1, sizeof(cl_mem), (void *) &Layer2_Weights_GPU);
    clSetKernelArg(kernelB, 2, sizeof(cl_mem), (void *) &Layer3_Neurons_GPU);
    clSetKernelArg(kernelB, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernelB, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernelB, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernelB, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernelB, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernelB, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernelB, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernelB, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernelB, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernelB, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernelB, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernelB, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernelB, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernelB, 16, sizeof(cl_mem), (void *) &Layer2_Mean_GPU);
    clSetKernelArg(kernelB, 17, sizeof(cl_mem), (void *) &Layer2_StanDev_GPU);
    clSetKernelArg(kernelB, 18, sizeof(cl_mem), (void *) &Layer2_Gamma_GPU);
    clSetKernelArg(kernelB, 19, sizeof(cl_mem), (void *) &Layer2_Beta_GPU);


    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_B[3] = {32, 7 * 16, 1 * 16};
    size_t localWorkSize_B[3] = {1, 16, 16};

    err = clEnqueueNDRangeKernel(queue, kernelB, 3, NULL, globalWorkSize_B, localWorkSize_B,
                                                              0, NULL, NULL);

    clFinish(queue);

    cl_kernel kernelC = clCreateKernel(program, "executeGenericFunctions_DSC", &err);

    inputWidth = 114, blockMultiplier1 = 0, blockMultiplier2 = 16, blockOffset1 = 96 * 114, blockOffset2 = 0;
    outputWidth = 112, blockMultiplier1Out = 0, blockMultiplier2Out = 16, blockOffset1Out = 96 * 112, blockOffset2Out = 0;

    // Setting arguments
    clSetKernelArg(kernelC, 0, sizeof(cl_mem), (void *) &Layer2_Neurons_GPU);
    clSetKernelArg(kernelC, 1, sizeof(cl_mem), (void *) &Layer2_Weights_GPU);
    clSetKernelArg(kernelC, 2, sizeof(cl_mem), (void *) &Layer3_Neurons_GPU);
    clSetKernelArg(kernelC, 3, sizeof(int), (void *) &inputWidth);
    clSetKernelArg(kernelC, 4, sizeof(int), (void *) &blockMultiplier1);
    clSetKernelArg(kernelC, 5, sizeof(int), (void *) &blockMultiplier2);
    clSetKernelArg(kernelC, 6, sizeof(int), (void *) &blockOffset1);
    clSetKernelArg(kernelC, 7, sizeof(int), (void *) &blockOffset2);
    clSetKernelArg(kernelC, 8, sizeof(int), (void *) &outputWidth);
    clSetKernelArg(kernelC, 9, sizeof(int), (void *) &blockMultiplier1Out);
    clSetKernelArg(kernelC, 10, sizeof(int), (void *) &blockMultiplier2Out);
    clSetKernelArg(kernelC, 11, sizeof(int), (void *) &blockOffset1Out);
    clSetKernelArg(kernelC, 12, sizeof(int), (void *) &blockOffset2Out);
    clSetKernelArg(kernelC, 13, sizeof(int), (void *) &weight_size);
    clSetKernelArg(kernelC, 14, sizeof(int), (void *) &kernelSize);
    clSetKernelArg(kernelC, 15, sizeof(int), (void *) &stride);
    clSetKernelArg(kernelC, 16, sizeof(cl_mem), (void *) &Layer2_Mean_GPU);
    clSetKernelArg(kernelC, 17, sizeof(cl_mem), (void *) &Layer2_StanDev_GPU);
    clSetKernelArg(kernelC, 18, sizeof(cl_mem), (void *) &Layer2_Gamma_GPU);
    clSetKernelArg(kernelC, 19, sizeof(cl_mem), (void *) &Layer2_Beta_GPU);

    // Execute the kernel over the entire range of the data set 
    size_t globalWorkSize_C[3] = {32, 1 * 16, 6 * 16};
    size_t localWorkSize_C[3] = {1, 16, 16};

    err = clEnqueueNDRangeKernel(queue, kernelC, 3, NULL, globalWorkSize_C, localWorkSize_C,
                                                              0, NULL, NULL);
    

    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    clReleaseMemObject(Layer2_Weights_GPU);
    clReleaseMemObject(Layer2_Mean_GPU);
    clReleaseMemObject(Layer2_StanDev_GPU);
    clReleaseMemObject(Layer2_Gamma_GPU);
    clReleaseMemObject(Layer2_Beta_GPU);

    free(Layer2_Weights_CPU);
    free(Layer2_Mean_CPU);
    free(Layer2_StanDev_CPU);
    free(Layer2_Gamma_CPU);
    free(Layer2_Beta_CPU);
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
    read = 0;
    fclose(fp);
}

char * read_Kernel_File(){
    size_t size = 0;

    char * buffer;
    /* Open your_file in read-only mode */
    FILE *fp = fopen("MobileNets_kernel.cl", "rb");

    /* Get the buffer size */
    fseek(fp, 0, SEEK_END); /* Go to end of file */
    size = ftell(fp); /* How many bytes did we pass ? */

    /* Set position of stream to the beginning */
    rewind(fp);

    /* Allocate the buffer (no need to initialize it with calloc) */
    buffer = malloc((size + 1) * sizeof(*buffer)); /* size + 1 byte for the \0 */

    /* Read the file into the buffer */
    fread(buffer, size, 1, fp); /* Read 1 chunk of size bytes from fp into buffer */

    /* NULL-terminate the buffer */
    buffer[size] = '\0';

    return buffer;
}