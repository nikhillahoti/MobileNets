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
    /* ************************************************ THIRD LAYER COMPLETE *********************************************** */\

    /* ************************************************ FOURTH LAYER START ******************************************************** */
    cl_mem Layer5_Neurons_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, FOURTH_LAYER_OUTPUT_SIZE * sizeof(double), NULL, NULL);

    Execute_Fourth_Layer(Layer4_Neurons_GPU, Layer5_Neurons_GPU, context, queue, program);

    // Read the results from the device
    bool SAVE_FOURTH_LAYER_WEIGHTS = true;
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

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    clReleaseMemObject(Layer5_Neurons_GPU);
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