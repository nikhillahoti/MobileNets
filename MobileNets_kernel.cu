#include <stdio.h>

__global__ void executeFirstLayer_partA(double *Layer1_Neurons_GPU,
                            double *Layer1_Weights_GPU,
                            double *Layer2_Neurons_GPU,
                            double *Layer1_Mean_GPU,
                            double *Layer1_StanDev_GPU,
                            double *Layer1_Gamma_GPU,
                            double *Layer1_Beta_GPU
                        )
{
	double product = 0.0;
    int outputOffset = 115;
    int stride = 2;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 114 * 114)   // channel to work with
                        + (blockIdx.y * 32 * 114)    // Position in the grid row-wise
                        + (blockIdx.z * 32)          // Position in the grid column-wise
                        + (threadIdx.x * 114)
                        + (threadIdx.y);

    int weight_Position = filter_number * 27;

    int input_Position = ((blockIdx.y * 32 * 225) * stride) // Position in the grid row-wise
                       + (blockIdx.z * 32 * stride)         // Position in the grid column-wise
                       + (threadIdx.x * 225 * stride )
                       + (threadIdx.y * stride);

    /* RGB weights and input 3*3*3 */
    for(int channel = 0; channel < 3; channel++) // This is the Channel loop
    {
        for(int row = 0; row < 3; row++)       // This is the Row Loop
        {
            product += ((Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225)] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3)])
                    + (Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225) + 1] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3) + 1])
                    + (Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225) + 2] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3) + 2]));
        }
    }

    double Z = (product - Layer1_Mean_GPU[filter_number]) / Layer1_StanDev_GPU[filter_number];
    Z = (Z * Layer1_Gamma_GPU[filter_number]) + Layer1_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0; 

    Layer2_Neurons_GPU[output_Position + outputOffset] = Z;
}


__global__ void executeFirstLayer_partB(double *Layer1_Neurons_GPU,
                            double *Layer1_Weights_GPU,
                            double *Layer2_Neurons_GPU,
                            double *Layer1_Mean_GPU,
                            double *Layer1_StanDev_GPU,
                            double *Layer1_Gamma_GPU,
                            double *Layer1_Beta_GPU
                        )
{
	double product = 0.0;
    int outputOffset = 115;
    int stride = 2;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 114 * 114)   // channel to work with
                        + (blockIdx.y * 16 * 114 + 96)  // Position in the grid row-wise and there is no column-wise position
                        + (threadIdx.x * 114)           // Position inside the 256 (16 * 16) block
                        + (threadIdx.y);

    int weight_Position = filter_number * 27;

    int input_Position = ((blockIdx.y * 16 * 225) * stride) + (96 * stride) // Position in the grid row-wise and column-wise
                       + (threadIdx.x * 225 * stride)
                       + (threadIdx.y * stride);

    /* RGB weights and input 3*3*3 */
    for(int channel = 0; channel < 3; channel++) // This is the Channel loop
    {
        for(int row = 0; row < 3; row++)       // This is the Row Loop
        {
            product += ((Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225)] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3)])
                    + (Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225) + 1] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3) + 1])
                    + (Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225) + 2] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3) + 2]));
        }
    }

    double Z = (product - Layer1_Mean_GPU[filter_number]) / Layer1_StanDev_GPU[filter_number];
    Z = (Z * Layer1_Gamma_GPU[filter_number]) + Layer1_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0; 

    Layer2_Neurons_GPU[output_Position + outputOffset] = Z;
}

__global__ void executeFirstLayer_partC(double *Layer1_Neurons_GPU,
                            double *Layer1_Weights_GPU,
                            double *Layer2_Neurons_GPU,
                            double *Layer1_Mean_GPU,
                            double *Layer1_StanDev_GPU,
                            double *Layer1_Gamma_GPU,
                            double *Layer1_Beta_GPU
                        )
{
	double product = 0.0;
    int outputOffset = 115;
    int stride = 2;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 114 * 114)   // channel to work with
                        + (96 * 114)                    // Position in the grid row-wise as row is last
                        + (blockIdx.y * 16)             // Position in the grid column-wise
                        + (threadIdx.x * 114)           // Position inside the 256 (16 * 16) block
                        + (threadIdx.y);

    int weight_Position = filter_number * 27;

    int input_Position = ((96 * 225) * stride)
                       + (blockIdx.y * 16 * stride)     // Position in the grid row-wise and column-wise
                       + (threadIdx.x * 225 * stride)
                       + (threadIdx.y * stride);

    /* RGB weights and input 3*3*3 */
    for(int channel = 0; channel < 3; channel++) // This is the Channel loop
    {
        for(int row = 0; row < 3; row++)       // This is the Row Loop
        {
            product += ((Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225)] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3)])
                    + (Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225) + 1] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3) + 1])
                    + (Layer1_Neurons_GPU[(channel * 225 * 225) + input_Position + (row * 225) + 2] * Layer1_Weights_GPU[weight_Position + (channel * 3 * 3) + (row * 3) + 2]));
        }
    }

    double Z = (product - Layer1_Mean_GPU[filter_number]) / Layer1_StanDev_GPU[filter_number];
    Z = (Z * Layer1_Gamma_GPU[filter_number]) + Layer1_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0; 

    Layer2_Neurons_GPU[output_Position + outputOffset] = Z;
}


// Second Layer
__global__ void executeSecondLayer_partA(double *Layer2_Neurons_GPU,
                            double *Layer2_Weights_GPU,
                            double *Layer3_Neurons_GPU
                        )
{
	double product = 0.0;

    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 112 * 112)   // channel to work with
                        + (blockIdx.y * 32 * 112)    // Position in the grid row-wise
                        + (blockIdx.z * 32)          // Position in the grid column-wise
                        + (threadIdx.x * 112)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (blockIdx.y * 32 * 114) // Position in the grid row-wise
                       + (blockIdx.z * 32)         // Position in the grid column-wise
                       + (threadIdx.x * 114)
                       + (threadIdx.y);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer2_Neurons_GPU[(filter_number * 114 * 114) + input_Position + (row * 114)] * Layer2_Weights_GPU[weight_Position + (row * 3)])
                + (Layer2_Neurons_GPU[(filter_number * 114 * 114) + input_Position + (row * 114) + 1] * Layer2_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer2_Neurons_GPU[(filter_number * 114 * 114) + input_Position + (row * 114) + 2] * Layer2_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    Layer3_Neurons_GPU[output_Position] = product;
}

__global__ void executeSecondLayer_partB(double *Layer2_Neurons_GPU,
                                    double *Layer2_Weights_GPU,
                                    double *Layer3_Neurons_GPU
                                )
{
    double product = 0.0;

    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 112 * 112)   // channel to work with
                        + (blockIdx.y * 16 * 112 + 96)  // Position in the grid row-wise and there is no column-wise position
                        + (threadIdx.x * 112)           // Position inside the 256 (16 * 16) block
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position  = (blockIdx.y * 16 * 114) 
                        + (96) // Position in the grid row-wise and column-wise
                        + (threadIdx.x * 114)
                        + (threadIdx.y);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer2_Neurons_GPU[(filter_number * 114 * 114) + input_Position + (row * 114)] * Layer2_Weights_GPU[weight_Position + (row * 3)])
                + (Layer2_Neurons_GPU[(filter_number * 114 * 114) + input_Position + (row * 114) + 1] * Layer2_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer2_Neurons_GPU[(filter_number * 114 * 114) + input_Position + (row * 114) + 2] * Layer2_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    Layer3_Neurons_GPU[output_Position] = product;
}

__global__ void executeSecondLayer_partC(double *Layer2_Neurons_GPU,
                                    double *Layer2_Weights_GPU,
                                    double *Layer3_Neurons_GPU
                                )
{
    double product = 0.0;

    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 112 * 112)   // channel to work with
                        + (96 * 112)                    // Position in the grid row-wise as row is last
                        + (blockIdx.y * 16)             // Position in the grid column-wise
                        + (threadIdx.x * 112)           // Position inside the 256 (16 * 16) block
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (96 * 114)
                        + (blockIdx.y * 16)     // Position in the grid row-wise and column-wise
                        + (threadIdx.x * 114)
                        + (threadIdx.y);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer2_Neurons_GPU[(filter_number * 114 * 114) + input_Position + (row * 114)] * Layer2_Weights_GPU[weight_Position + (row * 3)])
                + (Layer2_Neurons_GPU[(filter_number * 114 * 114) + input_Position + (row * 114) + 1] * Layer2_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer2_Neurons_GPU[(filter_number * 114 * 114) + input_Position + (row * 114) + 2] * Layer2_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    Layer3_Neurons_GPU[output_Position] = product;
}

