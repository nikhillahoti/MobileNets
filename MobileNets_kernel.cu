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

    if(Z > 6.0)
      Z = 6.0;

    Layer2_Neurons_GPU[output_Position] = Z;
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

    if(Z > 6.0)
      Z = 6.0;

    Layer2_Neurons_GPU[output_Position] = Z;
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

    if(Z > 6.0)
      Z = 6.0;

    Layer2_Neurons_GPU[output_Position] = Z;
}
