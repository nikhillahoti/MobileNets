#include <stdio.h>

/*  ************************************************** FIRST LAYER START ********************************************************* */
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
/*  ************************************************** FIRST LAYER END ************************************************************ */

/*  ************************************************** SECOND LAYER START ********************************************************* */
__global__ void executeSecondLayer_partA(double *Layer2_Neurons_GPU,
                            double *Layer2_Weights_GPU,
                            double *Layer3_Neurons_GPU,
                            double *Layer2_Mean_GPU,
                            double *Layer2_StanDev_GPU,
                            double *Layer2_Gamma_GPU,
                            double *Layer2_Beta_GPU
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

    double Z = (product - Layer2_Mean_GPU[filter_number]) / Layer2_StanDev_GPU[filter_number];
    Z = (Z * Layer2_Gamma_GPU[filter_number]) + Layer2_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0; 

    Layer3_Neurons_GPU[output_Position] = Z;
}

__global__ void executeSecondLayer_partB(double *Layer2_Neurons_GPU,
                                    double *Layer2_Weights_GPU,
                                    double *Layer3_Neurons_GPU,
                                    double *Layer2_Mean_GPU,
                                    double *Layer2_StanDev_GPU,
                                    double *Layer2_Gamma_GPU,
                                    double *Layer2_Beta_GPU
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

    double Z = (product - Layer2_Mean_GPU[filter_number]) / Layer2_StanDev_GPU[filter_number];
    Z = (Z * Layer2_Gamma_GPU[filter_number]) + Layer2_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0; 
    
    Layer3_Neurons_GPU[output_Position] = Z;
}

__global__ void executeSecondLayer_partC(double *Layer2_Neurons_GPU,
                                    double *Layer2_Weights_GPU,
                                    double *Layer3_Neurons_GPU,
                                    double *Layer2_Mean_GPU,
                                    double *Layer2_StanDev_GPU,
                                    double *Layer2_Gamma_GPU,
                                    double *Layer2_Beta_GPU
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

    double Z = (product - Layer2_Mean_GPU[filter_number]) / Layer2_StanDev_GPU[filter_number];
    Z = (Z * Layer2_Gamma_GPU[filter_number]) + Layer2_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0; 

    Layer3_Neurons_GPU[output_Position] = Z;
}
/*  ************************************************** SECOND LAYER END ********************************************************* */

/*  ************************************************** THIRD LAYER START ******************************************************** */
__global__ void executeThirdLayer_partA(double *Layer3_Neurons_GPU,
    double *Layer3_Weights_GPU,
    double *Layer4_Neurons_GPU,
    double *Layer3_Mean_GPU,
    double *Layer3_StanDev_GPU,
    double *Layer3_Gamma_GPU,
    double *Layer3_Beta_GPU
)
{
    double product = 0.0;

    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 113 * 113)   // channel to work with
                        + (blockIdx.y * 32 * 113)    // Position in the grid row-wise
                        + (blockIdx.z * 32)          // Position in the grid column-wise
                        + (threadIdx.x * 113)
                        + (threadIdx.y);

    int weight_Position = filter_number * 32;

    int input_Position = (blockIdx.y * 32 * 112) // Position in the grid row-wise
                       + (blockIdx.z * 32)         // Position in the grid column-wise
                       + (threadIdx.x * 112)
                       + (threadIdx.y);

    for(int channel = 0; channel < 32; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer3_Neurons_GPU[(channel * 112 * 112) + input_Position] * Layer3_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer3_Mean_GPU[filter_number]) / Layer3_StanDev_GPU[filter_number];
    Z = (Z * Layer3_Gamma_GPU[filter_number]) + Layer3_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0; 

    Layer4_Neurons_GPU[output_Position] = Z;
}

__global__ void executeThirdLayer_partB(double *Layer3_Neurons_GPU,
    double *Layer3_Weights_GPU,
    double *Layer4_Neurons_GPU,
    double *Layer3_Mean_GPU,
    double *Layer3_StanDev_GPU,
    double *Layer3_Gamma_GPU,
    double *Layer3_Beta_GPU
)
{
    double product = 0.0;

    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 113 * 113)   // channel to work with
                        + (blockIdx.y * 16 * 113 + 96)  // Position in the grid row-wise and there is no column-wise position
                        + (threadIdx.x * 113)           // Position inside the 256 (16 * 16) block
                        + (threadIdx.y);

    int weight_Position = filter_number * 32;

    int input_Position = (blockIdx.y * 16 * 112)         // Position in the grid row-wise
                       + (96)                   // Position in the grid column-wise
                       + (threadIdx.x * 112)
                       + (threadIdx.y);

    for(int channel = 0 ; channel < 32 ; channel++) // Channel loop as we have 32 input channels to work with
    {
        product += (Layer3_Neurons_GPU[(channel * 112 * 112) + input_Position] * Layer3_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer3_Mean_GPU[filter_number]) / Layer3_StanDev_GPU[filter_number];
    Z = (Z * Layer3_Gamma_GPU[filter_number]) + Layer3_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0; 

    Layer4_Neurons_GPU[output_Position] = Z;
}

__global__ void executeThirdLayer_partC(double *Layer3_Neurons_GPU,
    double *Layer3_Weights_GPU,
    double *Layer4_Neurons_GPU,
    double *Layer3_Mean_GPU,
    double *Layer3_StanDev_GPU,
    double *Layer3_Gamma_GPU,
    double *Layer3_Beta_GPU
)
{
    double product = 0.0;

    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 113 * 113)   // channel to work with
                        + (96 * 113)                    // Position in the grid row-wise as row is last
                        + (blockIdx.y * 16)             // Position in the grid column-wise
                        + (threadIdx.x * 113)           // Position inside the 256 (16 * 16) block
                        + (threadIdx.y);

    int weight_Position = filter_number * 32;

    int input_Position = (96 * 112)            // row-wise: the bottom part of the grid after 96th row
                       + (blockIdx.y * 16)     // column-wise: block number in the 6 blocks of 16 * 16 threads
                       + (threadIdx.x * 112)   // Position inside one the above block row-wise
                       + (threadIdx.y);        // Position inside one the above block column-wise
    
    for(int channel = 0 ; channel < 32 ; channel++) // Channel loop as we have 32 input channels to work with
    {
        product += (Layer3_Neurons_GPU[(channel * 112 * 112) + input_Position] * Layer3_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer3_Mean_GPU[filter_number]) / Layer3_StanDev_GPU[filter_number];
    Z = (Z * Layer3_Gamma_GPU[filter_number]) + Layer3_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0; 

    Layer4_Neurons_GPU[output_Position] = Z;
}
/*  ************************************************** THIRD LAYER END ********************************************************* */

/*  ************************************************** FOURTH LAYER START ****************************************************** */
__global__ void executeFourthLayer_partA(double *Layer4_Neurons_GPU,
    double *Layer4_Weights_GPU,
    double *Layer5_Neurons_GPU,
    double *Layer4_Mean_GPU,
    double *Layer4_StanDev_GPU,
    double *Layer4_Gamma_GPU,
    double *Layer4_Beta_GPU
)
{
    double product = 0.0;
    int stride = 2;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 56 * 56)   // channel to work with
                        + (threadIdx.x * 56)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (threadIdx.x * 113 * stride )
                       + (threadIdx.y * stride);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113)] * Layer4_Weights_GPU[weight_Position + (row * 3)])
                + (Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113) + 1] * Layer4_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113) + 2] * Layer4_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer4_Mean_GPU[filter_number]) / Layer4_StanDev_GPU[filter_number];
    Z = (Z * Layer4_Gamma_GPU[filter_number]) + Layer4_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0; 

    Layer5_Neurons_GPU[output_Position] = Z;
}

__global__ void executeFourthLayer_partB(double *Layer4_Neurons_GPU,
    double *Layer4_Weights_GPU,
    double *Layer5_Neurons_GPU,
    double *Layer4_Mean_GPU,
    double *Layer4_StanDev_GPU,
    double *Layer4_Gamma_GPU,
    double *Layer4_Beta_GPU
)
{
    double product = 0.0;
    int stride = 2;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 56 * 56)   // channel to work with
                        + (threadIdx.x * 56)
                        + (threadIdx.y + 32);

    int weight_Position = filter_number * 9;

    int input_Position = (threadIdx.x * 113 * stride)
                       + (threadIdx.y * stride) 
                       + (32 * stride);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113)] * Layer4_Weights_GPU[weight_Position + (row * 3)])
                + (Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113) + 1] * Layer4_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113) + 2] * Layer4_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer4_Mean_GPU[filter_number]) / Layer4_StanDev_GPU[filter_number];
    Z = (Z * Layer4_Gamma_GPU[filter_number]) + Layer4_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0; 

    Layer5_Neurons_GPU[output_Position] = Z;
}

__global__ void executeFourthLayer_partC(double *Layer4_Neurons_GPU,
    double *Layer4_Weights_GPU,
    double *Layer5_Neurons_GPU,
    double *Layer4_Mean_GPU,
    double *Layer4_StanDev_GPU,
    double *Layer4_Gamma_GPU,
    double *Layer4_Beta_GPU
)
{
    double product = 0.0;
    int stride = 2;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 56 * 56)   // channel to work with
                        + (56 * 32)
                        + (threadIdx.x * 56)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (113 * 32 * stride)
                       + (threadIdx.x * 113 * stride)
                       + (threadIdx.y * stride);

    for(int row = 0; row < 3; row++)       // This is the Row Loop
    {
        product += ((Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113)] * Layer4_Weights_GPU[weight_Position + (row * 3)])
                + (Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113) + 1] * Layer4_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113) + 2] * Layer4_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer4_Mean_GPU[filter_number]) / Layer4_StanDev_GPU[filter_number];
    Z = (Z * Layer4_Gamma_GPU[filter_number]) + Layer4_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0; 

    Layer5_Neurons_GPU[output_Position] = Z;
}

__global__ void executeFourthLayer_partD(double *Layer4_Neurons_GPU,
    double *Layer4_Weights_GPU,
    double *Layer5_Neurons_GPU,
    double *Layer4_Mean_GPU,
    double *Layer4_StanDev_GPU,
    double *Layer4_Gamma_GPU,
    double *Layer4_Beta_GPU
)
{
    double product = 0.0;
    int stride = 2;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 56 * 56)   // channel to work with
                        + (56 * 32) 
                        + 32
                        + (threadIdx.x * 56)
                        + (threadIdx.y);

    int weight_Position = filter_number * 9;

    int input_Position = (113 * 32 * stride)
                       + (32 * stride)
                       + (threadIdx.x * 113 * stride)
                       + (threadIdx.y * stride);

    for(int row = 0; row < 3; row++) // This is the Row Loop
    {
        product += ((Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113)] * Layer4_Weights_GPU[weight_Position + (row * 3)])
                + (Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113) + 1] * Layer4_Weights_GPU[weight_Position + (row * 3) + 1])
                + (Layer4_Neurons_GPU[(filter_number * 113 * 113) + input_Position + (row * 113) + 2] * Layer4_Weights_GPU[weight_Position + (row * 3) + 2]));
    }

    double Z = (product - Layer4_Mean_GPU[filter_number]) / Layer4_StanDev_GPU[filter_number];
    Z = (Z * Layer4_Gamma_GPU[filter_number]) + Layer4_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0; 

    Layer5_Neurons_GPU[output_Position] = Z;
}


/*  ************************************************** FOURTH LAYER END ****************************************************** */

/*  *************************************************** FIFTH LAYER START **************************************************** */

__global__ void executeFifthLayer_partA(double *Layer5_Neurons_GPU,
    double *Layer5_Weights_GPU,
    double *Layer6_Neurons_GPU,
    double *Layer5_Mean_GPU,
    double *Layer5_StanDev_GPU,
    double *Layer5_Gamma_GPU,
    double *Layer5_Beta_GPU
)
{
    double product = 0.0;

    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 56 * 56)   // channel to work with
                        + (threadIdx.x * 56)
                        + (threadIdx.y);

    int weight_Position = filter_number * 64;

    int input_Position = (threadIdx.x * 56)
                       + (threadIdx.y);

    for(int channel = 0; channel < 64; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer5_Neurons_GPU[(channel * 56 * 56) + input_Position] * Layer5_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer5_Mean_GPU[filter_number]) / Layer5_StanDev_GPU[filter_number];
    Z = (Z * Layer5_Gamma_GPU[filter_number]) + Layer5_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0; 

    Layer6_Neurons_GPU[output_Position] = Z;
}

__global__ void executeFifthLayer_partB(double *Layer5_Neurons_GPU,
    double *Layer5_Weights_GPU,
    double *Layer6_Neurons_GPU,
    double *Layer5_Mean_GPU,
    double *Layer5_StanDev_GPU,
    double *Layer5_Gamma_GPU,
    double *Layer5_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 56 * 56)   // channel to work with
                        + (threadIdx.x * 56)
                        + (threadIdx.y + 32);

    int weight_Position = filter_number * 64;

    int input_Position = (threadIdx.x * 56)
                       + (threadIdx.y) 
                       + (32);

    for(int channel = 0; channel < 64; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer5_Neurons_GPU[(channel * 56 * 56) + input_Position] * Layer5_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer5_Mean_GPU[filter_number]) / Layer5_StanDev_GPU[filter_number];
    Z = (Z * Layer5_Gamma_GPU[filter_number]) + Layer5_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0; 

    Layer6_Neurons_GPU[output_Position] = Z;
}

__global__ void executeFifthLayer_partC(double *Layer5_Neurons_GPU,
    double *Layer5_Weights_GPU,
    double *Layer6_Neurons_GPU,
    double *Layer5_Mean_GPU,
    double *Layer5_StanDev_GPU,
    double *Layer5_Gamma_GPU,
    double *Layer5_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 56 * 56)   // channel to work with
                        + (56 * 32)
                        + (threadIdx.x * 56)
                        + (threadIdx.y);

    int weight_Position = filter_number * 64;

    int input_Position = (56 * 32)
                       + (threadIdx.x * 56)
                       + (threadIdx.y);

    for(int channel = 0; channel < 64; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer5_Neurons_GPU[(channel * 56 * 56) + input_Position] * Layer5_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer5_Mean_GPU[filter_number]) / Layer5_StanDev_GPU[filter_number];
    Z = (Z * Layer5_Gamma_GPU[filter_number]) + Layer5_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0; 

    Layer6_Neurons_GPU[output_Position] = Z;
}

__global__ void executeFifthLayer_partD(double *Layer5_Neurons_GPU,
    double *Layer5_Weights_GPU,
    double *Layer6_Neurons_GPU,
    double *Layer5_Mean_GPU,
    double *Layer5_StanDev_GPU,
    double *Layer5_Gamma_GPU,
    double *Layer5_Beta_GPU
)
{
    double product = 0.0;
    int filter_number = blockIdx.x;

    // Output position
    int output_Position = (filter_number * 56 * 56)   // channel to work with
                        + (56 * 32) 
                        + 32
                        + (threadIdx.x * 56)
                        + (threadIdx.y);

    int weight_Position = filter_number * 64;

    int input_Position = (56 * 32)
                       + (32)
                       + (threadIdx.x * 56)
                       + (threadIdx.y);

    for(int channel = 0; channel < 64; channel++)       // This is the channel loop as we have 32 channels to work with
    {
        product += (Layer5_Neurons_GPU[(channel * 56 * 56) + input_Position] * Layer5_Weights_GPU[weight_Position + channel]);
    }

    double Z = (product - Layer5_Mean_GPU[filter_number]) / Layer5_StanDev_GPU[filter_number];
    Z = (Z * Layer5_Gamma_GPU[filter_number]) + Layer5_Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
        Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
        Z = 6.0; 

    Layer6_Neurons_GPU[output_Position] = Z;
}

/*  *************************************************** FIFTH LAYER END **************************************************** */