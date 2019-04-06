/*
    Generic Function Depthwise Separable Convolution
    Parameters:
     1) Input Matrix
     2) Weight Matrix
     3) Output Matrix
     4) Input Dimension
     5) Input Block Multiplier 1
     6) Input Block Multiplier 2
     7) Input Block Offset 1
     8) Input Block Offset 2
     9) Output Dimension
    10) Output Block Multiplier 1
    11) Output Block Multiplier 2
    12) Output Block Offset 1
    13) Output Block Offset 2
    14) Weight Dimension
    15) Kernel Size
    16) Beta Matrix
    17) Gamma Matrix
    18) Standard Deviation Matrix
    19) Mean Matrix
*/

__kernel void executeGenericFunctions_DSC(__global double *Neurons_GPU,
    __global double *Weights_GPU,
    __global double *Output_GPU,
    const int input_dimension,
    const int input_block_multiplier1,
    const int input_block_multiplier2,
    const int input_block_offset1,
    const int input_block_offset2,
    const int output_dimension,
    const int output_block_multiplier1,
    const int output_block_multiplier2,
    const int output_block_offset1,
    const int output_block_offset2,
    const int weight_dimension,
    const int kernel_size,
    const int stride,
    __global double *Mean_GPU,
    __global double *StanDev_GPU,
    __global double *Gamma_GPU,
    __global double *Beta_GPU
)
{
    double product = 0.0;

    int filter_number = get_group_id(0);

    int output_Position = (filter_number * output_dimension * output_dimension)   
                        + (get_group_id(1) * output_block_multiplier1 * output_dimension + output_block_offset1)    
                        + (get_group_id(2) * output_block_multiplier2 + output_block_offset2)          
                        + (get_local_id(1) * output_dimension)
                        + (get_local_id(2));

    int weight_Position = filter_number * weight_dimension;

    int input_Position = (get_group_id(1)  * input_block_multiplier1 * input_dimension * stride)
                       + (input_block_offset1 * stride) 
                       + (get_group_id(2)  * input_block_multiplier2 * stride)
                       + input_block_offset2 * stride
                       + (get_local_id(1) * input_dimension * stride)
                       + (get_local_id(2) * stride);

    for(int row = 0 ; row < kernel_size ; row++)       
        for(int col = 0 ; col < kernel_size ; col++)
            product += (Neurons_GPU[(filter_number * input_dimension * input_dimension) + input_Position + (row * input_dimension) + col] * Weights_GPU[weight_Position + (row * kernel_size) + col]);

    double Z = (product - Mean_GPU[filter_number]) / StanDev_GPU[filter_number];
    Z = (Z * Gamma_GPU[filter_number]) + Beta_GPU[filter_number];

    if(Z < 0)
    Z = 0;

    if(Z > 6)
    Z = 6.0; 

    Output_GPU[output_Position] = Z;
}

/*
    Generic Function Pointwise Separable Convolution
    Parameters:
     1) Input Matrix
     2) Weight Matrix
     3) Output Matrix
     4) Input Dimension
     5) Input Block Multiplier 1
     6) Input Block Multiplier 2
     7) Input Block Offset 1
     8) Input Block Offset 2
     9) Output Dimension
    10) Output Block Multiplier 1
    11) Output Block Multiplier 2
    12) Output Block Offset 1
    13) Output Block Offset 2
    14) Weight Dimension
    15) Kernel Size
    16) Beta Matrix
    17) Gamma Matrix
    18) Standard Deviation Matrix
    19) Mean Matrix
*/
__kernel void executeGenericFunctions_PSC(__global double *Neurons_GPU,
    __global double *Weights_GPU,
    __global double *Output_GPU,
    const int input_dimension,
    const int input_block_multiplier1,
    const int input_block_multiplier2,
    const int input_block_offset1,
    const int input_block_offset2,
    const int output_dimension,
    const int output_block_multiplier1,
    const int output_block_multiplier2,
    const int output_block_offset1,
    const int output_block_offset2,
    const int weight_dimension,
    const int channel_size,
    const int stride,
    const int offset,
    __global double *Mean_GPU,
    __global double *StanDev_GPU,
    __global double *Gamma_GPU,
    __global double *Beta_GPU
)
{
    double product = 0.0;

    int filter_number = get_group_id(0);

    // Output position
    int output_Position = (filter_number * output_dimension * output_dimension)   // channel to work with
                        + (get_group_id(1)  * output_block_multiplier1 * output_dimension + output_block_offset1)    // Position in the grid row-wise
                        + (get_group_id(2)  * output_block_multiplier2 + output_block_offset2)          // Position in the grid column-wise
                        + (get_local_id(1) * output_dimension)
                        + (get_local_id(2));

    int weight_Position = filter_number * weight_dimension;

    int input_Position = (get_group_id(1)  * input_block_multiplier1 * input_dimension * stride)
                       + (input_block_offset1 * stride) // Position in the grid row-wise
                       + (get_group_id(2)  * input_block_multiplier2 * stride)         // Position in the grid column-wise
                       + input_block_offset2 * stride
                       + (get_local_id(1) * input_dimension * stride)
                       + (get_local_id(2) * stride);

    for(int channel = 0 ; channel < channel_size ; channel++)       // This is the Channel Loop
        product += (Neurons_GPU[(channel * input_dimension * input_dimension) + input_Position] * Weights_GPU[weight_Position + channel]);

    double Z = (product - Mean_GPU[filter_number]) / StanDev_GPU[filter_number];
    Z = (Z * Gamma_GPU[filter_number]) + Beta_GPU[filter_number];

    // ReLU Layer
    if(Z < 0)
    Z = 0; // max(0,x)

    // ReLU 6 Layer
    if(Z > 6)
    Z = 6.0; 

    Output_GPU[output_Position + offset] = Z;
}

/*  ************************************************** FIRST LAYER START ********************************************************* */
/*
    Layer 1: Normal 3D Convolution Layer
    Input: 225 * 225 * 3 (Padding of 1)
    Weight: 3 * 3 * 3 with a Stride of 2
    Output: 112 * 112 * 32 
    Next Layer is a padding layer, so padding operation is handled in this layer itself & hence
    Final Output = 114 * 114 * 32 
*/
__kernel void executeFirstLayer_CONV3D_partA(__global double *Layer1_Neurons_GPU,
                            __global double *Layer1_Weights_GPU,
                            __global double *Layer2_Neurons_GPU,
                            __global double *Layer1_Mean_GPU,
                            __global double *Layer1_StanDev_GPU,
                            __global double *Layer1_Gamma_GPU,
                            __global  double *Layer1_Beta_GPU
                        )
{
	double product = 0.0;
    int outputOffset = 115;
    int stride = 2;
    int filter_number = get_group_id(0);

    /*if(get_local_id(1) == 0 && get_local_id(2) == 0){
        printf("\ngroup x - %d", get_group_id(0));
        printf("\ngroup doing - %d", get_local_id(0));
    }*/

    // Output position
    int output_Position = (filter_number * 114 * 114)   // channel to work with
                        + (get_group_id(1)  * 32 * 114)    // Position in the grid row-wise
                        + (get_group_id(2)  * 32)          // Position in the grid column-wise
                        + (get_local_id(1) * 114)
                        + (get_local_id(2));

    int weight_Position = filter_number * 27;

    int input_Position = ((get_group_id(1)  * 32 * 225) * stride) // Position in the grid row-wise
                       + (get_group_id(2)  * 32 * stride)         // Position in the grid column-wise
                       + (get_local_id(1) * 225 * stride )
                       + (get_local_id(2) * stride);

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

__kernel void executeFirstLayer_CONV3D_partB(__global double *Layer1_Neurons_GPU,
                            __global double *Layer1_Weights_GPU,
                            __global double *Layer2_Neurons_GPU,
                            __global double *Layer1_Mean_GPU,
                            __global double *Layer1_StanDev_GPU,
                            __global double *Layer1_Gamma_GPU,
                            __global double *Layer1_Beta_GPU
                        )
{
	double product = 0.0;
    int outputOffset = 115;
    int stride = 2;
    int filter_number = get_group_id(0);

    // Output position
    int output_Position = (filter_number * 114 * 114)   // channel to work with
                        + (get_group_id(1)  * 16 * 114 + 96)  // Position in the grid row-wise and there is no column-wise position
                        + (get_local_id(1) * 114)           // Position inside the 256 (16 * 16) block
                        + (get_local_id(2));

    int weight_Position = filter_number * 27;

    int input_Position = ((get_group_id(1)  * 16 * 225) * stride) + (96 * stride) // Position in the grid row-wise and column-wise
                       + (get_local_id(1) * 225 * stride)
                       + (get_local_id(2) * stride);

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

__kernel void executeFirstLayer_CONV3D_partC(__global double *Layer1_Neurons_GPU,
                            __global double *Layer1_Weights_GPU,
                            __global double *Layer2_Neurons_GPU,
                            __global double *Layer1_Mean_GPU,
                            __global double *Layer1_StanDev_GPU,
                            __global double *Layer1_Gamma_GPU,
                            __global double *Layer1_Beta_GPU
                        )
{
	double product = 0.0;
    int outputOffset = 115;
    int stride = 2;
    int filter_number = get_group_id(0);

    // Output position
    int output_Position = (filter_number * 114 * 114)   // channel to work with
                        + (96 * 114)                    // Position in the grid row-wise as row is last
                        + (get_group_id(1)  * 16)             // Position in the grid column-wise
                        + (get_local_id(1) * 114)           // Position inside the 256 (16 * 16) block
                        + (get_local_id(2));

    int weight_Position = filter_number * 27;

    int input_Position = ((96 * 225) * stride)
                       + (get_group_id(1)  * 16 * stride)     // Position in the grid row-wise and column-wise
                       + (get_local_id(1) * 225 * stride)
                       + (get_local_id(2) * stride);

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


/*  *************************************************** TWENTYEIGHT LAYER START ************************************************** */
/*
    Layer 28: Global Average Pooling Layer
    Input: 7 * 7 * 1024 
    Weight: None 
    Output: 1 * 1 * 1024 
*/
__kernel void executeTwentyEightLayer_AvgPooling(__global double *Layer28_Neurons_GPU,
    __global double *Layer29_Neurons_GPU
)
{
    double sum = 0.0;
    int filter_number = get_local_id(1) * 32 + get_local_id(2);

    // Output position
    int output_Position = filter_number;

    int input_Position_start = filter_number * 49;
    for(int row = 0 ; row < 7 ; row++) 
        for(int col = 0 ; col < 7 ; col++) 
            sum += Layer28_Neurons_GPU[input_Position_start + row * 7 + col];
          
    sum = sum / 49;
    Layer29_Neurons_GPU[output_Position] = sum;
}
/*  *************************************************** TWENTYEIGHT LAYER END **************************************************** */

/*  *************************************************** TWENTYNINE LAYER START ************************************************** */
/*
    Layer 29: Fully Connected Layer
    Input: 1 * 1 * 1024 
    Weight: 1000 * 1024 
    Bias: 1000 
    Output: 1000 
*/
__kernel void executeTwentyNineLayer_FullyConnected(__global double *Layer29_Neurons_GPU,
    __global double *Layer30_Neurons_GPU,
    __global double *Layer29_Weights_GPU,
    __global double *Layer29_Bias_GPU
)
{
    double product = 0.0;
    int filter_number = get_global_id(1);

    // Output position
    int output_Position = filter_number;

    int weight_Position = filter_number * 1024;

    for(int channel = 0; channel < 1024 ; channel++)      
    {
        product += (Layer29_Neurons_GPU[channel] * Layer29_Weights_GPU[weight_Position + channel]);
    }         
    
    //Adding Bias to the output
    product += Layer29_Bias_GPU[filter_number];

    Layer30_Neurons_GPU[output_Position] = product;
}
/*  *************************************************** TWENTYNINE LAYER END **************************************************** */