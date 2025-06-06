#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <filesystem>
#include <cuda_runtime.h>

/*----------------------------------------------------------------------------------------------------------------------------------------

                                             Function to get the image in order to visualize it 
                                                                
------------------------------------------------------------------------------------------------------------------------------------------*/
cv::Mat createG_x_y_Matrix(int channelId, float* gxy, int C, int R){
    cv::Mat gxy_cpu, tempMat;
    cv::Mat gxy_normalized, gxy_8U;
    std::vector<float> temp(3 * R * C);

    //copy all the channels from device to host
    cudaMemcpy(temp.data(), gxy, 3 * R * C * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Choose the right channel from the temp vector
    tempMat = cv::Mat(R, C, CV_32F, temp.data() + channelId  * R * C);
    gxy_cpu = tempMat.clone();

    // Normalize the matrix to the range [0, 255] and convert to CV_8U for visualization
    cv::normalize(gxy_cpu, gxy_normalized, 0, 255, cv::NORM_MINMAX);
    gxy_normalized.convertTo(gxy_8U, CV_8U);
    
    return gxy_8U;
}


/* ----------------------------------------------------------------------------------------------------------------------------------------

                                                                CUDA Kernel for gxy calculation

------------------------------------------------------------------------------------------------------------------------------------------*/

__global__ void g_x_y_calculation(float *channel, float *gxy, int CH, int R, int CO){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z; //channel index


    int idx = z * (R * CO) + x * (CO) + y;
    if (x > R || y > CO || z > CH) return;

    if (x <= 0 || x >= R - 1 || y <= 0 || y >= CO - 1) {
        gxy[idx] = 0.0;
        return;
    }

    int W[3][3] = {
        {1, 2, 1},
        {3, 4, 3},
        {1, 2, 1}
    };

    float sum = 0.0f;

    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++) {
            sum += (1.0/16.0) *  W[i][j] * channel[idx + (i - 1) * CO + (j - 1)];
        }
    }
    gxy[idx] = sum;
}


/* ----------------------------------------------------------------------------------------------------------------------------------------------
    
                                                                Function to process the image

    ------------------------------------------------------------------------------------------------------------------------------------------------*/



cv::Mat GetResult(std::string imagePath) {
    /* ----------------------------------------------------------------------------------------------------------------------------------------------
    
                                                                        OpenCV Setup

    ------------------------------------------------------------------------------------------------------------------------------------------------*/
    
    //Read the Image
    cv::Mat image = cv::imread("ImageBlurred.png", cv::IMREAD_COLOR);

    // Decomposition of the image into its RGB channels
    if(image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }  

    std::vector<cv::Mat> channels;
    cv::split(image, channels);


    /* ----------------------------------------------------------------------------------------------------------------------------------------------
    
                                                                        CUDA Setup

    ------------------------------------------------------------------------------------------------------------------------------------------------*/
    
    // get the number of threads from the environment variable
    const int nThreads = std::atoi(std::getenv("CUDA_N_THREADS"));

    // instatiate cv matrix from which you will get the data to be stored in CUDA memory
    std::vector<cv::Mat> ch32(3);
    std::vector<float> channel_host(3 * channels[0].rows * channels[0].cols);    

    // Convert the channels to CV_32F for CUDA processing and Flat the channels into a single vector
    for (int i = 0; i < 3; i++) {        
        // create a matrix to hold the channel data converted to CV_32F
        channels[i].convertTo(ch32[i], CV_32F);
        std::memcpy(
            channel_host.data() + i * channels[0].rows * channels[0].cols,
            ch32[i].ptr<float>(),
            channels[0].rows * channels[0].cols * sizeof(float)
        );
    }
    // Initialize the gxy vector with zeros
    std::vector<float> gxy_channels(3 * channels[0].rows * channels[0].cols, 0.0f);
    
    // Allocate memory for the channels (data as input) and gxy (data processed) on the device
    float *channel, *gxy;

    // Define the number of threads per block and the number of blocks
    dim3 threadsPerBlock(nThreads, nThreads);
    dim3 numBlocks(
        (channels[0].cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (channels[0].rows + threadsPerBlock.y - 1) / threadsPerBlock.y,
        3 // 3 channels (R, G, B)
    );
    
    // Allocate memory on the device for the channels and gxy
    cudaMalloc(&channel, 3 * channels[0].rows * channels[0].cols * sizeof(float));
    cudaMalloc(&gxy, 3 * channels[0].rows * channels[0].cols * sizeof(float));

    // copy the data from the image
    cudaMemcpy(channel, channel_host.data(), 3 * channels[0].rows * channels[0].cols * sizeof(float), cudaMemcpyHostToDevice);
    // copy the all 0s matrix that host the processed data
    cudaMemcpy(gxy, gxy_channels.data(), 3 * channels[0].rows * channels[0].cols * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch the kernel to calculate gxy for each channel
    g_x_y_calculation<<<numBlocks, threadsPerBlock>>>(channel, gxy, 3, channels[0].rows, channels[0].cols);
    

    /*-----------------------------------------------------------------------------------------------------------------------------------------------
                                                                        
                                                                    Display and Save Results

    ------------------------------------------------------------------------------------------------------------------------------------------------*/

    /* Local debugging
    cv::imshow("Red Channel", channels[2]);
    cv::imshow("Green Channel", channels[1]);
    cv::imshow("Blue Channel", channels[0]);
    cv::waitKey(0);
    */


    //                                                     Create the gxy matrices for each channel
    
    //Red Channel
    std::cout << "Red channel " << std::endl;
    cv::Mat Redgxy = createG_x_y_Matrix(2, gxy, channels[2].cols, channels[2].rows);

    //Green Channel
    std::cout << "Green channel " << std::endl;
    cv::Mat Greengxy = createG_x_y_Matrix(1, gxy, channels[1].cols, channels[1].rows);
    
    //Blue Channel
    std::cout << "Blue channel " << std::endl;
    cv::Mat Bluegxy = createG_x_y_Matrix(0, gxy, channels[0].cols, channels[0].rows);
      
    // Cuda free the memory
    cudaFree(channel);
    cudaFree(gxy);


    //Recombine the image
    cv::Mat gxyResult;
    cv::merge(std::vector<cv::Mat>{Bluegxy, Greengxy, Redgxy}, gxyResult);
    
    return gxyResult;
}



namespace fs = std::filesystem;

/* ----------------------------------------------------------------------------------------------------------------------------------------------
    
                                                                    Main Function

    ------------------------------------------------------------------------------------------------------------------------------------------------*/

int main() {
    fs::path input_dir = "./input"; // Directory containing the input images
    fs::path output_dir = "./output"; // Directory to save the processed images
    // Create the output directory if it doesn't exist
    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }
    for (const auto& subdir_entry : fs::directory_iterator(input_dir)){
        if (subdir_entry.is_directory()) {
            fs::path subdir_path = subdir_entry.path();
            for (const auto& file_entry : fs::directory_iterator(subdir_path)) {
                if (file_entry.is_regular_file() && file_entry.path().extension() == ".jpg") {
                    std::cout << "Processing image: " << file_entry.path() << std::endl;
                    cv::Mat result = GetResult(file_entry.path().string());
                    // Save the result to the output directory
                    fs::path output_path = output_dir / subdir_path.filename() / file_entry.path().filename();
                    // Create the subdirectory in the output directory if it doesn't exist
                    fs::create_directories(output_path.parent_path());
                    cv::imwrite(output_path.string(), result);
                    std::cout << "Saved result to: " << output_path << std::endl;
                }
            }
        }
    }
    return 0;
}