#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
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

__global__ void g_x_y_calculation(float *channel, float *gxy, int CH, int R, int CO, int nThreads){
    const int HALO_SIZE = 1; // Size of the halo for a 3x3 kernel
    __shared__ float tile[nThreads + 2*HALO_SIZE][nThreads + 2*HALO_SIZE]; // Shared memory tile with padding for Halo exchange

    // Calculate thread indices
    int tx = threadIdx.x; // Thread's x-index within the block (0 to 15)
    int ty = threadIdx.y; // Thread's y-index within the block (0 to 15)
    int z = blockIdx.z;   // Channel index

    // Calculate the top-left corner of the tile this thread will help load
    int x_start_in = blockIdx.x * nThreads - HALO_SIZE;
    int y_start_in = blockIdx.y * nThreads - HALO_SIZE;

    // Each thread loads one pixel into the shared memory tile.
    // We calculate the source coordinates in the global 'channel' buffer.
    int x_in = x_start_in + tx;
    int y_in = y_start_in + ty;

    // Cooperatively load data from global to shared memory
    // Check image boundaries before loading. If out of bounds, pad with 0.
    if (x_in >= 0 && x_in < CO && y_in >= 0 && y_in < R) {
        tile[ty][tx] = channel[z * (R * CO) + y_in * CO + x_in];
    } else {
        tile[ty][tx] = 0.0f;
    }
    // To fill the entire 18x18 tile with a 16x16 block, some threads must load more than one pixel.
    // Here, we handle the right and bottom halo edges.
    // Load right halo columns
    if (tx < 2 * HALO_SIZE) {
        x_in = x_start_in + nThreads + tx;
        if (x_in >= 0 && x_in < CO && y_in >= 0 && y_in < R) {
            tile[ty][tx + nThreads] = channel[z * (R * CO) + y_in * CO + x_in];
        } else {
            tile[ty][tx + nThreads] = 0.0f;
        }
    }
    // Load bottom halo rows
    if (ty < 2 * HALO_SIZE) {
        y_in = y_start_in + nThreads + ty;
        if (y_in >= 0 && y_in < R && x_in >= 0 && x_in < CO) {
            tile[ty + nThreads][tx] = channel[z * (R * CO) + y_in * CO + x_in];
        } else {
            tile[ty + nThreads][tx] = 0.0f;
        }
    }

    // Synchronize threads
    // This barrier ensures that the entire tile is loaded into shared memory
    // before any thread proceeds to the computation phase.
    __syncthreads();

    // Compute the convolution from shared memory
    // Calculate the global output coordinates for this thread
    int x_out = blockIdx.x * nThreads + tx;
    int y_out = blockIdx.y * nThreads + ty;

    // Ensure the output pixel is within the image bounds
    if (x_out < CO && y_out < R) {
        // Handle image borders (set to 0)
        if (x_out == 0 || x_out >= CO - 1 || y_out == 0 || y_out >= R - 1) {
            gxy[z * (R * CO) + y_out * CO + x_out] = 0.0f;
        } else {
            // The kernel weights (could be moved to __constant__ memory for another small boost)
            int W[3][3] = {{1, 2, 1}, {3, 4, 3}, {1, 2, 1}};
            float sum = 0.0f;

            // The thread's position in the shared memory tile corresponds to the center of its 3x3 window
            int shared_y = ty + HALO_SIZE;
            int shared_x = tx + HALO_SIZE;

            // Loop over the 3x3 kernel, reading from the fast shared memory tile
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    sum += (1.0f / 16.0f) * W[i][j] * tile[shared_y + i - HALO_SIZE][shared_x + j - HALO_SIZE];
                }
            }
            // Write the final result back to global memory
            gxy[z * (R * CO) + y_out * CO + x_out] = sum;
        }
    }
}


/* ----------------------------------------------------------------------------------------------------------------------------------------------
    
                                                                Function to process the image

    ------------------------------------------------------------------------------------------------------------------------------------------------*/



cv::Mat GetResult(std::string imagePath) {
    /* ----------------------------------------------------------------------------------------------------------------------------------------------
    
                                                                        OpenCV Setup

    ------------------------------------------------------------------------------------------------------------------------------------------------*/
    
    //Read the Image
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

    // Decomposition of the image into its RGB channels
    if(image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        exit(EXIT_FAILURE);
    }  

    std::vector<cv::Mat> channels;
    cv::split(image, channels);


    /* ----------------------------------------------------------------------------------------------------------------------------------------------
    
                                                                        CUDA Setup

    ------------------------------------------------------------------------------------------------------------------------------------------------*/
    // start the timer
    auto start = std::chrono::high_resolution_clock::now();
    // get the number of threads from the environment variable
    const int nThreads = std::atoi(std::getenv("CUDA_N_THREADS"));
    std::cout << "Thread used: " << nThreads * nThreads << std::endl;

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
    g_x_y_calculation<<<numBlocks, threadsPerBlock>>>(channel, gxy, 3, channels[0].rows, channels[0].cols, nThreads);
    

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

    //end the timer
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the elapsed time in milliseconds
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time: " << elapsed << " ms" << std::endl;
    //Recombine the image
    cv::Mat gxyResult;
    cv::merge(std::vector<cv::Mat>{Bluegxy, Greengxy, Redgxy}, gxyResult);
    
    return gxyResult;
}

/* ----------------------------------------------------------------------------------------------------------------------------------------------
    
                                                                    Main Function

    ------------------------------------------------------------------------------------------------------------------------------------------------*/

int main(int argc, char** argv) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Dispositivo " << i << ": " << prop.name << std::endl;
        std::cout << "  Multiprocessori: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max thread per blocco: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max thread per multiprocessore: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Dimensioni massime di un blocco: "
                  << prop.maxThreadsDim[0] << " x "
                  << prop.maxThreadsDim[1] << " x "
                  << prop.maxThreadsDim[2] << std::endl;
        std::cout << "  Dimensioni massime della griglia: "
                  << prop.maxGridSize[0] << " x "
                  << prop.maxGridSize[1] << " x "
                  << prop.maxGridSize[2] << std::endl;
        std::cout << "-------------------------------\n";
    }


    for (int i = 1; i < argc; i+=2) {
        std::cout << "Processing image: " << argv[i] << std::endl;
        cv::Mat result = GetResult(argv[i]);
        std::cout << "Saving result to: " << argv[i+1] << std::endl;
        cv::imwrite(argv[i+1], result);
    }
/*
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
*/
    return 0;
}
