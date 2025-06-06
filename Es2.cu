#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

/*Ho creato la funzione kernel cuda e dovrebbe andare: 
z è il canale, x è la riga e y è la colonna


Ho creato pure il numero di thread per blocco e il numero di blocchi in base alla dimensione dell'immagine.
La funzione kernel calcola il valore di gxy per ogni pixel dell'immagine in base alla formula data.
La funzione main legge l'immagine, la divide nei canali RGB, converte i canali in float e li copia nella memoria del dispositivo.
La funzione main deve ancora chiamare la funzione kernel per calcolare gxy per ogni canale e poi ricompone l'immagine finale.
La funzione main deve ancora mostrare l'immagine finale e salvarla su disco.
La funzione main deve ancora gestire la memoria del dispositivo e liberarla alla fine.
La funzione main deve ancora gestire gli errori di OpenCV e CUDA.


*/

cv::Mat createG_x_y_Matrix(int channelId, float* gxy, int C, int R){
    cv::Mat gxy_cpu, tempMat;
    cv::Mat gxy_normalized, gxy_8U;
    std::vector<float> temp(3 * R * C);

    //copy all the channels from device to host
    cudaMemcpy(temp.data(), gxy, 3 * R * C * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Choose the right channel from the temp vector
    tempMat = cv::Mat(R, C, CV_32F, temp + channelId  * R * C);
    gxy_cpu = tempMat.clone();

    // Normalize the matrix to the range [0, 255] and convert to CV_8U for visualization
    cv::normalize(gxy_cpu, gxy_normalized, 0, 255, cv::NORM_MINMAX);
    gxy_normalized.convertTo(gxy_8U, CV_8U);
    
    return gxy_8U;
}




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




int main(){
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
    std::vector<cv::Mat> gxy_channels(3);
    std::vector<cv::Mat> ch32(3);

    // Convert the channels to CV_32F for CUDA processing

    for (int i = 0; i < 3; i++) {

        // create a matrix to hold the processed data initialized to all 0s
        gxy_channels[i] = cv::Mat::zeros(channels[i].rows, channels[i].cols, CV_32F);
        
        // create a matrix to hold the channel data converted to CV_32F
        channels[i].convertTo(ch32[i], CV_32F);
    }
    
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
    cudaMemcpy(channel, ch32.data, 3 * channels[0].rows * channels[0].cols * sizeof(float), cudaMemcpyHostToDevice);
    // copy the all 0s matrix that host the processed data
    cudaMemcpy(gxy, gxy.data, 3 * channels[0].rows * channels[0].cols * sizeof(float), cudaMemcpyHostToDevice);
    
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
    
    //Save the result
    cv::imwrite("./output/result.jpg", gxyResult);
    
    return 0;
}