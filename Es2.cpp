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
    //Read the Image
    cv::Mat image = cv::imread("ImageBlurred.png", cv::IMREAD_COLOR);

    // Decomposition of the image into its RGB channels
    if(image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }  


    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    const char* nThreads = std::getenv("CUDA_N_THREADS")

    std::vector<cv::Mat> gxy_channels(3);
    std::vector<cv::Mat> ch64(3);

    for (int i = 0; i < 3; i++) {
        gxy_channels[i] = cv::Mat::zeros(channels[i].rows, channels[i].cols, CV_64F);
        channel[i].convertTo(ch64[i], CV_64F);
    }
    

    float *channel, *gxy;
    dim3 threadsPerBlock(nThreads, nThreads);
    dim3 numBlocks(
        (channel[0].cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (channel[0].rows + threadsPerBlock.y - 1) / threadsPerBlock.y,
        3 // Assuming 3 channels
    );

    cudaMalloc(&channel, 3 * channel[0].rows * channel[0].cols * sizeof(float));
    cudaMalloc(&gxy, 3 * channel[0].rows * channel[0].cols * sizeof(float));

    cudaMemcpy(channel, ch64.data, 3 * channel[0].rows * channel[0].cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gxy, gxy.data, 3 * channel[0].rows * channel[0].cols * sizeof(float), cudaMemcpyHostToDevice);

    g_x_y_calculation<<<numBlocks, threadsPerBlock>>>(channel, gxy, 3, channel[0].rows, channel[0].cols);
    

    cv::Mat gxy_normalized, gxy_8U;
    cv::normalize(gxy, gxy_normalized, 255, 0, cv::NORM_MINMAX);
    gxy_normalized.convertTo(gxy_8U, CV_8U);
    
    cv::imshow("Red Channel", channels[2]);
    cv::imshow("Green Channel", channels[1]);
    cv::imshow("Blue Channel", channels[0]);
    cv::waitKey(0);
    std::cout << "Red channel " << std::endl;
    cv::Mat Redgxy = createG_x_y_Matrix(channels[2]);
    scanf("%*c"); // Wait for user input to continue
    //std::cout<<"Computed the red channel:" << std::endl;

    std::cout << "Green channel " << std::endl;
    cv::Mat Greengxy = createG_x_y_Matrix(channels[1]);
    scanf("%*c"); // Wait for user input to continue
    //std::cout<<"Computed the green channel:" << std::endl;
    std::cout << "Blue channel " << std::endl;
    cv::Mat Bluegxy = createG_x_y_Matrix(channels[0]);
    //std::cout<<"Computed the blue channel:" << std::endl;
    scanf("%*c"); // Wait for user input to continue

    

    //Recombine the image
    cv::Mat gxy;
    cv::merge(std::vector<cv::Mat>{Bluegxy, Greengxy, Redgxy}, gxy);
    cv::imwrite("./output/result.jpg", gxy);
    return 0;
}