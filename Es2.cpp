#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>

double g_x_y_calculation(int x, int y, cv::Mat channel){
    std::vector<std::vector<int>> W = {
        {1, 2, 1},
        {3, 4, 3},
        {1, 2, 1}
    };

    double sum = 0.0f;
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++) {
            ////std::cout << "M[" << i << "][" << j << "] = " << M[i][j] << std::endl;
            sum += (1.0/16.0) *  W[i][j] * channel.at<double>(x + i - 1, y + j - 1);
        }
    }
    ////std::cout << "Sum: " << sum << std::endl;
    return sum;
}



cv::Mat createG_x_y_Matrix(cv::Mat channel) {
    cv::Mat gxy = cv::Mat::zeros(channel.rows, channel.cols, CV_64F);
    
    cv::Mat ch64;
    channel.convertTo(ch64, CV_64F);
    for (int x = 1; x < channel.rows - 1; x++) {
        for (int y = 1; y < channel.cols - 1; y++) {

            gxy.at<double>(x, y) = g_x_y_calculation(x, y, ch64);

        }
    }

    cv::Mat gxy_normalized, gxy_8U;
    cv::normalize(gxy, gxy_normalized, 255, 0, cv::NORM_MINMAX);
    gxy_normalized.convertTo(gxy_8U, CV_8U);
    /*
    for (int x = 0; x < channel.rows; x++) {
        for (int y = 0; y < channel.cols; y++) {
            std::cout << "gxy at (" << x << ", " << y << "): " << gxy.at<double>(x, y) << std::endl;
        }
    }
    */
    return gxy_8U;
}


int main(int argc, char** argv) {
    //Read the Image

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);

    // Decomposition of the image into its RGB channels
    if(image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }


    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    // start the timer
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat Redgxy = createG_x_y_Matrix(channels[2]);

    cv::Mat Greengxy = createG_x_y_Matrix(channels[1]);

    cv::Mat Bluegxy = createG_x_y_Matrix(channels[0]);

    // stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken to process the image: " << elapsed.count() << " seconds" << std::endl;
    //Recombine the image
    cv::Mat gxy;
    cv::merge(std::vector<cv::Mat>{Bluegxy, Greengxy, Redgxy}, gxy);


    cv::imwrite(argv[2], gxy);


    return 0;
}