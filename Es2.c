#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

double g_x_y_calculation(int x, int y, cv::Mat channel){
    std::vector<std::vector<int>> M = {
        {1, 2, 1},
        {3, 4, 3},
        {1, 2, 1}
    };

    double sum = 0.0f;
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++) {
            ////std::cout << "M[" << i << "][" << j << "] = " << M[i][j] << std::endl;
            sum += M[i][j] * channel.at<double>(x + i - 1, y + j - 1);
        }
    }
    ////std::cout << "Sum: " << sum << std::endl;
    return sum;
}



cv::Mat createG_x_y_Matrix(cv::Mat channel) {
    cv::Mat gxy = cv::Mat::zeros(channel.rows, channel.cols, CV_32F);

    for (int x = 0; x < channel.rows; x++) {
        for (int y = 0; y < channel.cols; y++) {
            if (x > 0 && x + 1 <= channel.rows && y > 0 && y + 1 <= channel.cols) {
                gxy.at<double>(x, y) = 1/16.0 * g_x_y_calculation(x, y, channel);
            }
        }
    }
    for (int x = 0; x < channel.rows; x++) {
        for (int y = 0; y < channel.cols; y++) {
            //std::cout << "gxy at (" << x << ", " << y << "): " << gxy.at<double>(x, y) << std::endl;
        }
    }
    return gxy;
}


int main(){
    //Read the Image
    cv::Mat image = cv::imread("image.png", cv::IMREAD_COLOR);

    // Decomposition of the image into its RGB channels
    if(image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    // Display the result
    cv::imshow("Red Channel", channels[2]);
    cv::imshow("Green Channel", channels[1]);
    cv::imshow("Blue Channel", channels[0]);
    cv::waitKey(0);


    // Check if t
    cv::Mat Redgxy = createG_x_y_Matrix(channels[2]);
    //std::cout<<"Computed the red channel:" << std::endl;
    cv::imshow("Red Channel Comp", Redgxy);
    cv::waitKey(0);

    cv::Mat Greengxy = createG_x_y_Matrix(channels[1]);
    //std::cout<<"Computed the green channel:" << std::endl;
    cv::imshow("Green Channel Comp", Greengxy);
    cv::waitKey(0);

    cv::Mat Bluegxy = createG_x_y_Matrix(channels[0]);
    //std::cout<<"Computed the blue channel:" << std::endl;
    cv::imshow("Blue Channel Comp", Bluegxy);
    cv::waitKey(0);


    //Recombine the image
    cv::Mat gxy;
    cv::merge(std::vector<cv::Mat>{Redgxy, Greengxy, Bluegxy}, gxy);
    // Display the result
    cv::imshow("Gxy Image", gxy);
    cv::waitKey(0);
    return 0;

}