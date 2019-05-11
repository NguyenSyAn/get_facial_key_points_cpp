#include "src/model.h"

#include <iostream>
#include<vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

std::vector<int> get_facial_points(cv::Mat image){
    std::vector<float> flat;

    for (int i = 0; i < image.rows; i++){
        for (int j = 0; j < image.cols; j++){
            auto x = image.at<uchar>(i, j);
            flat.push_back(x/255);
        }
    }

    keras2cpp::Tensor in{96, 96, 1};

    // Initialize model
    auto model = keras2cpp::Model::load("../example.model");
    in.data_ = flat;

    // Run prediction.
    keras2cpp::Tensor out = model(in);
    // out.print();

    std::vector<int> facial_points;

    for (int i=0; i<30; i++){
        int x = 48*out(i) + 48;
        facial_points.push_back(x);
    }

    return facial_points;
}

int main() {
    std::string imageName("../images.jpeg"); // by default
    cv::Mat image;

    image = cv::imread(imageName.c_str(), cv::IMREAD_GRAYSCALE);

    if( image.empty() )                      // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    cv::resize(image, image, cv::Size(96, 96)); // The input image must be 96*96
    std::vector<int> facial_points = get_facial_points(image);  // using trained model to detect all points
    cv::cvtColor(image, image, CV_GRAY2RGB);    // Convert to RGB color image

    // Draw all facial points on image
    int num_points = facial_points.size()/2;
    for(int i=0; i<num_points; i++){
        int x = facial_points[i*2];
        int y = facial_points[i*2+1];
        cv::Point p1 = cv::Point(x, y);
        cv::circle(image, p1, 1, cv::Scalar(0, 255, 0), 1, 8, 0);
    }

    cv::resize(image, image, cv::Size(500, 500));

    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Display window", image );                   // Show our image inside it.

    cv::waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}