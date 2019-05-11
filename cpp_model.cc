#include "src/model.h"

#include <iostream>
#include<vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

std::vector<int> get_facial_points(cv::Mat image, keras2cpp::Model *pointer_model){
    std::vector<float> flat;

    for (int i = 0; i < image.rows; i++){
        for (int j = 0; j < image.cols; j++){
            auto x = image.at<uchar>(i, j);
            flat.push_back(x/255);
        }
    }

    keras2cpp::Tensor in{96, 96, 1};

    in.data_ = flat;

    // Run prediction.
    keras2cpp::Tensor out = (*pointer_model)(in);
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

    // ==========================================================================================
    //  LOAD MODEL AND GET RESULT HERE
    keras2cpp::Model model = keras2cpp::Model::load("../example.model");    // Initialize model
    keras2cpp::Model *pointer_model = &model;   // Because the model can not be coppied, so we use the pointer to pass to function

    cv::resize(image, image, cv::Size(96, 96));                                 // The input image must be 96*96
    std::vector<int> facial_points = get_facial_points(image, pointer_model);   // using trained model to detect all points
    // ==========================================================================================

    int ratio = 5;
    cv::cvtColor(image, image, CV_GRAY2RGB);    // Convert to RGB color image
    cv::resize(image, image, cv::Size(96*ratio, 96*ratio));

    // Draw all facial points on image
    int num_points = facial_points.size()/2;
    for(int i=0; i<num_points; i++){
        int x = facial_points[i*2] * ratio;
        int y = facial_points[i*2+1] * ratio;

        std::cout << "point " << i << " at " << x << " " << y << std::endl;

        cv::Point p1 = cv::Point(x, y);
        cv::circle(image, p1, 1, cv::Scalar(0, 0, 255), 3, 8, 0);
        cv::putText(image, std::to_string(i), p1, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1, 8, false);
    }

    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Display window", image );                   // Show our image inside it.

    cv::waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}