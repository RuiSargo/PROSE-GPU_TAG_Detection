#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <vector>

void detectFiducialMarker(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Image could not be loaded!" << std::endl;
        return;
    }

    // Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Step 1: Edge detection using Canny (optional, as ArUco handles it internally)
    // cv::Mat edges;
    // cv::Canny(gray, edges, 100, 200); // Optional, ArUco already handles detection

    // Step 2: Create the ArUco dictionary and detector parameters
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();

    // Step 3: Detect markers in the image
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    cv::aruco::detectMarkers(gray, dictionary, corners, ids, parameters);

    // Step 4: If markers are found, draw the bounding boxes around them
    if (!ids.empty()) {
        // Draw detected markers and their ids
        cv::aruco::drawDetectedMarkers(image, corners, ids);

        // Optionally, draw the pose if you have the camera matrix and distortion coefficients
        // cv::aruco::estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);
    } else {
        std::cout << "No markers detected." << std::endl;
    }

    // Display results
    cv::imshow("Detected Fiducial Markers", image);
    cv::waitKey(0);
}

int main() {
    std::string imagePath = "tags_example2.png"; // Provide the correct path to your image
    detectFiducialMarker(imagePath);
    return 0;
}
