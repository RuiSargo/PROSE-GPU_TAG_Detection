#include <iostream>
#include <opencv2/opencv.hpp>
#include "lib/SquareFinder.cpp"
#include "lib/CornerRefinement.cpp"
#include "lib/ArucoMarker.cpp"
#include "lib/ArucoDetector.cpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // Check for image path argument
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }

    // Load the input image
    string imagePath = argv[1];
    Mat image = imread(imagePath);

    if (image.empty()) {
        cerr << "Error: Could not load image at " << imagePath << endl;
        return -1;
    }

    // Detect Aruco markers in the image
    vector<ArucoMarker> markers = ArucoDetector::getMarkers(image);

    // Display detected marker information
    if (markers.empty()) {
        cout << "No Aruco markers detected." << endl;
    } else {
        cout << "Detected " << markers.size() << " Aruco marker(s):" << endl;

        for (size_t i = 0; i < markers.size(); ++i) {
            cout << "Marker " << i + 1 << ":" << endl;
            cout << "  ID: " << markers[i].id << endl;
            cout << "  Corners:" << endl;

            for (const auto& point : markers[i].projected) {
                cout << "    (" << point.x << ", " << point.y << ")" << endl;
            }
        }
    }

    // Display the image with marker corners drawn
    for (const auto& marker : markers) {
        for (size_t i = 0; i < 4; ++i) {
            line(image, marker.projected[i], marker.projected[(i + 1) % 4], Scalar(0, 255, 0), 2);
        }
    }

    namedWindow("Detected Markers", WINDOW_AUTOSIZE);
    imshow("Detected Markers", image);

    waitKey(0);

    return 0;
}
