#include <cuda_runtime.h>
#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

#define THREADS_PER_BLOCK 16

// CUDA kernel for Sobel edge detection
__global__ void sobelEdgeDetectionKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && y >= 1 && x < width - 1 && y < height - 1) {
        int gx = -1 * input[(y - 1) * width + (x - 1)] + 1 * input[(y - 1) * width + (x + 1)] +
                 -2 * input[y * width + (x - 1)] + 2 * input[y * width + (x + 1)] +
                 -1 * input[(y + 1) * width + (x - 1)] + 1 * input[(y + 1) * width + (x + 1)];

        int gy = -1 * input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x] - 1 * input[(y - 1) * width + (x + 1)] +
                  1 * input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + 1 * input[(y + 1) * width + (x + 1)];

        output[y * width + x] = min(255, (int)sqrtf((float)(gx * gx + gy * gy)));
    }
}

void detectFiducialMarker(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Image could not be loaded!" << std::endl;
        return;
    }

    int width = image.cols;
    int height = image.rows;

    // Allocate memory on the device
    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, width * height * sizeof(unsigned char));

    // Copy data to the device
    cudaMemcpy(d_input, image.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocksPerGrid((width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 
                       (height + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // Launch Sobel edge detection kernel
    sobelEdgeDetectionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cv::Mat edges(height, width, CV_8UC1);
    cudaMemcpy(edges.data, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Find contours on the CPU
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.02 * cv::arcLength(contour, true), true);

        // Check if the contour is a quadrilateral (fiducial marker)
        if (approx.size() == 4) { // Quadrilateral detected
            cv::Rect boundingBox = cv::boundingRect(approx);
            cv::Mat roi = image(boundingBox);

            // You can add additional logic for fiducial marker recognition here
            cv::polylines(image, approx, true, cv::Scalar(255, 0, 0), 2);
        }
    }

    // Display the result
    cv::imshow("Detected Fiducial Markers", image);
    cv::waitKey(0);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    std::string imagePath = "tags_example.png";
    detectFiducialMarker(imagePath);
    return 0;
}
