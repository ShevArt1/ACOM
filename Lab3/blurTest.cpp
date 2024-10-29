#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp> 

using namespace std;

vector < vector < double>> gaussianKernel(int size, double sigma) {
	vector < vector < double>> kernel(size, vector<double>(size));
	int a = size / 2;
	double sum = 0.0; 

	
	for (int x = -a; x <= a; ++x) {
		for (int y = -a; y <= a; ++y) {
			kernel[x + a][y + a] = (1.0 / (2.0 * 3.14 * sigma * sigma)) * exp(-(x * x + y * y) / (2 * sigma * sigma));
			sum += kernel[x + a][y + a]; 
		}
	}

	
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			kernel[i][j] /= sum;
		}
	}

	return kernel;
}


cv::Mat applyGaussianBlur(const cv::Mat& image, const vector < vector < double>>& kernel) {
	int kernel_size = kernel.size();
	int a = kernel_size / 2;

	cv::Mat blurred_image = image.clone(); 
	for (int i = a; i < image.rows - a; ++i) {
		for (int j = a; j < image.cols - a; ++j) {
			double pixel_value = 0.0;

			for (int kx = -a; kx <= a; ++kx) {
				for (int ky = -a; ky <= a; ++ky) {
					pixel_value += image.at<uchar>(i + kx, j + ky) * kernel[kx + a][ky + a];
				}
			}
			blurred_image.at<uchar>(i, j) = static_cast<uchar>(pixel_value); 
		}
	}
	return blurred_image;
}

int main() {
	cv::Mat image = cv::imread("C:/Users/sevak/PycharmProjects/pythonACOM/Images/bRelzN7xP7.jpg", cv::IMREAD_GRAYSCALE);

	if (image.empty()) {
		cout << "Не удалось загрузить изображение!" << endl;
		return -1;
	}

	//vector < vector < double>> kernel = gaussianKernel(11, 5.0);
	vector < vector < double>> kernel = gaussianKernel(27, 15.0);

	cv::Mat blurred_image = applyGaussianBlur(image, kernel);

	cv::imwrite("blurred_image.jpg", blurred_image);

	cv::imshow("Original Image", image);
	cv::imshow("Blurred Image", blurred_image);
	cv::waitKey(0); 

	return 0;
}
