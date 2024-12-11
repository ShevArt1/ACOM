#include <iostream>
#include <cstdio>
#include <cmath>

#include <vector>
#include <opencv2/opencv.hpp> 

using namespace std;

cv::Mat conv(const cv::Mat& img, const cv::Mat& kernel) {
	cv::Mat result = cv::Mat::zeros(2, img.size, CV_32SC1);
	int h = img.size[0], w = img.size[1];
	for (int y = 1; y < h - 1; y++) {
		for (int x = 1; x < w - 1; x++) {
			double val = 0;
			for (int k = -1; k <= 1; k++) {
				for (int l = -1; l <= 1; l++) {
					val += img.at<uint8_t>(y + k, x + l) * kernel.at<int8_t>(1 + k, 1 + l);
				}
			}
			result.at<int32_t>(y, x) = (int32_t)val;
		}
	}
	return result;
}
bool ok(int neighbour, double a, double gr) {
	return neighbour > 0 && a >= gr;
}
uint8_t angle_num(int32_t x, int32_t y) {
	double tg = (double)y / x;
	double v1 = 0.414, v2 = 2.414;
	

	if ((x > 0 && y < 0 && tg < -v2) || (x < 0 && y < 0 && tg > v2))
		return 0;
	else if (x > 0 && y < 0 && tg < -v1)
		return 1;
	else if ((x > 0 && y < 0 && tg > -v1) || (x > 0 && y > 0 && tg < v1))
		return 2;
	else if (x > 0 && y > 0 && tg < v2)
		return 3;
	else if ((x > 0 && y > 0 && tg > v2) || (x < 0 && y > 0 && tg < -v2))
		return 4;
	else if (x < 0 && y > 0 && tg < -v1)
		return 5;
	else if (x < 0 && y < 0 && tg < v2)
		return 7;
	else
		return 6;
}

void cannyAlgorithm(cv::Mat& img, cv::Mat& grad, cv::Mat& nms, cv::Mat& edges, float lowerThresholdPercent, float upperThresholdPercent) {
	// Sobel
	const cv::Mat kerX = (cv::Mat_<int8_t>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	const cv::Mat kerY = (cv::Mat_<int8_t>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
	cv::Mat gX = conv(img, kerX);
	cv::Mat gY = conv(img, kerY);

	// Calculate gradient lengths and tangents
	cv::Mat gradLen = cv::Mat(2, img.size, CV_32FC1);
	for (int y = 0; y < gradLen.size[0]; y++) {
		for (int x = 0; x < gradLen.size[1]; x++) {
			gradLen.at<float>(y, x) = (float)sqrt(pow(gX.at<int32_t>(y, x), 2) + pow(gY.at<int32_t>(y, x), 2));
		}
	}
	double maxGradLen;
	cv::minMaxIdx(gradLen, nullptr, &maxGradLen, nullptr, nullptr);
	grad = gradLen * 255 / maxGradLen;
	grad.convertTo(grad, CV_8UC1);

	// Non-maximum suppression
	edges = cv::Mat::zeros(2, img.size, img.type());
	for (int y = 1; y < edges.size[0] - 1; y++) {
		for (int x = 1; x < edges.size[1] - 1; x++) {
			uint8_t angle = angle_num(gX.at<int32_t>(y, x), gY.at<int32_t>(y, x));
			int neighbor1[2] = { y, x };
			int neighbor2[2] = { y, x };
			if (angle == 0 || angle == 4) {
				neighbor1[0] = y - 1;
				neighbor1[1] = x;
				neighbor2[0] = y + 1;
				neighbor2[1] = x;
			}
			else if (angle == 1 || angle == 5) {
				neighbor1[0] = y - 1;
				neighbor1[1] = x + 1;
				neighbor2[0] = y + 1;
				neighbor2[1] = x - 1;
			}
			else if (angle == 2 || angle == 6) {
				neighbor1[0] = y;
				neighbor1[1] = x + 1;
				neighbor2[0] = y;
				neighbor2[1] = x - 1;
			}
			else if (angle == 3 || angle == 7) {
				neighbor1[0] = y + 1;
				neighbor1[1] = x + 1;
				neighbor2[0] = y - 1;
				neighbor2[1] = x - 1;
			}
			if (gradLen.at<float>(y, x) >= gradLen.at<float>(neighbor1[0], neighbor1[1]) && gradLen.at<float>(y, x) > gradLen.at<float>(neighbor2[0], neighbor2[1])) {
				edges.at<uint8_t>(y, x) = 255;
			}
		}
	}
	edges.copyTo(nms);

	// Double threshold filtering
	int lowLevel = int(maxGradLen * lowerThresholdPercent);
	int highLevel = int(maxGradLen * upperThresholdPercent);

	for (int y = 1; y < edges.size[0] - 1; y++) {
		for (int x = 1; x < edges.size[1] - 1; x++) {
			if (edges.at<uint8_t>(y, x) > 0) {
				if (gradLen.at<float>(y, x) < lowLevel) {
					edges.at<uint8_t>(y, x) = 0;
				}
				else if (gradLen.at<float>(y, x) < highLevel) {

					if (!(ok(edges.at<uchar>(y - 1, x - 1), gradLen.at<float>(y - 1, x - 1), highLevel) ||
					ok(edges.at<uchar>(y - 1, x), gradLen.at<float>(y - 1, x), highLevel) ||
					ok(edges.at<uchar>(y - 1, x + 1), gradLen.at<float>(y - 1, x + 1), highLevel) ||
					ok(edges.at<uchar>(y, x - 1), gradLen.at<float>(y, x - 1), highLevel) ||
					ok(edges.at<uchar>(y, x + 1), gradLen.at<float>(y, x + 1), highLevel) ||
					ok(edges.at<uchar>(y + 1, x - 1), gradLen.at<float>(y + 1, x - 1), highLevel) ||
					ok(edges.at<uchar>(y + 1, x), gradLen.at<float>(y + 1, x), highLevel) ||
					ok(edges.at<uchar>(y + 1, x + 1), gradLen.at<float>(y + 1, x + 1), highLevel))) {edges.at<uchar>(y, x) = 0;}
			
				}
			}
		}
	}
}

int main(int argc, char* argv[]) {
	const char* path = "C:/Users/sevak/PycharmProjects/pythonACOM/Images/bRelzN7xP7.jpg";
	int blurKernelSize = 5;
	float lowerThresholdPercent = 0.04;
	float upperThresholdPercent = 0.2;
	

	if (blurKernelSize < 1 || blurKernelSize % 2 == 0) {
		printf("Error: blur kernel size must be a positive odd integer (1 means no blur)\n");
		exit(1);
	}
	if (lowerThresholdPercent < 0 || lowerThresholdPercent > 1) {
		printf("Error: lower threshold percentage must be between 0 and 1\n");
		exit(1);
	}
	if (upperThresholdPercent < 0 || upperThresholdPercent > 1) {
		printf("Error: upper threshold percentage must be between 0 and 1\n");
		exit(1);
	}
	

	cv::Mat img;
	img = cv::imread(path, cv::IMREAD_GRAYSCALE);
	if (img.empty()) {
		printf("Error: image not loaded\n");
		exit(1);
	}
	if (blurKernelSize > 1) {
		GaussianBlur(img, img, cv::Size(blurKernelSize, blurKernelSize), 0.0);
	}
	

	cv::Mat grad, nms, edges;
	cannyAlgorithm(img, grad, nms, edges, lowerThresholdPercent, upperThresholdPercent);

	cv::imshow("Preprocessed Image", img);
	cv::imshow("Gradients", grad);
	cv::imshow("NMS result", nms);
	char* canny_title = new char[128];
	std::snprintf(canny_title, 128, "Edges (image size: %dx%d, blur kernel size: %d, lower threshold: %g%%, upper threshold: %g%%", img.size[1], img.size[0], blurKernelSize, lowerThresholdPercent * 100, upperThresholdPercent * 100);
	cv::imshow(canny_title, edges);
	delete[] canny_title;
	cv::waitKey(0);
}
