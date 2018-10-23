
#include "opencv2/opencv.hpp"
#include<opencv2/ml/ml.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/videoio.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	int label_num = 0;
	Mat src = imread(argv[1], 0);
	Mat resource;
	resize(src, resource, Size(), 5, 5, INTER_CUBIC);
	/*imshow("origin", resource);
	cvtColor(resource, resource, CV_RGB2GRAY);
	imshow("src", resource);
	Mat thre;
	threshold(resource, thre, 120, 255, 0);
	imshow("thre", thre);
	Mat labels, stats, centroids;
	label_num = connectedComponentsWithStats(thre, labels, stats, centroids);
	printf("%d\n", label_num);
	vector<cv::Vec3b> colors(label_num + 1);
	colors[0] = Vec3b(0, 0, 0); // background pixels remain black.
	for (int i = 1; i < label_num; i++) {
		colors[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
	}
	Mat img_color = Mat::zeros(resource.size(), CV_8UC3);
	for (int y = 0; y < img_color.rows; y++)
		for (int x = 0; x < img_color.cols; x++)
		{
			int label = labels.at<int>(y, x);
			CV_Assert(0 <= label && label <= label_num);
			img_color.at<cv::Vec3b>(y, x) = colors[label];
		}
	cv::imshow("Labeled map", img_color);*/
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat gray;
	threshold(resource, gray, 100, 255,0);
	imshow("grayscale", gray);
	findContours(gray, contours, hierarchy, 1,CV_CHAIN_APPROX_NONE);
	Mat imageContours = Mat::zeros(resource.size(), CV_8UC1);
	drawContours(imageContours,contours,-1,Scalar::all(255));
	imshow("src", resource);
	imshow("contours", imageContours);
	waitKey();
	return 0;
}