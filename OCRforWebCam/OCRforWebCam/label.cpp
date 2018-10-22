#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{

	//Process image to extract contour
	Mat thr, gray, con;
	Mat src = imread(argv[1], 1);
	cvtColor(src, gray, CV_BGR2GRAY);
	threshold(gray, thr, 200, 255, THRESH_BINARY_INV); //Threshold to find contour
	thr.copyTo(con);

	// Create sample and label data
	vector< vector <Point> > contours; // Vector for storing contour
	vector< Vec4i > hierarchy;
	Mat sample;
	Mat response_array;
	findContours(con, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); //Find contour

	for (int i = 0; i < contours.size(); i = hierarchy[i][0]) // iterate through first hierarchy level contours
	{
		Rect r = boundingRect(contours[i]); //Find bounding rect for each contour
		rectangle(src, Point(r.x, r.y), Point(r.x + r.width, r.y + r.height), Scalar(0, 0, 255), 2, 8, 0);
		Mat ROI = thr(r); //Crop the image
		Mat tmp1, tmp2;
		resize(ROI, tmp1, Size(10, 10), 0, 0, INTER_LINEAR); //resize to 10X10
		tmp1.convertTo(tmp2, CV_32FC1); //convert to float
		sample.push_back(tmp2.reshape(1, 1)); // Store  sample data
		imshow("src", src);
		int c = waitKey(0); // Read corresponding label for contour from keyoard
		c -= 0x30;     // Convert ascii to intiger value
		response_array.push_back(c); // Store label to a mat
		rectangle(src, Point(r.x, r.y), Point(r.x + r.width, r.y + r.height), Scalar(0, 255, 0), 2, 8, 0);
	}

	// Store the data to file
	Mat response, tmp;
	tmp = response_array.reshape(1, 1); //make continuous
	tmp.convertTo(response, CV_32FC1); // Convert  to float

	FileStorage Data("TrainingData.yml", FileStorage::WRITE); // Store the sample data in a file
	Data << "data" << sample;
	Data.release();

	FileStorage Label("LabelData.yml", FileStorage::WRITE); // Store the label data in a file
	Label << "label" << response;
	Label.release();
	cout << "Training and Label data created successfully....!! " << endl;

	imshow("src", src);
	waitKey(0);

	return 0;
}