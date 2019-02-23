#define _CRT_SECURE_NO_WARNINGS

#include "opencv2/opencv.hpp"
#include<opencv2/ml/ml.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/videoio.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <io.h>
#include <string.h>

using namespace cv;
using namespace std;

void FileScan(const char*,const char*);
int readFrame(VideoCapture&, Mat&);

int main(int argc, char** argv)
{
	if (argc == 1) {
		cout<<"usage:OCR.exe videostream [left top width height threshold area_max area_min w/h_max w/h_min ]"<<endl;
		std::cout << "For example: DriveLabel.exe d:\\sth.mp4" << std::endl;
	}
	Mat img, gray, imgThres, labels, seeLabels, stats, centroids;
	int frame_count = 0, frame_cur = 0, label_num;
	int threshold = 150;
	VideoCapture cap;
	Rect myROI(10, 500, 300, 580);
	int area_max = 3000, area_min = 1000;
	double wh_max = 0.6, wh_min = 0.3;
	if (argc > 6) {
		myROI.x = atol(argv[2]);
		myROI.y = atol(argv[3]);
		myROI.width = atol(argv[4]);
		myROI.height = atol(argv[5]);
		threshold = atol(argv[6]);
		area_max = atol(argv[7]);
		area_min = atol(argv[8]);
		wh_max = atol(argv[9]);
		wh_min = atol(argv[10]);
	}
	if (!cap.open(argv[1])) {
		time_t tm = time(NULL);
		char tmBuf[50];
		ctime_s(tmBuf, 50, &tm);
		return -1;
	}
	while (cap.isOpened()) {
		frame_cur = readFrame(cap, img);
		if (frame_cur == 0 || img.empty()) {
			time_t tm = time(NULL);
			char tmBuf[50];
			ctime_s(tmBuf, 50, &tm);
			break;
		}
		clock_t dct = clock();
		img = img(myROI);
		cvtColor(img, gray, CV_BGR2GRAY);
		cv::threshold(gray, imgThres, threshold, 255, CV_THRESH_BINARY);
		label_num = connectedComponentsWithStats(imgThres, labels, stats, centroids);

		for (int i = 0; i < label_num; i++) {
			int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
			int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
			int area = width * height;//stats.at<int>(i, cv::CC_STAT_AREA);
			if (area > area_min && area < area_max) {
				double iou = area / (double)(width*height);
				double wh = width / (double)(height);
				if (wh < wh_max&& wh> wh_min) {
					cv::Rect dect(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP), width, height);
					cv::rectangle(img, dect, cv::Scalar(0, 255, 0), 1);
					cv::Mat retImg = img(dect);
					char file[256];
					sprintf_s(file, "%d-%d.jpg", frame_count, i);
					string str = file;
					imwrite(file, retImg);
				}
			}
		}
		dct = clock() - dct;		dct = dct * 1000 / CLOCKS_PER_SEC;
		frame_count += frame_cur;
		cv::imshow("Img", img);
		cv::waitKey(40);
	}
	if (cap.isOpened()) {
		cap.release();
	}

	Mat sample;
	Mat response_array;
	char dir[200] = { 0,0,0,0, };
	//strcat(dir, argv[1]);
	strcat(dir, "./");
	char dirwithformat[200];
	int the = 100;
	//if(argc>2)
	//the	= atoi(argv[2]);
	strcpy(dirwithformat, dir);
	strcat_s(dirwithformat, "*.jpg");
	FileScan(dirwithformat,dir);
	
	FILE *file = fopen("filelist.txt", "r");
	char name[200];
	while (fscanf(file, "%[^\n]%*c", &name) != EOF) {
		//imgproc(name,the,sample, response_array);
		//Process image to extract contour
		Mat thr, gray, con;
		Mat src = imread(name, 1);
	//	if(src.cols*src.cols< 484000)
	//		resize(src, src,Size(),1.5,1.5, INTER_CUBIC);
	//	imshow("file", src);
		cvtColor(src, gray, CV_BGR2GRAY);
		cv::threshold(gray, thr, the, 255, THRESH_BINARY_INV); //Threshold to find contour
	//	imshow("thr", thr);
		thr.copyTo(con);

		// Create sample and label data
		vector< vector <Point> > contours; // Vector for storing contour
		vector< Vec4i > hierarchy;

		findContours(con, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); //Find contour

		for (int i = 0; i < contours.size(); i = hierarchy[i][0]) // iterate through first hierarchy level contours
		{
			Rect r = boundingRect(contours[i]); //Find bounding rect for each contour
			rectangle(src, Point(r.x, r.y), Point(r.x + r.width, r.y + r.height), Scalar(0, 0, 255), 2, 8, 0);
			Mat ROI = gray(r); //Crop the image
			Mat tmp1, tmp2;
			resize(ROI, tmp1, Size(10, 10), 0, 0, INTER_LINEAR); //resize to 10X10
			tmp1.copyTo(tmp2); //convert to float
			sample.push_back(tmp2.reshape(1, 1)); // Store sample data
			imshow("src", src);
			int c = waitKey(0); // Read corresponding label for contour from keyoard
			c -= 0x30;     // Convert ascii to intiger value
			if (c >= 0 && c <= 9) {
				response_array.push_back(c); // Store label to a mat
				rectangle(src, Point(r.x, r.y), Point(r.x + r.width, r.y + r.height), Scalar(0, 255, 0), 2, 8, 0);
			}
			else {
				//response_array.push_back(-1); // Store label to a mat
				sample.pop_back(1);
			//	printf("illegal\n");
				rectangle(src, Point(r.x, r.y), Point(r.x + r.width, r.y + r.height), Scalar(255, 0, 0), 2, 8, 0);
			}
		}
		imshow("src", src);
		waitKey(0);
		remove(name);
	}
	Mat response, tmp;
	tmp = response_array.reshape(1, 1); //make continuous
	tmp.copyTo(response);
	sample.copyTo(sample);
	FileStorage Data("TrainingData.yml", FileStorage::WRITE);
	Data << "data" << sample;
	Data.release();

	FileStorage Label("LabelData.yml", FileStorage::WRITE);
	Label << "label" << response;
	Label.release();
	cout << "Training and Label data created successfully....!! " << endl;
	fclose(file);
	remove("filelist.txt");
	return 0;
	
}

void FileScan(const char * dirwithformat,const char* dir)
{
	intptr_t handle;
	_finddata_t findData;

	handle = _findfirst(dirwithformat, &findData);    
	if (handle == -1)
	{
		cout << "Failed to find first file!\n";
		return;
	}

	do
	{
		if (findData.attrib & _A_SUBDIR
			&& strcmp(findData.name, ".") == 0
			&& strcmp(findData.name, "..") == 0
			)
			cout << findData.name << "\t<dir>\n";
		else {
			cout << findData.name << "\t" << findData.size << endl;
			FILE *input = fopen("filelist.txt", "a+");
			fprintf(input, "%s%s\n",dir,findData.name);
			fclose(input);
		}
	} while (_findnext(handle, &findData) == 0);    

	cout << "Done!\n";
	_findclose(handle);
}

int readFrame(VideoCapture &cap, Mat &img) {
	assert(cap.isOpened());
	int frames = 0;
	if (cap.grab()) {
		frames += 1;
		if (cap.grab()) {
			frames += 1;
			cap.retrieve(img);
		}
	}
	return frames;
}