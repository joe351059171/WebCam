#define _CRT_SECURE_NO_WARNINGS

#include "opencv2/opencv.hpp"
#include<opencv2/ml/ml.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/videoio.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <io.h>
#include <string.h>

using namespace cv;
using namespace std;

void FileScan(const char*,const char*);
//void imgproc(const char*,int, Mat, Mat);

int main(int argc, char** argv)
{
	Mat sample;
	Mat response_array;
	char dir[200] = { 0,0,0,0, };
	strcat(dir, argv[1]);
//	printf("%s\n",argv[1]);
	char dirwithformat[200];
	int the = atoi(argv[2]);
	//cin.getline(dir, 200);
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
		//if(file.cols*file.cols< 484000)
		//	resize(file, src,Size(),8,8, INTER_CUBIC);
		imshow("file", src);
		cvtColor(src, gray, CV_BGR2GRAY);
		threshold(gray, thr, the, 255, THRESH_BINARY_INV); //Threshold to find contour
		imshow("thr", thr);
		thr.copyTo(con);

		// Create sample and label data
		vector< vector <Point> > contours; // Vector for storing contour
		vector< Vec4i > hierarchy;

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
			if (c >= 0 && c <= 9) {
				response_array.push_back(c); // Store label to a mat
				rectangle(src, Point(r.x, r.y), Point(r.x + r.width, r.y + r.height), Scalar(0, 255, 0), 2, 8, 0);
			}
		}
		imshow("src", src);
		waitKey(0);

	}
	Mat response, tmp;
	tmp = response_array.reshape(1, 1); //make continuous
	tmp.convertTo(response, CV_32FC1); // Convert  to float
	sample = sample.reshape(1, 1);
	sample.convertTo(sample, CV_32FC1);
	FileStorage Data("TrainingData.yml", FileStorage::WRITE); // !!!!!!!!!!!!!!!!!!!!!!!!
	Data << "data" << sample;
	Data.release();

	FileStorage Label("LabelData.yml", FileStorage::WRITE); //!!!!!!!!!!!!!!!!!!!!!!!!
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

	handle = _findfirst(dirwithformat, &findData);    // 查找目录中的第一个文件
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
			)    // 是否是子目录并且不为"."或".."
			cout << findData.name << "\t<dir>\n";
		else {
			cout << findData.name << "\t" << findData.size << endl;
			FILE *input = fopen("filelist.txt", "a+");
			fprintf(input, "%s%s\n",dir,findData.name);
			fclose(input);
		}
	} while (_findnext(handle, &findData) == 0);    // 查找目录中的下一个文件

	cout << "Done!\n";
	_findclose(handle);    // 关闭搜索句柄
}

/*void imgproc(const char* filename,int thresh) {
	//Process image to extract contour
	Mat thr, gray, con;
	Mat src = imread(filename, 1);
	//if(file.cols*file.cols< 484000)
	//	resize(file, src,Size(),8,8, INTER_CUBIC);
	imshow("file",src);
	cvtColor(src, gray, CV_BGR2GRAY);
	threshold(gray, thr, thresh, 255, THRESH_BINARY_INV); //Threshold to find contour
	imshow("thr", thr);
	thr.copyTo(con);

	// Create sample and label data
	vector< vector <Point> > contours; // Vector for storing contour
	vector< Vec4i > hierarchy;

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
			if (c >= 0 && c <= 9) {
				response_array.push_back(c); // Store label to a mat
				rectangle(src, Point(r.x, r.y), Point(r.x + r.width, r.y + r.height), Scalar(0, 255, 0), 2, 8, 0);
			}
	}
	imshow("src", src);
	waitKey(0);
}*/