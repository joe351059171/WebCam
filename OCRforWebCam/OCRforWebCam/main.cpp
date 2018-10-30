// DriveLabel.cpp : Defines the entry point for the console application.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>
#include <ctime>
#include <iostream>
#include <vector>
#include <cstdarg>
#include <string>
using namespace std;
using namespace cv;

static int num = 0;

int writeLog(char *fmt, ...)
{
	int n = 0;
	FILE *fp = NULL;
	va_list args = NULL;
	char sprint_buf[1024] = { 0 };
	if (fopen_s(&fp, "log.txt", "a+") != 0)
		return -1;
	va_start(args, fmt);
	fprintf(fp, "-------------------------\n");
	n = vfprintf(fp, fmt, args);
	fprintf(fp, "\n-------------------------\n");
	va_end(args);
	fclose(fp);
	return n;
}

int readFrame(cv::VideoCapture &cap, cv::Mat &img) {
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

void imgprocess(string filename) {

	Mat thr, gray, con;
	Mat src = imread(filename);
	cvtColor(src, gray, CV_BGR2GRAY);
	threshold(gray, thr, 100, 255, THRESH_BINARY_INV); // Threshold to create input
	thr.copyTo(con);


	// Read stored sample and label for training
	Mat sample;
	Mat response, tmp;
	FileStorage Data("TrainingData.yml", FileStorage::READ); // Read traing data to a Mat
	Data["data"] >> sample;
	Data.release();

	FileStorage Label("LabelData.yml", FileStorage::READ); // Read label data to a Mat
	Label["label"] >> response;
	Label.release();

	Ptr<ml::KNearest>  knn(ml::KNearest::create());
	//ml::KNearest knn();
	knn->train(sample, ml::ROW_SAMPLE, response); // Train with sample and responses
	//cout << "Training compleated.....!!" << endl;

	vector< vector <Point> > contours; // Vector for storing contour
	vector< Vec4i > hierarchy;

	//Create input sample by contour finding and cropping
	findContours(con, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	Mat dst(src.rows, src.cols, CV_8UC3, Scalar::all(0));

	for (int i = 0; i < contours.size(); i = hierarchy[i][0]) // iterate through each contour for first hierarchy level .
	{
		Rect r = boundingRect(contours[i]);
		Mat ROI = thr(r);
		Mat tmp1, tmp2;
		resize(ROI, tmp1, Size(10, 10), 0, 0, INTER_LINEAR);
		tmp1.convertTo(tmp2, CV_32FC1);
		Mat response;
		float p = knn->findNearest(tmp2.reshape(1, 1), 1, response);//If only a single input vector is passed, 
												//all output matrices are optional and the predicted value is returned by the method.
		char name[4];
		sprintf_s(name, "%d", (int)p);
		putText(dst, name, Point(r.x, r.y + r.height), 0, 1, Scalar(0, 255, 0), 1, 8);
		printf("%d", int(p));
	}
	printf("\n");

	filename = filename + "x.jpg";
	imwrite(filename, dst);
}

// prog.exe rtsp://.... oblique.model [gpuID threshold 10 10 10 10 101 101 101 101]
int main(int argc, char** argv)
{
	if (argc == 1) {
		std::cout << "DriveLabel.exe videostream [left top width height threshold]" << std::endl;
		std::cout << "For example: DriveLabel.exe d:\\sth.mp4" << std::endl;
	}
	cv::Mat img, gray, imgThres, labels, seeLabels, stats, centroids;
	int frame_count = 0, frame_cur = 0, label_num;
	int threshold = 150;
	cv::VideoCapture cap;
	cv::VideoWriter cvw;
	cv::Rect myROI(10, 500, 300, 580);
	if (argc > 6) {
		myROI.x = atol(argv[2]);
		myROI.y = atol(argv[3]);
		myROI.width = atol(argv[4]);
		myROI.height = atol(argv[5]);
		threshold = atol(argv[6]);
	}
	//cvw.open("test.mp4", CV_FOURCC('M', 'J', 'P', 'G'), 25, cv::Size(300, 580));
	if (!cap.open(argv[1])) {
		//cout << "video stream failed! please check it!" << endl;
		time_t tm = time(NULL);
		char tmBuf[50];
		ctime_s(tmBuf, 50, &tm);
		//writeLog("%s \n %s video stream failed! please check it!", argv[1], tmBuf);
		return -1;
	}
	while (cap.isOpened()) {
		frame_cur = readFrame(cap, img);
		if (frame_cur == 0 || img.empty()) {
			time_t tm = time(NULL);
			char tmBuf[50];
			ctime_s(tmBuf, 50, &tm);
			//writeLog("%s \n %s read failed! frames:%d, %d", argv[1], tmBuf, frame_cur, img.empty());
			break;
		}
		clock_t dct = clock();
		img = img(myROI);
		cv::cvtColor(img, gray, CV_BGR2GRAY);
		//gray = gray(myROI);
		cv::threshold(gray, imgThres, threshold, 255, CV_THRESH_BINARY);
		//cv::adaptiveThreshold(gray, imgThres, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 125, 5);
		//label_num = cv::connectedComponents(imgThres, labels);
		label_num = cv::connectedComponentsWithStats(imgThres, labels, stats, centroids);
		//cv::normalize(labels, seeLabels, 0, 255, cv::NORM_MINMAX, CV_8U);

		for (int i = 0; i < label_num; i++) {
			int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
			int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
			int area = width * height;//stats.at<int>(i, cv::CC_STAT_AREA);
			if (area > 1000 && area < 5000) {
				double iou = area / (double)(width*height);
				//cout << "IOU" << iou << endl;//½»²¢±È
				double wh = width / (double)(height);
				//cout << "W/H" << wh << endl;
				if (wh < 0.6&& wh> 0.3) {
					//	cout << "(" << stats.at<int>(i, cv::CC_STAT_LEFT) << " x " << stats.at<int>(i, cv::CC_STAT_TOP) << ")(" << width << " x " << height << ")" << endl;
					cv::Rect dect(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP), width, height);
					cv::rectangle(img, dect, cv::Scalar(0, 255, 0), 1);
					cv::Mat retImg = img(dect);
					char file[256];
					sprintf_s(file, "%d-%d.jpg", frame_count, i);
					string str = file;
					imwrite(file, retImg);
				//	imgprocess(str);
				}
			}
		}
		dct = clock() - dct;		dct = dct * 1000 / CLOCKS_PER_SEC;
		// Print the detection results. 
		//cout << "Time: " << dct << "Frame: " << frame_count << " " << "score " << label_num << endl;
		frame_count += frame_cur;
		cvw << img;
		cv::imshow("Img", img);

		//if (cv::waitKey(40) >= 0)
			//break;
		cv::waitKey(0);
	}
	if (cap.isOpened()) {
		cap.release();
	}
	cvw.release();
	return 0;
}