
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

void  listFiles(const char*);

int main(int argc, char** argv)
{
	char dir[200];
	cout << "Enter a directory (ends with \'\\\'): ";
	cin.getline(dir, 200);

	strcat_s(dir, "*.jpg");        // ��Ҫ������Ŀ¼�����ͨ���
	listFiles(dir);

	return 0;
	
}

void listFiles(const char * dir)
{
	intptr_t handle;
	_finddata_t findData;

	handle = _findfirst(dir, &findData);    // ����Ŀ¼�еĵ�һ���ļ�
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
			)    // �Ƿ�����Ŀ¼���Ҳ�Ϊ"."��".."
			cout << findData.name << "\t<dir>\n";
		else
			cout << findData.name << "\t" << findData.size << endl;
	} while (_findnext(handle, &findData) == 0);    // ����Ŀ¼�е���һ���ļ�

	cout << "Done!\n";
	_findclose(handle);    // �ر��������
}